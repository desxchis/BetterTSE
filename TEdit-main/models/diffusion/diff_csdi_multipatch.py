import copy
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from linear_attention_transformer import LinearAttentionTransformer
import numpy as np

from models.conditioning import StrengthProjector


def _format_scalar_key(value):
    return f"{float(value):.4f}"

def get_torch_trans(heads=8, layers=1, channels=64):
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=channels, nhead=heads, dim_feedforward=64, activation="gelu", batch_first=True
    )
    return nn.TransformerEncoder(encoder_layer, num_layers=layers)

def get_linear_trans(heads=8,layers=1,channels=64,localheads=0,localwindow=0):

  return LinearAttentionTransformer(
        dim = channels,
        depth = layers,
        heads = heads,
        max_seq_len = 256,
        n_local_attn_heads = 0, 
        local_attn_window_size = 0,
    )

def Conv1d_with_init(in_channels, out_channels, kernel_size):
    layer = nn.Conv1d(in_channels, out_channels, kernel_size)
    nn.init.kaiming_normal_(layer.weight)
    return layer


class DiffusionEmbedding(nn.Module):
    def __init__(self, num_steps, embedding_dim=128, projection_dim=None):
        super().__init__()
        if projection_dim is None:
            projection_dim = embedding_dim
        self.register_buffer(
            "embedding",
            self._build_embedding(num_steps, embedding_dim / 2),
            persistent=False,
        )
        self.projection1 = nn.Linear(embedding_dim, projection_dim)
        self.projection2 = nn.Linear(projection_dim, projection_dim)

    def forward(self, diffusion_step):
        x = self.embedding[diffusion_step]
        x = self.projection1(x)
        x = F.silu(x)
        x = self.projection2(x)
        x = F.silu(x)
        return x

    def _build_embedding(self, num_steps, dim=64):
        steps = torch.arange(num_steps).unsqueeze(1)  # (T,1)
        frequencies = 10.0 ** (torch.arange(dim) / (dim - 1) * 4.0).unsqueeze(0)  # (1,dim)
        table = steps * frequencies  # (T,dim)
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)  # (T,dim*2)
        return table

class ResidualBlock(nn.Module):
    _diag_enabled = False
    _diag_records = []
    _diag_max_records = 64
    _diag_stage_records = []
    _diag_stage_max_records = 1024
    _flip_beta_sign_inference = False

    @classmethod
    def enable_diagnostics(
        cls,
        enabled: bool = True,
        max_records: int = 64,
        stage_max_records: int = 1024,
    ) -> None:
        cls._diag_enabled = bool(enabled)
        cls._diag_max_records = max(1, int(max_records))
        cls._diag_stage_max_records = max(1, int(stage_max_records))
        cls._diag_records = []
        cls._diag_stage_records = []

    @classmethod
    def disable_diagnostics(cls) -> None:
        cls._diag_enabled = False

    @classmethod
    def set_flip_beta_sign_inference(cls, enabled: bool = False) -> None:
        cls._flip_beta_sign_inference = bool(enabled)

    @classmethod
    def consume_diagnostics(cls):
        records = copy.deepcopy(cls._diag_records) + copy.deepcopy(cls._diag_stage_records)
        cls._diag_records = []
        cls._diag_stage_records = []
        return records

    def __init__(
        self,
        side_dim,
        attr_dim,
        channels,
        diffusion_embedding_dim,
        nheads,
        is_linear=False,
        is_attr_proj=False,
        strength_cond_dim=0,
        strength_mode="modulation_residual",
        strength_gain_multiplier=4.0,
    ):
        super().__init__()
        self.diffusion_projection = nn.Linear(diffusion_embedding_dim, channels)
        if is_attr_proj:
            self.attr_projection = nn.Linear(attr_dim, channels)
        else:
            self.attr_projection = nn.Identity()
        self.base_modulation = nn.Sequential(
            nn.Linear(channels + channels, channels),
            nn.SiLU(),
            nn.Linear(channels, 2 * channels),
        )
        nn.init.zeros_(self.base_modulation[-1].weight)
        nn.init.zeros_(self.base_modulation[-1].bias)
        self.strength_mode = strength_mode
        self.strength_gain_multiplier = float(strength_gain_multiplier)
        self.strength_modulation = None
        if strength_cond_dim > 0:
            self.strength_modulation = nn.Sequential(
                nn.Linear(strength_cond_dim, channels),
                nn.SiLU(),
                nn.Linear(channels, 2 * channels),
            )
            nn.init.normal_(self.strength_modulation[-1].weight, mean=0.0, std=1.0e-3)
            nn.init.normal_(self.strength_modulation[-1].bias, mean=0.0, std=1.0e-3)
        self.side_projection = Conv1d_with_init(side_dim, 2 * channels, 1)
        self.mid_projection = Conv1d_with_init(channels, 2 * channels, 1)
        self.output_projection = Conv1d_with_init(channels, 2 * channels, 1)

        self.is_linear = is_linear
        if is_linear:
            self.time_layer = get_linear_trans(heads=nheads,layers=1,channels=channels)
            self.feature_layer = get_linear_trans(heads=nheads,layers=1,channels=channels)
        else:
            self.time_layer = get_torch_trans(heads=nheads, layers=1, channels=channels)
            self.feature_layer = get_torch_trans(heads=nheads, layers=1, channels=channels)


    def forward_time(self, y, base_shape, attention_mask=None):
        B, channel, K, L = base_shape
        if L == 1:
            return y
        y = y.reshape(B, channel, K, L).permute(0, 2, 1, 3).reshape(B * K, channel, L)
        if self.is_linear:
            y = self.time_layer(y.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            y = self.time_layer(y.permute(0, 2, 1), mask=attention_mask).permute(0, 2, 1)
        y = y.reshape(B, K, channel, L).permute(0, 2, 1, 3).reshape(B, channel, K * L)
        return y

    def forward_feature(self, y, base_shape, attention_mask=None):
        B, channel, K, L = base_shape
        if K == 1:
            return y
        y = y.reshape(B, channel, K, L).permute(0, 3, 1, 2).reshape(B * L, channel, K)
        if self.is_linear:
            y = self.feature_layer(y.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            y = self.time_layer(y.permute(0, 2, 1), mask=attention_mask).permute(0, 2, 1)
        y = y.reshape(B, L, channel, K).permute(0, 2, 3, 1).reshape(B, channel, K * L)
        return y

    def _record_modulation_diagnostics(
        self,
        gamma_orig,
        beta_orig,
        delta_gamma,
        delta_beta,
        strength_cond,
        strength_label=None,
        strength_scalar=None,
        gamma_final=None,
        beta_final=None,
        gamma_scale=None,
        beta_scale=None,
    ):
        if not self.__class__._diag_enabled:
            return
        if len(self.__class__._diag_records) >= self.__class__._diag_max_records:
            return
        with torch.no_grad():
            gamma_orig_cpu = gamma_orig.detach().cpu().float()
            beta_orig_cpu = beta_orig.detach().cpu().float()
            delta_gamma_cpu = delta_gamma.detach().cpu().float()
            delta_beta_cpu = delta_beta.detach().cpu().float()
            strength_cond_cpu = strength_cond.detach().cpu().float()
            gamma_orig_norm = gamma_orig_cpu.norm(dim=-1)
            beta_orig_norm = beta_orig_cpu.norm(dim=-1)
            delta_gamma_norm = delta_gamma_cpu.norm(dim=-1)
            delta_beta_norm = delta_beta_cpu.norm(dim=-1)
            eps = 1.0e-8
            record = {
                "layer_id": id(self),
                "gamma_orig_norm_mean": float(gamma_orig_norm.mean().item()),
                "beta_orig_norm_mean": float(beta_orig_norm.mean().item()),
                "delta_gamma_norm_mean": float(delta_gamma_norm.mean().item()),
                "delta_beta_norm_mean": float(delta_beta_norm.mean().item()),
                "delta_gamma_abs_mean": float(delta_gamma_cpu.abs().mean().item()),
                "delta_beta_abs_mean": float(delta_beta_cpu.abs().mean().item()),
                "delta_gamma_over_base_mean": float((delta_gamma_norm / torch.clamp(gamma_orig_norm, min=eps)).mean().item()),
                "delta_beta_over_base_mean": float((delta_beta_norm / torch.clamp(beta_orig_norm, min=eps)).mean().item()),
                "strength_cond_norm_mean": float(strength_cond_cpu.norm(dim=-1).mean().item()),
            }
            if gamma_scale is not None:
                gamma_scale_cpu = gamma_scale.detach().cpu().float()
                record["gamma_scale_mean"] = float(gamma_scale_cpu.mean().item())
                record["gamma_scale_abs_mean"] = float(gamma_scale_cpu.abs().mean().item())
            if beta_scale is not None:
                beta_scale_cpu = beta_scale.detach().cpu().float()
                record["beta_scale_mean"] = float(beta_scale_cpu.mean().item())
                record["beta_scale_abs_mean"] = float(beta_scale_cpu.abs().mean().item())
            if gamma_final is not None:
                gamma_final_cpu = gamma_final.detach().cpu().float()
                record["gamma_final_norm_mean"] = float(gamma_final_cpu.norm(dim=-1).mean().item())
                record["gamma_final_abs_mean"] = float(gamma_final_cpu.abs().mean().item())
            if beta_final is not None:
                beta_final_cpu = beta_final.detach().cpu().float()
                record["beta_final_norm_mean"] = float(beta_final_cpu.norm(dim=-1).mean().item())
                record["beta_final_abs_mean"] = float(beta_final_cpu.abs().mean().item())
            stage_tensors = {
                "strength_cond": strength_cond_cpu,
                "gamma_orig": gamma_orig_cpu,
                "beta_orig": beta_orig_cpu,
                "delta_gamma": delta_gamma_cpu,
                "delta_beta": delta_beta_cpu,
            }
            if gamma_scale is not None:
                stage_tensors["gamma_scale"] = gamma_scale_cpu
            if beta_scale is not None:
                stage_tensors["beta_scale"] = beta_scale_cpu
            if gamma_final is not None:
                stage_tensors["gamma_final"] = gamma_final_cpu
            if beta_final is not None:
                stage_tensors["beta_final"] = beta_final_cpu
            for stage_name, tensor in stage_tensors.items():
                record[f"{stage_name}_norm_mean"] = float(tensor.norm(dim=-1).mean().item())
                record[f"{stage_name}_abs_mean"] = float(tensor.abs().mean().item())
            labels = None if strength_label is None else strength_label.detach().cpu().long().view(-1)
            if labels is not None:
                for name, tensor in (("delta_gamma", delta_gamma_cpu), ("delta_beta", delta_beta_cpu)):
                    means = {}
                    for label in sorted(set(labels.tolist())):
                        mask = labels == label
                        if int(mask.sum().item()) == 0:
                            continue
                        mean_vec = tensor[mask].mean(dim=0)
                        means[str(label)] = {
                            "count": int(mask.sum().item()),
                            "norm": float(mean_vec.norm().item()),
                            "mean_abs": float(mean_vec.abs().mean().item()),
                        }
                    pairwise = {}
                    unique_labels = sorted(set(labels.tolist()))
                    for idx, left in enumerate(unique_labels):
                        for right in unique_labels[idx + 1:]:
                            if str(left) not in means or str(right) not in means:
                                continue
                            left_mean = tensor[labels == left].mean(dim=0)
                            right_mean = tensor[labels == right].mean(dim=0)
                            diff = left_mean - right_mean
                            pairwise[f"{left}_{right}"] = float(diff.norm().item())
                            pairwise[f"{left}_{right}_mean_abs"] = float(diff.abs().mean().item())
                    record[f"{name}_mean_by_strength"] = means
                    record[f"{name}_pairwise_l2"] = pairwise
            scalar_values = None if strength_scalar is None else strength_scalar.detach().cpu().float().view(-1)
            if scalar_values is not None:
                for name, tensor in (("delta_gamma", delta_gamma_cpu), ("delta_beta", delta_beta_cpu), ("gamma_orig", gamma_orig_cpu), ("beta_orig", beta_orig_cpu)):
                    means = {}
                    pairwise = {}
                    unique_scalars = sorted({float(value) for value in scalar_values.tolist()})
                    for scalar_value in unique_scalars:
                        mask = torch.isclose(scalar_values, scalar_values.new_tensor(float(scalar_value)), atol=1.0e-6, rtol=0.0)
                        if int(mask.sum().item()) == 0:
                            continue
                        mean_vec = tensor[mask].mean(dim=0)
                        scalar_key = _format_scalar_key(scalar_value)
                        means[scalar_key] = {
                            "count": int(mask.sum().item()),
                            "norm": float(mean_vec.norm().item()),
                            "mean_abs": float(mean_vec.abs().mean().item()),
                            "mean": float(mean_vec.mean().item()),
                            "positive_frac": float((mean_vec > 0).float().mean().item()),
                            "negative_frac": float((mean_vec < 0).float().mean().item()),
                        }
                    for idx, left in enumerate(unique_scalars):
                        for right in unique_scalars[idx + 1:]:
                            left_mask = torch.isclose(scalar_values, scalar_values.new_tensor(float(left)), atol=1.0e-6, rtol=0.0)
                            right_mask = torch.isclose(scalar_values, scalar_values.new_tensor(float(right)), atol=1.0e-6, rtol=0.0)
                            if int(left_mask.sum().item()) == 0 or int(right_mask.sum().item()) == 0:
                                continue
                            left_mean = tensor[left_mask].mean(dim=0)
                            right_mean = tensor[right_mask].mean(dim=0)
                            diff = left_mean - right_mean
                            pair_key = f"{_format_scalar_key(left)}_{_format_scalar_key(right)}"
                            pairwise[pair_key] = float(diff.norm().item())
                            pairwise[f"{pair_key}_mean_abs"] = float(diff.abs().mean().item())
                            pairwise[f"{pair_key}_mean"] = float(diff.mean().item())
                    record[f"{name}_mean_by_scalar"] = means
                    record[f"{name}_pairwise_l2_by_scalar"] = pairwise
                for name, tensor in (("gamma_final", gamma_final_cpu), ("beta_final", beta_final_cpu)):
                    means = {}
                    pairwise = {}
                    unique_scalars = sorted({float(value) for value in scalar_values.tolist()})
                    for scalar_value in unique_scalars:
                        mask = torch.isclose(scalar_values, scalar_values.new_tensor(float(scalar_value)), atol=1.0e-6, rtol=0.0)
                        if int(mask.sum().item()) == 0:
                            continue
                        mean_vec = tensor[mask].mean(dim=0)
                        scalar_key = _format_scalar_key(scalar_value)
                        means[scalar_key] = {
                            "count": int(mask.sum().item()),
                            "norm": float(mean_vec.norm().item()),
                            "mean_abs": float(mean_vec.abs().mean().item()),
                            "mean": float(mean_vec.mean().item()),
                            "positive_frac": float((mean_vec > 0).float().mean().item()),
                            "negative_frac": float((mean_vec < 0).float().mean().item()),
                        }
                    for idx, left in enumerate(unique_scalars):
                        for right in unique_scalars[idx + 1:]:
                            left_mask = torch.isclose(scalar_values, scalar_values.new_tensor(float(left)), atol=1.0e-6, rtol=0.0)
                            right_mask = torch.isclose(scalar_values, scalar_values.new_tensor(float(right)), atol=1.0e-6, rtol=0.0)
                            if int(left_mask.sum().item()) == 0 or int(right_mask.sum().item()) == 0:
                                continue
                            diff = tensor[left_mask].mean(dim=0) - tensor[right_mask].mean(dim=0)
                            pair_key = f"{_format_scalar_key(left)}_{_format_scalar_key(right)}"
                            pairwise[pair_key] = float(diff.norm().item())
                            pairwise[f"{pair_key}_mean_abs"] = float(diff.abs().mean().item())
                            pairwise[f"{pair_key}_mean"] = float(diff.mean().item())
                    record[f"{name}_mean_by_scalar"] = means
                    record[f"{name}_pairwise_l2_by_scalar"] = pairwise
            self.__class__._diag_records.append(record)

    def _record_response_diagnostics(self, stage_name, tensor, strength_label=None, strength_scalar=None):
        if not self.__class__._diag_enabled:
            return
        if len(self.__class__._diag_stage_records) >= self.__class__._diag_stage_max_records:
            return
        with torch.no_grad():
            tensor_cpu = tensor.detach().cpu().float().reshape(tensor.shape[0], -1)
            record = {
                "stage_name": stage_name,
                "stage_norm_mean": float(tensor_cpu.norm(dim=-1).mean().item()),
                "stage_abs_mean": float(tensor_cpu.abs().mean().item()),
                "stage_feature_std_mean": float(tensor_cpu.std(dim=-1).mean().item()),
            }
            labels = None if strength_label is None else strength_label.detach().cpu().long().view(-1)
            if labels is not None:
                means = {}
                pairwise = {}
                unique_labels = sorted(set(labels.tolist()))
                for label in unique_labels:
                    mask = labels == label
                    if int(mask.sum().item()) == 0:
                        continue
                    mean_vec = tensor_cpu[mask].mean(dim=0)
                    means[str(label)] = {
                        "count": int(mask.sum().item()),
                        "norm": float(mean_vec.norm().item()),
                        "mean_abs": float(mean_vec.abs().mean().item()),
                    }
                for idx, left in enumerate(unique_labels):
                    for right in unique_labels[idx + 1:]:
                        if str(left) not in means or str(right) not in means:
                            continue
                        diff = tensor_cpu[labels == left].mean(dim=0) - tensor_cpu[labels == right].mean(dim=0)
                        pairwise[f"{left}_{right}"] = float(diff.norm().item())
                        pairwise[f"{left}_{right}_mean_abs"] = float(diff.abs().mean().item())
                record["stage_mean_by_strength"] = means
                record["stage_pairwise_l2"] = pairwise
            scalar_values = None if strength_scalar is None else strength_scalar.detach().cpu().float().view(-1)
            if scalar_values is not None:
                means = {}
                pairwise = {}
                unique_scalars = sorted({float(value) for value in scalar_values.tolist()})
                for scalar_value in unique_scalars:
                    mask = torch.isclose(scalar_values, scalar_values.new_tensor(float(scalar_value)), atol=1.0e-6, rtol=0.0)
                    if int(mask.sum().item()) == 0:
                        continue
                    mean_vec = tensor_cpu[mask].mean(dim=0)
                    scalar_key = _format_scalar_key(scalar_value)
                    means[scalar_key] = {
                        "count": int(mask.sum().item()),
                        "norm": float(mean_vec.norm().item()),
                        "mean_abs": float(mean_vec.abs().mean().item()),
                    }
                for idx, left in enumerate(unique_scalars):
                    for right in unique_scalars[idx + 1:]:
                        left_mask = torch.isclose(scalar_values, scalar_values.new_tensor(float(left)), atol=1.0e-6, rtol=0.0)
                        right_mask = torch.isclose(scalar_values, scalar_values.new_tensor(float(right)), atol=1.0e-6, rtol=0.0)
                        if int(left_mask.sum().item()) == 0 or int(right_mask.sum().item()) == 0:
                            continue
                        diff = tensor_cpu[left_mask].mean(dim=0) - tensor_cpu[right_mask].mean(dim=0)
                        pair_key = f"{_format_scalar_key(left)}_{_format_scalar_key(right)}"
                        pairwise[pair_key] = float(diff.norm().item())
                        pairwise[f"{pair_key}_mean_abs"] = float(diff.abs().mean().item())
                record["stage_mean_by_scalar"] = means
                record["stage_pairwise_l2_by_scalar"] = pairwise
            self.__class__._diag_stage_records.append(record)

    def _record_forward_response_diagnostics(self, stage_values, strength_label=None, strength_scalar=None):
        for stage_name, tensor in stage_values.items():
            if tensor is None:
                continue
            self._record_response_diagnostics(stage_name, tensor, strength_label=strength_label, strength_scalar=strength_scalar)

    def _apply_mid_gate_filter(self, y, strength_label=None, strength_scalar=None):
        gate, filter = torch.chunk(y, 2, dim=1)
        gated = torch.sigmoid(gate) * torch.tanh(filter)
        self._record_forward_response_diagnostics(
            {
                "pre_mid_gatefilter": y,
                "mid_gate_logits": gate,
                "mid_filter_logits": filter,
                "mid_gate_activation": torch.sigmoid(gate),
                "mid_filter_activation": torch.tanh(filter),
                "mid_gated_output": gated,
            },
            strength_label=strength_label,
            strength_scalar=strength_scalar,
        )
        return gated

    def _run_post_modulation_stack(self, y, base_shape, attention_mask=None, strength_label=None, strength_scalar=None):
        self._record_forward_response_diagnostics(
            {"post_modulation": y},
            strength_label=strength_label,
            strength_scalar=strength_scalar,
        )
        y = self.forward_time(y, base_shape, attention_mask)
        self._record_forward_response_diagnostics(
            {"post_time": y},
            strength_label=strength_label,
            strength_scalar=strength_scalar,
        )
        y = self.forward_feature(y, base_shape, attention_mask)
        self._record_forward_response_diagnostics(
            {"post_feature": y},
            strength_label=strength_label,
            strength_scalar=strength_scalar,
        )
        return y

    def _run_output_head(self, y, side_emb, base_shape, strength_label=None, strength_scalar=None):
        B, channel, K, L = base_shape
        y = y.reshape(B, channel, K * L)
        y = self.mid_projection(y)
        self._record_forward_response_diagnostics(
            {"post_mid_projection": y},
            strength_label=strength_label,
            strength_scalar=strength_scalar,
        )
        _, side_dim, _, _ = side_emb.shape
        side_emb = side_emb.reshape(B, side_dim, K * L)
        side_emb = self.side_projection(side_emb)
        combined = y + side_emb
        self._record_forward_response_diagnostics(
            {
                "side_projection": side_emb,
                "post_side_add": combined,
            },
            strength_label=strength_label,
            strength_scalar=strength_scalar,
        )
        gated = self._apply_mid_gate_filter(combined, strength_label=strength_label, strength_scalar=strength_scalar)
        y = self.output_projection(gated)
        self._record_forward_response_diagnostics(
            {"post_output_projection": y},
            strength_label=strength_label,
            strength_scalar=strength_scalar,
        )
        residual, skip = torch.chunk(y, 2, dim=1)
        self._record_forward_response_diagnostics(
            {
                "residual_branch": residual,
                "skip_branch": skip,
            },
            strength_label=strength_label,
            strength_scalar=strength_scalar,
        )
        return residual.reshape(base_shape), skip.reshape(base_shape)

    def _record_residual_merge(self, x, residual, strength_label=None, strength_scalar=None):
        merged = (x + residual) / math.sqrt(2.0)
        self._record_forward_response_diagnostics(
            {"residual_merge": merged},
            strength_label=strength_label,
            strength_scalar=strength_scalar,
        )
        return merged

    def _forward_with_strength_diagnostics(self, x, side_emb, base_shape, y, attention_mask=None, strength_label=None, strength_scalar=None):
        y = self._run_post_modulation_stack(
            y,
            base_shape,
            attention_mask=attention_mask,
            strength_label=strength_label,
            strength_scalar=strength_scalar,
        )
        residual, skip = self._run_output_head(
            y,
            side_emb,
            base_shape,
            strength_label=strength_label,
            strength_scalar=strength_scalar,
        )
        return self._record_residual_merge(x, residual, strength_label=strength_label, strength_scalar=strength_scalar), skip

    def _compute_modulation(self, attr_emb, diffusion_cond, strength_cond, strength_label=None, strength_scalar=None):
        pooled_attr = torch.mean(attr_emb, dim=1)
        base_input = torch.cat([diffusion_cond, pooled_attr], dim=-1)
        gamma_orig, beta_orig = torch.chunk(self.base_modulation(base_input), 2, dim=-1)
        if self.strength_modulation is None or strength_cond is None or self.strength_mode != "modulation_residual":
            return gamma_orig, beta_orig
        delta_gamma, delta_beta = torch.chunk(self.strength_modulation(strength_cond), 2, dim=-1)
        delta_gamma = delta_gamma * self.strength_gain_multiplier
        delta_beta = delta_beta * self.strength_gain_multiplier
        strength_ablation_mode = os.environ.get("TEDIT_STRENGTH_ABLATION_MODE", "").strip().lower()
        if strength_ablation_mode == "gamma_only":
            delta_beta = torch.zeros_like(delta_beta)
        elif strength_ablation_mode == "beta_only":
            delta_gamma = torch.zeros_like(delta_gamma)
        if strength_scalar is not None:
            strength_scalar = strength_scalar.float().view(-1, 1)
            delta_gamma = delta_gamma * strength_scalar
            delta_beta = delta_beta * strength_scalar
        flip_beta_sign = bool(self.__class__._flip_beta_sign_inference) and not self.training
        gamma_final = gamma_orig + delta_gamma
        beta_final = beta_orig - delta_beta if flip_beta_sign else beta_orig + delta_beta
        self._record_modulation_diagnostics(
            gamma_orig,
            beta_orig,
            delta_gamma,
            delta_beta,
            strength_cond,
            strength_label=strength_label,
            strength_scalar=strength_scalar,
            gamma_final=gamma_final,
            beta_final=beta_final,
        )
        return gamma_final, beta_final

    def forward(self, x, side_emb, attr_emb, diffusion_emb, attention_mask=None, strength_cond=None, strength_label=None, strength_scalar=None):
        B, channel, K, L = x.shape
        base_shape = x.shape
        
        # concatenate attr_emb and x
        attr_emb = self.attr_projection(attr_emb)  # (B,N,channel)
        attr_emb = attr_emb.permute(0, 2, 1).unsqueeze(2).expand(-1, -1, K, -1)  # (B,channel,K,N)
        N = attr_emb.shape[-1]

        a_x = torch.concat((attr_emb, x), dim=-1)  # (B,channel,K,N+L)
        base_shape_a = a_x.shape
        a_x = a_x.reshape(B, channel, -1)  # (B,channel,K*(N+L))

        diffusion_cond = self.diffusion_projection(diffusion_emb)
        gamma, beta = self._compute_modulation(
            attr_emb,
            diffusion_cond,
            strength_cond,
            strength_label=strength_label,
            strength_scalar=strength_scalar,
        )
        diffusion_emb = diffusion_cond.unsqueeze(-1)  # (B,channel,1)
        gamma = gamma.unsqueeze(-1)
        beta = beta.unsqueeze(-1)
        y = (a_x + diffusion_emb) * (1.0 + gamma) + beta
        y = self._run_post_modulation_stack(
            y,
            base_shape_a,
            attention_mask=attention_mask,
            strength_label=strength_label,
            strength_scalar=strength_scalar,
        )
        y = y.reshape(base_shape_a)[..., N:]
        residual, skip = self._run_output_head(
            y,
            side_emb,
            base_shape,
            strength_label=strength_label,
            strength_scalar=strength_scalar,
        )
        x = x.reshape(base_shape)
        return self._record_residual_merge(x, residual, strength_label=strength_label, strength_scalar=strength_scalar), skip

class TsPatchEmbedding(nn.Module):
    def __init__(self, L_patch_len, channels, d_model, dropout):
        super(TsPatchEmbedding, self).__init__()
        self.L_patch_len = L_patch_len
        self.padding_patch_layer = nn.ReplicationPad2d((0, L_patch_len, 0, 0))
        self.value_embedding = nn.Sequential(
            nn.Linear(L_patch_len*channels, d_model),
            nn.ReLU(),
        )

    def forward(self, x_in):
        # B, C, n_var, L = x.shape
        if x_in.shape[-1] % self.L_patch_len:
            x = self.padding_patch_layer(x_in)
        else:
            x = x_in
        x = x.unfold(dimension=3, size=self.L_patch_len, step=self.L_patch_len) # (B, C, n_var, Nl, Pl)
        B, C, n_var, Nl, Pl = x.shape
        x = x.permute(0, 3, 4, 2, 1).contiguous().reshape(B, 1, Nl, Pl*n_var*C)
        x = self.value_embedding(x) # (B, 1, Nl, D)
        x = x.permute(0, 3, 1, 2).contiguous() # (B, D, 1, Nl)
        return x

class CondPatchEmbedding(nn.Module):
    def __init__(self, L_patch_len, channels, d_model, dropout):
        super(CondPatchEmbedding, self).__init__()
        self.L_patch_len = L_patch_len
        self.padding_patch_layer = nn.ReplicationPad2d((0, L_patch_len, 0, 0))
        self.value_embedding = nn.Sequential(
            nn.Linear(L_patch_len*channels, d_model),
        )

    def forward(self, x_in):
        if x_in.shape[-1] % self.L_patch_len:
            x = self.padding_patch_layer(x_in)
        else:
            x = x_in
        x = x.unfold(dimension=3, size=self.L_patch_len, step=self.L_patch_len) # (B, C, n_var, Nl, Pl)
        B, C, n_var, Nl, Pl = x.shape
        x = x.permute(0, 3, 4, 2, 1).contiguous().reshape(B, 1, Nl, Pl*n_var*C)
        x = self.value_embedding(x) # (B, 1, Nl, D)
        x = x.permute(0, 3, 1, 2).contiguous() # (B, D, 1, Nl)
        return x

class PatchDecoder(nn.Module):
    def __init__(self, L_patch_len, n_var, d_model, channels):
        super().__init__()
        self.L_patch_len = L_patch_len
        self.channels = channels
        self.n_var = n_var
        self.linear = nn.Linear(d_model, L_patch_len*n_var*channels)

    def forward(self, x):
        B, D, _, Nl = x.shape
        x = x.permute(0, 2, 3, 1).contiguous()# (B, 1, Nl, D)
        x = self.linear(x) #(B, 1, Nl, Pl*n_var*C)
        x = x.reshape(B, Nl, self.L_patch_len, self.n_var, self.channels).permute(0, 4, 3, 1, 2).contiguous() #(B, C, K, Nl, Pl)
        x = x.reshape(B, self.channels, self.n_var, Nl*self.L_patch_len)
        return x

class Diff_CSDI_Patch(nn.Module):
    def __init__(self, config, inputdim=2):
        super().__init__()
        self.channels = config["channels"]

        self.diffusion_embedding = DiffusionEmbedding(
            num_steps=config["num_steps"],
            embedding_dim=config["diffusion_embedding_dim"],
        )

        self.output_projection1 = Conv1d_with_init(self.channels, self.channels, 1)
        self.ts_downsample = TsPatchEmbedding(L_patch_len=config["L_patch_len"], channels=inputdim, d_model=self.channels, dropout=0)
        self.side_downsample = CondPatchEmbedding(L_patch_len=config["L_patch_len"], channels=config["side_dim"], d_model=config["side_dim"], dropout=0)
        self.patch_decoder = PatchDecoder(L_patch_len=config["L_patch_len"], n_var=config["n_var"], d_model=self.channels, channels=1)

        self.residual_layers = nn.ModuleList(
            [
                ResidualBlock(
                    side_dim=config["side_dim"],
                    attr_dim=config["attr_dim"],
                    channels=self.channels,
                    diffusion_embedding_dim=config["diffusion_embedding_dim"],
                    nheads=config["nheads"],
                    is_linear=config["is_linear"],
                )
                for _ in range(config["layers"])
            ]
        )

    def forward(self, x, side_emb, attr_emb, diffusion_step, **kwargs):
        B, inputdim, K, L = x.shape
        x = self.ts_downsample(x)
        side_emb = self.side_downsample(side_emb)
        B, _, Nk, Nl = x.shape

        diffusion_emb = self.diffusion_embedding(diffusion_step)
        skip = []
        for layer in self.residual_layers:
            x, skip_connection = layer(x, side_emb, attr_emb, diffusion_emb)
            skip.append(skip_connection)

        x = torch.sum(torch.stack(skip), dim=0) / math.sqrt(len(self.residual_layers))
        x = x.reshape(B, self.channels, Nk * Nl)
        x = self.output_projection1(x)  # (B,channel,K*L)
        x = F.relu(x)
        x = x.reshape(B, self.channels, Nk, Nl) # (B, channel, Nk, Nl)
        x = self.patch_decoder(x) 
        x = x[:, :, :, :L] # (B, 1, K, L)
        x = x.reshape(B, K, L)
        return x
    
class Diff_CSDI_MultiPatch(nn.Module):
    def __init__(self, config, inputdim=2):
        super().__init__()
        self.channels = config["channels"]
        self.multipatch_num = config["multipatch_num"]
        self.diffusion_embedding = DiffusionEmbedding(
            num_steps=config["num_steps"],
            embedding_dim=config["diffusion_embedding_dim"],
        )
        self.output_projection1 = Conv1d_with_init(self.channels, self.channels, 1)
        self.ts_downsample = nn.ModuleList([])
        self.side_downsample = nn.ModuleList([])
        self.patch_decoder = nn.ModuleList([])
        for i in range(self.multipatch_num):
            self.ts_downsample.append(TsPatchEmbedding(L_patch_len=config["L_patch_len"]**i, channels=inputdim, d_model=self.channels, dropout=0))
            self.side_downsample.append(CondPatchEmbedding(L_patch_len=config["L_patch_len"]**i, channels=config["side_dim"], d_model=config["side_dim"], dropout=0))
            self.patch_decoder.append(PatchDecoder(L_patch_len=config["L_patch_len"]**i, n_var=config["n_var"], d_model=self.channels, channels=1))
        self.multipatch_mixer = nn.Linear(self.multipatch_num, 1)

        self.residual_layers = nn.ModuleList(
            [
                ResidualBlock(
                    side_dim=config["side_dim"],
                    attr_dim=config["attr_dim"],
                    channels=self.channels,
                    diffusion_embedding_dim=config["diffusion_embedding_dim"],
                    nheads=config["nheads"],
                    is_linear=config["is_linear"],
                )
                for _ in range(config["layers"])
            ]
        )

    def forward(self, x_raw, side_emb_raw, attr_emb_raw, diffusion_step, **kwargs):
        B, inputdim, K, L = x_raw.shape
        diffusion_emb = self.diffusion_embedding(diffusion_step)
        all_out = []
        for i in range(self.multipatch_num):
            x = self.ts_downsample[i](x_raw)
            side_emb = self.side_downsample[i](side_emb_raw)
            attr_emb = attr_emb_raw

            B, _, Nk, Nl = x.shape
            skip = []
            for layer in self.residual_layers:
                x, skip_connection = layer(x, side_emb, attr_emb, diffusion_emb)
                skip.append(skip_connection)

            x = torch.sum(torch.stack(skip), dim=0) / math.sqrt(len(self.residual_layers))
            x = x.reshape(B, self.channels, Nk * Nl)
            x = self.output_projection1(x)  # (B,channel,Nk*Nl)
            x = F.relu(x)
            x = x.reshape(B, self.channels, Nk, Nl) # (B, channel, Nk, Nl)
            x = self.patch_decoder[i](x) 
            x = x[:, :, :, :L] # (B, 1, K, L)
            all_out.append(x)
        all_out = torch.cat(all_out, dim=1) # (B, M, K, L)
        all_out = self.multipatch_mixer(all_out.permute(0, 2, 3, 1).contiguous()) # (B, K, L, 1)
        all_out = all_out.reshape(B, K, L)
        return all_out

class Diff_CSDI_MultiPatch_Parallel(nn.Module):
    def __init__(self, config, inputdim=2):
        super().__init__()
        self.channels = config["channels"]
        self.multipatch_num = config["multipatch_num"]
        strength_cfg = dict(config.get("strength_control") or {})
        self.strength_enabled = bool(strength_cfg.get("enabled", False))
        self.strength_mode = str(strength_cfg.get("mode", "modulation_residual"))
        self.strength_gain_multiplier = float(strength_cfg.get("gain_multiplier", 4.0))
        self.strength_cond_dim = int(strength_cfg.get("out_dim", self.channels))
        self.strength_projector = None
        self.strength_input_projection = None
        if self.strength_enabled:
            self.strength_projector = StrengthProjector(
                num_strength_bins=int(strength_cfg.get("num_strength_bins", 3)),
                num_tasks=int(strength_cfg.get("num_tasks", 8)),
                emb_dim=int(strength_cfg.get("emb_dim", 32)),
                hidden_dim=int(strength_cfg.get("hidden_dim", 64)),
                out_dim=self.strength_cond_dim,
                use_text_context=bool(strength_cfg.get("use_text_context", True)),
                text_dim=int(strength_cfg.get("text_dim", self.strength_cond_dim)),
                dropout=float(strength_cfg.get("dropout", 0.0)),
                use_task_id=bool(strength_cfg.get("use_task_id", False)),
                text_num_buckets=int(strength_cfg.get("text_num_buckets", 4096)),
            )
            self.strength_input_projection = nn.Linear(self.strength_cond_dim, self.channels)
            nn.init.zeros_(self.strength_input_projection.weight)
            nn.init.zeros_(self.strength_input_projection.bias)
        self.diffusion_embedding = DiffusionEmbedding(
            num_steps=config["num_steps"],
            embedding_dim=config["diffusion_embedding_dim"],
        )
        self.attention_mask_type = config["attention_mask_type"]
        self.output_projection1 = Conv1d_with_init(self.channels, self.channels, 1)
        self.ts_downsample = nn.ModuleList([])
        self.side_downsample = nn.ModuleList([])
        self.patch_decoder = nn.ModuleList([])
        for i in range(self.multipatch_num):
            self.ts_downsample.append(TsPatchEmbedding(L_patch_len=config["L_patch_len"]**i, channels=inputdim, d_model=self.channels, dropout=0))
            self.side_downsample.append(CondPatchEmbedding(L_patch_len=config["L_patch_len"]**i, channels=config["side_dim"], d_model=config["side_dim"], dropout=0))
            self.patch_decoder.append(PatchDecoder(L_patch_len=config["L_patch_len"]**i, n_var=config["n_var"], d_model=self.channels, channels=1))
        self.multipatch_mixer = nn.Linear(self.multipatch_num, 1)
        self.residual_layers = nn.ModuleList(
            [
                ResidualBlock(
                    side_dim=config["side_dim"],
                    attr_dim=config["attr_dim"],
                    is_attr_proj=config["is_attr_proj"],
                    channels=self.channels,
                    diffusion_embedding_dim=config["diffusion_embedding_dim"],
                    nheads=config["nheads"],
                    is_linear=config["is_linear"],
                    strength_cond_dim=self.strength_cond_dim if self.strength_enabled else 0,
                    strength_mode=self.strength_mode,
                    strength_gain_multiplier=self.strength_gain_multiplier,
                )
                for _ in range(config["layers"])
            ]
        )
    
    def _build_strength_condition(self, strength_label=None, task_id=None, text_context=None, strength_scalar=None):
        if not self.strength_enabled:
            return None
        has_text = text_context is not None
        has_task = task_id is not None and self.strength_projector is not None and self.strength_projector.use_task_id
        has_label = strength_label is not None
        if not (has_label or has_task or has_text):
            return None
        return self.strength_projector(
            strength_label=strength_label,
            task_id=task_id,
            text_context=text_context,
            strength_scalar=None,
        )

    def forward(self, x_raw, side_emb_raw, attr_emb_raw, diffusion_step,
                strength_label=None, task_id=None, text_context=None, strength_scalar=None):
        B, inputdim, K, L = x_raw.shape
        diffusion_emb = self.diffusion_embedding(diffusion_step)
        x_list = []
        side_list = []
        for i in range(self.multipatch_num):
            x = self.ts_downsample[i](x_raw)
            side_emb = self.side_downsample[i](side_emb_raw)
            x_list.append(x)
            side_list.append(side_emb)
        if self.attention_mask_type == "full":
            attention_mask = None
        elif self.attention_mask_type == "parallel":
            attention_mask = self.get_mask(attr_emb_raw.shape[1], [x_list[i].shape[-1] for i in range(len(x_list))], device=x_raw.device)
        
        x_in = torch.cat(x_list, dim=-1)
        side_in = torch.cat(side_list, dim=-1)
        B, _, Nk, Nl = x_in.shape
        strength_cond = self._build_strength_condition(
            strength_label=strength_label,
            task_id=task_id,
            text_context=text_context,
            strength_scalar=strength_scalar,
        )
        if strength_cond is not None and self.strength_mode == "input_concat_baseline":
            strength_map = self.strength_input_projection(strength_cond).unsqueeze(-1).unsqueeze(-1)
            x_in = x_in + strength_map

        skip = []
        for layer in self.residual_layers:
            x_in, skip_connection = layer(
                x_in,
                side_in,
                attr_emb_raw,
                diffusion_emb,
                attention_mask=attention_mask,
                strength_cond=strength_cond,
                strength_label=strength_label,
                strength_scalar=strength_scalar,
            )
            skip.append(skip_connection)        
        
        x = torch.sum(torch.stack(skip), dim=0) / math.sqrt(len(self.residual_layers))
        x = x.reshape(B, self.channels, Nk * Nl)
        x = self.output_projection1(x)  # (B,channel,Nk*Nl)
        x = F.relu(x)
        x = x.reshape(B, self.channels, Nk, Nl)

        start_id = 0
        all_out = []
        for i in range(len(x_list)):
            x_out = x[:,:,:,start_id:start_id+x_list[i].shape[-1]]
            x_out = self.patch_decoder[i](x_out)
            x_out = x_out[:, :, :, :L]
            all_out.append(x_out)
            start_id += x_list[i].shape[-1]

        all_out = torch.cat(all_out, dim=1) # (B, M, K, L)
        all_out = self.multipatch_mixer(all_out.permute(0, 2, 3, 1).contiguous()) # (B, K, L, 1)
        all_out = all_out.reshape(B, K, L)
        return all_out
    
    def get_mask(self, attr_len, len_list, device=None):
        if device is None:
            device = self.output_projection1.weight.device
        total_len = sum(len_list) + attr_len
        mask = torch.zeros(total_len, total_len, device=device) - np.inf
        mask[:attr_len, :] = 0
        mask[:, :attr_len] = 0
        start_id = attr_len
        for i in range(len(len_list)):
            mask[start_id:start_id+len_list[i], start_id:start_id+len_list[i]] = 0
            start_id += len_list[i]
        return mask
