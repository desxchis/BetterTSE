import copy
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from linear_attention_transformer import LinearAttentionTransformer
import numpy as np
from einops import rearrange

from models.conditioning import StrengthProjector


def _format_scalar_key(value):
    return f"{float(value):.4f}"


class AttentionInjectionLayer(nn.Module):
    """
    Transformer layer with Soft-Boundary Attention Injection support.
    
    This layer implements the core innovation of BetterTSE:
    - Semantic Isolation: Edit instructions do not leak into background regions
    - Context-Aware Blending: Edge regions automatically "see" surrounding unmodified regions
    
    Core Formula:
        A_inj = Λ ⊙ A(Q, K_edit) + (I - Λ) ⊙ A(Q, K_null)
        V_inj = Λ ⊙ (A_edit @ V_edit) + (I - Λ) ⊙ (A_null @ V_null)
    """
    
    def __init__(self, d_model, nhead, dim_feedforward=64, dropout=0.1, activation="gelu"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = getattr(F, activation)
        
    def forward_with_injection(self, src, src_mask=None, 
                                soft_mask=None, keys_null=None, values_null=None):
        """
        Forward pass with attention injection for soft-boundary editing.
        
        Args:
            src: [B, L, D] - Input tensor
            src_mask: Attention mask
            soft_mask: [B, L] or [B, L, 1] - Soft boundary mask
            keys_null: [B, L, D] - Keys from null/background condition
            values_null: [B, L, D] - Values from null/background condition
        """
        B, L, D = src.shape
        
        src2 = self.self_attn(src, src, src, attn_mask=src_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        
        return src
    
    def forward(self, src, src_mask=None, 
                soft_mask=None, keys_null=None, values_null=None):
        return self.forward_with_injection(src, src_mask, soft_mask, keys_null, values_null)


class AttentionInjectionEncoder(nn.Module):
    """
    Transformer Encoder with Attention Injection support.
    """
    
    def __init__(self, d_model, nhead, dim_feedforward=64, num_layers=1, dropout=0.1, activation="gelu"):
        super().__init__()
        self.layers = nn.ModuleList([
            AttentionInjectionLayer(d_model, nhead, dim_feedforward, dropout, activation)
            for _ in range(num_layers)
        ])
        
    def forward(self, src, mask=None, soft_mask=None, keys_null=None, values_null=None):
        output = src
        for layer in self.layers:
            output = layer(output, src_mask=mask, soft_mask=soft_mask, 
                          keys_null=keys_null, values_null=values_null)
        return output


def get_torch_trans(heads=8, layers=1, channels=64):
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=channels, nhead=heads, dim_feedforward=64, activation="gelu", batch_first=True
    )
    return nn.TransformerEncoder(encoder_layer, num_layers=layers)


def get_injection_trans(heads=8, layers=1, channels=64):
    return AttentionInjectionEncoder(
        d_model=channels, nhead=heads, dim_feedforward=64, num_layers=layers
    )

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


def _normalize_output_branch_carrier_config(config):
    config = dict(config or {})
    config.setdefault("enabled", False)
    config.setdefault("mode", "skip_residual_mix")
    config.setdefault("skip_scale", 0.0)
    config.setdefault("min_residual_to_skip_ratio", 0.0)
    config.setdefault("scalar_order_margin", 0.0)
    if str(config["mode"]) != "skip_residual_mix":
        raise ValueError(f"Unsupported output_branch_carrier mode: {config['mode']}")
    config["mode"] = "skip_residual_mix"
    config["enabled"] = bool(config["enabled"])
    config["skip_scale"] = float(config["skip_scale"])
    config["min_residual_to_skip_ratio"] = float(config["min_residual_to_skip_ratio"])
    config["scalar_order_margin"] = float(config["scalar_order_margin"])
    if config["skip_scale"] < 0.0:
        raise ValueError("output_branch_carrier.skip_scale must be non-negative")
    if config["min_residual_to_skip_ratio"] < 0.0:
        raise ValueError("output_branch_carrier.min_residual_to_skip_ratio must be non-negative")
    if config["scalar_order_margin"] < 0.0:
        raise ValueError("output_branch_carrier.scalar_order_margin must be non-negative")
    return config


def _normalize_final_output_strength_scale_config(config):
    config = dict(config or {})
    config.setdefault("enabled", False)
    config.setdefault("mode", "linear_scalar")
    config.setdefault("scale_per_unit", 0.0)
    config.setdefault("scalar_center", 1.0)
    config.setdefault("min_gain", 1.0)
    config.setdefault("max_gain", 1.0)
    if str(config["mode"]) != "linear_scalar":
        raise ValueError(f"Unsupported final_output_strength_scale mode: {config['mode']}")
    config["mode"] = "linear_scalar"
    config["enabled"] = bool(config["enabled"])
    config["scale_per_unit"] = float(config["scale_per_unit"])
    config["scalar_center"] = float(config["scalar_center"])
    config["min_gain"] = float(config["min_gain"])
    config["max_gain"] = float(config["max_gain"])
    if config["min_gain"] <= 0.0:
        raise ValueError("final_output_strength_scale.min_gain must be positive")
    if config["max_gain"] <= 0.0:
        raise ValueError("final_output_strength_scale.max_gain must be positive")
    if config["max_gain"] < config["min_gain"]:
        raise ValueError("final_output_strength_scale.max_gain must be >= min_gain")
    return config


def _normalize_final_output_strength_mapping_config(config):
    config = dict(config or {})
    config.setdefault("enabled", False)
    config.setdefault("mode", "bounded_scalar_gain")
    config.setdefault("scalar_center", 1.0)
    config.setdefault("scalar_prior_scale", 0.0)
    config.setdefault("learned_max_delta", 0.0)
    config.setdefault("min_gain", 1.0)
    config.setdefault("max_gain", 1.0)
    config.setdefault("hidden_dim", 64)
    config.setdefault("gain_order_margin", 0.0)
    config.setdefault("gain_order_weight", 0.0)
    config.setdefault("gain_order_direction", "increasing")
    config.setdefault("scope", "global")
    scalar_transform = dict(config.get("scalar_transform") or {})
    scalar_transform.setdefault("enabled", False)
    scalar_transform.setdefault("scale", 1.0)
    scalar_transform.setdefault("offset", 0.0)
    scalar_transform.setdefault("name", "identity")
    scalar_transform.setdefault("auto_legacy_half_step", True)
    if str(config["mode"]) != "bounded_scalar_gain":
        raise ValueError(f"Unsupported final_output_strength_mapping mode: {config['mode']}")
    config["mode"] = "bounded_scalar_gain"
    config["enabled"] = bool(config["enabled"])
    config["scalar_center"] = float(config["scalar_center"])
    config["scalar_prior_scale"] = float(config["scalar_prior_scale"])
    config["learned_max_delta"] = float(config["learned_max_delta"])
    config["min_gain"] = float(config["min_gain"])
    config["max_gain"] = float(config["max_gain"])
    config["hidden_dim"] = int(config["hidden_dim"])
    config["gain_order_margin"] = float(config["gain_order_margin"])
    config["gain_order_weight"] = float(config["gain_order_weight"])
    config["gain_order_direction"] = str(config["gain_order_direction"])
    config["scope"] = str(config["scope"])
    scalar_transform["enabled"] = bool(scalar_transform["enabled"])
    scalar_transform["scale"] = float(scalar_transform["scale"])
    scalar_transform["offset"] = float(scalar_transform["offset"])
    scalar_transform["name"] = str(scalar_transform["name"])
    scalar_transform["auto_legacy_half_step"] = bool(scalar_transform["auto_legacy_half_step"])
    config["scalar_transform"] = scalar_transform
    if config["gain_order_direction"] not in {"increasing", "decreasing"}:
        raise ValueError("final_output_strength_mapping.gain_order_direction must be increasing or decreasing")
    if config["scope"] not in {"global", "edit_region"}:
        raise ValueError("final_output_strength_mapping.scope must be global or edit_region")
    if config["min_gain"] <= 0.0:
        raise ValueError("final_output_strength_mapping.min_gain must be positive")
    if config["max_gain"] <= 0.0:
        raise ValueError("final_output_strength_mapping.max_gain must be positive")
    if config["max_gain"] < config["min_gain"]:
        raise ValueError("final_output_strength_mapping.max_gain must be >= min_gain")
    if config["hidden_dim"] <= 0:
        raise ValueError("final_output_strength_mapping.hidden_dim must be positive")
    if config["learned_max_delta"] < 0.0:
        raise ValueError("final_output_strength_mapping.learned_max_delta must be non-negative")
    if config["gain_order_margin"] < 0.0:
        raise ValueError("final_output_strength_mapping.gain_order_margin must be non-negative")
    if config["gain_order_weight"] < 0.0:
        raise ValueError("final_output_strength_mapping.gain_order_weight must be non-negative")
    if not math.isfinite(config["scalar_transform"]["scale"]):
        raise ValueError("final_output_strength_mapping.scalar_transform.scale must be finite")
    if not math.isfinite(config["scalar_transform"]["offset"]):
        raise ValueError("final_output_strength_mapping.scalar_transform.offset must be finite")
    return config


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
        output_branch_carrier=None,
        final_output_gain_gate=None,
    ):
        super().__init__()
        self.diffusion_projection = nn.Linear(diffusion_embedding_dim, channels)
        if is_attr_proj:
            self.attr_projection = nn.Linear(attr_dim, channels)
        else:
            self.attr_projection = nn.Identity()
        self.base_modulation = nn.Sequential(
            nn.Linear(channels + attr_dim, channels),
            nn.SiLU(),
            nn.Linear(channels, 2 * channels),
        )
        nn.init.zeros_(self.base_modulation[-1].weight)
        nn.init.zeros_(self.base_modulation[-1].bias)
        self.strength_mode = strength_mode
        self.strength_gain_multiplier = float(strength_gain_multiplier)
        output_branch_carrier = _normalize_output_branch_carrier_config(output_branch_carrier)
        self.output_branch_carrier_enabled = bool(output_branch_carrier["enabled"])
        self.output_branch_carrier_skip_scale = float(output_branch_carrier["skip_scale"])
        self.output_branch_carrier_min_residual_to_skip_ratio = float(
            output_branch_carrier["min_residual_to_skip_ratio"]
        )
        self.output_branch_carrier_scalar_order_margin = float(
            output_branch_carrier["scalar_order_margin"]
        )
        self._latest_output_branch_regularizer_loss = None
        self._latest_output_branch_scalar_order_loss = None
        final_output_gain_gate = dict(final_output_gain_gate or {})
        self.final_output_gain_gate_enabled = bool(final_output_gain_gate.get("enabled", False)) and strength_cond_dim > 0
        self.final_output_gain_gate_mode = str(final_output_gain_gate.get("mode", "sigmoid_range"))
        self.final_output_gain_gate_min = float(final_output_gain_gate.get("min_gain", 1.0))
        self.final_output_gain_gate_max = float(final_output_gain_gate.get("max_gain", 1.0))
        if self.final_output_gain_gate_mode != "sigmoid_range":
            raise ValueError(f"Unsupported final_output_gain_gate mode: {self.final_output_gain_gate_mode}")
        self.strength_modulation = None
        self.strength_amplitude_head = None
        self.final_output_gain_gate_head = None
        self.use_amplitude_head = strength_cond_dim > 0 and self.strength_mode == "amplitude_decomposition"
        if strength_cond_dim > 0:
            self.strength_modulation = nn.Sequential(
                nn.Linear(strength_cond_dim, channels),
                nn.SiLU(),
                nn.Linear(channels, 2 * channels),
            )
            nn.init.normal_(self.strength_modulation[-1].weight, mean=0.0, std=1.0e-3)
            nn.init.normal_(self.strength_modulation[-1].bias, mean=0.0, std=1.0e-3)
            if self.use_amplitude_head:
                self.strength_amplitude_head = nn.Sequential(
                    nn.Linear(strength_cond_dim, channels),
                    nn.SiLU(),
                    nn.Linear(channels, 2 * channels),
                )
                nn.init.normal_(self.strength_amplitude_head[-1].weight, mean=0.0, std=1.0e-3)
                nn.init.normal_(self.strength_amplitude_head[-1].bias, mean=0.0, std=1.0e-3)
            if self.final_output_gain_gate_enabled:
                self.final_output_gain_gate_head = nn.Sequential(
                    nn.Linear(strength_cond_dim, channels),
                    nn.SiLU(),
                    nn.Linear(channels, channels),
                )
                nn.init.zeros_(self.final_output_gain_gate_head[-1].weight)
                nn.init.zeros_(self.final_output_gain_gate_head[-1].bias)
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


    def forward_time(self, y, base_shape, attention_mask=None, soft_mask=None, keys_null=None, values_null=None):
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

    def forward_feature(self, y, base_shape, attention_mask=None, soft_mask=None, keys_null=None, values_null=None):
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

    def _record_scalar_gain_diagnostics(self, record_name, gain, strength_scalar=None, transformed_scalar=None, scalar_transform=None):
        if not self.__class__._diag_enabled:
            return
        if len(self.__class__._diag_records) >= self.__class__._diag_max_records:
            return
        with torch.no_grad():
            gain_cpu = gain.detach().cpu().float().reshape(gain.shape[0], -1)
            record = {
                f"{record_name}_mean": float(gain_cpu.mean().item()),
                f"{record_name}_min": float(gain_cpu.min().item()),
                f"{record_name}_max": float(gain_cpu.max().item()),
            }
            if strength_scalar is not None:
                scalar_values = strength_scalar.detach().cpu().float().view(-1)
                unique_scalars = sorted({float(value) for value in scalar_values.tolist()})
                by_scalar = {}
                for scalar_value in unique_scalars:
                    mask = torch.isclose(
                        scalar_values,
                        scalar_values.new_tensor(float(scalar_value)),
                        atol=1.0e-6,
                        rtol=0.0,
                    )
                    if int(mask.sum().item()) == 0:
                        continue
                    scalar_key = _format_scalar_key(scalar_value)
                    scalar_gain = gain_cpu[mask]
                    by_scalar[scalar_key] = {
                        "count": int(mask.sum().item()),
                        "mean": float(scalar_gain.mean().item()),
                        "min": float(scalar_gain.min().item()),
                        "max": float(scalar_gain.max().item()),
                    }
                record[f"{record_name}_by_scalar"] = by_scalar
            if transformed_scalar is not None:
                transformed_values = transformed_scalar.detach().cpu().float().view(-1)
                record[f"{record_name}_transformed_scalar"] = [float(value) for value in transformed_values.tolist()]
                if strength_scalar is not None and transformed_values.numel() == scalar_values.numel():
                    transformed_by_scalar = {}
                    for scalar_value in unique_scalars:
                        mask = torch.isclose(
                            scalar_values,
                            scalar_values.new_tensor(float(scalar_value)),
                            atol=1.0e-6,
                            rtol=0.0,
                        )
                        if int(mask.sum().item()) == 0:
                            continue
                        scalar_key = _format_scalar_key(scalar_value)
                        transformed_by_scalar[scalar_key] = {
                            "count": int(mask.sum().item()),
                            "mean": float(transformed_values[mask].mean().item()),
                            "min": float(transformed_values[mask].min().item()),
                            "max": float(transformed_values[mask].max().item()),
                        }
                    record[f"{record_name}_transformed_scalar_by_scalar"] = transformed_by_scalar
            if isinstance(scalar_transform, dict):
                record[f"{record_name}_scalar_transform"] = dict(scalar_transform)
            self.__class__._diag_records.append(record)

    def _compute_scalar_order_loss(self, tensor, strength_scalar):
        if strength_scalar is None or self.output_branch_carrier_scalar_order_margin <= 0.0:
            return None
        scalar_values = strength_scalar.float().view(-1)
        if scalar_values.numel() != tensor.shape[0]:
            return None
        unique_scalars = sorted({float(value) for value in scalar_values.detach().cpu().tolist()})
        if len(unique_scalars) < 2:
            return None
        sample_abs = tensor.float().reshape(tensor.shape[0], -1).abs().mean(dim=1)
        scalar_means = []
        for scalar_value in unique_scalars:
            mask = torch.isclose(
                scalar_values,
                scalar_values.new_tensor(float(scalar_value)),
                atol=1.0e-6,
                rtol=0.0,
            )
            if int(mask.sum().item()) == 0:
                continue
            scalar_means.append(sample_abs[mask].mean())
        if len(scalar_means) < 2:
            return None
        losses = []
        margin = tensor.new_tensor(float(self.output_branch_carrier_scalar_order_margin))
        for idx in range(len(scalar_means) - 1):
            losses.append(torch.relu(margin - (scalar_means[idx + 1] - scalar_means[idx])))
        if not losses:
            return None
        return torch.stack(losses).mean()

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

    def _run_post_modulation_stack(self, y, base_shape, attention_mask=None, soft_mask=None, keys_null=None, values_null=None, strength_label=None, strength_scalar=None):
        self._record_forward_response_diagnostics(
            {"post_modulation": y},
            strength_label=strength_label,
            strength_scalar=strength_scalar,
        )
        y = self.forward_time(y, base_shape, attention_mask, soft_mask, keys_null, values_null)
        self._record_forward_response_diagnostics(
            {"post_time": y},
            strength_label=strength_label,
            strength_scalar=strength_scalar,
        )
        y = self.forward_feature(y, base_shape, attention_mask, soft_mask, keys_null, values_null)
        self._record_forward_response_diagnostics(
            {"post_feature": y},
            strength_label=strength_label,
            strength_scalar=strength_scalar,
        )
        return y

    def _run_output_head(self, y, side_emb, base_shape, strength_cond=None, strength_label=None, strength_scalar=None):
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
        residual, skip = torch.chunk(y, 2, dim=1)
        self._record_forward_response_diagnostics(
            {
                "post_output_projection": y,
                "residual_content_branch": residual,
                "skip_branch": skip,
            },
            strength_label=strength_label,
            strength_scalar=strength_scalar,
        )
        residual_content = residual
        if self.output_branch_carrier_min_residual_to_skip_ratio > 0.0:
            residual_content_train_abs = residual_content.float().reshape(B, -1).abs().mean(dim=1)
            skip_train_abs = skip.detach().float().reshape(B, -1).abs().mean(dim=1)
            branch_floor = self.output_branch_carrier_min_residual_to_skip_ratio * skip_train_abs
            self._latest_output_branch_regularizer_loss = torch.relu(branch_floor - residual_content_train_abs).mean()
        else:
            self._latest_output_branch_regularizer_loss = None
        carrier_source = torch.zeros_like(residual)
        if self.output_branch_carrier_enabled and self.output_branch_carrier_skip_scale > 0.0:
            carrier_source = skip * self.output_branch_carrier_skip_scale
            residual = residual + carrier_source
        self._record_forward_response_diagnostics(
            {
                "residual_carrier_source_branch": carrier_source,
                "residual_carrier_restored_branch": residual,
            },
            strength_label=strength_label,
            strength_scalar=strength_scalar,
        )
        gate_gain = None
        if self.use_amplitude_head and self.strength_amplitude_head is not None and strength_cond is not None:
            amp_gamma, amp_beta = torch.chunk(self.strength_amplitude_head(strength_cond), 2, dim=-1)
            amp_gamma = (1.0 + self.strength_gain_multiplier * amp_gamma).unsqueeze(-1)
            amp_beta = (self.strength_gain_multiplier * amp_beta).unsqueeze(-1)
            residual = residual * amp_gamma + amp_beta
            self._record_forward_response_diagnostics(
                {
                    "amplitude_gamma": amp_gamma,
                    "amplitude_beta": amp_beta,
                    "residual_amplitude_branch": residual,
                },
                strength_label=strength_label,
                strength_scalar=strength_scalar,
            )
        else:
            self._record_forward_response_diagnostics(
                {"residual_amplitude_branch": residual},
                strength_label=strength_label,
                strength_scalar=strength_scalar,
            )
        self._latest_output_branch_scalar_order_loss = self._compute_scalar_order_loss(residual, strength_scalar)
        if self.final_output_gain_gate_enabled and self.final_output_gain_gate_head is not None and strength_cond is not None:
            gate_logits = self.final_output_gain_gate_head(strength_cond)
            gate_gain = torch.sigmoid(gate_logits)
            gate_gain = self.final_output_gain_gate_min + (self.final_output_gain_gate_max - self.final_output_gain_gate_min) * gate_gain
            gate_gain = gate_gain.unsqueeze(-1)
            residual = residual * gate_gain
            self._record_forward_response_diagnostics(
                {
                    "final_output_gain_gate_logits": gate_logits.unsqueeze(-1),
                    "final_output_gain_gate": gate_gain,
                    "residual_gain_gated_branch": residual,
                },
                strength_label=strength_label,
                strength_scalar=strength_scalar,
            )
        else:
            gain_fill = residual.new_full((B, channel, 1), 1.0)
            self._record_forward_response_diagnostics(
                {
                    "final_output_gain_gate": gain_fill,
                    "residual_gain_gated_branch": residual,
                },
                strength_label=strength_label,
                strength_scalar=strength_scalar,
            )
        if self.__class__._diag_enabled and len(self.__class__._diag_records) < self.__class__._diag_max_records:
            residual_content_stats = residual_content.detach().cpu().float().reshape(B, -1).abs().mean(dim=1)
            skip_stats = skip.detach().cpu().float().reshape(B, -1).abs().mean(dim=1)
            restored_stats = residual.detach().cpu().float().reshape(B, -1).abs().mean(dim=1)
            carrier_stats = carrier_source.detach().cpu().float().reshape(B, -1).abs().mean(dim=1)
            gate_stats = gate_gain if gate_gain is not None else residual.new_full((B, channel, 1), 1.0)
            gate_stats = gate_stats.detach().cpu().float().reshape(B, -1)
            record = {
                "output_branch_carrier_enabled": float(self.output_branch_carrier_enabled),
                "output_branch_carrier_skip_scale": float(self.output_branch_carrier_skip_scale),
                "residual_content_abs_mean": float(residual_content_stats.mean().item()),
                "skip_branch_abs_mean": float(skip_stats.mean().item()),
                "residual_carrier_source_abs_mean": float(carrier_stats.mean().item()),
                "residual_carrier_restored_abs_mean": float(restored_stats.mean().item()),
                "residual_content_to_skip_abs_ratio": float((residual_content_stats / torch.clamp(skip_stats, min=1.0e-8)).mean().item()),
                "residual_restored_to_skip_abs_ratio": float((restored_stats / torch.clamp(skip_stats, min=1.0e-8)).mean().item()),
                "final_output_gain_gate_mean": float(gate_stats.mean().item()),
                "final_output_gain_gate_min": float(gate_stats.min().item()),
                "final_output_gain_gate_max": float(gate_stats.max().item()),
            }
            if strength_scalar is not None:
                scalar_values = strength_scalar.detach().cpu().float().view(-1)
                unique_scalars = sorted({float(value) for value in scalar_values.tolist()})
                mean_by_scalar = {}
                min_by_scalar = {}
                max_by_scalar = {}
                residual_content_by_scalar = {}
                skip_by_scalar = {}
                residual_restored_by_scalar = {}
                residual_ratio_by_scalar = {}
                for scalar_value in unique_scalars:
                    mask = torch.isclose(scalar_values, scalar_values.new_tensor(float(scalar_value)), atol=1.0e-6, rtol=0.0)
                    if int(mask.sum().item()) == 0:
                        continue
                    scalar_key = _format_scalar_key(scalar_value)
                    scalar_gate = gate_stats[mask]
                    mean_by_scalar[scalar_key] = {
                        "count": int(mask.sum().item()),
                        "mean": float(scalar_gate.mean().item()),
                    }
                    min_by_scalar[scalar_key] = {
                        "count": int(mask.sum().item()),
                        "mean": float(scalar_gate.min().item()),
                    }
                    max_by_scalar[scalar_key] = {
                        "count": int(mask.sum().item()),
                        "mean": float(scalar_gate.max().item()),
                    }
                    scalar_residual = residual_content_stats[mask]
                    scalar_skip = skip_stats[mask]
                    scalar_restored = restored_stats[mask]
                    residual_content_by_scalar[scalar_key] = {
                        "count": int(mask.sum().item()),
                        "mean": float(scalar_residual.mean().item()),
                    }
                    skip_by_scalar[scalar_key] = {
                        "count": int(mask.sum().item()),
                        "mean": float(scalar_skip.mean().item()),
                    }
                    residual_restored_by_scalar[scalar_key] = {
                        "count": int(mask.sum().item()),
                        "mean": float(scalar_restored.mean().item()),
                    }
                    residual_ratio_by_scalar[scalar_key] = {
                        "count": int(mask.sum().item()),
                        "mean": float((scalar_restored / torch.clamp(scalar_skip, min=1.0e-8)).mean().item()),
                    }
                record["final_output_gain_gate_mean_by_scalar"] = mean_by_scalar
                record["final_output_gain_gate_min_by_scalar"] = min_by_scalar
                record["final_output_gain_gate_max_by_scalar"] = max_by_scalar
                record["residual_content_abs_mean_by_scalar"] = residual_content_by_scalar
                record["skip_branch_abs_mean_by_scalar"] = skip_by_scalar
                record["residual_carrier_restored_abs_mean_by_scalar"] = residual_restored_by_scalar
                record["residual_restored_to_skip_abs_ratio_by_scalar"] = residual_ratio_by_scalar
            self.__class__._diag_records.append(record)
        return residual.reshape(base_shape), skip.reshape(base_shape)

    def _record_residual_merge(self, x, residual, strength_label=None, strength_scalar=None):
        merged = (x + residual) / math.sqrt(2.0)
        self._record_forward_response_diagnostics(
            {"residual_merge": merged},
            strength_label=strength_label,
            strength_scalar=strength_scalar,
        )
        return merged

    def _forward_with_strength_diagnostics(self, x, side_emb, base_shape, y, attention_mask=None, soft_mask=None, keys_null=None, values_null=None, strength_cond=None, strength_label=None, strength_scalar=None):
        y = self._run_post_modulation_stack(
            y,
            base_shape,
            attention_mask=attention_mask,
            soft_mask=soft_mask,
            keys_null=keys_null,
            values_null=values_null,
            strength_label=strength_label,
            strength_scalar=strength_scalar,
        )
        residual, skip = self._run_output_head(
            y,
            side_emb,
            base_shape,
            strength_cond=strength_cond,
            strength_label=strength_label,
            strength_scalar=strength_scalar,
        )
        return self._record_residual_merge(x, residual, strength_label=strength_label, strength_scalar=strength_scalar), skip

    def _compute_modulation(self, attr_emb, diffusion_cond, strength_cond, strength_label=None, strength_scalar=None):
        pooled_attr = torch.mean(self.attr_projection(attr_emb), dim=1)
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
        elif strength_ablation_mode == "beta_upweight":
            delta_beta = delta_beta * 2.0
        if strength_scalar is not None:
            strength_scalar = strength_scalar.float().view(-1, 1)
        flip_beta_sign = bool(self.__class__._flip_beta_sign_inference) and not self.training
        gamma_final = gamma_orig + delta_gamma
        beta_final = beta_orig + delta_beta if flip_beta_sign else beta_orig - delta_beta
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

    def _normalize_diffusion_timestep_embedding(self, diffusion_emb, batch_size):
        if not torch.is_tensor(diffusion_emb):
            raise TypeError(
                f"diffusion_emb must be a torch.Tensor of integer timesteps or timestep embeddings, got {type(diffusion_emb).__name__}"
            )
        if diffusion_emb.ndim == 0:
            if not diffusion_emb.dtype.is_floating_point and diffusion_emb.dtype != torch.bool:
                diffusion_emb = diffusion_emb.expand(batch_size)
            else:
                raise ValueError(
                    "diffusion_emb scalar input is only supported for integer-like timesteps; batched timestep embeddings must be rank-1 or rank-2"
                )
        if diffusion_emb.ndim == 1:
            if diffusion_emb.shape[0] != batch_size:
                raise ValueError(
                    f"diffusion_emb batch mismatch: expected {batch_size} timesteps, got shape {tuple(diffusion_emb.shape)}"
                )
            if diffusion_emb.dtype.is_floating_point or diffusion_emb.dtype == torch.bool:
                raise TypeError(
                    f"diffusion_emb rank-1 input must contain integer-like timesteps, got dtype {diffusion_emb.dtype}"
                )
        elif diffusion_emb.ndim != 2:
            raise ValueError(
                f"diffusion_emb must be a scalar timestep, shape ({batch_size},), or shape ({batch_size}, D); got shape {tuple(diffusion_emb.shape)}"
            )
        elif diffusion_emb.shape[0] != batch_size:
            raise ValueError(
                f"diffusion_emb batch mismatch: expected leading dimension {batch_size}, got shape {tuple(diffusion_emb.shape)}"
            )
        return diffusion_emb

    def forward(self, x, side_emb, attr_emb, diffusion_emb, attention_mask=None,
                soft_mask=None, keys_null=None, values_null=None, strength_cond=None, strength_label=None, strength_scalar=None):
        B, channel, K, L = x.shape
        base_shape = x.shape

        diffusion_emb = self._normalize_diffusion_timestep_embedding(diffusion_emb, B)
        diffusion_cond = self.diffusion_projection(diffusion_emb)
        gamma, beta = self._compute_modulation(
            attr_emb,
            diffusion_cond,
            strength_cond,
            strength_label=strength_label,
            strength_scalar=strength_scalar,
        )
        diffusion_emb = diffusion_cond.unsqueeze(-1).unsqueeze(-1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        y = (x + diffusion_emb) * (1.0 + gamma) + beta
        x = x.reshape(base_shape)
        return self._forward_with_strength_diagnostics(
            x,
            side_emb,
            base_shape,
            y,
            attention_mask=attention_mask,
            soft_mask=soft_mask,
            keys_null=keys_null,
            values_null=values_null,
            strength_cond=strength_cond,
            strength_label=strength_label,
            strength_scalar=strength_scalar,
        )

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

class Diff_CSDI_MultiPatch_Weaver_Parallel(nn.Module):
    def __init__(self, config, inputdim=2):
        super().__init__()
        self.channels = config["channels"]
        self.multipatch_num = config["multipatch_num"]
        strength_cfg = dict(config.get("strength_control") or {})
        self.strength_enabled = bool(strength_cfg.get("enabled", False))
        self.strength_mode = str(strength_cfg.get("mode", "modulation_residual"))
        self.strength_gain_multiplier = float(strength_cfg.get("gain_multiplier", 4.0))
        self.strength_cond_dim = int(strength_cfg.get("out_dim", self.channels))
        self.output_branch_carrier_cfg = _normalize_output_branch_carrier_config(
            strength_cfg.get("output_branch_carrier")
        )
        strength_cfg["output_branch_carrier"] = dict(self.output_branch_carrier_cfg)
        self.latest_output_branch_regularizer_loss = None
        self.latest_output_branch_scalar_order_loss = None
        self.latest_final_output_strength_mapping_order_loss = None
        self.latest_final_output_strength_mapping_scalar_transform = None
        self.final_output_gain_gate_cfg = dict(strength_cfg.get("final_output_gain_gate") or {})
        self.final_output_gain_gate_cfg.setdefault("enabled", False)
        self.final_output_gain_gate_cfg.setdefault("mode", "sigmoid_range")
        self.final_output_gain_gate_cfg.setdefault("min_gain", 1.0)
        self.final_output_gain_gate_cfg.setdefault("max_gain", 1.0)
        if str(self.final_output_gain_gate_cfg.get("mode", "sigmoid_range")) != "sigmoid_range":
            raise ValueError(f"Unsupported final_output_gain_gate mode: {self.final_output_gain_gate_cfg.get('mode')}")
        self.final_output_gain_gate_cfg["mode"] = "sigmoid_range"
        self.final_output_gain_gate_cfg["enabled"] = bool(self.final_output_gain_gate_cfg.get("enabled", False))
        self.final_output_gain_gate_cfg["min_gain"] = float(self.final_output_gain_gate_cfg.get("min_gain", 1.0))
        self.final_output_gain_gate_cfg["max_gain"] = float(self.final_output_gain_gate_cfg.get("max_gain", 1.0))
        strength_cfg["final_output_gain_gate"] = dict(self.final_output_gain_gate_cfg)
        self.final_output_strength_scale_cfg = _normalize_final_output_strength_scale_config(
            strength_cfg.get("final_output_strength_scale")
        )
        strength_cfg["final_output_strength_scale"] = dict(self.final_output_strength_scale_cfg)
        self.final_output_strength_mapping_cfg = _normalize_final_output_strength_mapping_config(
            strength_cfg.get("final_output_strength_mapping")
        )
        strength_cfg["final_output_strength_mapping"] = dict(self.final_output_strength_mapping_cfg)
        config["strength_control"] = strength_cfg
        self.strength_control_config = strength_cfg
        self.strength_projector = None
        self.strength_input_projection = None
        self.final_output_strength_mapping_head = None
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
                include_strength_scalar=bool(strength_cfg.get("include_strength_scalar", True)),
            )
            self.strength_input_projection = nn.Linear(self.strength_cond_dim, self.channels)
            nn.init.zeros_(self.strength_input_projection.weight)
            nn.init.zeros_(self.strength_input_projection.bias)
            if self.final_output_strength_mapping_cfg["enabled"]:
                self.final_output_strength_mapping_head = nn.Sequential(
                    nn.Linear(self.strength_cond_dim, int(self.final_output_strength_mapping_cfg["hidden_dim"])),
                    nn.SiLU(),
                    nn.Linear(int(self.final_output_strength_mapping_cfg["hidden_dim"]), 1),
                )
                nn.init.zeros_(self.final_output_strength_mapping_head[-1].weight)
                nn.init.zeros_(self.final_output_strength_mapping_head[-1].bias)
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
                    output_branch_carrier=self.output_branch_carrier_cfg,
                    final_output_gain_gate=self.final_output_gain_gate_cfg,
                )
                for _ in range(config["layers"])
            ]
        )

        self.attr_projector = AttrProjectorAvg(
                dim_in=config["attr_dim"], 
                dim_hid=config["channels"], 
                dim_out=config["channels"],
        )

    def _apply_final_output_strength_scale(self, output, strength_scalar=None):
        if not self.final_output_strength_scale_cfg.get("enabled", False):
            return output, None
        if strength_scalar is None:
            return output, None
        scalar = strength_scalar.float().view(-1)
        if scalar.numel() != output.shape[0]:
            return output, None
        gain = 1.0 + float(self.final_output_strength_scale_cfg["scale_per_unit"]) * (
            scalar - float(self.final_output_strength_scale_cfg["scalar_center"])
        )
        gain = torch.clamp(
            gain,
            min=float(self.final_output_strength_scale_cfg["min_gain"]),
            max=float(self.final_output_strength_scale_cfg["max_gain"]),
        )
        view_shape = [output.shape[0]] + [1] * (output.ndim - 1)
        gain = gain.view(*view_shape).to(device=output.device, dtype=output.dtype)
        return output * gain, gain

    def _compute_final_output_strength_mapping_order_loss(self, gain, strength_scalar):
        cfg = self.final_output_strength_mapping_cfg
        if strength_scalar is None or not cfg.get("enabled", False):
            return None
        if float(cfg.get("gain_order_margin", 0.0)) <= 0.0:
            return None
        scalar_values = strength_scalar.float().view(-1)
        if scalar_values.numel() != gain.shape[0]:
            return None
        unique_scalars = sorted({float(value) for value in scalar_values.detach().cpu().tolist()})
        if len(unique_scalars) < 2:
            return None
        gain_values = gain.float().reshape(gain.shape[0], -1).mean(dim=1)
        scalar_means = []
        for scalar_value in unique_scalars:
            mask = torch.isclose(
                scalar_values,
                scalar_values.new_tensor(float(scalar_value)),
                atol=1.0e-6,
                rtol=0.0,
            )
            if int(mask.sum().item()) == 0:
                continue
            scalar_means.append(gain_values[mask].mean())
        if len(scalar_means) < 2:
            return None
        margin = gain.new_tensor(float(cfg["gain_order_margin"]))
        losses = [
            torch.relu(
                margin
                - (
                    scalar_means[idx + 1] - scalar_means[idx]
                    if str(cfg.get("gain_order_direction", "increasing")) == "increasing"
                    else scalar_means[idx] - scalar_means[idx + 1]
                )
            )
            for idx in range(len(scalar_means) - 1)
        ]
        return torch.stack(losses).mean()

    def _transform_final_output_strength_scalar(self, strength_scalar):
        cfg = self.final_output_strength_mapping_cfg
        transform = dict(cfg.get("scalar_transform") or {})
        if strength_scalar is None:
            return strength_scalar
        if bool(transform.get("enabled", False)):
            return strength_scalar.float() * float(transform.get("scale", 1.0)) + float(transform.get("offset", 0.0))

        auto_legacy_half_step = bool(transform.get("auto_legacy_half_step", True))
        scalar_center = float(cfg.get("scalar_center", 1.0))
        if not auto_legacy_half_step or not math.isclose(scalar_center, 1.0, rel_tol=0.0, abs_tol=1.0e-6):
            return strength_scalar

        scalar_values = strength_scalar.detach().cpu().float().view(-1)
        if scalar_values.numel() == 0:
            return strength_scalar
        scalar_min = float(scalar_values.min().item())
        scalar_max = float(scalar_values.max().item())
        if scalar_min < -1.0e-6 or scalar_max > 1.0 + 1.0e-6:
            return strength_scalar

        unique_scalars = sorted({round(float(value), 6) for value in scalar_values.tolist()})
        has_half_step = any(abs(value - round(value)) > 1.0e-6 for value in unique_scalars)
        if not has_half_step:
            return strength_scalar
        return strength_scalar.float() * 2.0

    def _broadcast_final_strength_mask(self, final_strength_mask, output):
        if final_strength_mask is None:
            return None
        mask = final_strength_mask.to(device=output.device, dtype=output.dtype)
        if mask.ndim == output.ndim - 1:
            mask = mask.unsqueeze(-1)
        while mask.ndim < output.ndim:
            mask = mask.unsqueeze(1)
        if mask.shape[0] != output.shape[0]:
            raise ValueError(
                f"final_strength_mask batch size {mask.shape[0]} does not match output batch size {output.shape[0]}"
            )
        try:
            return mask.expand_as(output)
        except RuntimeError as exc:
            raise ValueError(
                f"final_strength_mask shape {tuple(final_strength_mask.shape)} cannot broadcast to output shape {tuple(output.shape)}"
            ) from exc

    def _apply_final_output_strength_mapping(self, output, strength_cond=None, strength_scalar=None, final_strength_mask=None):
        cfg = self.final_output_strength_mapping_cfg
        self.latest_final_output_strength_mapping_order_loss = None
        self.latest_final_output_strength_mapping_scalar_transform = None
        if not cfg.get("enabled", False):
            return output, None
        batch_size = output.shape[0]
        gain = output.new_ones(batch_size)
        mapping_strength_scalar = self._transform_final_output_strength_scalar(strength_scalar)
        if strength_scalar is not None and mapping_strength_scalar is not strength_scalar:
            transform_cfg = dict(cfg.get("scalar_transform") or {})
            transform_name = str(transform_cfg.get("name", "identity"))
            transform_scale = float(transform_cfg.get("scale", 1.0))
            transform_offset = float(transform_cfg.get("offset", 0.0))
            transform_enabled = bool(transform_cfg.get("enabled", False))
            if not transform_enabled:
                transform_name = "auto_legacy_half_step_0_0p5_1_to_legacy_0_1_2"
                transform_scale = 2.0
                transform_offset = 0.0
            self.latest_final_output_strength_mapping_scalar_transform = {
                "name": transform_name,
                "enabled": True,
                "scale": transform_scale,
                "offset": transform_offset,
                "original_scalar": strength_scalar.detach().cpu().float().view(-1),
                "transformed_scalar": mapping_strength_scalar.detach().cpu().float().view(-1),
            }
        if mapping_strength_scalar is not None:
            scalar = mapping_strength_scalar.float().view(-1)
            if scalar.numel() == batch_size:
                scalar = scalar.to(device=output.device, dtype=output.dtype)
                gain = gain + float(cfg["scalar_prior_scale"]) * (scalar - float(cfg["scalar_center"]))
        if (
            self.final_output_strength_mapping_head is not None
            and strength_cond is not None
            and float(cfg["learned_max_delta"]) > 0.0
        ):
            learned_delta = torch.tanh(self.final_output_strength_mapping_head(strength_cond).view(-1))
            if learned_delta.numel() == batch_size:
                gain = gain + float(cfg["learned_max_delta"]) * learned_delta.to(device=output.device, dtype=output.dtype)
        gain = torch.clamp(gain, min=float(cfg["min_gain"]), max=float(cfg["max_gain"]))
        view_shape = [batch_size] + [1] * (output.ndim - 1)
        gain = gain.view(*view_shape).to(device=output.device, dtype=output.dtype)
        self.latest_final_output_strength_mapping_order_loss = self._compute_final_output_strength_mapping_order_loss(
            gain,
            mapping_strength_scalar,
        )
        if str(cfg.get("scope", "global")) == "edit_region" and final_strength_mask is not None:
            mask = self._broadcast_final_strength_mask(final_strength_mask, output).clamp(0.0, 1.0)
            gain = 1.0 + (gain - 1.0) * mask
        return output * gain, gain
    
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
            strength_scalar=strength_scalar,
        )

    def forward(self, x_raw, side_emb_raw, attr_emb_raw, diffusion_step,
                soft_mask=None, keys_null=None, values_null=None,
                strength_label=None, task_id=None, text_context=None, strength_scalar=None,
                final_strength_mask=None):
        """
        Forward pass with Soft-Boundary Attention Injection support.
        
        Args:
            x_raw: [B, inputdim, K, L] - Input tensor
            side_emb_raw: Side information embedding
            attr_emb_raw: Attribute embedding
            diffusion_step: Diffusion timestep
            soft_mask: [B, L] or [B, L, 1] - Soft boundary mask for attention injection
            keys_null: [B, L, D] - Keys from null/background condition
            values_null: [B, L, D] - Values from null/background condition
        """
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
            attention_mask = self.get_mask(
                0,
                [x_list[i].shape[-1] for i in range(len(x_list))],
                device=x_raw.device,
            )
        
        x_in = torch.cat(x_list, dim=-1)
        side_in = torch.cat(side_list, dim=-1)
        B, _, Nk, Nl = x_in.shape

        _x_in = x_in

        attr_emb = self.attr_projector(attr_emb_raw, length=x_in.shape[-1])
        attr_emb = attr_emb.permute(0,2,1)  # (B,L,C) -> (B,C,L)
        attr_emb = attr_emb.unsqueeze(2)  # (B,C,1,L)
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
        output_branch_regularizer_losses = []
        output_branch_scalar_order_losses = []
        for layer in self.residual_layers:
            x_in, skip_connection = layer(
                x_in+_x_in+attr_emb, side_in, attr_emb_raw, diffusion_emb, 
                attention_mask=attention_mask,
                soft_mask=soft_mask, keys_null=keys_null, values_null=values_null,
                strength_cond=strength_cond,
                strength_label=strength_label,
                strength_scalar=strength_scalar,
            )
            skip.append(skip_connection)        
            branch_loss = getattr(layer, "_latest_output_branch_regularizer_loss", None)
            if torch.is_tensor(branch_loss):
                output_branch_regularizer_losses.append(branch_loss)
            scalar_order_loss = getattr(layer, "_latest_output_branch_scalar_order_loss", None)
            if torch.is_tensor(scalar_order_loss):
                output_branch_scalar_order_losses.append(scalar_order_loss)
        if output_branch_regularizer_losses:
            self.latest_output_branch_regularizer_loss = torch.stack(output_branch_regularizer_losses).mean()
        else:
            self.latest_output_branch_regularizer_loss = None
        if output_branch_scalar_order_losses:
            self.latest_output_branch_scalar_order_loss = torch.stack(output_branch_scalar_order_losses).mean()
        else:
            self.latest_output_branch_scalar_order_loss = None
        
        x = torch.sum(torch.stack(skip), dim=0) / math.sqrt(len(self.residual_layers))
        if self.residual_layers:
            self.residual_layers[0]._record_forward_response_diagnostics(
                {"skip_aggregate": x},
                strength_label=strength_label,
                strength_scalar=strength_scalar,
            )
        x = x.reshape(B, self.channels, Nk * Nl)
        x = self.output_projection1(x)  # (B,channel,Nk*Nl)
        if self.residual_layers:
            self.residual_layers[0]._record_forward_response_diagnostics(
                {"final_head_projection": x},
                strength_label=strength_label,
                strength_scalar=strength_scalar,
            )
        x = F.relu(x)
        if self.residual_layers:
            self.residual_layers[0]._record_forward_response_diagnostics(
                {"final_head_relu": x},
                strength_label=strength_label,
                strength_scalar=strength_scalar,
            )
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
        if self.residual_layers:
            self.residual_layers[0]._record_forward_response_diagnostics(
                {"patch_decoder_concat": all_out},
                strength_label=strength_label,
                strength_scalar=strength_scalar,
            )
        all_out = self.multipatch_mixer(all_out.permute(0, 2, 3, 1).contiguous()) # (B, K, L, 1)
        if self.residual_layers:
            self.residual_layers[0]._record_forward_response_diagnostics(
                {"final_multipatch_output": all_out},
                strength_label=strength_label,
                strength_scalar=strength_scalar,
            )
        all_out, final_mapping_gain = self._apply_final_output_strength_mapping(
            all_out,
            strength_cond=strength_cond,
            strength_scalar=strength_scalar,
            final_strength_mask=final_strength_mask,
        )
        if self.residual_layers:
            self.residual_layers[0]._record_forward_response_diagnostics(
                {"final_strength_mapped_output": all_out},
                strength_label=strength_label,
                strength_scalar=strength_scalar,
            )
            if final_mapping_gain is not None:
                self.residual_layers[0]._record_scalar_gain_diagnostics(
                    "final_output_strength_mapping",
                    final_mapping_gain,
                    strength_scalar=strength_scalar,
                    transformed_scalar=getattr(self, "latest_final_output_strength_mapping_scalar_transform", None).get("transformed_scalar")
                    if isinstance(getattr(self, "latest_final_output_strength_mapping_scalar_transform", None), dict)
                    else None,
                    scalar_transform=(self.final_output_strength_mapping_cfg.get("scalar_transform") or None),
                )
        all_out, final_strength_gain = self._apply_final_output_strength_scale(all_out, strength_scalar=strength_scalar)
        if self.residual_layers:
            self.residual_layers[0]._record_forward_response_diagnostics(
                {"final_strength_scaled_output": all_out},
                strength_label=strength_label,
                strength_scalar=strength_scalar,
            )
            if final_strength_gain is not None:
                self.residual_layers[0]._record_scalar_gain_diagnostics(
                    "final_output_strength_scale",
                    final_strength_gain,
                    strength_scalar=strength_scalar,
                )
        all_out = all_out.reshape(B, K, L)
        return all_out
    
    def get_mask(self, attr_len, len_list, device=None):
        if device is None:
            device = self.output_projection1.weight.device
        total_len = sum(len_list) + attr_len
        mask = torch.zeros(total_len, total_len, device=device) - float("inf")
        mask[:attr_len, :] = 0
        mask[:, :attr_len] = 0
        start_id = attr_len
        for i in range(len(len_list)):
            mask[start_id:start_id+len_list[i], start_id:start_id+len_list[i]] = 0
            start_id += len_list[i]
        return mask


class AttrProjector(nn.Module):
    def __init__(self, n_attrs=3, dim_in=128, dim_hid=128, dim_out=128, n_heads=8, n_layers=2):
        super().__init__()
        self.n_heads = n_heads
        self.n_layers = n_layers

        self.dim_in = dim_in
        self.dim_hid = dim_hid
        self.dim_out = dim_out

        self.proj_in = nn.Linear(n_attrs*self.dim_in, self.dim_hid)
        self.proj_mid = nn.Linear(self.dim_hid, self.dim_hid)
        self.proj_out = nn.Linear(self.dim_hid, self.dim_out)
        self.attns = nn.ModuleList(
            [torch.nn.MultiheadAttention(dim_hid, n_heads, batch_first=True) 
             for i in range(n_layers)])

    def forward(self, attr, length):
        # input project
        B = attr.shape[0]
        attr = torch.reshape(attr, (B, -1))
        h = F.gelu(self.proj_in(attr))  # (B,d)
        h = F.gelu(self.proj_mid(h))
        h = h.unsqueeze(1)  # (B,d) -> (B,1,d)
        h = h.expand([-1, length, -1])  # (B,L,d)

        # out project
        out = self.proj_out(h)
        return out


class AttrProjectorAvg(nn.Module):
    def __init__(self, dim_in=128, dim_hid=128, dim_out=128):
        super().__init__()
        self.dim_in = dim_in
        self.dim_hid = dim_hid
        self.dim_out = dim_out

        self.proj_out = nn.Linear(self.dim_hid, self.dim_out)

    def forward(self, attr, length):
        # input project
        B = attr.shape[0]
        h = torch.mean(attr, dim=1, keepdim=True)  # (B,1,d)
        h = h.expand([-1, length, -1])  # (B,L,d)

        # out project
        out = self.proj_out(h)
        return out
