import copy
import math
import torch
import torch.nn as nn

from models.diffusion.diff_csdi import Diff_CSDI
from models.diffusion.diff_csdi_multipatch import Diff_CSDI_Patch, Diff_CSDI_MultiPatch, Diff_CSDI_MultiPatch_Parallel
from models.diffusion.diff_csdi_time_weaver import Diff_CSDI_TimeWeaver
from models.encoders.attr_encoder import AttributeEncoder
from models.encoders.side_encoder import SideEncoder
from models.diffusion.diff_csdi_multipatch_weaver import Diff_CSDI_MultiPatch_Weaver_Parallel

from samplers import DDPMSampler, DDIMSampler


class ConditionalGenerator(nn.Module):
    _strength_diag_enabled = False
    _strength_diag_records = []
    _strength_diag_max_records = 32

    @classmethod
    def enable_strength_diagnostics(cls, enabled: bool = True, max_records: int = 32) -> None:
        cls._strength_diag_enabled = bool(enabled)
        cls._strength_diag_max_records = max(1, int(max_records))
        cls._strength_diag_records = []

    @classmethod
    def disable_strength_diagnostics(cls) -> None:
        cls._strength_diag_enabled = False

    @classmethod
    def consume_strength_diagnostics(cls):
        records = copy.deepcopy(cls._strength_diag_records)
        cls._strength_diag_records = []
        return records

    def __init__(self, configs):
        super().__init__()
        self.device = configs["device"]
        strength_cfg = dict(configs.get("diffusion", {}).get("strength_control") or {})
        self.diffusion_loss_weight = float(strength_cfg.get("diffusion_loss_weight", 1.0))
        self.edit_region_loss_weight = float(strength_cfg.get("edit_region_loss_weight", 0.0))
        self.background_loss_weight = float(strength_cfg.get("background_loss_weight", 0.0))
        self.monotonic_loss_weight = float(strength_cfg.get("monotonic_loss_weight", 0.0))
        self.monotonic_margin = float(strength_cfg.get("monotonic_margin", 1.0e-3))
        self.gain_match_loss_weight = float(strength_cfg.get("gain_match_loss_weight", 0.0))
        self.family_gap_match_loss_weight = float(strength_cfg.get("family_gap_match_loss_weight", 0.0))
        self.family_relative_gain_loss_weight = float(strength_cfg.get("family_relative_gain_loss_weight", 0.0))
        self.family_relative_margin_scale = float(strength_cfg.get("family_relative_margin_scale", 1.0))
        self.constant_gain_penalty_weight = float(strength_cfg.get("constant_gain_penalty_weight", 0.0))
        self.minimum_family_gain_std = float(strength_cfg.get("minimum_family_gain_std", 0.0))
        self.numeric_only_loss_weight = float(strength_cfg.get("numeric_only_loss_weight", 0.0))
        self.beta_direction_loss_weight = float(strength_cfg.get("beta_direction_loss_weight", 0.0))
        self.beta_direction_margin = float(strength_cfg.get("beta_direction_margin", 0.0))
        self.beta_direction_target = str(strength_cfg.get("beta_direction_target", "family_signed_gain"))
        output_branch_carrier_cfg = dict(strength_cfg.get("output_branch_carrier") or {})
        self.output_branch_regularizer_weight = float(output_branch_carrier_cfg.get("regularizer_weight", 0.0))
        self.output_branch_scalar_order_weight = float(output_branch_carrier_cfg.get("scalar_order_weight", 0.0))
        final_mapping_cfg = dict(strength_cfg.get("final_output_strength_mapping") or {})
        self.final_output_strength_mapping_order_weight = float(final_mapping_cfg.get("gain_order_weight", 0.0))
        self._latest_loss_breakdown = None

        self._init_condition_encoders(configs["attrs"], configs["side"])
        self._init_diff(configs["diffusion"])

    def _init_condition_encoders(self, configs_attr, configs_side):
        configs_attr["device"] = self.device
        configs_side["device"] = self.device
        self.attr_en = AttributeEncoder(configs_attr).to(self.device)
        self.side_en = SideEncoder(configs_side).to(self.device)

    def _init_diff(self, configs):
        configs["side_dim"] = self.side_en.total_emb_dim
        configs["attr_dim"] = self.attr_en.emb_dim
        configs["n_attrs"] = self.attr_en.n_attr

        # input_dim = 1 if self.is_unconditional == True else 2
        input_dim = 1
        if configs["type"] == "CSDI":
            self.diff_model = Diff_CSDI(configs, input_dim).to(self.device)
        if configs["type"] == "CSDI_Patch":
            self.diff_model = Diff_CSDI_Patch(configs, input_dim).to(self.device)
        if configs["type"] == "CSDI_MultiPatch":
            self.diff_model = Diff_CSDI_MultiPatch(configs, input_dim).to(self.device)
        if configs["type"] == "CSDI_MultiPatch_Parallel":
            self.diff_model = Diff_CSDI_MultiPatch_Parallel(configs, input_dim).to(self.device)
        if configs["type"] == "CSDI_MultiPatch_Weaver_Parallel":
            self.diff_model = Diff_CSDI_MultiPatch_Weaver_Parallel(configs, input_dim).to(self.device)
        if configs["type"] == "CSDI_TimeWeaver":
            self.diff_model = Diff_CSDI_TimeWeaver(configs, input_dim).to(self.device)

        # steps
        self.num_steps = configs["num_steps"]

        # sampler
        self.ddpm = DDPMSampler(self.num_steps, configs["beta_start"], configs["beta_end"], configs["schedule"], self.device)
        self.ddim = DDIMSampler(self.num_steps, configs["beta_start"], configs["beta_end"], configs["schedule"], self.device)

        # edit
        self.edit_steps = configs["edit_steps"]
        self.bootstrap_ratio = configs["bootstrap_ratio"]

    def predict_noise(
        self,
        xt,
        side_emb,
        src_attr_emb,
        t,
        strength_label=None,
        task_id=None,
        text_context=None,
        strength_scalar=None,
        soft_mask=None,
        keys_null=None,
        values_null=None,
        final_strength_mask=None,
    ):
        """
        Predict noise with optional attention injection for soft-boundary editing.
        
        Args:
            xt: Noisy latent [B, K, L]
            side_emb: Side information embedding
            src_attr_emb: Attribute embedding
            t: Diffusion timestep
            soft_mask: [B, L] or [B, L, 1] - Soft boundary mask for attention injection
            keys_null: [B, L, D] - Keys from null/background condition
            values_null: [B, L, D] - Values from null/background condition
        """
        noisy_x = torch.unsqueeze(xt, 1)  # (B,1,K,L): this is required by the diff model
        model_kwargs = {}
        if soft_mask is not None:
            model_kwargs["soft_mask"] = soft_mask
        if keys_null is not None:
            model_kwargs["keys_null"] = keys_null
        if values_null is not None:
            model_kwargs["values_null"] = values_null
        if strength_label is not None:
            model_kwargs["strength_label"] = strength_label
        if task_id is not None:
            model_kwargs["task_id"] = task_id
        if text_context is not None:
            model_kwargs["text_context"] = text_context
        if strength_scalar is not None:
            model_kwargs["strength_scalar"] = strength_scalar
        if final_strength_mask is not None:
            model_kwargs["final_strength_mask"] = final_strength_mask

        pred_noise = self.diff_model(noisy_x, side_emb, src_attr_emb, t, **model_kwargs)  # (B,K,L)
        return pred_noise

    def _output_branch_regularizer_loss(self, reference_tensor):
        branch_loss = getattr(self.diff_model, "latest_output_branch_regularizer_loss", None)
        if torch.is_tensor(branch_loss):
            return branch_loss.to(device=reference_tensor.device, dtype=reference_tensor.dtype)
        return reference_tensor.new_tensor(0.0)

    def _output_branch_scalar_order_loss(self, reference_tensor):
        branch_loss = getattr(self.diff_model, "latest_output_branch_scalar_order_loss", None)
        if torch.is_tensor(branch_loss):
            return branch_loss.to(device=reference_tensor.device, dtype=reference_tensor.dtype)
        return reference_tensor.new_tensor(0.0)

    def _final_output_strength_mapping_order_loss(self, reference_tensor):
        gain_loss = getattr(self.diff_model, "latest_final_output_strength_mapping_order_loss", None)
        if torch.is_tensor(gain_loss):
            return gain_loss.to(device=reference_tensor.device, dtype=reference_tensor.dtype)
        return reference_tensor.new_tensor(0.0)

    def _noise_estimation_loss(self, x, side_emb, attr_emb, t, strength_label=None, task_id=None, text_context=None, strength_scalar=None, final_strength_mask=None):
        noise = torch.randn_like(x)
        noisy_x = self.ddpm.forward(x, t, noise)
        pred_noise = self.predict_noise(
            noisy_x,
            side_emb,
            attr_emb,
            t,
            strength_label=strength_label,
            task_id=task_id,
            text_context=text_context,
            strength_scalar=strength_scalar,
            final_strength_mask=final_strength_mask,
        )
        
        residual = noise - pred_noise
        loss = (residual ** 2).mean()
        return loss
    
    def forward(self, batch, is_train=False, mode="pretrain"):
        """
        Training.
        """
        if mode == "pretrain":
            return self.pretrain(batch, is_train)
        elif mode == "finetune":
            return self.fintune(batch, is_train)
    
    """
    Pretrain.
    """
    def pretrain(self, batch, is_train):
        x, tp, attrs = self._unpack_data_cond_gen(batch)
    
        side_emb = self.side_en(tp)
        attr_emb = self.attr_en(attrs)
        B = x.shape[0]

        if is_train:
            t = torch.randint(0, self.num_steps, [B], device=self.device)
            return self._noise_estimation_loss(x, side_emb, attr_emb, t)
        
        # valid
        loss = 0
        for t in range(self.num_steps):
            t = (torch.ones(B, device=self.device) * t).long()
            loss += self._noise_estimation_loss(x, side_emb, attr_emb, t)
        return loss/self.num_steps

    def _unpack_data_cond_gen(self, batch):
        x = batch["x"].to(self.device).float()
        tp = batch["tp"].to(self.device).float()
        attrs = batch["attrs"].to(self.device).long()
    
        x = x.permute(0, 2, 1)  # (B,L,K) -> (B,K,L)
        return x, tp, attrs

    """
    Finetune.
    """

    def fintune(self, batch, is_train):
        src_x, tp, src_attrs, tgt_attrs, tgt_x, strength_label, strength_scalar, task_id, instruction_text = self._unpack_data_edit(batch)
        final_strength_mask = self._prepare_final_strength_mask(batch, src_x.shape)

        side_emb = self.side_en(tp)
        src_attr_emb = self.attr_en(src_attrs)
        tgt_attr_emb = self.attr_en(tgt_attrs)

        # src -> tgt
        tgt_x_pred = self._edit(
            src_x,
            side_emb,
            src_attr_emb,
            tgt_attr_emb,
            sampler="ddim",
            strength_label=strength_label,
            strength_scalar=strength_scalar,
            task_id=task_id,
            text_context=instruction_text,
            final_strength_mask=final_strength_mask,
        )  # (B,V,L)
        output_branch_regularizer_loss = (
            self._output_branch_regularizer_loss(src_x)
            if self.output_branch_regularizer_weight > 0.0
            else src_x.new_tensor(0.0)
        )
        output_branch_scalar_order_loss = (
            self._output_branch_scalar_order_loss(src_x)
            if self.output_branch_scalar_order_weight > 0.0
            else src_x.new_tensor(0.0)
        )
        final_output_strength_mapping_order_loss = (
            self._final_output_strength_mapping_order_loss(src_x)
            if self.final_output_strength_mapping_order_weight > 0.0
            else src_x.new_tensor(0.0)
        )
        if self.bootstrap_ratio > 0:
            bs_ids = self._bootstrap(tgt_x_pred, src_x, side_emb, src_attr_emb, tgt_attr_emb)  # (B)
        else:
            bs_ids = None

        if is_train:
            # tgt
            if bs_ids is not None:
                t = torch.randint(0, self.num_steps, [len(bs_ids)], device=self.device)
                loss_tgt = self._noise_estimation_loss(
                    tgt_x_pred[bs_ids],
                    side_emb[bs_ids],
                    tgt_attr_emb[bs_ids],
                    t,
                    strength_label=None if strength_label is None else strength_label[bs_ids],
                    strength_scalar=None if strength_scalar is None else strength_scalar[bs_ids],
                    task_id=None if task_id is None else task_id[bs_ids],
                    final_strength_mask=None if final_strength_mask is None else final_strength_mask[bs_ids],
                    text_context=(
                        None
                        if instruction_text is None
                        else [instruction_text[int(i)] for i in bs_ids.detach().cpu().tolist()]
                        if isinstance(instruction_text, list)
                        else instruction_text[bs_ids]
                    ),
                )
            else:
                loss_tgt = 0.0
        else:
            loss_tgt = 0
            B = tgt_x_pred.shape[0]
            for t in range(self.num_steps):
                t = (torch.ones(B, device=self.device) * t).long()
                if self.bootstrap_ratio > 0:
                    loss_tgt += self._noise_estimation_loss(
                        tgt_x_pred,
                        side_emb,
                        tgt_attr_emb,
                        t,
                        strength_label=strength_label,
                        strength_scalar=strength_scalar,
                        task_id=task_id,
                        text_context=instruction_text,
                        final_strength_mask=final_strength_mask,
                    )
            loss_tgt = loss_tgt/self.num_steps
        weighted_loss_tgt = self.diffusion_loss_weight * loss_tgt
        weighted_output_branch_regularizer_loss = self.output_branch_regularizer_weight * output_branch_regularizer_loss
        weighted_output_branch_scalar_order_loss = self.output_branch_scalar_order_weight * output_branch_scalar_order_loss
        weighted_final_output_strength_mapping_order_loss = (
            self.final_output_strength_mapping_order_weight * final_output_strength_mapping_order_loss
        )
        total_loss = (
            weighted_loss_tgt
            + weighted_output_branch_regularizer_loss
            + weighted_output_branch_scalar_order_loss
            + weighted_final_output_strength_mapping_order_loss
        )
        loss_breakdown = {
            "loss_tgt": float(loss_tgt.detach().item()) if isinstance(loss_tgt, torch.Tensor) else float(loss_tgt),
            "diffusion_realism_loss": float(loss_tgt.detach().item()) if isinstance(loss_tgt, torch.Tensor) else float(loss_tgt),
            "weighted_diffusion_realism_loss": float(weighted_loss_tgt.detach().item()) if isinstance(weighted_loss_tgt, torch.Tensor) else float(weighted_loss_tgt),
            "diffusion_loss_weight": float(self.diffusion_loss_weight),
            "output_branch_regularizer_loss": float(output_branch_regularizer_loss.detach().item()),
            "weighted_output_branch_regularizer_loss": float(weighted_output_branch_regularizer_loss.detach().item()),
            "output_branch_regularizer_weight": float(self.output_branch_regularizer_weight),
            "output_branch_scalar_order_loss": float(output_branch_scalar_order_loss.detach().item()),
            "weighted_output_branch_scalar_order_loss": float(weighted_output_branch_scalar_order_loss.detach().item()),
            "output_branch_scalar_order_weight": float(self.output_branch_scalar_order_weight),
            "final_output_strength_mapping_order_loss": float(final_output_strength_mapping_order_loss.detach().item()),
            "weighted_final_output_strength_mapping_order_loss": float(weighted_final_output_strength_mapping_order_loss.detach().item()),
            "final_output_strength_mapping_order_weight": float(self.final_output_strength_mapping_order_weight),
            "edit_region_loss": 0.0,
            "background_loss": 0.0,
            "monotonic_loss": 0.0,
            "gain_match_loss": 0.0,
            "amplitude_control_loss": 0.0,
            "local_fidelity_loss": 0.0,
            "preservation_loss": 0.0,
            "order_regularization_loss": 0.0,
            "weighted_edit_region_loss": 0.0,
            "weighted_background_loss": 0.0,
            "weighted_monotonic_loss": 0.0,
            "weighted_gain_match_loss": 0.0,
            "weighted_amplitude_control_loss": 0.0,
            "weighted_local_fidelity_loss": 0.0,
            "weighted_preservation_loss": 0.0,
            "weighted_order_regularization_loss": 0.0,
            "total_loss": float(total_loss.detach().item()) if isinstance(total_loss, torch.Tensor) else float(total_loss),
            "bootstrap_selected": None if bs_ids is None else int(len(bs_ids)),
            "batch_size": int(tgt_x_pred.shape[0]),
        }
        mask_gt = batch.get("mask_gt")
        family_sizes = batch.get("family_sizes")
        if mask_gt is not None:
            mask_gt = mask_gt.to(self.device).float()
            if mask_gt.dim() == 3:
                mask_gt = mask_gt.permute(0, 2, 1)
            elif mask_gt.dim() == 2:
                mask_gt = mask_gt.unsqueeze(1)
            strength_loss, strength_breakdown = self._strength_supervision_loss(
                src_x=src_x,
                tgt_x=tgt_x,
                tgt_x_pred=tgt_x_pred,
                mask_gt=mask_gt,
                family_sizes=family_sizes,
                strength_label=strength_label,
                strength_scalar=strength_scalar,
                instruction_text=instruction_text,
                return_breakdown=True,
            )
            total_loss = total_loss + strength_loss
            loss_breakdown.update(strength_breakdown)
            loss_breakdown["total_loss"] = float(total_loss.detach().item())
        loss_breakdown["objective_groups"] = {
            "amplitude_control": float(loss_breakdown.get("weighted_amplitude_control_loss", 0.0)),
            "local_fidelity": float(loss_breakdown.get("weighted_local_fidelity_loss", 0.0)),
            "preservation": float(loss_breakdown.get("weighted_preservation_loss", 0.0)),
            "order_regularization": float(loss_breakdown.get("weighted_order_regularization_loss", 0.0)),
            "diffusion_realism": float(loss_breakdown.get("weighted_diffusion_realism_loss", 0.0)),
            "branch_carrier": float(loss_breakdown.get("weighted_output_branch_regularizer_loss", 0.0)),
            "branch_scalar_order": float(loss_breakdown.get("weighted_output_branch_scalar_order_loss", 0.0)),
            "final_strength_mapping": float(loss_breakdown.get("weighted_final_output_strength_mapping_order_loss", 0.0)),
        }
        self._latest_loss_breakdown = loss_breakdown
        return total_loss

    def _masked_mean(self, values, mask, eps=1.0e-8):
        mask = mask.float()
        denom = torch.clamp(mask.sum(dim=(1, 2)), min=eps)
        numer = (values * mask).sum(dim=(1, 2))
        return numer / denom

    def _format_scalar_key(self, value):
        return f"{float(value):.4f}"

    def _safe_spearman(self, x, y):
        if x.numel() < 2 or y.numel() < 2 or x.numel() != y.numel():
            return None
        x = x.detach().float().view(-1)
        y = y.detach().float().view(-1)
        x_rank = torch.argsort(torch.argsort(x)).float()
        y_rank = torch.argsort(torch.argsort(y)).float()
        x_rank = x_rank - x_rank.mean()
        y_rank = y_rank - y_rank.mean()
        denom = torch.sqrt(torch.clamp((x_rank.square().sum() * y_rank.square().sum()), min=1.0e-12))
        if float(denom.item()) <= 1.0e-12:
            return None
        return float((x_rank * y_rank).sum().item() / denom.item())

    def _strength_supervision_loss(self, src_x, tgt_x, tgt_x_pred, mask_gt, family_sizes=None, strength_label=None, strength_scalar=None, instruction_text=None, return_breakdown=False):
        abs_pred_target = torch.abs(tgt_x_pred - tgt_x)
        abs_pred_source = torch.abs(tgt_x_pred - src_x)
        edit_mask = mask_gt
        bg_mask = 1.0 - edit_mask

        edit_region_l1 = src_x.new_tensor(0.0)
        background_l1 = src_x.new_tensor(0.0)
        monotonic_loss = src_x.new_tensor(0.0)
        gain_match_loss = src_x.new_tensor(0.0)
        family_gap_match_loss = src_x.new_tensor(0.0)
        family_relative_gain_loss = src_x.new_tensor(0.0)
        constant_gain_penalty = src_x.new_tensor(0.0)
        numeric_only_gain_match_loss = src_x.new_tensor(0.0)
        beta_direction_loss = src_x.new_tensor(0.0)
        weighted_losses = []

        gains = self._masked_mean(abs_pred_source, edit_mask)
        target_gains = self._masked_mean(torch.abs(tgt_x - src_x), edit_mask)

        if self.edit_region_loss_weight > 0.0:
            edit_region_l1 = self._masked_mean(abs_pred_target, edit_mask).mean()
            weighted_losses.append(self.edit_region_loss_weight * edit_region_l1)
        if self.background_loss_weight > 0.0:
            background_l1 = self._masked_mean(abs_pred_source, bg_mask).mean()
            weighted_losses.append(self.background_loss_weight * background_l1)
        if self.gain_match_loss_weight > 0.0:
            gain_match_loss = torch.mean(torch.abs(gains - target_gains))
            weighted_losses.append(self.gain_match_loss_weight * gain_match_loss)

        family_monotonic_losses = []
        family_gap_losses = []
        family_relative_losses = []
        family_constant_penalties = []
        family_beta_direction_losses = []
        family_hits = []
        family_rhos = []
        family_hard_cases = []
        numeric_only_mask = None
        if isinstance(instruction_text, list):
            numeric_only_mask = torch.tensor(
                [len(str(text).strip()) == 0 for text in instruction_text],
                device=src_x.device,
                dtype=torch.bool,
            )
        elif isinstance(instruction_text, str):
            numeric_only_mask = torch.tensor(
                [len(str(instruction_text).strip()) == 0] * gains.shape[0],
                device=src_x.device,
                dtype=torch.bool,
            )

        if family_sizes is not None and strength_scalar is not None:
            start_idx = 0
            for family_size in family_sizes.detach().cpu().tolist():
                family_size = int(family_size)
                family_slice = slice(start_idx, start_idx + family_size)
                family_scalar = strength_scalar[family_slice].view(-1)
                family_gains = gains[family_slice]
                family_target_gains = target_gains[family_slice]
                family_numeric_only = None if numeric_only_mask is None else numeric_only_mask[family_slice]
                start_idx += family_size
                if family_scalar.numel() < 2:
                    continue
                order = torch.argsort(family_scalar)
                family_scalar = family_scalar[order]
                family_gains = family_gains[order]
                family_target_gains = family_target_gains[order]
                if family_numeric_only is not None:
                    family_numeric_only = family_numeric_only[order]

                family_monotonic_hit = float(all(
                    float(family_gains[idx + 1].item()) + 1.0e-6 >= float(family_gains[idx].item())
                    for idx in range(family_gains.numel() - 1)
                ))
                family_hits.append(family_monotonic_hit)
                rho = self._safe_spearman(family_scalar, family_gains)
                if rho is not None:
                    family_rhos.append(float(rho))

                pred_gain_seq = [float(value) for value in family_gains.detach().cpu().tolist()]
                target_gain_seq = [float(value) for value in family_target_gains.detach().cpu().tolist()]
                strength_scalar_seq = [float(value) for value in family_scalar.detach().cpu().tolist()]
                pred_minus_target_seq = []
                pred_gap_seq = []
                target_gap_seq = []
                gap_error_seq = []

                for idx in range(family_gains.numel() - 1):
                    pred_gap = family_gains[idx + 1] - family_gains[idx]
                    target_gap = family_target_gains[idx + 1] - family_target_gains[idx]
                    required_gap = torch.clamp(target_gap * self.family_relative_margin_scale, min=self.monotonic_margin)
                    pred_gap_seq.append(float(pred_gap.detach().item()))
                    target_gap_seq.append(float(target_gap.detach().item()))
                    gap_error_seq.append(float(torch.abs(pred_gap - target_gap).detach().item()))
                    if self.monotonic_loss_weight > 0.0:
                        family_monotonic_losses.append(torch.relu(required_gap - pred_gap))
                    if self.family_gap_match_loss_weight > 0.0:
                        family_gap_losses.append(torch.abs(pred_gap - target_gap))
                    if self.family_relative_gain_loss_weight > 0.0:
                        norm_gap = required_gap / torch.clamp(family_target_gains[idx + 1] + family_target_gains[idx], min=1.0e-6)
                        pred_norm_gap = pred_gap / torch.clamp(family_gains[idx + 1].detach() + family_gains[idx].detach(), min=1.0e-6)
                        family_relative_losses.append(torch.abs(pred_norm_gap - norm_gap))
                    if self.beta_direction_loss_weight > 0.0 and self.beta_direction_target == "family_signed_gain":
                        direction_target = torch.sign(target_gap.detach())
                        if float(direction_target.abs().item()) > 0.0:
                            direction_margin = torch.clamp(target_gap.detach().abs(), min=self.beta_direction_margin)
                            family_beta_direction_losses.append(torch.relu(direction_margin - (direction_target * pred_gap)))

                pred_minus_target_seq = [
                    float(pred_value - target_value)
                    for pred_value, target_value in zip(pred_gain_seq, target_gain_seq)
                ]
                family_hard_cases.append({
                    "strength_scalar_seq": strength_scalar_seq,
                    "pred_gain_seq": pred_gain_seq,
                    "target_gain_seq": target_gain_seq,
                    "pred_minus_target_seq": pred_minus_target_seq,
                    "pred_gap_seq": pred_gap_seq,
                    "target_gap_seq": target_gap_seq,
                    "family_monotonic_hit": family_monotonic_hit,
                    "family_spearman": None if rho is None else float(rho),
                    "worst_gap_error": float(max(gap_error_seq)) if gap_error_seq else 0.0,
                    "worst_gain_error": float(max(abs(value) for value in pred_minus_target_seq)) if pred_minus_target_seq else 0.0,
                })

                if self.constant_gain_penalty_weight > 0.0 and family_gains.numel() > 1:
                    family_gain_std = torch.std(family_gains, unbiased=False)
                    family_constant_penalties.append(torch.relu(self.minimum_family_gain_std - family_gain_std))

                if (
                    self.numeric_only_loss_weight > 0.0
                    and family_numeric_only is not None
                    and bool(torch.any(family_numeric_only).item())
                ):
                    numeric_family_gains = family_gains[family_numeric_only]
                    numeric_family_targets = family_target_gains[family_numeric_only]
                    if numeric_family_gains.numel() > 0:
                        family_loss = torch.mean(torch.abs(numeric_family_gains - numeric_family_targets))
                        weighted_losses.append(self.numeric_only_loss_weight * family_loss)
                        numeric_only_gain_match_loss = numeric_only_gain_match_loss + family_loss

        if family_monotonic_losses:
            monotonic_loss = torch.stack(family_monotonic_losses).mean()
            weighted_losses.append(self.monotonic_loss_weight * monotonic_loss)
        if family_gap_losses:
            family_gap_match_loss = torch.stack(family_gap_losses).mean()
            weighted_losses.append(self.family_gap_match_loss_weight * family_gap_match_loss)
        if family_relative_losses:
            family_relative_gain_loss = torch.stack(family_relative_losses).mean()
            weighted_losses.append(self.family_relative_gain_loss_weight * family_relative_gain_loss)
        if family_constant_penalties:
            constant_gain_penalty = torch.stack(family_constant_penalties).mean()
            weighted_losses.append(self.constant_gain_penalty_weight * constant_gain_penalty)
        if family_beta_direction_losses:
            beta_direction_loss = torch.stack(family_beta_direction_losses).mean()
            weighted_losses.append(self.beta_direction_loss_weight * beta_direction_loss)

        diag_payload = {
            'train_edit_gain_mean': float(gains.detach().mean().item()),
            'train_target_gain_mean': float(target_gains.detach().mean().item()),
            'train_gain_gap_abs_mean': float(torch.abs(gains - target_gains).detach().mean().item()),
            'train_family_gap_match_loss': float(family_gap_match_loss.detach().item()),
            'train_family_relative_gain_loss': float(family_relative_gain_loss.detach().item()),
            'train_constant_gain_penalty': float(constant_gain_penalty.detach().item()),
            'train_numeric_only_gain_match_loss': float(numeric_only_gain_match_loss.detach().item()),
            'train_beta_direction_loss': float(beta_direction_loss.detach().item()),
        }
        if family_hard_cases:
            sorted_hard_cases = sorted(
                family_hard_cases,
                key=lambda row: (
                    0 if not row['family_monotonic_hit'] else 1,
                    -float(row['worst_gap_error']),
                    -float(row['worst_gain_error']),
                ),
            )
            diag_payload['train_family_hard_cases'] = sorted_hard_cases[: min(3, len(sorted_hard_cases))]
            diag_payload['train_family_worst_gap_error_mean'] = float(
                sum(float(row['worst_gap_error']) for row in family_hard_cases) / len(family_hard_cases)
            )
            diag_payload['train_family_worst_gain_error_mean'] = float(
                sum(float(row['worst_gain_error']) for row in family_hard_cases) / len(family_hard_cases)
            )
            diag_payload['train_family_non_monotonic_count'] = int(
                sum(1 for row in family_hard_cases if not row['family_monotonic_hit'])
            )
            valid_family_spearman = [float(row['family_spearman']) for row in family_hard_cases if row['family_spearman'] is not None]
            if valid_family_spearman:
                diag_payload['train_family_spearman_min'] = float(min(valid_family_spearman))

        if strength_scalar is not None:
            scalar_diag = {}
            target_scalar_diag = {}
            scalar_view = strength_scalar.detach().float().view(-1)
            unique_scalars = sorted({float(value) for value in scalar_view.detach().cpu().tolist()})
            for scalar_value in unique_scalars:
                scalar_mask = torch.isclose(strength_scalar.view(-1), strength_scalar.new_tensor(float(scalar_value)), atol=1.0e-6, rtol=0.0)
                if int(scalar_mask.sum().item()) == 0:
                    continue
                scalar_key = self._format_scalar_key(scalar_value)
                scalar_diag[scalar_key] = float(gains[scalar_mask].detach().mean().item())
                target_scalar_diag[scalar_key] = float(target_gains[scalar_mask].detach().mean().item())
            diag_payload['train_edit_gain_by_scalar'] = scalar_diag
            diag_payload['train_target_gain_by_scalar'] = target_scalar_diag
            diag_payload['train_strength_scalar_mean'] = float(strength_scalar.detach().float().mean().item())
            if strength_scalar.numel() > 1:
                scalar_vec = strength_scalar.detach().float().view(-1)
                gain_vec = gains.detach().float().view(-1)
                scalar_centered = scalar_vec - scalar_vec.mean()
                gain_centered = gain_vec - gain_vec.mean()
                denom = torch.sqrt(torch.clamp((scalar_centered.square().sum() * gain_centered.square().sum()), min=1.0e-12))
                diag_payload['train_edit_gain_scalar_corr'] = float((scalar_centered * gain_centered).sum().item() / denom.item())
                spearman = self._safe_spearman(scalar_vec, gain_vec)
                if spearman is not None:
                    diag_payload['train_edit_gain_scalar_spearman'] = spearman
        if strength_label is not None:
            strength_diag = {}
            target_strength_diag = {}
            for label in torch.unique(strength_label.detach()).detach().cpu().tolist():
                label = int(label)
                label_mask = strength_label == label
                if int(label_mask.sum().item()) == 0:
                    continue
                strength_diag[str(label)] = float(gains[label_mask].detach().mean().item())
                target_strength_diag[str(label)] = float(target_gains[label_mask].detach().mean().item())
            diag_payload['train_edit_gain_by_strength'] = strength_diag
            diag_payload['train_target_gain_by_strength'] = target_strength_diag
        if family_hits:
            diag_payload['train_family_monotonic_hit_rate'] = float(sum(family_hits) / len(family_hits))
        if family_rhos:
            diag_payload['train_family_gain_scalar_spearman_mean'] = float(sum(family_rhos) / len(family_rhos))
        if numeric_only_mask is not None:
            diag_payload['train_numeric_only_batch_fraction'] = float(numeric_only_mask.float().mean().item())
        if self.__class__._strength_diag_enabled:
            self._record_strength_diagnostics(diag_payload)

        if weighted_losses:
            total_loss = torch.stack(weighted_losses).sum()
        else:
            total_loss = src_x.new_tensor(0.0)
        if not return_breakdown:
            return total_loss
        weighted_gain_match_loss = float((self.gain_match_loss_weight * gain_match_loss).detach().item()) if self.gain_match_loss_weight > 0.0 else 0.0
        weighted_family_gap_match_loss = float((self.family_gap_match_loss_weight * family_gap_match_loss).detach().item()) if self.family_gap_match_loss_weight > 0.0 else 0.0
        weighted_family_relative_gain_loss = float((self.family_relative_gain_loss_weight * family_relative_gain_loss).detach().item()) if self.family_relative_gain_loss_weight > 0.0 else 0.0
        weighted_constant_gain_penalty = float((self.constant_gain_penalty_weight * constant_gain_penalty).detach().item()) if self.constant_gain_penalty_weight > 0.0 else 0.0
        weighted_numeric_only_gain_match_loss = float((self.numeric_only_loss_weight * numeric_only_gain_match_loss).detach().item()) if self.numeric_only_loss_weight > 0.0 else 0.0
        weighted_beta_direction_loss = float((self.beta_direction_loss_weight * beta_direction_loss).detach().item()) if self.beta_direction_loss_weight > 0.0 else 0.0
        weighted_amplitude_control_loss = (
            weighted_gain_match_loss
            + weighted_family_gap_match_loss
            + weighted_family_relative_gain_loss
            + weighted_constant_gain_penalty
            + weighted_numeric_only_gain_match_loss
            + weighted_beta_direction_loss
        )
        breakdown = {
            'edit_region_loss': float(edit_region_l1.detach().item()),
            'background_loss': float(background_l1.detach().item()),
            'monotonic_loss': float(monotonic_loss.detach().item()),
            'gain_match_loss': float(gain_match_loss.detach().item()),
            'family_gap_match_loss': float(family_gap_match_loss.detach().item()),
            'family_relative_gain_loss': float(family_relative_gain_loss.detach().item()),
            'constant_gain_penalty': float(constant_gain_penalty.detach().item()),
            'numeric_only_gain_match_loss': float(numeric_only_gain_match_loss.detach().item()),
            'beta_direction_loss': float(beta_direction_loss.detach().item()),
            'amplitude_control_loss': float((gain_match_loss + family_gap_match_loss + family_relative_gain_loss + constant_gain_penalty + numeric_only_gain_match_loss + beta_direction_loss).detach().item()),
            'local_fidelity_loss': float(edit_region_l1.detach().item()),
            'preservation_loss': float(background_l1.detach().item()),
            'order_regularization_loss': float(monotonic_loss.detach().item()),
            'weighted_edit_region_loss': float((self.edit_region_loss_weight * edit_region_l1).detach().item()) if self.edit_region_loss_weight > 0.0 else 0.0,
            'weighted_background_loss': float((self.background_loss_weight * background_l1).detach().item()) if self.background_loss_weight > 0.0 else 0.0,
            'weighted_monotonic_loss': float((self.monotonic_loss_weight * monotonic_loss).detach().item()) if self.monotonic_loss_weight > 0.0 else 0.0,
            'weighted_gain_match_loss': weighted_gain_match_loss,
            'weighted_family_gap_match_loss': weighted_family_gap_match_loss,
            'weighted_family_relative_gain_loss': weighted_family_relative_gain_loss,
            'weighted_constant_gain_penalty': weighted_constant_gain_penalty,
            'weighted_numeric_only_gain_match_loss': weighted_numeric_only_gain_match_loss,
            'weighted_beta_direction_loss': weighted_beta_direction_loss,
            'weighted_amplitude_control_loss': weighted_amplitude_control_loss,
            'weighted_local_fidelity_loss': float((self.edit_region_loss_weight * edit_region_l1).detach().item()) if self.edit_region_loss_weight > 0.0 else 0.0,
            'weighted_preservation_loss': float((self.background_loss_weight * background_l1).detach().item()) if self.background_loss_weight > 0.0 else 0.0,
            'weighted_order_regularization_loss': float((self.monotonic_loss_weight * monotonic_loss).detach().item()) if self.monotonic_loss_weight > 0.0 else 0.0,
            'family_gap_match_loss_weight': float(self.family_gap_match_loss_weight),
            'family_relative_gain_loss_weight': float(self.family_relative_gain_loss_weight),
            'constant_gain_penalty_weight': float(self.constant_gain_penalty_weight),
            'numeric_only_loss_weight': float(self.numeric_only_loss_weight),
            'beta_direction_loss_weight': float(self.beta_direction_loss_weight),
            'strength_total_loss': float(total_loss.detach().item()),
        }
        return total_loss, breakdown

    def _record_strength_diagnostics(self, payload):
        if not self.__class__._strength_diag_enabled:
            return
        if len(self.__class__._strength_diag_records) >= self.__class__._strength_diag_max_records:
            return
        self.__class__._strength_diag_records.append(copy.deepcopy(payload))

    def _bootstrap(self, tgt_x_pred, src_x, side_emb, src_attr_emb, tgt_attr_emb):
        """
        Translate tgt_x_pred back to src_x_pred.
        Calculate similarity score between src_x and src_x_pred as the confidence score for tgt_x_pred.
        Return the idx of top bootstrap_ratio samples.
        """
        B = src_x.shape[0]
        tgt_x_pred.detach()

        with torch.no_grad():
            src_x_pred = self._edit(tgt_x_pred, side_emb, tgt_attr_emb, src_attr_emb, sampler="ddim")
            src_pred = src_x_pred.detach()
        score = -torch.sum(torch.sum((src_pred - src_x)**2, dim=-1), dim=-1)  # (B)
        
        B_bs = int(B*self.bootstrap_ratio)
        ids = torch.topk(score, B_bs, dim=0)[1]  # select top B_bs samples
        return ids        

    def _unpack_data_edit(self, batch):
        src_x = batch["src_x"].to(self.device).float()
        src_attrs = batch["src_attrs"].to(self.device).long()
        
        tgt_x = batch["tgt_x"].to(self.device).float()
        tgt_attrs = batch["tgt_attrs"].to(self.device).long()

        tp = batch["tp"].to(self.device).float()
        
        src_x = src_x.permute(0, 2, 1)  # (B,L,K) -> (B,K,L)
        tgt_x = tgt_x.permute(0, 2, 1)
        strength_label = batch.get("strength_label")
        if strength_label is not None:
            strength_label = strength_label.to(self.device).long()
        strength_scalar = batch.get("strength_scalar")
        if strength_scalar is not None:
            strength_scalar = strength_scalar.to(self.device).float().view(-1)
        task_id = batch.get("task_id")
        if task_id is not None:
            task_id = task_id.to(self.device).long()
        instruction_text = batch.get("instruction_text")
        if isinstance(instruction_text, torch.Tensor):
            instruction_text = instruction_text.to(self.device).float()
        elif isinstance(instruction_text, str):
            instruction_text = [instruction_text]
        elif isinstance(instruction_text, (list, tuple)):
            instruction_text = [str(item) for item in instruction_text]
        else:
            instruction_text = None
        return src_x, tp, src_attrs, tgt_attrs, tgt_x, strength_label, strength_scalar, task_id, instruction_text

    def _prepare_final_strength_mask(self, batch, x_shape):
        mask = batch.get("mask_gt")
        if mask is None:
            return None
        mask = mask.to(self.device).float()
        batch_size, num_vars, seq_len = x_shape
        if mask.dim() == 2:
            mask = mask.unsqueeze(1)
        elif mask.dim() == 3:
            if mask.shape[1] == seq_len and mask.shape[2] == num_vars:
                mask = mask.permute(0, 2, 1)
            elif mask.shape[1] == num_vars and mask.shape[2] == seq_len:
                pass
            elif mask.shape[-1] == 1 and mask.shape[1] == seq_len:
                mask = mask.permute(0, 2, 1)
        if mask.shape[0] != batch_size:
            raise ValueError(f"mask_gt batch size {mask.shape[0]} does not match input batch size {batch_size}")
        if mask.shape[1] == 1 and num_vars != 1:
            mask = mask.expand(-1, num_vars, -1)
        if tuple(mask.shape) != (batch_size, num_vars, seq_len):
            raise ValueError(f"mask_gt shape {tuple(mask.shape)} cannot broadcast to {(batch_size, num_vars, seq_len)}")
        return mask

    """
    Generation.
    """
    @torch.no_grad()
    def generate(self, batch, n_samples, mode="edit", sampler="ddim", return_diagnostics=False):
        return self.__getattribute__(mode)(batch, n_samples, sampler, return_diagnostics=return_diagnostics)

    def _prepare_generation_output(self, samples, diagnostics=None, return_diagnostics=False):
        if return_diagnostics:
            return samples, diagnostics
        return samples

    def _stack_sample_outputs(self, outputs, return_diagnostics=False):
        sample_tensors = [item["sample"] for item in outputs]
        if not return_diagnostics:
            return torch.stack(sample_tensors)
        diagnostics = [item["diagnostics"] for item in outputs]
        return torch.stack(sample_tensors), diagnostics

    def cond_gen(self, batch, n_samples, sampler="ddpm", return_diagnostics=False):
        src_x, tp, src_attrs, tgt_attrs, tgt_x, strength_label, strength_scalar, task_id, instruction_text = self._unpack_data_edit(batch)

        side_emb = self.side_en(tp)
        attr_emb = self.attr_en(tgt_attrs)

        samples = []
        B = src_x.shape[0]
        for i in range(n_samples):
            x = torch.randn_like(src_x)
            for t in range(self.num_steps-1, -1, -1):
                noise = torch.randn_like(x)  # noise for std
                t_tensor = torch.full((B,), t, device=self.device, dtype=torch.long)
                pred_noise = self.predict_noise(
                    x,
                    side_emb,
                    attr_emb,
                    t_tensor,
                    strength_label=strength_label,
                    task_id=task_id,
                    text_context=instruction_text,
                )
                if sampler == "ddpm":
                    x = self.ddpm.reverse(x, pred_noise, t_tensor, noise)
                else:
                    x = self.ddim.reverse(x, pred_noise, t_tensor, noise, is_determin=True)
            samples.append(x)
        return torch.stack(samples)
    
    def edit(self, batch, n_samples, sampler="ddim-ddim", return_diagnostics=False):
        """
        Args:
           sampler: forward-backward: ddim-ddim or ddim.
        """
        src_x, tp, src_attrs, tgt_attrs, tgt_x, strength_label, strength_scalar, task_id, instruction_text = self._unpack_data_edit(batch)
        final_strength_mask = self._prepare_final_strength_mask(batch, src_x.shape)

        side_emb = self.side_en(tp)
        src_attr_emb = self.attr_en(src_attrs)
        tgt_attr_emb = self.attr_en(tgt_attrs)

        samples = []
        for i in range(n_samples):
            tgt_x_pred = self._edit(
                src_x,
                side_emb,
                src_attr_emb,
                tgt_attr_emb,
                sampler,
                strength_label=strength_label,
                strength_scalar=strength_scalar,
                task_id=task_id,
                text_context=instruction_text,
                final_strength_mask=final_strength_mask,
                return_diagnostics=return_diagnostics,
            )
            if return_diagnostics:
                sample_tensor, diagnostics = tgt_x_pred
                samples.append({"sample": sample_tensor, "diagnostics": diagnostics})
            else:
                samples.append({"sample": tgt_x_pred, "diagnostics": None})
        return self._stack_sample_outputs(samples, return_diagnostics=return_diagnostics)

    def _edit(self, src_x, side_emb, src_attr_emb, tgt_attr_emb, sampler, strength_label=None, strength_scalar=None, task_id=None, text_context=None, final_strength_mask=None, return_diagnostics=False):
        B = src_x.shape[0]

        # forward
        xt = src_x
        if sampler[:4] == "ddpm":
            noise = torch.randn_like(src_x)
            xt = self.ddpm.forward(xt, self.edit_steps-1, noise=noise)
        else:
            for t in range(-1, self.edit_steps-1):
                if t == -1:
                    pred_noise = 0
                    t = (torch.ones(B, device=self.device) * t).long()
                else:
                    t = (torch.ones(B, device=self.device) * t).long()
                    pred_noise = self.predict_noise(
                        xt,
                        side_emb,
                        src_attr_emb,
                        t,
                        strength_label=strength_label,
                        strength_scalar=strength_scalar,
                        task_id=task_id,
                        text_context=text_context,
                        final_strength_mask=final_strength_mask,
                    )
                xt = self.ddim.forward(xt, pred_noise, t)

        # reverse
        raw_reverse = None
        for t in range(self.edit_steps-1, -1, -1):
            noise = torch.randn_like(xt)
            t = (torch.ones(B, device=self.device) * t).long()
            pred_noise = self.predict_noise(
                xt,
                side_emb,
                tgt_attr_emb,
                t,
                strength_label=strength_label,
                strength_scalar=strength_scalar,
                task_id=task_id,
                text_context=text_context,
                final_strength_mask=final_strength_mask,
            )
            if sampler[-4:] == "ddpm":
                xt = self.ddpm.reverse(xt, pred_noise, t, noise)
            else:
                xt = self.ddim.reverse(xt, pred_noise, t, noise, is_determin=True)
            if t[0].item() == 0:
                raw_reverse = xt.detach().clone()
        if return_diagnostics:
            final_output = xt
            abs_delta = torch.abs(final_output - src_x)
            diagnostics = {
                "raw_reverse_output": raw_reverse.detach().cpu(),
                "blended_output": final_output.detach().cpu(),
                "final_output": final_output.detach().cpu(),
                "raw_edit_region_mean_abs_delta": float(abs_delta.mean().item()),
                "final_edit_region_mean_abs_delta": float(abs_delta.mean().item()),
                "raw_background_mean_abs_delta": 0.0,
                "final_background_mean_abs_delta": 0.0,
                "blend_gap_edit_region_mean_abs": 0.0,
                "blend_gap_background_mean_abs": 0.0,
            }
            self._record_strength_diagnostics(diagnostics)
            return final_output, diagnostics
        return xt

    @staticmethod
    def create_soft_mask(hard_mask, smooth_width=5, smooth_type="gaussian"):
        """
        Create soft boundary mask from hard mask with smooth transition.
        
        This implements the soft boundary processing to avoid cliff effect:
        - Hard mask: 0 or 1 values cause abrupt transitions
        - Soft mask: Smooth transition (0 → 0.2 → 0.5 → 0.8 → 1) at boundaries
        
        Args:
            hard_mask: Binary mask (numpy array, shape: [L]), 1=edit, 0=preserve
            smooth_width: Width of smooth transition region
            smooth_type: "gaussian" or "linear"
        
        Returns:
            numpy.ndarray: Soft boundary mask with smooth transitions
        """
        import numpy as np
        from scipy.ndimage import gaussian_filter1d
        
        hard_mask = np.array(hard_mask, dtype=np.float32)
        
        if smooth_type == "gaussian":
            soft_mask = gaussian_filter1d(hard_mask, sigma=smooth_width / 3.0)
        elif smooth_type == "linear":
            kernel = np.ones(smooth_width) / smooth_width
            soft_mask = np.convolve(hard_mask, kernel, mode='same')
        else:
            soft_mask = hard_mask
            
        soft_mask = np.clip(soft_mask, 0.0, 1.0)
        
        return soft_mask

    def edit_soft(self, batch, n_samples, sampler="ddim", soft_mask=None,
                  hard_mask=None, smooth_width=5, smooth_type="gaussian", return_diagnostics=False):
        """
        Entry point: Execute local editing with soft boundary mask.
        
        Implements Noise/Score Blending for variance-preserving editing.
        
        Args:
            batch: Data batch containing src_x, src_attrs, tgt_attrs, tp
            n_samples: Number of samples to generate
            sampler: Sampler type ("ddim" or "ddpm")
            soft_mask: Soft boundary mask (numpy array, shape: [L])
            hard_mask: Hard boundary mask (numpy array, shape: [L]) - will be converted to soft
            smooth_width: Width of smooth transition region
            smooth_type: "gaussian" or "linear"
        
        Returns:
            torch.Tensor: Edited samples (n_samples, B, K, L)
        """
        src_x, tp, src_attrs, tgt_attrs, tgt_x, strength_label, strength_scalar, task_id, instruction_text = self._unpack_data_edit(batch)

        side_emb = self.side_en(tp)
        src_attr_emb = self.attr_en(src_attrs)
        tgt_attr_emb = self.attr_en(tgt_attrs)
        
        # Convert hard mask to soft mask if needed
        if soft_mask is None and hard_mask is not None:
            soft_mask = self.create_soft_mask(hard_mask, smooth_width, smooth_type)

        samples = []
        for i in range(n_samples):
            tgt_x_pred = self._edit_soft(
                src_x,
                side_emb,
                src_attr_emb,
                tgt_attr_emb,
                sampler,
                soft_mask,
                strength_label=strength_label,
                strength_scalar=strength_scalar,
                task_id=task_id,
                text_context=instruction_text,
                return_diagnostics=return_diagnostics,
            )
            if return_diagnostics:
                sample_tensor, diagnostics = tgt_x_pred
                samples.append({"sample": sample_tensor, "diagnostics": diagnostics})
            else:
                samples.append({"sample": tgt_x_pred, "diagnostics": None})

        return self._stack_sample_outputs(samples, return_diagnostics=return_diagnostics)

    @torch.no_grad()
    def _edit_soft(self, src_x, side_emb, src_attr_emb, tgt_attr_emb, sampler, soft_mask, strength_label=None, strength_scalar=None, task_id=None, text_context=None, return_diagnostics=False):
        """
        Core Innovation: Latent Blending (State-Space Mixing) for Ground-Truth Preservation.

        Math formula:
        z_{t-1} = M ⊙ z_{t-1}^{pred} + (1-M) ⊙ z_{t-1}^{GT}

        Key Mechanism:
        - Foreground (z^{pred}): Denoised from TEdit conditioned on edit prompt.
        - Background (z^{GT}): Forward-diffused DIRECTLY from original time series x_0:
              z_{t-1}^{GT} = sqrt(ᾱ_{t-1}) * x_0 + sqrt(1-ᾱ_{t-1}) * ε_fixed
          This is the PHYSICAL TRUTH (closed-form, no model involved), guaranteeing
          zero reconstruction error outside the edit region.

        Bug fixed: Previous version used a model prediction with src_attr_emb as the
        "background GT", which introduced reconstruction errors outside the edit region.
        The correct approach is to directly forward-diffuse src_x to step t-1.

        Args:
            src_x: Source time series (B, K, L) - Ground Truth
            side_emb: Side information embedding
            src_attr_emb: Source attribute embedding (unused for background, kept for signature)
            tgt_attr_emb: Target attribute embedding (for foreground generation)
            sampler: Sampler type
            soft_mask: Soft boundary mask (numpy array, shape: [L])

        Returns:
            torch.Tensor: Edited time series (B, K, L)
        """
        B, K, L = src_x.shape

        # Convert numpy mask to tensor: (1, 1, L)
        mask_tensor = torch.as_tensor(soft_mask, dtype=torch.float32, device=self.device)
        mask_tensor = mask_tensor.view(1, 1, L)

        # Prepare soft_mask for attention injection: (B, L)
        soft_mask_attn = torch.as_tensor(soft_mask, dtype=torch.float32, device=self.device)
        soft_mask_attn = soft_mask_attn.unsqueeze(0).expand(B, -1)  # (B, L)

        # ==================== Step 1: Forward Diffusion to obtain starting noisy latent ====================
        # Pre-sample a FIXED background noise (reused every step to keep the trajectory consistent)
        noise_bg = torch.randn_like(src_x)

        xt = src_x.clone()
        if sampler[:4] == "ddpm":
            xt = self.ddpm.forward(xt, self.edit_steps - 1, noise=noise_bg)
        else:
            # DDIM Inversion: deterministic forward pass
            for t in range(-1, self.edit_steps - 1):
                if t == -1:
                    pred_noise = 0
                    t_tensor = (torch.ones(B, device=self.device) * 0).long()
                else:
                    t_tensor = (torch.ones(B, device=self.device) * t).long()
                    pred_noise = self.predict_noise(xt, side_emb, src_attr_emb, t_tensor)
                xt = self.ddim.forward(xt, pred_noise, t_tensor)

        xt_orig = xt.clone()  # Store noisy latent for attention injection

        raw_reverse = None
        blended_reverse = None

        # ==================== Step 2: Reverse Denoising with Latent Blending ====================
        # At every step t:
        #   z_{t-1}^{GT} = sqrt(ᾱ_{t-1}) * x_0 + sqrt(1-ᾱ_{t-1}) * ε_fixed   (direct formula)
        #   z_{t-1}      = M ⊙ z_{t-1}^{pred} + (1-M) ⊙ z_{t-1}^{GT}
        for t in range(self.edit_steps - 1, -1, -1):
            noise = torch.randn_like(xt)
            t_tensor = (torch.ones(B, device=self.device) * t).long()

            # Foreground: model prediction with target attributes
            pred_noise_tgt = self.predict_noise(
                xt, side_emb, tgt_attr_emb, t_tensor,
                strength_label=strength_label,
                strength_scalar=strength_scalar,
                task_id=task_id,
                text_context=text_context,
                soft_mask=soft_mask_attn,
                keys_null=xt_orig,
                values_null=xt_orig,
            )
            if sampler[-4:] == "ddpm":
                xt_pred = self.ddpm.reverse(xt, pred_noise_tgt, t_tensor, noise)
            else:
                xt_pred = self.ddim.reverse(xt, pred_noise_tgt, t_tensor, noise, is_determin=True)

            # Background (GROUND TRUTH): direct forward diffusion of original src_x to step t-1
            # Using the closed-form q(x_{t-1} | x_0) = N(sqrt(ᾱ_{t-1}) * x_0, (1-ᾱ_{t-1}) * I)
            # This guarantees ZERO reconstruction error outside the edit region.
            if t > 0:
                t_prev_tensor = (torch.ones(B, device=self.device) * (t - 1)).long()
                xt_gt_step = self.ddpm.forward(src_x, t_prev_tensor, noise=noise_bg)
            else:
                # At t=0: background is exactly the original signal (no noise)
                xt_gt_step = src_x

            # ==================== CORE: Latent Blending ====================
            # z_{t-1} = M ⊙ z_{t-1}^{pred} + (1-M) ⊙ z_{t-1}^{GT}
            xt = mask_tensor * xt_pred + (1.0 - mask_tensor) * xt_gt_step
            if t == 0:
                raw_reverse = xt_pred.detach().clone()
                blended_reverse = xt.detach().clone()

        final_output = xt
        diagnostics = None
        if return_diagnostics:
            raw_edit_delta = torch.abs(raw_reverse - src_x)
            final_edit_delta = torch.abs(final_output - src_x)
            blend_gap = torch.abs(final_output - raw_reverse)
            edit_mask = mask_tensor.expand(B, K, L)
            bg_mask = (1.0 - mask_tensor).expand(B, K, L)
            diagnostics = {
                "raw_reverse_output": raw_reverse.detach().cpu(),
                "blended_output": blended_reverse.detach().cpu(),
                "final_output": final_output.detach().cpu(),
                "raw_edit_region_mean_abs_delta": float((raw_edit_delta * edit_mask).sum().item() / torch.clamp(edit_mask.sum(), min=1.0).item()),
                "final_edit_region_mean_abs_delta": float((final_edit_delta * edit_mask).sum().item() / torch.clamp(edit_mask.sum(), min=1.0).item()),
                "raw_background_mean_abs_delta": float((raw_edit_delta * bg_mask).sum().item() / torch.clamp(bg_mask.sum(), min=1.0).item()),
                "final_background_mean_abs_delta": float((final_edit_delta * bg_mask).sum().item() / torch.clamp(bg_mask.sum(), min=1.0).item()),
                "blend_gap_edit_region_mean_abs": float((blend_gap * edit_mask).sum().item() / torch.clamp(edit_mask.sum(), min=1.0).item()),
                "blend_gap_background_mean_abs": float((blend_gap * bg_mask).sum().item() / torch.clamp(bg_mask.sum(), min=1.0).item()),
            }
            self._record_strength_diagnostics(diagnostics)
            return final_output, diagnostics

        return final_output
