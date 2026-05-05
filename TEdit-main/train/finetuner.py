import os
import time
import json
import copy
import math
from collections import defaultdict
from typing import Any, Dict

import torch.nn as nn

import torch
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from data import EditDataset
from evaluation.base_evaluator import BaseEvaluator
from models.conditioning.numeric_projector import StrengthProjector
from models.diffusion.diff_csdi_multipatch import ResidualBlock as ResidualBlockBase
from models.diffusion.diff_csdi_multipatch_weaver import ResidualBlock as ResidualBlockWeaver
from models.conditional_generator import ConditionalGenerator

class Finetuner:
    def __init__(self, configs, eval_configs, dataset, model, c_mean):
        self._init_cfgs(configs)
        self._init_model(model)
        self._init_opt()
        self._init_data(dataset)
        self._init_eval(eval_configs, c_mean)
        self._best_valid_loss = 1e10
        self._best_valid_score = float("-inf")
        self._best_valid_collapse = None
        self._best_valid_selection = None

        self.tf_writer = SummaryWriter(log_dir=self.output_folder)

    def _init_eval(self, eval_configs, c_mean):
        self.eval_configs = eval_configs
        self.c_mean = c_mean
        self.evaluator = None
        if not self.run_generation_eval:
            return
        dataset = EditDataset(eval_configs["data"])
        self.evaluator = BaseEvaluator(eval_configs["eval"], dataset, None)

    def _init_cfgs(self, configs):
        self.configs = configs
        
        self.n_epochs = self.configs["epochs"]
        self.itr_per_epoch = self.configs["itr_per_epoch"]
        self.valid_epoch_interval = self.configs["val_epoch_interval"]
        self.display_epoch_interval = self.configs["display_interval"]

        self.lr = self.configs["lr"]
        self.strength_lr_scale = float(self.configs.get("strength_lr_scale", 1.0))
        self.batch_size = self.configs["batch_size"]
        self.strength_diagnostics_enabled = bool(self.configs.get("strength_diagnostics_enabled", False))
        self.strength_diagnostics_interval = int(self.configs.get("strength_diagnostics_interval", 50))
        self.freeze_backbone_for_strength = bool(self.configs.get("freeze_backbone_for_strength", False))
        self.beta_only_repair = bool(self.configs.get("beta_only_repair", False))
        self.reset_beta_output_head = bool(self.configs.get("reset_beta_output_head", False))
        self.conditioning_composition = dict(self.configs.get("conditioning_composition") or {})
        if self.beta_only_repair and self.freeze_backbone_for_strength:
            print("beta_only_repair enabled; overriding freeze_backbone_for_strength with stricter beta-only freeze")
            self.freeze_backbone_for_strength = False
        self.instruction_text_dropout_prob = float(self.configs.get("instruction_text_dropout_prob", 0.0))
        if "numeric_only_ratio" not in self.conditioning_composition and self.instruction_text_dropout_prob > 0.0:
            self.conditioning_composition["numeric_only_ratio"] = self.instruction_text_dropout_prob
        self.conditioning_composition_enabled = bool(self.conditioning_composition.get("enabled", False))
        self.conditioning_composition_numeric_only_ratio = float(self.conditioning_composition.get("numeric_only_ratio", 0.0))
        self.conditioning_composition_both_ratio = float(self.conditioning_composition.get("both_ratio", max(0.0, 1.0 - self.conditioning_composition_numeric_only_ratio)))
        self.run_generation_eval = bool(self.configs.get("run_generation_eval", True))
        self.spacing_early_stop_enabled = bool(self.configs.get("spacing_early_stop_enabled", False))
        self.spacing_early_stop_patience = int(self.configs.get("spacing_early_stop_patience", 2))
        self.spacing_best_window_require_positive_score = bool(
            self.configs.get("spacing_best_window_require_positive_score", True)
        )
        self.spacing_best_window_require_positive_medium_long_gap = bool(
            self.configs.get("spacing_best_window_require_positive_medium_long_gap", True)
        )

        self.include_self = self.configs["include_self"]

        self.model_path = self.configs["model_path"]  # resume training
        self.output_folder = configs["output_folder"]
        self.strength_diag_path = os.path.join(self.output_folder, "strength_diagnostics.jsonl")
        
        os.makedirs(self.output_folder, exist_ok=True)

    def _init_model(self, model):
        self.model = model.to(model.device)
        if self.model_path != "":
            print("Laoding pretrained model from {}".format(self.model_path))
            self.model.load_state_dict(torch.load(self.model_path), strict=False)
        self._beta_only_repair_targets = []
        self._beta_only_repair_masks = {}
        if self.beta_only_repair:
            self._configure_beta_only_repair()
        elif self.freeze_backbone_for_strength:
            self._freeze_backbone_for_strength()
        if self.beta_only_repair and self.reset_beta_output_head:
            self._reset_beta_only_output_heads()
        self._register_beta_only_repair_masks()
        self._log_beta_only_repair_targets()

    def _configure_beta_only_repair(self):
        for _, param in self.model.named_parameters():
            param.requires_grad = False
        self._beta_only_repair_targets = self._collect_beta_only_repair_targets()
        if not self._beta_only_repair_targets:
            raise ValueError("beta_only_repair enabled but no strength_modulation[-1] targets were found")
        for target in self._beta_only_repair_targets:
            target["param"].requires_grad = True

    def _collect_beta_only_repair_targets(self):
        targets = []
        for module_name, module in self.model.named_modules():
            if not isinstance(module, nn.Sequential) or len(module) == 0:
                continue
            if not module_name.endswith("strength_modulation"):
                continue
            output_head = module[-1]
            if not isinstance(output_head, nn.Linear):
                continue
            if output_head.out_features % 2 != 0:
                raise ValueError(f"Expected even out_features for beta split at {module_name}, got {output_head.out_features}")
            beta_start = output_head.out_features // 2
            for param_name in ("weight", "bias"):
                param = getattr(output_head, param_name, None)
                if param is None:
                    continue
                targets.append(
                    {
                        "name": f"{module_name}[-1].{param_name}",
                        "param": param,
                        "beta_start": beta_start,
                    }
                )
        return targets

    def _reset_beta_only_output_heads(self):
        for target in self._beta_only_repair_targets:
            with torch.no_grad():
                target["param"][target["beta_start"]:] = 0

    def _register_beta_only_repair_masks(self):
        self._beta_only_repair_masks = {}
        if not self.beta_only_repair:
            return
        for target in self._beta_only_repair_targets:
            mask = torch.zeros_like(target["param"].detach())
            mask[target["beta_start"]:] = 1
            self._beta_only_repair_masks[id(target["param"])] = mask

    def _log_beta_only_repair_targets(self):
        if not self.beta_only_repair:
            return
        print("Beta-only repair targets:")
        for target in self._beta_only_repair_targets:
            print(f"  - {target['name']} (beta rows: {target['beta_start']}:{target['param'].shape[0]})")

    def _apply_beta_only_repair_grad_mask(self):
        if not self.beta_only_repair:
            return
        for target in self._beta_only_repair_targets:
            grad = target["param"].grad
            if grad is None:
                continue
            grad.mul_(self._beta_only_repair_masks[id(target["param"])].to(device=grad.device, dtype=grad.dtype))

    def _mask_grad_tensor(self, param, grad_norm):
        mask = getattr(self, "_beta_only_repair_masks", {}).get(id(param))
        if mask is None:
            return grad_norm
        grad = param.grad
        if grad is None:
            return grad_norm
        masked_grad = grad.detach() * mask.to(device=grad.device, dtype=grad.dtype)
        return float(masked_grad.norm().item())

    def _init_guider(self, guider):
        self.guider = guider.to(self.model.device)
        print("Laoding pretrained guider from {}".format(self.configs["guider_path"]))
        self.guider.load_state_dict(torch.load(self.model_path), strict=False)

    def _init_opt(self):
        named_params = list(self.model.named_parameters())
        strength_params = []
        backbone_params = []
        self._strength_param_meta = []
        self._tracked_diag_param_meta = []
        for name, param in named_params:
            if not param.requires_grad:
                continue
            if self._is_strength_parameter(name) or self._is_modulation_neighbor_parameter(name):
                self._tracked_diag_param_meta.append((name, param))
            if self._is_strength_parameter(name):
                strength_params.append(param)
                self._strength_param_meta.append((name, param))
            else:
                backbone_params.append(param)

        param_groups = []
        if self.beta_only_repair:
            if strength_params:
                param_groups.append({"params": strength_params, "lr": self.lr * self.strength_lr_scale})
        else:
            if backbone_params:
                param_groups.append({"params": backbone_params, "lr": self.lr})
            if strength_params:
                param_groups.append({"params": strength_params, "lr": self.lr * self.strength_lr_scale})
        self.opt = Adam(param_groups, lr=self.lr, weight_decay=1e-6)
        if not param_groups:
            raise ValueError("No trainable parameters were configured for finetuning")
        self._strength_param_init = {
            name: param.detach().cpu().clone()
            for name, param in self._tracked_diag_param_meta
        }

    def _init_data(self, dataset):
        self.dataset = dataset
        self.train_loader = dataset.get_loader(split="train", batch_size=self.batch_size, shuffle=True, 
                                               include_self=self.include_self)
        self.valid_loader = dataset.get_loader(split="valid", batch_size=self.batch_size, shuffle=False, 
                                               include_self=self.include_self)

    def _reset_train(self):
        self._best_valid_loss = 1e10
        self._best_valid_collapse = None
        self._best_valid_score = float("-inf")
        self._best_valid_selection = None
        self._global_batch_no = 0
        self._spacing_early_stop_state = {
            "enabled": bool(self.spacing_early_stop_enabled),
            "patience": int(max(1, self.spacing_early_stop_patience)),
            "best_window_active": False,
            "best_window_enter_epoch": None,
            "no_improve_count": 0,
            "triggered": False,
            "trigger_epoch": None,
        }
        if self.strength_diagnostics_enabled and os.path.exists(self.strength_diag_path):
            os.remove(self.strength_diag_path)
        if self.strength_diagnostics_enabled:
            with open(self.strength_diag_path, "w", encoding="utf-8") as f:
                f.write("")
            StrengthProjector.enable_diagnostics(True)
            ResidualBlockBase.enable_diagnostics(True, max_records=512)
            ResidualBlockWeaver.enable_diagnostics(True, max_records=512)
            ConditionalGenerator.enable_strength_diagnostics(True)
        else:
            StrengthProjector.disable_diagnostics()
            ResidualBlockBase.disable_diagnostics()
            ResidualBlockWeaver.disable_diagnostics()
            ConditionalGenerator.disable_strength_diagnostics()

    def _summarize_numeric_dicts(self, records):
        scalars = defaultdict(list)
        for record in records:
            if not isinstance(record, dict):
                continue
            for key, value in record.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    scalars[key].append(float(value))
        summary = {}
        for key, values in scalars.items():
            if values:
                summary[key] = float(sum(values) / len(values))
        return summary

    def _summarize_nested_metric(self, records, key):
        bucket = defaultdict(list)
        for record in records:
            nested = record.get(key)
            if not isinstance(nested, dict):
                continue
            for sub_key, sub_value in nested.items():
                if isinstance(sub_value, (int, float)) and not isinstance(sub_value, bool):
                    bucket[sub_key].append(float(sub_value))
        return {sub_key: float(sum(values) / len(values)) for sub_key, values in bucket.items() if values}

    def _summarize_strength_means(self, records, key):
        stats = defaultdict(lambda: defaultdict(list))
        for record in records:
            nested = record.get(key)
            if not isinstance(nested, dict):
                continue
            for strength, payload in nested.items():
                if not isinstance(payload, dict):
                    continue
                for metric_name, metric_value in payload.items():
                    if isinstance(metric_value, (int, float)) and not isinstance(metric_value, bool):
                        stats[str(strength)][metric_name].append(float(metric_value))
        return {
            strength: {
                metric_name: float(sum(values) / len(values))
                for metric_name, values in metric_dict.items() if values
            }
            for strength, metric_dict in stats.items()
        }

    def _format_scalar_key(self, value):
        return f"{float(value):.4f}"

    def _safe_diff(self, high, low):
        if high is None or low is None:
            return None
        return float(high - low)

    def _collapse_indicator(self, gap_a, gap_b, tol=1.0e-6):
        valid = [float(gap) for gap in (gap_a, gap_b) if gap is not None]
        if not valid:
            return None
        return float(min(valid) <= tol)

    def _gap_balance_ratio(self, gap_a, gap_b, tol=1.0e-6):
        if gap_a is None or gap_b is None:
            return None
        denom = max(abs(float(gap_a)), abs(float(gap_b)), tol)
        return float(min(abs(float(gap_a)), abs(float(gap_b))) / denom)

    def _safe_spearman(self, x_values, y_values):
        if len(x_values) < 2 or len(y_values) < 2 or len(x_values) != len(y_values):
            return None
        x = torch.as_tensor(x_values, dtype=torch.float32).view(-1)
        y = torch.as_tensor(y_values, dtype=torch.float32).view(-1)
        x_rank = torch.argsort(torch.argsort(x)).float()
        y_rank = torch.argsort(torch.argsort(y)).float()
        x_rank -= x_rank.mean()
        y_rank -= y_rank.mean()
        denom = torch.sqrt(torch.clamp(x_rank.square().sum() * y_rank.square().sum(), min=1.0e-12))
        if float(denom.item()) <= 1.0e-12:
            return None
        return float((x_rank * y_rank).sum().item() / denom.item())

    def _summarize_stage_records(self, records):
        by_stage = defaultdict(list)
        for record in records:
            stage_name = record.get("stage_name")
            if isinstance(stage_name, str) and stage_name:
                by_stage[stage_name].append(record)
        return {
            stage_name: {
                "count": int(len(stage_records)),
                "scalars": self._summarize_numeric_dicts(stage_records),
                "pairwise_l2": self._summarize_nested_metric(stage_records, "stage_pairwise_l2"),
                "pairwise_l2_by_scalar": self._summarize_nested_metric(stage_records, "stage_pairwise_l2_by_scalar"),
                "mean_by_strength": self._summarize_strength_means(stage_records, "stage_mean_by_strength"),
                "mean_by_scalar": self._summarize_strength_means(stage_records, "stage_mean_by_scalar"),
            }
            for stage_name, stage_records in by_stage.items()
        }

    def _summarize_condition_diagnostics(self):
        projector_records = StrengthProjector.consume_diagnostics()
        modulation_base_records = ResidualBlockBase.consume_diagnostics()
        modulation_weaver_records = ResidualBlockWeaver.consume_diagnostics()
        generator_records = ConditionalGenerator.consume_strength_diagnostics()
        modulation_base_stage_records = [record for record in modulation_base_records if isinstance(record.get("stage_name"), str)]
        modulation_weaver_stage_records = [record for record in modulation_weaver_records if isinstance(record.get("stage_name"), str)]
        modulation_base_records = [record for record in modulation_base_records if not isinstance(record.get("stage_name"), str)]
        modulation_weaver_records = [record for record in modulation_weaver_records if not isinstance(record.get("stage_name"), str)]
        return {
            "projector": {
                "count": int(len(projector_records)),
                "scalars": self._summarize_numeric_dicts(projector_records),
                "pairwise_l2": self._summarize_nested_metric(projector_records, "projector_output_pairwise_l2"),
                "pairwise_l2_by_scalar": self._summarize_nested_metric(projector_records, "projector_output_pairwise_l2_by_scalar"),
                "mean_norm_by_strength": self._summarize_strength_means(projector_records, "projector_output_mean_norm_by_strength"),
                "mean_norm_by_scalar": self._summarize_strength_means(projector_records, "projector_output_mean_norm_by_scalar"),
            },
            "modulation_base": {
                "count": int(len(modulation_base_records)),
                "scalars": self._summarize_numeric_dicts(modulation_base_records),
                "delta_gamma_pairwise_l2": self._summarize_nested_metric(modulation_base_records, "delta_gamma_pairwise_l2"),
                "delta_beta_pairwise_l2": self._summarize_nested_metric(modulation_base_records, "delta_beta_pairwise_l2"),
                "delta_gamma_pairwise_l2_by_scalar": self._summarize_nested_metric(modulation_base_records, "delta_gamma_pairwise_l2_by_scalar"),
                "delta_beta_pairwise_l2_by_scalar": self._summarize_nested_metric(modulation_base_records, "delta_beta_pairwise_l2_by_scalar"),
                "delta_gamma_mean_by_strength": self._summarize_strength_means(modulation_base_records, "delta_gamma_mean_by_strength"),
                "delta_beta_mean_by_strength": self._summarize_strength_means(modulation_base_records, "delta_beta_mean_by_strength"),
                "delta_gamma_mean_by_scalar": self._summarize_strength_means(modulation_base_records, "delta_gamma_mean_by_scalar"),
                "delta_beta_mean_by_scalar": self._summarize_strength_means(modulation_base_records, "delta_beta_mean_by_scalar"),
                "downstream_stages": self._summarize_stage_records(modulation_base_stage_records),
            },
            "modulation_weaver": {
                "count": int(len(modulation_weaver_records)),
                "scalars": self._summarize_numeric_dicts(modulation_weaver_records),
                "delta_gamma_pairwise_l2": self._summarize_nested_metric(modulation_weaver_records, "delta_gamma_pairwise_l2"),
                "delta_beta_pairwise_l2": self._summarize_nested_metric(modulation_weaver_records, "delta_beta_pairwise_l2"),
                "delta_gamma_pairwise_l2_by_scalar": self._summarize_nested_metric(modulation_weaver_records, "delta_gamma_pairwise_l2_by_scalar"),
                "delta_beta_pairwise_l2_by_scalar": self._summarize_nested_metric(modulation_weaver_records, "delta_beta_pairwise_l2_by_scalar"),
                "delta_gamma_mean_by_strength": self._summarize_strength_means(modulation_weaver_records, "delta_gamma_mean_by_strength"),
                "delta_beta_mean_by_strength": self._summarize_strength_means(modulation_weaver_records, "delta_beta_mean_by_strength"),
                "delta_gamma_mean_by_scalar": self._summarize_strength_means(modulation_weaver_records, "delta_gamma_mean_by_scalar"),
                "delta_beta_mean_by_scalar": self._summarize_strength_means(modulation_weaver_records, "delta_beta_mean_by_scalar"),
                "downstream_stages": self._summarize_stage_records(modulation_weaver_stage_records),
            },
            "generator": {
                "count": int(len(generator_records)),
                "scalars": self._summarize_numeric_dicts(generator_records),
                "train_edit_gain_by_strength": self._summarize_nested_metric(generator_records, "train_edit_gain_by_strength"),
                "train_target_gain_by_strength": self._summarize_nested_metric(generator_records, "train_target_gain_by_strength"),
                "train_edit_gain_by_scalar": self._summarize_nested_metric(generator_records, "train_edit_gain_by_scalar"),
                "train_target_gain_by_scalar": self._summarize_nested_metric(generator_records, "train_target_gain_by_scalar"),
                "train_family_hard_cases": [
                    record.get("train_family_hard_cases")
                    for record in generator_records
                    if isinstance(record.get("train_family_hard_cases"), list) and record.get("train_family_hard_cases")
                ],
            },
        }

    def _family_gain_summaries(self, batch):
        src_x = batch.get("src_x")
        tgt_x = batch.get("tgt_x")
        mask_gt = batch.get("mask_gt")
        family_sizes = batch.get("family_sizes")
        strength_label = batch.get("strength_label")
        strength_scalar = batch.get("strength_scalar")
        required = (src_x, tgt_x, mask_gt, family_sizes)
        if not all(isinstance(item, torch.Tensor) for item in required):
            return {}
        src = src_x.detach().cpu().float()
        tgt = tgt_x.detach().cpu().float()
        mask = mask_gt.detach().cpu().float()
        if mask.dim() == 3 and mask.shape != src.shape:
            mask = mask.permute(0, 2, 1)
        if mask.dim() == 2:
            mask = mask.unsqueeze(1)
        denom = torch.clamp(mask.sum(dim=(1, 2)), min=1.0)
        gains = (torch.abs(tgt - src) * mask).sum(dim=(1, 2)) / denom
        summary = {"family_monotonic_target_hit_rate": 0.0}
        if isinstance(strength_label, torch.Tensor):
            labels = strength_label.detach().cpu().long().view(-1)
            summary["target_gain_by_strength"] = {
                str(label): float(gains[labels == label].mean().item())
                for label in sorted(set(labels.tolist()))
                if int((labels == label).sum().item()) > 0
            }
        scalar_values = None
        if isinstance(strength_scalar, torch.Tensor):
            scalar_values = strength_scalar.detach().cpu().float().view(-1)
            summary["target_gain_by_scalar"] = {}
            for scalar_value in sorted({float(value) for value in scalar_values.tolist()}):
                scalar_mask = torch.isclose(scalar_values, torch.tensor(float(scalar_value), dtype=scalar_values.dtype), atol=1.0e-6, rtol=0.0)
                if int(scalar_mask.sum().item()) == 0:
                    continue
                summary["target_gain_by_scalar"][self._format_scalar_key(scalar_value)] = float(gains[scalar_mask].mean().item())
            overall_rho = self._safe_spearman(scalar_values.tolist(), gains.tolist())
            if overall_rho is not None:
                summary["overall_target_gain_scalar_spearman"] = float(overall_rho)
        hits = []
        family_rhos = []
        adjacent_gap_rows = []
        duration_bucket_rows = defaultdict(list)
        duration_bucket_values = batch.get("duration_bucket")
        duration_bucket_list = None
        if isinstance(duration_bucket_values, (list, tuple)):
            duration_bucket_list = [str(value) for value in duration_bucket_values]
        start = 0
        for size_idx, size in enumerate(family_sizes.detach().cpu().tolist()):
            size = int(size)
            family_gains = gains[start:start + size]
            family_scalar = None if scalar_values is None else scalar_values[start:start + size]
            family_bucket = None
            if duration_bucket_list is not None and start < len(duration_bucket_list):
                family_bucket = str(duration_bucket_list[start])
            start += size
            if family_gains.numel() < 2:
                continue
            if family_scalar is not None:
                order = torch.argsort(family_scalar)
                family_gains = family_gains[order]
                family_scalar = family_scalar[order]
            hits.append(float(all(
                float(family_gains[idx + 1].item()) + 1.0e-6 >= float(family_gains[idx].item())
                for idx in range(family_gains.numel() - 1)
            )))
            if family_scalar is not None:
                rho = self._safe_spearman(family_scalar.tolist(), family_gains.tolist())
                if rho is not None:
                    family_rhos.append(float(rho))
                medium_minus_weak = None if family_gains.numel() < 2 else self._safe_diff(family_gains[1].item(), family_gains[0].item())
                strong_minus_medium = None if family_gains.numel() < 3 else self._safe_diff(family_gains[2].item(), family_gains[1].item())
                strong_minus_weak = None if family_gains.numel() < 2 else self._safe_diff(family_gains[-1].item(), family_gains[0].item())
                weak_le_medium_pass = bool(family_gains.numel() >= 2 and float(family_gains[1].item()) + 1.0e-6 >= float(family_gains[0].item()))
                medium_le_strong_pass = bool(family_gains.numel() >= 3 and float(family_gains[2].item()) + 1.0e-6 >= float(family_gains[1].item()))
                min_adjacent_gap = None
                if family_gains.numel() >= 2:
                    min_adjacent_gap = min(gap for gap in (medium_minus_weak, strong_minus_medium) if gap is not None)
                gap_balance_ratio = self._gap_balance_ratio(medium_minus_weak, strong_minus_medium)
                adjacent_gap_collapse = self._collapse_indicator(medium_minus_weak, strong_minus_medium)
                gap_row = {
                    "family_index": int(size_idx),
                    "duration_bucket": family_bucket,
                    "weak_le_medium_pass": float(weak_le_medium_pass),
                    "medium_le_strong_pass": float(medium_le_strong_pass),
                    "medium_minus_weak": medium_minus_weak,
                    "strong_minus_medium": strong_minus_medium,
                    "strong_minus_weak": strong_minus_weak,
                    "min_adjacent_gap": min_adjacent_gap,
                    "gap_balance_ratio": gap_balance_ratio,
                    "adjacent_gap_collapse": adjacent_gap_collapse,
                }
                adjacent_gap_rows.append(gap_row)
                if family_bucket:
                    duration_bucket_rows[family_bucket].append(gap_row)
        if hits:
            summary["family_monotonic_target_hit_rate"] = float(sum(hits) / len(hits))
        if family_rhos:
            summary["family_target_gain_spearman_mean"] = float(sum(family_rhos) / len(family_rhos))
        if adjacent_gap_rows:
            summary["spacing_metrics_primary"] = True
            summary["local_path_default_definition"] = "edit_time_series + edit_mask + final_output_strength_mapping.scope=edit_region"
            summary["target_weak_le_medium_pass_rate"] = float(sum(row["weak_le_medium_pass"] for row in adjacent_gap_rows) / len(adjacent_gap_rows))
            summary["target_medium_le_strong_pass_rate"] = float(sum(row["medium_le_strong_pass"] for row in adjacent_gap_rows) / len(adjacent_gap_rows))
            for metric_name in ("medium_minus_weak", "strong_minus_medium", "strong_minus_weak", "min_adjacent_gap", "gap_balance_ratio", "adjacent_gap_collapse"):
                values = [row[metric_name] for row in adjacent_gap_rows if row[metric_name] is not None]
                if values:
                    summary[f"target_{metric_name}_mean"] = float(sum(float(value) for value in values) / len(values))
            summary["target_duration_bucket_spacing"] = {}
            for bucket, bucket_rows in sorted(duration_bucket_rows.items()):
                bucket_summary = {
                    "num_families": int(len(bucket_rows)),
                    "weak_le_medium_pass_rate": float(sum(row["weak_le_medium_pass"] for row in bucket_rows) / len(bucket_rows)),
                    "medium_le_strong_pass_rate": float(sum(row["medium_le_strong_pass"] for row in bucket_rows) / len(bucket_rows)),
                }
                for metric_name in ("medium_minus_weak", "strong_minus_medium", "strong_minus_weak", "min_adjacent_gap", "gap_balance_ratio", "adjacent_gap_collapse"):
                    values = [row[metric_name] for row in bucket_rows if row[metric_name] is not None]
                    if values:
                        bucket_summary[f"{metric_name}_mean"] = float(sum(float(value) for value in values) / len(values))
                summary["target_duration_bucket_spacing"][bucket] = bucket_summary
        return summary

    def _family_flags(self, batch):
        return {
            "family_valid": bool(batch.get("family_valid", False)),
            "family_order_valid": bool(batch.get("family_order_valid", False)),
        }

    def _current_final_mapping_scope(self):
        mapping_cfg = getattr(self.model.diff_model, "final_output_strength_mapping_cfg", None)
        if isinstance(mapping_cfg, dict):
            scope = mapping_cfg.get("scope")
            if scope is not None:
                return str(scope)
        return None

    def _canonical_predicted_route_metadata(self):
        scope = self._current_final_mapping_scope()
        local_path_active = bool(scope == "edit_region")
        return {
            "route_statement": "edit_time_series + edit_mask + final_output_strength_mapping.scope=edit_region",
            "generation_route": "standard",
            "generation_route_label": "standard_edit_time_series",
            "acceptance_route": "local_path_mask_routed" if local_path_active else None,
            "eval_mask_routed": True,
            "local_path_route_active": local_path_active,
            "local_path_route_exclusions": [
                "edit_region_soft latent blending",
                "unmasked global standard route",
            ],
            "final_mapping_scope": scope,
        }

    def _summarize_family_spacing_rows(self, rows, prefix, duration_bucket_rows=None):
        summary = {"spacing_metrics_primary": True}
        if not rows:
            return summary
        summary[f"{prefix}_weak_le_medium_pass_rate"] = float(sum(row["weak_le_medium_pass"] for row in rows) / len(rows))
        summary[f"{prefix}_medium_le_strong_pass_rate"] = float(sum(row["medium_le_strong_pass"] for row in rows) / len(rows))
        for metric_name in ("medium_minus_weak", "strong_minus_medium", "strong_minus_weak", "min_adjacent_gap", "gap_balance_ratio", "adjacent_gap_collapse"):
            values = [row[metric_name] for row in rows if row[metric_name] is not None]
            if values:
                summary[f"{prefix}_{metric_name}_mean"] = float(sum(float(value) for value in values) / len(values))
        summary[f"{prefix}_duration_bucket_spacing"] = {}
        if isinstance(duration_bucket_rows, dict):
            for bucket, bucket_rows in sorted(duration_bucket_rows.items()):
                bucket_summary = {
                    "num_families": int(len(bucket_rows)),
                    "weak_le_medium_pass_rate": float(sum(row["weak_le_medium_pass"] for row in bucket_rows) / len(bucket_rows)),
                    "medium_le_strong_pass_rate": float(sum(row["medium_le_strong_pass"] for row in bucket_rows) / len(bucket_rows)),
                }
                for metric_name in ("medium_minus_weak", "strong_minus_medium", "strong_minus_weak", "min_adjacent_gap", "gap_balance_ratio", "adjacent_gap_collapse"):
                    values = [row[metric_name] for row in bucket_rows if row[metric_name] is not None]
                    if values:
                        bucket_summary[f"{metric_name}_mean"] = float(sum(float(value) for value in values) / len(values))
                summary[f"{prefix}_duration_bucket_spacing"][bucket] = bucket_summary
        return summary

    def _prediction_family_gain_summaries(self, batch, pred_x):
        src_x = batch.get("src_x")
        mask_gt = batch.get("mask_gt")
        family_sizes = batch.get("family_sizes")
        strength_scalar = batch.get("strength_scalar")
        required = (src_x, pred_x, mask_gt, family_sizes, strength_scalar)
        if not all(isinstance(item, torch.Tensor) for item in required):
            return {}
        src = src_x.detach().cpu().float()
        pred = pred_x.detach().cpu().float()
        mask = mask_gt.detach().cpu().float()
        if pred.dim() == 4 and pred.shape[0] == 1:
            pred = pred.squeeze(0)
        if pred.dim() != 3:
            return {}
        if mask.dim() == 3 and mask.shape != src.shape:
            mask = mask.permute(0, 2, 1)
        if mask.dim() == 2:
            mask = mask.unsqueeze(1)
        denom = torch.clamp(mask.sum(dim=(1, 2)), min=1.0)
        final_pred = src * (1.0 - mask) + pred * mask
        raw_gains = (torch.abs(pred - src) * mask).sum(dim=(1, 2)) / denom
        final_gains = (torch.abs(final_pred - src) * mask).sum(dim=(1, 2)) / denom
        scalar_values = strength_scalar.detach().cpu().float().view(-1)
        summary = {"spacing_metrics_primary": True}
        route_metadata = self._canonical_predicted_route_metadata()
        summary["local_path_default_definition"] = route_metadata["route_statement"]
        start = 0
        raw_hits = []
        final_hits = []
        raw_family_rhos = []
        final_family_rhos = []
        raw_adjacent_gap_rows = []
        final_adjacent_gap_rows = []
        raw_duration_bucket_rows = defaultdict(list)
        final_duration_bucket_rows = defaultdict(list)
        duration_bucket_values = batch.get("duration_bucket")
        duration_bucket_list = None
        if isinstance(duration_bucket_values, (list, tuple)):
            duration_bucket_list = [str(value) for value in duration_bucket_values]
        for size_idx, size in enumerate(family_sizes.detach().cpu().tolist()):
            size = int(size)
            family_raw_gains = raw_gains[start:start + size]
            family_final_gains = final_gains[start:start + size]
            family_scalar = scalar_values[start:start + size]
            family_bucket = None
            if duration_bucket_list is not None and start < len(duration_bucket_list):
                family_bucket = str(duration_bucket_list[start])
            start += size
            if family_raw_gains.numel() < 2:
                continue
            order = torch.argsort(family_scalar)
            family_raw_gains = family_raw_gains[order]
            family_final_gains = family_final_gains[order]
            family_scalar = family_scalar[order]
            raw_hits.append(float(all(
                float(family_raw_gains[idx + 1].item()) + 1.0e-6 >= float(family_raw_gains[idx].item())
                for idx in range(family_raw_gains.numel() - 1)
            )))
            final_hits.append(float(all(
                float(family_final_gains[idx + 1].item()) + 1.0e-6 >= float(family_final_gains[idx].item())
                for idx in range(family_final_gains.numel() - 1)
            )))
            raw_rho = self._safe_spearman(family_scalar.tolist(), family_raw_gains.tolist())
            if raw_rho is not None:
                raw_family_rhos.append(float(raw_rho))
            final_rho = self._safe_spearman(family_scalar.tolist(), family_final_gains.tolist())
            if final_rho is not None:
                final_family_rhos.append(float(final_rho))
            for gains_seq, rows_sink, bucket_sink in (
                (family_raw_gains, raw_adjacent_gap_rows, raw_duration_bucket_rows),
                (family_final_gains, final_adjacent_gap_rows, final_duration_bucket_rows),
            ):
                medium_minus_weak = None if gains_seq.numel() < 2 else self._safe_diff(gains_seq[1].item(), gains_seq[0].item())
                strong_minus_medium = None if gains_seq.numel() < 3 else self._safe_diff(gains_seq[2].item(), gains_seq[1].item())
                strong_minus_weak = None if gains_seq.numel() < 2 else self._safe_diff(gains_seq[-1].item(), gains_seq[0].item())
                weak_le_medium_pass = bool(gains_seq.numel() >= 2 and float(gains_seq[1].item()) + 1.0e-6 >= float(gains_seq[0].item()))
                medium_le_strong_pass = bool(gains_seq.numel() >= 3 and float(gains_seq[2].item()) + 1.0e-6 >= float(gains_seq[1].item()))
                min_adjacent_gap = None
                if gains_seq.numel() >= 2:
                    min_adjacent_gap = min(gap for gap in (medium_minus_weak, strong_minus_medium) if gap is not None)
                gap_row = {
                    "family_index": int(size_idx),
                    "duration_bucket": family_bucket,
                    "weak_le_medium_pass": float(weak_le_medium_pass),
                    "medium_le_strong_pass": float(medium_le_strong_pass),
                    "medium_minus_weak": medium_minus_weak,
                    "strong_minus_medium": strong_minus_medium,
                    "strong_minus_weak": strong_minus_weak,
                    "min_adjacent_gap": min_adjacent_gap,
                    "gap_balance_ratio": self._gap_balance_ratio(medium_minus_weak, strong_minus_medium),
                    "adjacent_gap_collapse": self._collapse_indicator(medium_minus_weak, strong_minus_medium),
                }
                rows_sink.append(gap_row)
                if family_bucket:
                    bucket_sink[family_bucket].append(gap_row)
        if raw_hits:
            summary["family_monotonic_pred_hit_rate"] = float(sum(raw_hits) / len(raw_hits))
            summary["family_monotonic_raw_hit_rate"] = float(sum(raw_hits) / len(raw_hits))
        if final_hits:
            summary["family_monotonic_final_hit_rate"] = float(sum(final_hits) / len(final_hits))
        if raw_family_rhos:
            summary["family_pred_gain_spearman_mean"] = float(sum(raw_family_rhos) / len(raw_family_rhos))
            summary["family_raw_gain_spearman_mean"] = float(sum(raw_family_rhos) / len(raw_family_rhos))
        if final_family_rhos:
            summary["family_final_gain_spearman_mean"] = float(sum(final_family_rhos) / len(final_family_rhos))
        summary.update(self._summarize_family_spacing_rows(raw_adjacent_gap_rows, "raw", raw_duration_bucket_rows))
        summary.update(self._summarize_family_spacing_rows(final_adjacent_gap_rows, "final", final_duration_bucket_rows))
        for metric_name in (
            "weak_le_medium_pass_rate",
            "medium_le_strong_pass_rate",
            "medium_minus_weak_mean",
            "strong_minus_medium_mean",
            "strong_minus_weak_mean",
            "min_adjacent_gap_mean",
            "gap_balance_ratio_mean",
            "adjacent_gap_collapse_mean",
            "duration_bucket_spacing",
        ):
            raw_key = f"raw_{metric_name}"
            if raw_key in summary:
                summary[f"pred_{metric_name}"] = summary[raw_key]
        return summary

    def _aggregate_duration_bucket_spacing(self, records, key):
        bucket = defaultdict(list)
        for record in records:
            nested = record.get(key)
            if not isinstance(nested, dict):
                continue
            for bucket_name, bucket_payload in nested.items():
                if isinstance(bucket_payload, dict):
                    bucket[str(bucket_name)].append(bucket_payload)
        summary = {}
        for bucket_name, payloads in bucket.items():
            metric_values = defaultdict(list)
            count_values = []
            for payload in payloads:
                num_families = payload.get("num_families")
                if isinstance(num_families, (int, float)) and not isinstance(num_families, bool):
                    count_values.append(float(num_families))
                for metric_name, metric_value in payload.items():
                    if isinstance(metric_value, (int, float)) and not isinstance(metric_value, bool):
                        metric_values[str(metric_name)].append(float(metric_value))
            bucket_summary = {}
            for metric_name, values in metric_values.items():
                if values:
                    bucket_summary[metric_name] = float(sum(values) / len(values))
            if count_values and "num_families" not in bucket_summary:
                bucket_summary["num_families"] = float(sum(count_values) / len(count_values))
            summary[bucket_name] = bucket_summary
        return summary

    def _bucket_metric(self, duration_spacing, bucket_name, metric_name):
        if not isinstance(duration_spacing, dict):
            return None
        bucket_payload = duration_spacing.get(str(bucket_name))
        if not isinstance(bucket_payload, dict):
            return None
        return bucket_payload.get(metric_name)

    def _extract_spacing_summary(self, summary, prefix):
        duration_key = f"{prefix}_duration_bucket_spacing"
        duration_spacing = summary.get(duration_key)
        return {
            "weak_le_medium_pass_rate": summary.get(f"{prefix}_weak_le_medium_pass_rate"),
            "medium_le_strong_pass_rate": summary.get(f"{prefix}_medium_le_strong_pass_rate"),
            "medium_minus_weak_mean": summary.get(f"{prefix}_medium_minus_weak_mean"),
            "strong_minus_medium_mean": summary.get(f"{prefix}_strong_minus_medium_mean"),
            "strong_minus_weak_mean": summary.get(f"{prefix}_strong_minus_weak_mean"),
            "min_adjacent_gap_mean": summary.get(f"{prefix}_min_adjacent_gap_mean"),
            "gap_balance_ratio_mean": summary.get(f"{prefix}_gap_balance_ratio_mean"),
            "adjacent_gap_collapse_rate": summary.get(f"{prefix}_adjacent_gap_collapse_mean"),
            "duration_bucket_spacing": duration_spacing,
            "medium_bucket_min_adjacent_gap_mean": self._bucket_metric(duration_spacing, "medium", "min_adjacent_gap_mean"),
            "medium_bucket_adjacent_gap_collapse_rate": self._bucket_metric(duration_spacing, "medium", "adjacent_gap_collapse_mean"),
            "long_bucket_min_adjacent_gap_mean": self._bucket_metric(duration_spacing, "long", "min_adjacent_gap_mean"),
            "long_bucket_adjacent_gap_collapse_rate": self._bucket_metric(duration_spacing, "long", "adjacent_gap_collapse_mean"),
        }

    def _selection_score_from_summary(self, summary):
        if not isinstance(summary, dict):
            return None
        min_gap = summary.get("raw_min_adjacent_gap_mean")
        collapse = summary.get("raw_adjacent_gap_collapse_mean")
        weak_pass = summary.get("raw_weak_le_medium_pass_rate")
        medium_pass = summary.get("raw_medium_le_strong_pass_rate")
        duration_spacing = summary.get("raw_duration_bucket_spacing")
        medium_bucket_min_gap = self._bucket_metric(duration_spacing, "medium", "min_adjacent_gap_mean")
        long_bucket_min_gap = self._bucket_metric(duration_spacing, "long", "min_adjacent_gap_mean")
        if any(value is None for value in (min_gap, collapse, weak_pass, medium_pass, medium_bucket_min_gap, long_bucket_min_gap)):
            return None
        return (
            1.5 * float(min_gap)
            - 1.0 * float(collapse)
            + 0.5 * float(weak_pass)
            + 0.5 * float(medium_pass)
            + 0.75 * float(medium_bucket_min_gap)
            + 1.0 * float(long_bucket_min_gap)
        )

    def _is_better_selection(self, candidate_score, candidate_loss, best_score, best_loss, candidate_collapse=None, best_collapse=None):
        if candidate_score is not None:
            if best_score is None:
                return True
            score_delta = float(candidate_score) - float(best_score)
            if score_delta > 1.0e-8:
                return True
            if abs(score_delta) <= 1.0e-8:
                if candidate_collapse is not None and best_collapse is not None:
                    collapse_delta = float(best_collapse) - float(candidate_collapse)
                    if collapse_delta > 1.0e-8:
                        return True
                    if collapse_delta < -1.0e-8:
                        return False
                if candidate_loss is not None and best_loss is not None:
                    return float(candidate_loss) < float(best_loss)
            return False
        if best_score is not None:
            return False
        if candidate_loss is None:
            return False
        if best_loss is None:
            return True
        return float(candidate_loss) < float(best_loss)

    def _build_spacing_selection_payload(self, summary):
        if not isinstance(summary, dict):
            return None
        selection_score = self._selection_score_from_summary(summary)
        raw_duration_spacing = summary.get("raw_duration_bucket_spacing")
        final_duration_spacing = summary.get("final_duration_bucket_spacing")
        return {
            "spacing_metrics_primary": bool(summary.get("spacing_metrics_primary", False)),
            "local_path_default_definition": summary.get("local_path_default_definition"),
            "selection_score": selection_score,
            "selection_score_v2": selection_score,
            "selection_primary_domain": "raw_local_spacing",
            "selection_tiebreak": "collapse_then_loss",
            "weak_le_medium_pass_rate": summary.get("raw_weak_le_medium_pass_rate"),
            "medium_le_strong_pass_rate": summary.get("raw_medium_le_strong_pass_rate"),
            "min_adjacent_gap_mean": summary.get("raw_min_adjacent_gap_mean"),
            "adjacent_gap_collapse_rate": summary.get("raw_adjacent_gap_collapse_mean"),
            "medium_minus_weak_mean": summary.get("raw_medium_minus_weak_mean"),
            "strong_minus_medium_mean": summary.get("raw_strong_minus_medium_mean"),
            "raw_min_adjacent_gap_mean": summary.get("raw_min_adjacent_gap_mean"),
            "raw_adjacent_gap_collapse_rate": summary.get("raw_adjacent_gap_collapse_mean"),
            "raw_weak_le_medium_pass_rate": summary.get("raw_weak_le_medium_pass_rate"),
            "raw_medium_le_strong_pass_rate": summary.get("raw_medium_le_strong_pass_rate"),
            "final_min_adjacent_gap_mean": summary.get("final_min_adjacent_gap_mean"),
            "final_adjacent_gap_collapse_rate": summary.get("final_adjacent_gap_collapse_mean"),
            "final_weak_le_medium_pass_rate": summary.get("final_weak_le_medium_pass_rate"),
            "final_medium_le_strong_pass_rate": summary.get("final_medium_le_strong_pass_rate"),
            "duration_bucket_spacing": raw_duration_spacing,
            "raw_duration_bucket_spacing": raw_duration_spacing,
            "final_duration_bucket_spacing": final_duration_spacing,
            "medium_bucket_min_adjacent_gap_mean": self._bucket_metric(raw_duration_spacing, "medium", "min_adjacent_gap_mean"),
            "medium_bucket_adjacent_gap_collapse_rate": self._bucket_metric(raw_duration_spacing, "medium", "adjacent_gap_collapse_mean"),
            "long_bucket_min_adjacent_gap_mean": self._bucket_metric(raw_duration_spacing, "long", "min_adjacent_gap_mean"),
            "long_bucket_adjacent_gap_collapse_rate": self._bucket_metric(raw_duration_spacing, "long", "adjacent_gap_collapse_mean"),
            "selection_metric_definition": "1.5 * raw_min_adjacent_gap_mean - raw_adjacent_gap_collapse_rate + 0.5 * raw_weak_le_medium_pass_rate + 0.5 * raw_medium_le_strong_pass_rate + 0.75 * medium_bucket_raw_min_adjacent_gap_mean + 1.0 * long_bucket_raw_min_adjacent_gap_mean",
            "selection_metric_definition_v2": "1.5 * raw_min_adjacent_gap_mean - raw_adjacent_gap_collapse_rate + 0.5 * raw_weak_le_medium_pass_rate + 0.5 * raw_medium_le_strong_pass_rate + 0.75 * medium_bucket_raw_min_adjacent_gap_mean + 1.0 * long_bucket_raw_min_adjacent_gap_mean",
            "predicted_spacing_summary": self._extract_spacing_summary(summary, "raw"),
            "raw_spacing_summary": self._extract_spacing_summary(summary, "raw"),
            "final_spacing_summary": self._extract_spacing_summary(summary, "final"),
            "target_spacing_reference": self._extract_spacing_summary(summary, "target"),
        }

    def _spacing_best_window_qualifies(self, spacing_selection):
        if not isinstance(spacing_selection, dict):
            return False
        selection_score = spacing_selection.get("selection_score_v2")
        medium_gap = spacing_selection.get("medium_bucket_min_adjacent_gap_mean")
        long_gap = spacing_selection.get("long_bucket_min_adjacent_gap_mean")
        if self.spacing_best_window_require_positive_score:
            if selection_score is None or not math.isfinite(float(selection_score)) or float(selection_score) <= 0.0:
                return False
        if self.spacing_best_window_require_positive_medium_long_gap:
            if medium_gap is None or long_gap is None:
                return False
            if not math.isfinite(float(medium_gap)) or not math.isfinite(float(long_gap)):
                return False
            if float(medium_gap) <= 0.0 or float(long_gap) <= 0.0:
                return False
        return True

    def _spacing_early_stop_payload(self, spacing_selection):
        state = dict(getattr(self, "_spacing_early_stop_state", {}) or {})
        qualifies = self._spacing_best_window_qualifies(spacing_selection)
        payload = {
            "enabled": bool(state.get("enabled", False)),
            "patience": int(state.get("patience", 0) or 0),
            "best_window_active": bool(state.get("best_window_active", False)),
            "best_window_enter_epoch": state.get("best_window_enter_epoch"),
            "no_improve_count": int(state.get("no_improve_count", 0) or 0),
            "triggered": bool(state.get("triggered", False)),
            "trigger_epoch": state.get("trigger_epoch"),
            "qualifies_current_epoch": qualifies,
        }
        if isinstance(spacing_selection, dict):
            payload["selection_score_v2"] = spacing_selection.get("selection_score_v2")
            payload["medium_bucket_min_adjacent_gap_mean"] = spacing_selection.get("medium_bucket_min_adjacent_gap_mean")
            payload["long_bucket_min_adjacent_gap_mean"] = spacing_selection.get("long_bucket_min_adjacent_gap_mean")
        return payload

    def _write_best_selection_with_early_stop_state(self):
        if not self._best_valid_selection:
            return
        selection_path = os.path.join(self.output_folder, "best_model_selection.json")
        payload = copy.deepcopy(self._best_valid_selection)
        payload["spacing_early_stop_final_state"] = self._spacing_early_stop_payload(
            payload.get("spacing_selection")
        )
        with open(selection_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def _update_spacing_early_stop_state(self, epoch_no, spacing_selection, is_best):
        state = getattr(self, "_spacing_early_stop_state", None)
        if not isinstance(state, dict) or not state.get("enabled", False):
            return False
        qualifies = self._spacing_best_window_qualifies(spacing_selection)
        if is_best and qualifies:
            state["best_window_active"] = True
            if state.get("best_window_enter_epoch") is None:
                state["best_window_enter_epoch"] = int(epoch_no)
            state["no_improve_count"] = 0
            return False
        if not state.get("best_window_active", False):
            return False
        if is_best:
            state["no_improve_count"] = 0
            return False
        state["no_improve_count"] = int(state.get("no_improve_count", 0) or 0) + 1
        if state["no_improve_count"] >= int(state.get("patience", 1)):
            state["triggered"] = True
            state["trigger_epoch"] = int(epoch_no)
            self._write_best_selection_with_early_stop_state()
            return True
        return False

    def _training_batch_diagnostics(self, batch):
        payload = {}
        payload.update(self._family_flags(batch))
        payload.update(self._family_gain_summaries(batch))
        payload["condition_path"] = self._summarize_condition_diagnostics()
        return payload

    def _is_strength_parameter(self, name):
        return (
            "strength_projector" in name
            or "strength_modulation" in name
            or "strength_input_projection" in name
            or "final_output_strength_mapping_head" in name
        )

    def _is_modulation_neighbor_parameter(self, name):
        return (
            "base_modulation" in name
            or "diffusion_projection" in name
            or "attr_projection" in name
        )

    def _freeze_backbone_for_strength(self):
        for name, param in self.model.named_parameters():
            param.requires_grad = self._is_strength_parameter(name) or self._is_modulation_neighbor_parameter(name)

    def _tensor_histogram(self, values):
        if values is None:
            return {}
        if not isinstance(values, torch.Tensor):
            values = torch.as_tensor(values)
        if values.numel() == 0:
            return {}
        unique, counts = torch.unique(values.detach().cpu().long(), return_counts=True)
        return {str(int(k)): int(v) for k, v in zip(unique.tolist(), counts.tolist())}

    def _batch_strength_stats(self, batch):
        stats = {
            "strength_hist": self._tensor_histogram(batch.get("strength_label")),
            "task_hist": self._tensor_histogram(batch.get("task_id")),
        }
        strength_scalar = batch.get("strength_scalar")
        if isinstance(strength_scalar, torch.Tensor):
            scalar_cpu = strength_scalar.detach().cpu().float().view(-1)
            stats["strength_scalar_range"] = {
                "min": float(scalar_cpu.min().item()),
                "max": float(scalar_cpu.max().item()),
                "unique": int(torch.unique(scalar_cpu).numel()),
            }
        tgt_x = batch.get("tgt_x")
        src_x = batch.get("src_x")
        strength_label = batch.get("strength_label")
        task_id = batch.get("task_id")
        if not isinstance(tgt_x, torch.Tensor) or not isinstance(strength_label, torch.Tensor):
            return stats

        tgt_cpu = tgt_x.detach().cpu().float()
        src_cpu = src_x.detach().cpu().float() if isinstance(src_x, torch.Tensor) else None
        strength_cpu = strength_label.detach().cpu().long()
        task_cpu = task_id.detach().cpu().long() if isinstance(task_id, torch.Tensor) else torch.zeros_like(strength_cpu)
        task_gap_stats = []
        for task_value in torch.unique(task_cpu):
            task_mask = task_cpu == task_value
            weak_mask = task_mask & (strength_cpu == 0)
            strong_mask = task_mask & (strength_cpu == 2)
            if int(weak_mask.sum()) == 0 or int(strong_mask.sum()) == 0:
                continue
            weak_target = tgt_cpu[weak_mask].mean(dim=0)
            strong_target = tgt_cpu[strong_mask].mean(dim=0)
            target_mse = float(torch.mean((weak_target - strong_target) ** 2).item())
            gap = {"task_id": int(task_value.item()), "target_mse": target_mse}
            if src_cpu is not None:
                weak_src = src_cpu[weak_mask].mean(dim=0)
                strong_src = src_cpu[strong_mask].mean(dim=0)
                weak_delta = weak_target - weak_src
                strong_delta = strong_target - strong_src
                gap["peak_delta_gap"] = float((strong_delta.abs().max() - weak_delta.abs().max()).item())
                gap["signed_area_gap"] = float((strong_delta.sum() - weak_delta.sum()).item())
            task_gap_stats.append(gap)
        stats["task_gap_stats"] = task_gap_stats
        return stats

    def _collect_strength_diagnostics(self, batch, loss):
        modules = []
        for name, param in self._strength_param_meta:
            grad_norm = None if param.grad is None else self._mask_grad_tensor(param, float(param.grad.detach().norm().item()))
            init_tensor = self._strength_param_init[name].to(param.device)
            delta_norm = float((param.detach() - init_tensor).norm().item())
            modules.append(
                {
                    "name": name,
                    "grad_norm": grad_norm,
                    "param_norm": float(param.detach().norm().item()),
                    "delta_from_init_norm": delta_norm,
                }
            )
        tracked_param_summary = self._summarize_tracked_param_state()
        payload = {
            "step": int(self._global_batch_no),
            "loss": float(loss.item()),
            "strength_lr_scale": float(self.strength_lr_scale),
            "strength_modules": modules,
            "grad_norm_by_param": tracked_param_summary["grad_norm_by_param"],
            "param_norm_by_param": tracked_param_summary["param_norm_by_param"],
            "param_update_l2_by_param": tracked_param_summary["param_update_l2_by_param"],
            "param_update_ratio_by_param": tracked_param_summary["param_update_ratio_by_param"],
            "batch_stats": self._batch_strength_stats(batch),
            "loss_breakdown": copy.deepcopy(getattr(self.model, "_latest_loss_breakdown", None)),
            "training_batch_diagnostics": self._training_batch_diagnostics(batch),
        }
        with open(self.strength_diag_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def _summarize_tracked_param_state(self):
        grad_norm_by_param = {}
        param_norm_by_param = {}
        param_update_l2_by_param = {}
        param_update_ratio_by_param = {}
        eps = 1.0e-12
        for name, param in getattr(self, "_tracked_diag_param_meta", []):
            grad_norm = None if param.grad is None else self._mask_grad_tensor(param, float(param.grad.detach().norm().item()))
            param_norm = float(param.detach().norm().item())
            init_tensor = self._strength_param_init[name].to(param.device)
            update_l2 = float((param.detach() - init_tensor).norm().item())
            update_ratio = update_l2 / max(param_norm, eps)
            grad_norm_by_param[name] = grad_norm
            param_norm_by_param[name] = param_norm
            param_update_l2_by_param[name] = update_l2
            param_update_ratio_by_param[name] = float(update_ratio)
        return {
            "grad_norm_by_param": grad_norm_by_param,
            "param_norm_by_param": param_norm_by_param,
            "param_update_l2_by_param": param_update_l2_by_param,
            "param_update_ratio_by_param": param_update_ratio_by_param,
        }

    def _apply_conditioning_composition(self, batch):
        instruction_text = batch.get("instruction_text")
        if instruction_text is None:
            return batch
        if not self.conditioning_composition_enabled and self.instruction_text_dropout_prob <= 0.0:
            return batch

        numeric_only_ratio = self.conditioning_composition_numeric_only_ratio
        both_ratio = self.conditioning_composition_both_ratio
        if not self.conditioning_composition_enabled:
            numeric_only_ratio = self.instruction_text_dropout_prob
            both_ratio = max(0.0, 1.0 - float(numeric_only_ratio))
        numeric_only_ratio = min(max(float(numeric_only_ratio), 0.0), 1.0)
        both_ratio = min(max(float(both_ratio), 0.0), 1.0)
        total = numeric_only_ratio + both_ratio
        if total <= 0.0:
            return batch
        numeric_only_prob = numeric_only_ratio / total

        composed_batch = dict(batch)
        if isinstance(instruction_text, torch.Tensor):
            mask = torch.rand(instruction_text.shape[0], device=instruction_text.device) < numeric_only_prob
            dropped = instruction_text.clone()
            if dropped.dim() >= 2:
                dropped[mask] = 0
            else:
                dropped[mask] = 0
            composed_batch["instruction_text"] = dropped
            composed_batch["conditioning_mode"] = torch.where(
                mask,
                torch.ones_like(mask, dtype=torch.long),
                torch.zeros_like(mask, dtype=torch.long),
            )
        elif isinstance(instruction_text, (list, tuple)):
            dropped = list(instruction_text)
            conditioning_mode = []
            for idx in range(len(dropped)):
                is_numeric_only = torch.rand(1).item() < numeric_only_prob
                if is_numeric_only:
                    dropped[idx] = ""
                    conditioning_mode.append(1)
                else:
                    conditioning_mode.append(0)
            composed_batch["instruction_text"] = dropped
            composed_batch["conditioning_mode"] = torch.tensor(conditioning_mode, dtype=torch.long)
        return composed_batch

    """
    Train.
    """
    def train(self):
        self._reset_train()
        for epoch_no in range(self.n_epochs):
            self._train_epoch(epoch_no)
            if self.valid_loader is not None and (epoch_no + 1) % self.valid_epoch_interval == 0:
                should_stop = self.valid(epoch_no)
                if self.run_generation_eval:
                    self.evaluate(epoch_no)
                if should_stop:
                    print(
                        f"\n*** Spacing-aware early stop triggered at epoch {epoch_no} "
                        f"(patience={self._spacing_early_stop_state.get('patience')}).\n"
                    )
                    break
    
    def evaluate(self, epoch_no):
        metric_list = ["cos", "rats", "auc"]
        self.model.eval()
        self.evaluator.model = self.model
        
        res_dict = self.evaluator.evaluate(mode="cond_gen", sampler="ddim", save_pred=False, c_mean=self.c_mean)
        for k in res_dict.keys():
            record_flag = False
            for metric_name in metric_list:
                if metric_name in k:
                    record_flag = True
                    break
            if record_flag == True:
                self.tf_writer.add_scalar(fr"Cond_gen/{k}", res_dict[k], epoch_no)
        
        res_dict = self.evaluator.evaluate(mode="edit", sampler="ddim", save_pred=False, c_mean=self.c_mean)
        for k in res_dict.keys():
            record_flag = False
            for metric_name in metric_list:
                if metric_name in k:
                    record_flag = True
                    break
            if record_flag == True:
                self.tf_writer.add_scalar(fr"Edit50/{k}", res_dict[k], epoch_no)

    def _train_epoch(self, epoch_no):
            start_time = time.time()
            avg_loss = 0
            self.model.train()
            for batch_no, train_batch in enumerate(self.train_loader):
                self._global_batch_no += 1
                train_batch = self._apply_conditioning_composition(train_batch)

                self.opt.zero_grad()
                loss = self.model(train_batch, is_train=True, mode="finetune")
                loss.backward()
                self._apply_beta_only_repair_grad_mask()
                self.opt.step()
                if (
                    self.strength_diagnostics_enabled
                    and self._global_batch_no % max(1, self.strength_diagnostics_interval) == 0
                ):
                    self._collect_strength_diagnostics(train_batch, loss)
                avg_loss += loss.item()
                self.tf_writer.add_scalar("Finetune/Train/batch_loss", loss.item(), self._global_batch_no)

                if batch_no >= self.itr_per_epoch:
                    break

            avg_loss /= len(self.train_loader)
            self.tf_writer.add_scalar("Finetune/Train/epoch_loss", avg_loss, epoch_no)
            end_time = time.time()
            
            if (epoch_no+1)%self.display_epoch_interval==0:
                print("Epoch:", epoch_no,
                      "Loss:", avg_loss,
                      "Time: {:.2f}s".format(end_time-start_time))

    """
    Valid.
    """
    def valid(self, epoch_no=-1):
        self.model.eval()
        avg_loss_valid = 0
        valid_diagnostics = []
        predicted_diagnostics = []
        with torch.no_grad():
            for batch_no, valid_batch in enumerate(self.valid_loader):
                loss = self.model(valid_batch, is_train=False, mode="finetune")
                avg_loss_valid += loss.item()
                batch_diag = self._training_batch_diagnostics(valid_batch)
                valid_diagnostics.append(batch_diag)
                pred_samples = self.model.edit(valid_batch, n_samples=1, sampler="ddim")
                if isinstance(pred_samples, torch.Tensor) and pred_samples.dim() == 4 and pred_samples.shape[0] == 1:
                    pred_samples = pred_samples[0]
                pred_diag = {}
                pred_diag.update(self._family_flags(valid_batch))
                pred_diag.update(self._prediction_family_gain_summaries(valid_batch, pred_samples))
                predicted_diagnostics.append(pred_diag)

        avg_loss_valid = avg_loss_valid/len(self.valid_loader)
        valid_summary = self._summarize_numeric_dicts(valid_diagnostics)
        valid_summary["target_duration_bucket_spacing"] = self._aggregate_duration_bucket_spacing(valid_diagnostics, "target_duration_bucket_spacing")
        predicted_summary = self._summarize_numeric_dicts(predicted_diagnostics)
        predicted_summary["pred_duration_bucket_spacing"] = self._aggregate_duration_bucket_spacing(predicted_diagnostics, "pred_duration_bucket_spacing")
        predicted_summary["raw_duration_bucket_spacing"] = self._aggregate_duration_bucket_spacing(predicted_diagnostics, "raw_duration_bucket_spacing")
        predicted_summary["final_duration_bucket_spacing"] = self._aggregate_duration_bucket_spacing(predicted_diagnostics, "final_duration_bucket_spacing")
        valid_summary.update(predicted_summary)
        route_metadata = self._canonical_predicted_route_metadata()
        valid_summary["spacing_metrics_primary"] = bool(
            valid_summary.get("spacing_metrics_primary", False) or predicted_summary.get("spacing_metrics_primary", False)
        )
        valid_summary["local_path_default_definition"] = route_metadata["route_statement"]
        valid_summary["generation_route_label"] = route_metadata["generation_route_label"]
        valid_summary["acceptance_route"] = route_metadata["acceptance_route"]
        valid_summary["local_path_route_active"] = route_metadata["local_path_route_active"]
        valid_summary["final_mapping_scope"] = route_metadata["final_mapping_scope"]
        spacing_selection = self._build_spacing_selection_payload(valid_summary)
        selection_score = None if not isinstance(spacing_selection, dict) else spacing_selection.get("selection_score")
        selection_collapse = None if not isinstance(spacing_selection, dict) else spacing_selection.get("raw_adjacent_gap_collapse_rate")

        self.tf_writer.add_scalar("Finetune/Valid/epoch_loss", avg_loss_valid, epoch_no)
        if selection_score is not None and math.isfinite(float(selection_score)):
            self.tf_writer.add_scalar("Finetune/Valid/spacing_selection_score", float(selection_score), epoch_no)
        if isinstance(spacing_selection, dict):
            for metric_name in (
                "weak_le_medium_pass_rate",
                "medium_le_strong_pass_rate",
                "min_adjacent_gap_mean",
                "adjacent_gap_collapse_rate",
                "raw_min_adjacent_gap_mean",
                "raw_adjacent_gap_collapse_rate",
                "final_min_adjacent_gap_mean",
                "final_adjacent_gap_collapse_rate",
                "medium_minus_weak_mean",
                "strong_minus_medium_mean",
                "medium_bucket_min_adjacent_gap_mean",
                "medium_bucket_adjacent_gap_collapse_rate",
                "long_bucket_min_adjacent_gap_mean",
                "long_bucket_adjacent_gap_collapse_rate",
            ):
                metric_value = spacing_selection.get(metric_name)
                if metric_value is not None and math.isfinite(float(metric_value)):
                    self.tf_writer.add_scalar(f"Finetune/Valid/{metric_name}", float(metric_value), epoch_no)

        selection_payload = {
            "epoch": int(epoch_no),
            "avg_loss_valid": float(avg_loss_valid),
            "spacing_selection": spacing_selection,
            "predicted_spacing_summary": self._extract_spacing_summary(valid_summary, "pred"),
            "target_spacing_summary": self._extract_spacing_summary(valid_summary, "target"),
            "valid_summary": valid_summary,
        }

        finite_selection_score = None
        if selection_score is not None and math.isfinite(float(selection_score)):
            finite_selection_score = float(selection_score)
        finite_selection_collapse = None
        if selection_collapse is not None and math.isfinite(float(selection_collapse)):
            finite_selection_collapse = float(selection_collapse)
        best_score = None if not math.isfinite(float(self._best_valid_score)) else float(self._best_valid_score)
        is_best = self._is_better_selection(
            candidate_score=finite_selection_score,
            candidate_loss=avg_loss_valid,
            best_score=best_score,
            best_loss=self._best_valid_loss,
            candidate_collapse=finite_selection_collapse,
            best_collapse=self._best_valid_collapse,
        )
        if is_best:
            if finite_selection_score is not None:
                self._best_valid_score = float(finite_selection_score)
                self._best_valid_loss = avg_loss_valid
                self._best_valid_collapse = finite_selection_collapse
                self._best_valid_selection = selection_payload
                if best_score is None or float(finite_selection_score) > float(best_score) + 1.0e-8:
                    print(f"\n*** Best spacing selection score is updated to {finite_selection_score} at {epoch_no} (loss={avg_loss_valid}).\n")
                else:
                    if self._best_valid_collapse is not None and best_score is not None and finite_selection_collapse is not None:
                        print(f"\n*** Spacing selection tie broken by lower raw collapse {finite_selection_collapse} at {epoch_no} (loss={avg_loss_valid}).\n")
                    else:
                        print(f"\n*** Spacing selection tie broken by lower valid loss {avg_loss_valid} at {epoch_no}.\n")
            else:
                self._best_valid_loss = avg_loss_valid
                self._best_valid_collapse = finite_selection_collapse
                self._best_valid_selection = selection_payload
                print(f"\n*** Best loss is updated to {avg_loss_valid} at {epoch_no}.\n")

        should_stop = self._update_spacing_early_stop_state(epoch_no, spacing_selection, is_best)
        selection_payload["spacing_early_stop"] = self._spacing_early_stop_payload(spacing_selection)
        if is_best and self._best_valid_selection is not None:
            self._best_valid_selection["spacing_early_stop"] = copy.deepcopy(selection_payload["spacing_early_stop"])
        selection_path = os.path.join(self.output_folder, "valid_epoch_summary.jsonl")
        with open(selection_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(selection_payload, ensure_ascii=False) + "\n")
        if is_best:
            self.save_model(epoch_no, selection_payload=selection_payload)
        return should_stop


    """
    Save.
    """
    def save_model(self, epoch_no, selection_payload=None):
        ckpt_dir = os.path.join(fr"{self.output_folder}", "ckpts")
        os.makedirs(ckpt_dir, exist_ok=True)
        path = os.path.join(ckpt_dir, "model_best.pth")
        torch.save(self.model.state_dict(), path)
        path = os.path.join(ckpt_dir, fr"model_best_{epoch_no}.pth")
        torch.save(self.model.state_dict(), path)
        if selection_payload is None:
            selection_payload = self._best_valid_selection
        if selection_payload is not None:
            selection_path = os.path.join(self.output_folder, "best_model_selection.json")
            with open(selection_path, "w", encoding="utf-8") as f:
                json.dump(selection_payload, f, ensure_ascii=False, indent=2)
