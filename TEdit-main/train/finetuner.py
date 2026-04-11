import os
import time
import json
import copy
from collections import defaultdict

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

        self.tf_writer = SummaryWriter(log_dir=self.output_folder)

    def _init_eval(self, eval_configs, c_mean):
        dataset = EditDataset(eval_configs["data"])
        self.evaluator = BaseEvaluator(eval_configs["eval"], dataset, None)
        self.eval_configs = eval_configs
        self.c_mean = c_mean

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
        self.instruction_text_dropout_prob = float(self.configs.get("instruction_text_dropout_prob", 0.0))
        self.run_generation_eval = bool(self.configs.get("run_generation_eval", True))

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
        if self.freeze_backbone_for_strength:
            self._freeze_backbone_for_strength()

    def _init_guider(self, guider):
        self.guider = guider.to(self.model.device)
        print("Laoding pretrained guider from {}".format(self.configs["guider_path"]))
        self.guider.load_state_dict(torch.load(self.model_path), strict=False)

    def _init_opt(self):
        named_params = list(self.model.named_parameters())
        strength_params = []
        backbone_params = []
        self._strength_param_meta = []
        for name, param in named_params:
            if not param.requires_grad:
                continue
            if self._is_strength_parameter(name):
                strength_params.append(param)
                self._strength_param_meta.append((name, param))
            else:
                backbone_params.append(param)

        param_groups = []
        if backbone_params:
            param_groups.append({"params": backbone_params, "lr": self.lr})
        if strength_params:
            param_groups.append({"params": strength_params, "lr": self.lr * self.strength_lr_scale})
        self.opt = Adam(param_groups, lr=self.lr, weight_decay=1e-6)
        self._strength_param_init = {
            name: param.detach().cpu().clone()
            for name, param in self._strength_param_meta
        }

    def _init_data(self, dataset):
        self.dataset = dataset
        self.train_loader = dataset.get_loader(split="train", batch_size=self.batch_size, shuffle=True, 
                                               include_self=self.include_self)
        self.valid_loader = dataset.get_loader(split="valid", batch_size=self.batch_size, shuffle=False, 
                                               include_self=self.include_self)

    def _reset_train(self):
        self._best_valid_loss = 1e10
        self._global_batch_no = 0
        if self.strength_diagnostics_enabled and os.path.exists(self.strength_diag_path):
            os.remove(self.strength_diag_path)
        if self.strength_diagnostics_enabled:
            StrengthProjector.enable_diagnostics(True)
            ResidualBlockBase.enable_diagnostics(True)
            ResidualBlockWeaver.enable_diagnostics(True)
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

    def _summarize_condition_diagnostics(self):
        projector_records = StrengthProjector.consume_diagnostics()
        modulation_base_records = ResidualBlockBase.consume_diagnostics()
        modulation_weaver_records = ResidualBlockWeaver.consume_diagnostics()
        generator_records = ConditionalGenerator.consume_strength_diagnostics()
        return {
            "projector": {
                "count": int(len(projector_records)),
                "scalars": self._summarize_numeric_dicts(projector_records),
                "pairwise_l2": self._summarize_nested_metric(projector_records, "projector_output_pairwise_l2"),
                "mean_norm_by_strength": self._summarize_strength_means(projector_records, "projector_output_mean_norm_by_strength"),
            },
            "modulation_base": {
                "count": int(len(modulation_base_records)),
                "scalars": self._summarize_numeric_dicts(modulation_base_records),
                "delta_gamma_pairwise_l2": self._summarize_nested_metric(modulation_base_records, "delta_gamma_pairwise_l2"),
                "delta_beta_pairwise_l2": self._summarize_nested_metric(modulation_base_records, "delta_beta_pairwise_l2"),
                "delta_gamma_mean_by_strength": self._summarize_strength_means(modulation_base_records, "delta_gamma_mean_by_strength"),
                "delta_beta_mean_by_strength": self._summarize_strength_means(modulation_base_records, "delta_beta_mean_by_strength"),
            },
            "modulation_weaver": {
                "count": int(len(modulation_weaver_records)),
                "scalars": self._summarize_numeric_dicts(modulation_weaver_records),
                "delta_gamma_pairwise_l2": self._summarize_nested_metric(modulation_weaver_records, "delta_gamma_pairwise_l2"),
                "delta_beta_pairwise_l2": self._summarize_nested_metric(modulation_weaver_records, "delta_beta_pairwise_l2"),
                "delta_gamma_mean_by_strength": self._summarize_strength_means(modulation_weaver_records, "delta_gamma_mean_by_strength"),
                "delta_beta_mean_by_strength": self._summarize_strength_means(modulation_weaver_records, "delta_beta_mean_by_strength"),
            },
            "generator": {
                "count": int(len(generator_records)),
                "scalars": self._summarize_numeric_dicts(generator_records),
                "train_edit_gain_by_strength": self._summarize_nested_metric(generator_records, "train_edit_gain_by_strength"),
                "train_target_gain_by_strength": self._summarize_nested_metric(generator_records, "train_target_gain_by_strength"),
            },
        }

    def _family_gain_summaries(self, batch):
        src_x = batch.get("src_x")
        tgt_x = batch.get("tgt_x")
        mask_gt = batch.get("mask_gt")
        family_sizes = batch.get("family_sizes")
        strength_label = batch.get("strength_label")
        if not all(isinstance(item, torch.Tensor) for item in (src_x, tgt_x, mask_gt, family_sizes, strength_label)):
            return {}
        src = src_x.detach().cpu().float()
        tgt = tgt_x.detach().cpu().float()
        mask = mask_gt.detach().cpu().float()
        if mask.dim() == 3:
            mask = mask.permute(0, 2, 1)
        if mask.dim() == 2:
            mask = mask.unsqueeze(1)
        denom = torch.clamp(mask.sum(dim=(1, 2)), min=1.0)
        gains = (torch.abs(tgt - src) * mask).sum(dim=(1, 2)) / denom
        labels = strength_label.detach().cpu().long().view(-1)
        summary = {
            "target_gain_by_strength": {
                str(label): float(gains[labels == label].mean().item())
                for label in [0, 1, 2] if int((labels == label).sum().item()) > 0
            },
            "family_monotonic_target_hit_rate": 0.0,
        }
        hits = []
        start = 0
        for size in family_sizes.detach().cpu().tolist():
            size = int(size)
            family_gains = gains[start:start + size]
            start += size
            if family_gains.numel() < 3:
                continue
            hits.append(float(family_gains[0] <= family_gains[1] and family_gains[1] <= family_gains[2]))
        if hits:
            summary["family_monotonic_target_hit_rate"] = float(sum(hits) / len(hits))
        return summary

    def _family_flags(self, batch):
        return {
            "family_valid": bool(batch.get("family_valid", False)),
            "family_order_valid": bool(batch.get("family_order_valid", False)),
        }

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
            grad_norm = None if param.grad is None else float(param.grad.detach().norm().item())
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
        payload = {
            "step": int(self._global_batch_no),
            "loss": float(loss.item()),
            "strength_lr_scale": float(self.strength_lr_scale),
            "strength_modules": modules,
            "batch_stats": self._batch_strength_stats(batch),
            "loss_breakdown": copy.deepcopy(getattr(self.model, "_latest_loss_breakdown", None)),
            "training_batch_diagnostics": self._training_batch_diagnostics(batch),
        }
        with open(self.strength_diag_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def _apply_instruction_text_dropout(self, batch):
        if self.instruction_text_dropout_prob <= 0.0:
            return batch
        instruction_text = batch.get("instruction_text")
        if instruction_text is None:
            return batch

        dropped_batch = dict(batch)
        if isinstance(instruction_text, torch.Tensor):
            mask = torch.rand(instruction_text.shape[0]) < self.instruction_text_dropout_prob
            dropped = instruction_text.clone()
            if dropped.dim() >= 2:
                dropped[mask] = 0
            dropped_batch["instruction_text"] = dropped
        elif isinstance(instruction_text, (list, tuple)):
            dropped = list(instruction_text)
            for idx in range(len(dropped)):
                if torch.rand(1).item() < self.instruction_text_dropout_prob:
                    dropped[idx] = ""
            dropped_batch["instruction_text"] = dropped
        return dropped_batch

    """
    Train.
    """
    def train(self):
        self._reset_train()
        for epoch_no in range(self.n_epochs):
            self._train_epoch(epoch_no)
            if self.valid_loader is not None and (epoch_no + 1) % self.valid_epoch_interval == 0:
                self.valid(epoch_no)
                if self.run_generation_eval:
                    self.evaluate(epoch_no)
    
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
                train_batch = self._apply_instruction_text_dropout(train_batch)

                self.opt.zero_grad()
                loss = self.model(train_batch, is_train=True, mode="finetune")
                loss.backward()
                if (
                    self.strength_diagnostics_enabled
                    and self._strength_param_meta
                    and self._global_batch_no % max(1, self.strength_diagnostics_interval) == 0
                ):
                    self._collect_strength_diagnostics(train_batch, loss)
                self.opt.step()
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
        with torch.no_grad():
            for batch_no, valid_batch in enumerate(self.valid_loader):
                loss = self.model(valid_batch, is_train=False, mode="finetune")
                avg_loss_valid += loss.item()

        avg_loss_valid = avg_loss_valid/len(self.valid_loader)

        self.tf_writer.add_scalar("Finetune/Valid/epoch_loss", avg_loss_valid, epoch_no)

        if self._best_valid_loss > avg_loss_valid:
            self._best_valid_loss = avg_loss_valid
            print(f"\n*** Best loss is updated to {avg_loss_valid} at {epoch_no}.\n")
            self.save_model(epoch_no)
    
    """
    Save.
    """
    def save_model(self, epoch_no):
        os.makedirs(fr"{self.output_folder}/ckpts", exist_ok=True)
        path = os.path.join(fr"{self.output_folder}/ckpts", "model_best.pth")
        torch.save(self.model.state_dict(), path)
        path = os.path.join(fr"{self.output_folder}/ckpts", fr"model_best_{epoch_no}.pth")
        torch.save(self.model.state_dict(), path)
