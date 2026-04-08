import os
import time
import json
from collections import defaultdict

import torch
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from data import EditDataset
from evaluation.base_evaluator import BaseEvaluator

class Finetuner:
    def __init__(self, configs, eval_configs, dataset, model, c_mean):
        self._init_cfgs(configs)
        self._init_model(model)
        self._init_opt()
        self._init_data(dataset)
        self.evaluator = None
        self.eval_configs = eval_configs
        self.c_mean = c_mean
        if self.run_generation_eval:
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
        self.trainable_scope = str(
            self.configs.get(
                "trainable_scope",
                "strength_only" if self.freeze_backbone_for_strength else "all",
            )
        )
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
        self._apply_trainable_scope()

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

    def _component_grad_norm(self, loss_tensor):
        if loss_tensor is None or not isinstance(loss_tensor, torch.Tensor) or not loss_tensor.requires_grad:
            return None
        grads = torch.autograd.grad(
            loss_tensor,
            [param for _, param in self._strength_param_meta],
            retain_graph=True,
            allow_unused=True,
        )
        sq_sum = 0.0
        has_grad = False
        for grad in grads:
            if grad is None:
                continue
            has_grad = True
            sq_sum += float(torch.sum(grad.detach() ** 2).item())
        if not has_grad:
            return None
        return sq_sum ** 0.5

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

    def _is_strength_parameter(self, name):
        return (
            "strength_projector" in name
            or "strength_modulation" in name
            or "strength_input_projection" in name
            or "strength_output_projection" in name
        )

    def _is_modulation_neighbor_parameter(self, name):
        return (
            "base_modulation" in name
            or "diffusion_projection" in name
            or "attr_projection" in name
        )

    def _is_output_amplitude_parameter(self, name):
        return (
            "output_projection1" in name
            or "output_projection" in name
            or "mid_projection" in name
            or "side_projection" in name
            or "patch_decoder" in name
            or "multipatch_mixer" in name
        )

    def _apply_trainable_scope(self):
        if self.trainable_scope == "all":
            return
        for name, param in self.model.named_parameters():
            allow = self._is_strength_parameter(name) or self._is_modulation_neighbor_parameter(name)
            if self.trainable_scope == "wider_unfreeze":
                allow = allow or self._is_output_amplitude_parameter(name)
            param.requires_grad = allow

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

    def _collect_strength_diagnostics(self, batch, loss, finetune_stats=None, loss_tensors=None):
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
        component_grad_norms = {}
        if loss_tensors:
            for key in ["denoising_loss", "edit_region_l1", "background_l1", "monotonic_loss", "total_loss"]:
                component_grad_norms[key] = self._component_grad_norm(loss_tensors.get(key))
        payload = {
            "step": int(self._global_batch_no),
            "loss": float(loss.item()),
            "strength_lr_scale": float(self.strength_lr_scale),
            "strength_modules": modules,
            "batch_stats": self._batch_strength_stats(batch),
            "finetune_stats": finetune_stats or {},
            "component_grad_norms": component_grad_norms,
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
                finetune_stats = self.model.get_last_finetune_stats() if hasattr(self.model, "get_last_finetune_stats") else {}
                loss_tensors = self.model.get_last_loss_tensors() if hasattr(self.model, "get_last_loss_tensors") else {}
                need_diag = (
                    self.strength_diagnostics_enabled
                    and self._strength_param_meta
                    and self._global_batch_no % max(1, self.strength_diagnostics_interval) == 0
                )
                loss.backward(retain_graph=need_diag)
                if need_diag:
                    self._collect_strength_diagnostics(train_batch, loss, finetune_stats=finetune_stats, loss_tensors=loss_tensors)
                self.opt.step()
                avg_loss += loss.item()
                self.tf_writer.add_scalar("Finetune/Train/batch_loss", loss.item(), self._global_batch_no)
                for key, value in finetune_stats.items():
                    self.tf_writer.add_scalar(f"Finetune/Train/{key}", value, self._global_batch_no)

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
