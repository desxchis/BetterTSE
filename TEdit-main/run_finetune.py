import os
import sys
import copy
import yaml
import json
import argparse
import subprocess
import pandas as pd

import random
import numpy as np
import torch
from datetime import datetime, timezone

from data import EditDataset
from models.conditional_generator import ConditionalGenerator
from train.pretrainer import PreTrainer
from train.finetuner import Finetuner
from evaluation.base_evaluator import BaseEvaluator
from evaluation.pretrain_stat import PretrainStat


def _to_jsonable(obj):
    if isinstance(obj, dict):
        return {str(key): _to_jsonable(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(value) for value in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


def save_runtime_config_json(payload, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_to_jsonable(payload), f, indent=2, sort_keys=True)


def save_configs(configs, path):
    print(json.dumps(_to_jsonable(configs), indent=4))
    with open(path, "w") as f:
        yaml.dump(_to_jsonable(configs), f, yaml.SafeDumper)


def _build_cli_overrides(args):
    overrides = {
        "lr": args.lr,
        "epochs": args.epochs,
        "include_self": bool(args.include_self),
        "bootstrap_ratio": args.bootstrap_ratio,
        "strength_diagnostics": bool(args.strength_diagnostics),
        "strength_diagnostics_interval": args.strength_diagnostics_interval,
        "only_evaluate": args.only_evaluate,
        "n_runs": args.n_runs,
    }
    optional_fields = {
        "freeze_backbone_for_strength": (args.freeze_backbone_for_strength, lambda value: value >= 0, lambda value: bool(value)),
        "instruction_text_dropout_prob": (args.instruction_text_dropout_prob, lambda value: value >= 0.0, None),
        "conditioning_composition_enabled": (args.conditioning_composition_enabled, lambda value: value >= 0, lambda value: bool(value)),
        "conditioning_composition_both_ratio": (args.conditioning_composition_both_ratio, lambda value: value >= 0.0, None),
        "conditioning_composition_numeric_only_ratio": (args.conditioning_composition_numeric_only_ratio, lambda value: value >= 0.0, None),
        "strength_lr_scale": (args.strength_lr_scale, lambda value: value > 0.0, None),
        "beta_direction_loss_weight": (args.beta_direction_loss_weight, lambda value: value >= 0.0, None),
        "beta_direction_margin": (args.beta_direction_margin, lambda value: value >= 0.0, None),
        "beta_direction_target": (args.beta_direction_target, lambda value: bool(value), None),
        "beta_only_repair": (args.beta_only_repair, lambda value: value >= 0, lambda value: bool(value)),
        "reset_beta_output_head": (args.reset_beta_output_head, lambda value: value >= 0, lambda value: bool(value)),
    }
    for key, (value, predicate, transform) in optional_fields.items():
        if predicate(value):
            overrides[key] = transform(value) if transform else value
    return overrides


def _get_git_provenance():
    repo_dir = os.path.dirname(__file__)
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_dir, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=repo_dir, text=True).strip())
        return {"commit": commit, "dirty": dirty}
    except Exception:
        return None


def _deep_merge_dict(base, override):
    merged = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge_dict(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _build_resolved_runtime_payload(*, cli_overrides, run_index, seed, ctrl_attrs, only_evaluate, skip_final_evaluate,
                                    output_folder, dataset_folder, pretrained_model_path, evaluation_model_path,
                                    finetune_configs, model_configs, eval_configs):
    payload = {
        "schema_version": 1,
        "entrypoint": "TEdit-main/run_finetune.py",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "argv": sys.argv,
        "cwd": os.getcwd(),
        "cli_overrides": cli_overrides,
        "run_context": {
            "run_index": run_index,
            "seed": seed,
            "ctrl_attrs": ctrl_attrs,
            "resolved_output_folder": output_folder,
            "resolved_dataset_folder": dataset_folder,
            "resolved_pretrained_checkpoint_path": pretrained_model_path,
            "resolved_evaluation_checkpoint_path": evaluation_model_path,
            "only_evaluate": only_evaluate,
            "skip_final_evaluate": skip_final_evaluate,
        },
        "resolved_configs": {
            "finetune": finetune_configs,
            "model": model_configs,
            "eval": eval_configs,
        },
    }
    git_info = _get_git_provenance()
    if git_info is not None:
        payload["git"] = git_info
    return payload


def _save_resolved_runtime_artifacts(finetune_configs, model_configs, eval_configs, output_folder, payload):
    save_configs(finetune_configs, os.path.join(output_folder, "finetune_configs.yaml"))
    save_configs(model_configs, os.path.join(output_folder, "model_configs.yaml"))
    save_configs(eval_configs, os.path.join(output_folder, "eval_configs.yaml"))
    save_runtime_config_json(payload, os.path.join(output_folder, "resolved_runtime_config.json"))


def pretrain(pretrain_configs, model_configs, output_folder):
    pretrain_configs["train"]["output_folder"] = output_folder

    dataset = EditDataset(pretrain_configs["data"])
    model_configs["attrs"]["num_attr_ops"] = dataset.num_attr_ops.tolist()
    model = ConditionalGenerator(model_configs)

    print("\n***** Pretrain Configs *****")
    save_configs(pretrain_configs, os.path.join(output_folder, "pretrain_configs.yaml"))

    print("\n***** Model Configs *****")
    save_configs(model_configs, os.path.join(output_folder, "model_configs.yaml"))

    pretrainer = PreTrainer(pretrain_configs["train"], dataset, model)
    pretrainer.train()


def finetune(finetune_configs, model_configs, eval_configs, output_folder, c_mean=None, runtime_payload=None):
    finetune_configs["train"]["output_folder"] = output_folder

    dataset = EditDataset(finetune_configs["data"])
    model_configs["attrs"]["num_attr_ops"] = dataset.num_attr_ops.tolist()
    model = ConditionalGenerator(model_configs)

    if runtime_payload is not None:
        runtime_payload["resolved_configs"]["model"] = model_configs
        _save_resolved_runtime_artifacts(finetune_configs, model_configs, eval_configs, output_folder, runtime_payload)
    else:
        save_configs(finetune_configs, os.path.join(output_folder, "finetune_configs.yaml"))
        save_configs(model_configs, os.path.join(output_folder, "model_configs.yaml"))

    finetuner = Finetuner(finetune_configs["train"], eval_configs, dataset, model, c_mean=c_mean)
    finetuner.train()


def evaluate(eval_configs, model_configs, output_folder, c_mean=None, runtime_payload=None):
    dataset = EditDataset(eval_configs["data"])
    model_configs["attrs"]["num_attr_ops"] = dataset.num_attr_ops.tolist()
    model = ConditionalGenerator(model_configs)

    print("\n***** Evaluate Configs *****")
    if runtime_payload is not None:
        runtime_payload["resolved_configs"]["model"] = model_configs
        _save_resolved_runtime_artifacts(runtime_payload["resolved_configs"]["finetune"], model_configs, eval_configs, output_folder, runtime_payload)
    else:
        save_configs(eval_configs, path=os.path.join(output_folder, "eval_configs.yaml"))

    evaluator = BaseEvaluator(eval_configs["eval"], dataset, model)
    df_cond = _evaluate_cond_gen(evaluator, c_mean=c_mean)
    df_edit = _evaluate_edit(evaluator, c_mean=c_mean)
    df = pd.concat([df_cond, df_edit], ignore_index=True)
    return df


def _evaluate_cond_gen(evaluator, sampler="ddim", n_sample=10, c_mean=None):
    evaluator.n_samples = n_sample
    res_dict = evaluator.evaluate(mode="cond_gen", sampler=sampler, save_pred=False, c_mean=c_mean)

    info = {
        "mode": "cond_gen",
        "sampler": sampler,
        "n_samples": evaluator.n_samples,
        "steps": -1,
    }
    info.update(res_dict)
    df = pd.DataFrame([info])
    df["steps"].astype(int)
    return df


def _evaluate_edit(evaluator, sampler="ddim", n_samples=1, c_mean=None):
    evaluator.n_samples = n_samples
    df = pd.DataFrame(columns=["mode"])
    info = {"mode": "edit", "sampler": sampler, "n_samples": evaluator.n_samples, "steps": -1}

    for steps in [50]:
        print("\n*******************")
        print(f"Edit steps: {steps}")
        evaluator.model.edit_steps = steps
        res_dict = evaluator.evaluate(mode="edit", sampler=sampler, save_pred=False, c_mean=c_mean)

        info["steps"] = steps
        res_dict.update(info)
        df_res = pd.DataFrame([res_dict])
        df = pd.concat([df, df_res], ignore_index=True)
        df["steps"].astype(int)
    return df


def _resolve_selector_name(attrs):
    return "_".join(str(attr) for attr in attrs)


def _resolve_finetune_dataset_folder(data_folder, attrs):
    selector = _resolve_selector_name(attrs)
    candidate = os.path.join(data_folder, selector)
    if os.path.exists(os.path.join(candidate, "meta.json")):
        return candidate
    if os.path.exists(os.path.join(data_folder, "meta.json")):
        return data_folder
    return candidate


def run(finetune_configs, eval_configs, model_configs, output_folder_all, data_folder="", only_evaluate="false", c_mean=None,
        cli_overrides=None, run_index=None, seed=None, pretrained_model_path=""):
    ctrl_attrs = finetune_configs["train"]["ctrl_attrs"]
    data_name = str(finetune_configs["data"]["name"])
    skip_final_evaluate = bool(finetune_configs["train"].get("skip_final_evaluate", False))

    df_list = []
    for attrs in ctrl_attrs:
        print("\n**************************************")
        print("*****", attrs)
        print("**************************************")

        selector = _resolve_selector_name(attrs)
        output_folder = os.path.join(output_folder_all, selector)
        os.makedirs(output_folder, exist_ok=True)

        resolved_finetune_configs = copy.deepcopy(finetune_configs)
        resolved_eval_configs = copy.deepcopy(eval_configs)
        resolved_model_configs = copy.deepcopy(model_configs)

        resolved_data_folder = _resolve_finetune_dataset_folder(data_folder, attrs)
        resolved_finetune_configs["data"]["folder"] = resolved_data_folder
        resolved_eval_configs["data"]["folder"] = resolved_data_folder
        resolved_finetune_configs["train"]["output_folder"] = output_folder

        evaluation_model_path = os.path.join(output_folder, "ckpts/model_best.pth")
        if only_evaluate == "true":
            configured_eval_model_path = resolved_eval_configs["eval"].get("model_path", "")
            if configured_eval_model_path:
                evaluation_model_path = configured_eval_model_path
            elif pretrained_model_path:
                evaluation_model_path = pretrained_model_path
            elif os.path.exists(os.path.join(output_folder, "ckpts", "model_best.pth")):
                evaluation_model_path = os.path.join(output_folder, "ckpts", "model_best.pth")
            else:
                candidate_attr_checkpoint = os.path.join(
                    os.path.dirname(output_folder_all),
                    os.path.basename(output_folder_all),
                    "_".join(attrs),
                    "ckpts",
                    "model_best.pth",
                )
                if os.path.exists(candidate_attr_checkpoint):
                    evaluation_model_path = candidate_attr_checkpoint
        if only_evaluate == "false" and not os.path.exists(evaluation_model_path):
            resolved_eval_configs["eval"]["model_path"] = ""
            runtime_payload = _build_resolved_runtime_payload(
                cli_overrides=cli_overrides,
                run_index=run_index,
                seed=seed,
                ctrl_attrs=attrs,
                only_evaluate=only_evaluate,
                skip_final_evaluate=skip_final_evaluate,
                output_folder=output_folder,
                dataset_folder=resolved_data_folder,
                pretrained_model_path=pretrained_model_path,
                evaluation_model_path="",
                finetune_configs=resolved_finetune_configs,
                model_configs=resolved_model_configs,
                eval_configs=resolved_eval_configs,
            )
            finetune(resolved_finetune_configs, resolved_model_configs, resolved_eval_configs, output_folder, c_mean, runtime_payload=runtime_payload)

        should_run_evaluate = only_evaluate == "true" or not skip_final_evaluate
        if should_run_evaluate:
            resolved_eval_for_eval = copy.deepcopy(resolved_eval_configs)
            resolved_model_for_eval = copy.deepcopy(resolved_model_configs)
            resolved_eval_for_eval["eval"]["model_path"] = evaluation_model_path
            runtime_payload = _build_resolved_runtime_payload(
                cli_overrides=cli_overrides,
                run_index=run_index,
                seed=seed,
                ctrl_attrs=attrs,
                only_evaluate=only_evaluate,
                skip_final_evaluate=skip_final_evaluate,
                output_folder=output_folder,
                dataset_folder=resolved_data_folder,
                pretrained_model_path=pretrained_model_path,
                evaluation_model_path=evaluation_model_path,
                finetune_configs=resolved_finetune_configs,
                model_configs=resolved_model_for_eval,
                eval_configs=resolved_eval_for_eval,
            )
            df = evaluate(resolved_eval_for_eval, resolved_model_for_eval, output_folder, c_mean=c_mean, runtime_payload=runtime_payload)
            n_records = df.shape[0]
            df.insert(0, column="ctrl_attrs", value=[str(attrs)] * n_records)
            df_list.append(df)
        else:
            runtime_payload = _build_resolved_runtime_payload(
                cli_overrides=cli_overrides,
                run_index=run_index,
                seed=seed,
                ctrl_attrs=attrs,
                only_evaluate=only_evaluate,
                skip_final_evaluate=skip_final_evaluate,
                output_folder=output_folder,
                dataset_folder=resolved_data_folder,
                pretrained_model_path=pretrained_model_path,
                evaluation_model_path="",
                finetune_configs=resolved_finetune_configs,
                model_configs=resolved_model_configs,
                eval_configs=resolved_eval_configs,
            )
            _save_resolved_runtime_artifacts(resolved_finetune_configs, resolved_model_configs, resolved_eval_configs, output_folder, runtime_payload)

    if df_list:
        df_finetune = pd.concat(df_list, ignore_index=True)
    else:
        df_finetune = pd.DataFrame({})
    path = os.path.join(output_folder_all, "results_finetune.csv")
    df_finetune.to_csv(path)
    return df_finetune


parser = argparse.ArgumentParser(description="TSE Finetune")
parser.add_argument("--pretrained_dir", type=str, default="save/synthetic/pretrain")
parser.add_argument("--model_config_path", type=str, default="model_configs.yaml")
parser.add_argument("--pretrained_model_path", type=str, default="ckpts/model_best.pth")
parser.add_argument("--finetune_config_path", type=str, default="configs/synthetic/fintune.yaml")
parser.add_argument("--evaluate_config_path", type=str, default="configs/synthetic/evaluate.yaml")
parser.add_argument("--data_type", type=str, default="synthetic")
parser.add_argument("--data_folder", type=str, default="datasets/synthetic")
parser.add_argument("--save_folder", type=str, default="./save")
parser.add_argument("--n_runs", type=int, default=1)
parser.add_argument("--only_evaluate", type=str, default="false")
parser.add_argument("--bootstrap_ratio", type=float, default=0.5)
parser.add_argument("--lr", type=float, default=1e-5)
parser.add_argument("--epochs", type=int, default=50)
parser.add_argument("--include_self", type=int, default=1)
parser.add_argument("--freeze-backbone-for-strength", type=int, default=-1)
parser.add_argument("--instruction-text-dropout-prob", type=float, default=-1.0)
parser.add_argument("--conditioning-composition-enabled", type=int, default=-1)
parser.add_argument("--conditioning-composition-both-ratio", type=float, default=-1.0)
parser.add_argument("--conditioning-composition-numeric-only-ratio", type=float, default=-1.0)
parser.add_argument("--strength-lr-scale", type=float, default=-1.0)
parser.add_argument("--strength-diagnostics", type=int, default=0)
parser.add_argument("--strength-diagnostics-interval", type=int, default=50)
parser.add_argument("--beta-direction-loss-weight", type=float, default=-1.0)
parser.add_argument("--beta-direction-margin", type=float, default=-1.0)
parser.add_argument("--beta-direction-target", type=str, default="")
parser.add_argument("--beta-only-repair", type=int, default=-1)
parser.add_argument("--reset-beta-output-head", type=int, default=-1)

args = parser.parse_args()
cli_overrides = _build_cli_overrides(args)

save_folder = args.save_folder
os.makedirs(save_folder, exist_ok=True)
print("All files will be saved to '{}'".format(save_folder))

finetune_configs = yaml.safe_load(open(args.finetune_config_path))
eval_configs = yaml.safe_load(open(args.evaluate_config_path))

finetune_configs["train"]["lr"] = args.lr
finetune_configs["train"]["epochs"] = args.epochs
finetune_configs["train"]["include_self"] = bool(args.include_self)
if args.freeze_backbone_for_strength >= 0:
    finetune_configs["train"]["freeze_backbone_for_strength"] = bool(args.freeze_backbone_for_strength)
if args.instruction_text_dropout_prob >= 0.0:
    finetune_configs["train"]["instruction_text_dropout_prob"] = args.instruction_text_dropout_prob
conditioning_composition_cfg = dict(finetune_configs["train"].get("conditioning_composition") or {})
if args.conditioning_composition_enabled >= 0:
    conditioning_composition_cfg["enabled"] = bool(args.conditioning_composition_enabled)
if args.conditioning_composition_both_ratio >= 0.0:
    conditioning_composition_cfg["both_ratio"] = args.conditioning_composition_both_ratio
if args.conditioning_composition_numeric_only_ratio >= 0.0:
    conditioning_composition_cfg["numeric_only_ratio"] = args.conditioning_composition_numeric_only_ratio
if conditioning_composition_cfg:
    if "enabled" not in conditioning_composition_cfg:
        conditioning_composition_cfg["enabled"] = True
    if "numeric_only_ratio" not in conditioning_composition_cfg and args.instruction_text_dropout_prob >= 0.0:
        conditioning_composition_cfg["numeric_only_ratio"] = args.instruction_text_dropout_prob
    if "both_ratio" not in conditioning_composition_cfg:
        conditioning_composition_cfg["both_ratio"] = max(0.0, 1.0 - float(conditioning_composition_cfg.get("numeric_only_ratio", 0.0)))
    finetune_configs["train"]["conditioning_composition"] = conditioning_composition_cfg
if args.strength_lr_scale > 0.0:
    finetune_configs["train"]["strength_lr_scale"] = args.strength_lr_scale
if args.beta_direction_loss_weight >= 0.0:
    finetune_configs["train"]["beta_direction_loss_weight"] = args.beta_direction_loss_weight
if args.beta_direction_margin >= 0.0:
    finetune_configs["train"]["beta_direction_margin"] = args.beta_direction_margin
if args.beta_direction_target:
    finetune_configs["train"]["beta_direction_target"] = args.beta_direction_target
finetune_configs["train"]["strength_diagnostics_enabled"] = bool(args.strength_diagnostics)
finetune_configs["train"]["strength_diagnostics_interval"] = args.strength_diagnostics_interval
if args.beta_only_repair >= 0:
    finetune_configs["train"]["beta_only_repair"] = bool(args.beta_only_repair)
if args.reset_beta_output_head >= 0:
    finetune_configs["train"]["reset_beta_output_head"] = bool(args.reset_beta_output_head)
if args.data_type != "synthetic":
    finetune_configs["data"]["name"] = args.data_type
    eval_configs["data"]["name"] = args.data_type

base_finetune_configs = copy.deepcopy(finetune_configs)
base_eval_configs = copy.deepcopy(eval_configs)

print("Constucting dataset...")
script_dir = os.path.dirname(__file__)
eval_data_folder = eval_configs["data"]["folder"]
if not os.path.isabs(eval_data_folder):
    eval_data_folder = os.path.join(script_dir, eval_data_folder)
pretrain_folder = os.path.join(os.path.dirname(eval_data_folder), "pretrain")
data_configs = {"name": "synthetic_pretrain", "folder": pretrain_folder}
dataset = EditDataset(data_configs)

print("Obtaining stats of the pretraining data...")
ctap_folder = eval_configs["eval"]["ctap_folder"]
if not os.path.isabs(ctap_folder):
    ctap_folder = os.path.join(script_dir, ctap_folder)
pretrain_stat = PretrainStat(ctap_folder)
c_mean = pretrain_stat.get_concept_mean(dataset, split="train", batch_size=256)

print("Started training...")
seed_list = [1, 7, 42]
df_list = []
eval_record_folder = base_eval_configs["data"]["folder"]
for n in range(args.n_runs):
    fix_seed = seed_list[n]
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    print(f"\nRun: {n}")

    model_config_path = args.model_config_path
    if not os.path.isabs(model_config_path):
        candidate_model_config_path = os.path.normpath(os.path.join(args.pretrained_dir, str(n), model_config_path))
        if os.path.exists(candidate_model_config_path):
            model_config_path = candidate_model_config_path
        else:
            model_config_path = os.path.normpath(model_config_path)
    model_configs = yaml.safe_load(open(model_config_path))
    model_configs["diffusion"]["bootstrap_ratio"] = args.bootstrap_ratio
    run_finetune_configs = copy.deepcopy(base_finetune_configs)
    run_eval_configs = copy.deepcopy(base_eval_configs)

    if bool(run_finetune_configs["train"].get("enable_strength_control", False)):
        strength_cfg = dict(model_configs["diffusion"].get("strength_control") or {})
        template_model_cfg_path = args.model_config_path if os.path.isabs(args.model_config_path) else os.path.join(script_dir, "configs/synthetic/model_multi_weaver.yaml")
        template_model_cfg = yaml.safe_load(open(template_model_cfg_path))
        template_strength_cfg = dict(template_model_cfg.get("diffusion", {}).get("strength_control") or {})
        merged_strength_cfg = _deep_merge_dict(template_strength_cfg, strength_cfg)
        train_strength_cfg = run_finetune_configs["train"].get("strength_control")
        if isinstance(train_strength_cfg, dict):
            merged_strength_cfg = _deep_merge_dict(merged_strength_cfg, train_strength_cfg)
        for key in [
            "diffusion_loss_weight",
            "edit_region_loss_weight",
            "background_loss_weight",
            "monotonic_loss_weight",
            "monotonic_margin",
            "gain_match_loss_weight",
            "family_gap_match_loss_weight",
            "family_relative_gain_loss_weight",
            "family_relative_margin_scale",
            "constant_gain_penalty_weight",
            "minimum_family_gain_std",
            "numeric_only_loss_weight",
            "beta_direction_loss_weight",
            "beta_direction_margin",
            "beta_direction_target",
        ]:
            if key in run_finetune_configs["train"]:
                merged_strength_cfg[key] = run_finetune_configs["train"][key]
        merged_strength_cfg["enabled"] = True
        model_configs["diffusion"]["strength_control"] = merged_strength_cfg
        print("Strength control runtime config:", json.dumps(_to_jsonable(model_configs["diffusion"]["strength_control"]), indent=2))

    pretrained_model_path = args.pretrained_model_path
    if not os.path.isabs(pretrained_model_path):
        candidate_pretrained_model_path = os.path.normpath(os.path.join(args.pretrained_dir, str(n), pretrained_model_path))
        if os.path.exists(candidate_pretrained_model_path):
            pretrained_model_path = candidate_pretrained_model_path
        else:
            pretrained_model_path = os.path.normpath(pretrained_model_path)
    run_finetune_configs["train"]["model_path"] = pretrained_model_path

    output_folder = os.path.join(save_folder, str(n))
    os.makedirs(output_folder, exist_ok=True)
    if args.only_evaluate == "true":
        configured_eval_model_path = run_eval_configs["eval"].get("model_path", "")
        if (not configured_eval_model_path) or (not os.path.isabs(configured_eval_model_path) and not os.path.exists(configured_eval_model_path)):
            candidate_attr_checkpoint = os.path.join(args.pretrained_dir, str(n), "trend_injection", args.pretrained_model_path)
            if os.path.exists(candidate_attr_checkpoint):
                run_eval_configs["eval"]["model_path"] = candidate_attr_checkpoint
            elif os.path.exists(pretrained_model_path):
                run_eval_configs["eval"]["model_path"] = pretrained_model_path
    else:
        run_eval_configs["eval"]["model_path"] = ""
    if args.data_type != "synthetic":
        run_eval_configs["data"]["folder"] = eval_record_folder
        run_eval_configs["data"]["name"] = args.data_type
    else:
        run_eval_configs["data"]["folder"] = args.data_folder or run_eval_configs["data"]["folder"]
        if not os.path.isabs(run_eval_configs["data"]["folder"]):
            run_eval_configs["data"]["folder"] = os.path.join(os.path.dirname(__file__), run_eval_configs["data"]["folder"])
        if not os.path.isabs(run_eval_configs["eval"].get("ctap_folder", "")):
            run_eval_configs["eval"]["ctap_folder"] = os.path.join(os.path.dirname(__file__), run_eval_configs["eval"]["ctap_folder"])

    df = run(
        run_finetune_configs,
        run_eval_configs,
        model_configs,
        output_folder_all=output_folder,
        data_folder=args.data_folder,
        only_evaluate=args.only_evaluate,
        c_mean=c_mean,
        cli_overrides=cli_overrides,
        run_index=n,
        seed=fix_seed,
        pretrained_model_path=pretrained_model_path,
    )

    n_records = df.shape[0]
    if n_records > 0:
        df.insert(0, column="run", value=[n] * n_records)
    df_list.append(df)

if df_list:
    df = pd.concat(df_list, ignore_index=True)
else:
    df = pd.DataFrame()
path = os.path.join(save_folder, "results_finetune.csv")
df.to_csv(path)

print("\n**************************************")
print("*****", "Simple Statistics")
print("**************************************")
if not df.empty and all(col in df.columns for col in ["ctrl_attrs", "mode", "sampler", "steps", "n_samples"]):
    df_stat = df.groupby(["ctrl_attrs", "mode", "sampler", "steps", "n_samples"], as_index=False).agg(["mean", "std"])
    df_stat.to_csv(os.path.join(save_folder, fr"results_finetune_stat.csv"))
else:
    print("Skip statistics because no evaluation records were produced.")
