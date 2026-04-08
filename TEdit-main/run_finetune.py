import os
import yaml
import json
import argparse
import pandas as pd
from pathlib import Path

import random
import numpy as np
import torch

from data import EditDataset
from models.conditional_generator import ConditionalGenerator
from train.pretrainer import PreTrainer
from train.finetuner import Finetuner
from evaluation.base_evaluator import BaseEvaluator
from evaluation.pretrain_stat import PretrainStat


def save_configs(configs, path):
    print(json.dumps(configs, indent=4))
    with open(path, "w") as f:
        yaml.dump(configs, f, yaml.SafeDumper)


def resolve_existing_path(value, *, anchors=None):
    path = Path(value)
    candidates = []
    if path.is_absolute():
        candidates.append(path)
    else:
        candidates.append(Path.cwd() / path)
        for anchor in anchors or []:
            candidates.append(Path(anchor) / path)
    for candidate in candidates:
        if candidate.exists():
            return str(candidate.resolve())
    return str(candidates[0].resolve()) if candidates else str(path)

def pretrain(pretrain_configs, model_configs, output_folder):
    pretrain_configs["train"]["output_folder"] = output_folder

    # data
    dataset = EditDataset(pretrain_configs["data"])

    # model
    model_configs["attrs"]["num_attr_ops"] = dataset.num_attr_ops.tolist()
    model = ConditionalGenerator(model_configs)

    # save configs
    print("\n***** Pretrain Configs *****")
    path = os.path.join(output_folder, "pretrain_configs.yaml")
    save_configs(pretrain_configs, path)

    print("\n***** Model Configs *****")
    path = os.path.join(output_folder, "model_configs.yaml")
    save_configs(model_configs, path)

    # train
    pretrainer = PreTrainer(pretrain_configs["train"], dataset, model)
    pretrainer.train()


def finetune(finetune_configs, model_configs, eval_configs, output_folder, c_mean=None):
    finetune_configs["train"]["output_folder"] = output_folder

    # data
    dataset = EditDataset(finetune_configs["data"])

    # model
    model_configs["attrs"]["num_attr_ops"] = dataset.num_attr_ops.tolist()
    model = ConditionalGenerator(model_configs)

    path = os.path.join(output_folder, "finetune_configs.yaml")
    save_configs(finetune_configs, path)

    path = os.path.join(output_folder, "model_configs.yaml")
    save_configs(model_configs, path)

    # train
    finetuner = Finetuner(finetune_configs["train"], eval_configs, dataset, model, c_mean=c_mean)
    finetuner.train()

def evaluate(eval_configs, model_configs, output_folder, c_mean=None):
    # data
    dataset = EditDataset(eval_configs["data"])

    # model
    model_configs["attrs"]["num_attr_ops"] = dataset.num_attr_ops
    model = ConditionalGenerator(model_configs)

    # save configs
    print("\n***** Evaluate Configs *****")
    path = os.path.join(output_folder, "eval_configs.yaml")
    save_configs(eval_configs, path=path)

    # eval
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
    evaluator.n_samples = n_samples # ddim is deterministic.
    df = pd.DataFrame(columns=["mode"]) 
    info = {"mode": "edit", 
            "sampler": sampler,
            "n_samples": evaluator.n_samples,
            "steps": -1}   

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


def run(finetune_configs, eval_configs, model_configs, output_folder_all, data_folder="", only_evaluate="false", c_mean=None):    
    ### finetune ###
    ctrl_attrs = finetune_configs["train"]["ctrl_attrs"]
    data_name = str(finetune_configs["data"]["name"])
    skip_final_evaluate = bool(finetune_configs["train"].get("skip_final_evaluate", False))

    df_list = []
    for attrs in ctrl_attrs:
        print("\n**************************************")
        print("*****", attrs)
        print("**************************************")

        # finetune
        output_folder = os.path.join(output_folder_all, "_".join(attrs))

        os.makedirs(output_folder, exist_ok=True)
        if data_name == "discrete_strength_family":
            finetune_configs["data"]["folder"] = data_folder
            eval_configs["data"]["folder"] = data_folder
        else:
            finetune_configs["data"]["folder"] = os.path.join(data_folder, "_".join(attrs))
            eval_configs["data"]["folder"] = os.path.join(data_folder, "_".join(attrs))
        if only_evaluate == "false" and not os.path.exists(os.path.join(output_folder, "ckpts/model_best.pth")):
            eval_configs["eval"]["model_path"] = ""
            finetune(finetune_configs, model_configs, eval_configs, output_folder, c_mean)

        if not skip_final_evaluate:
            eval_configs["eval"]["model_path"] = os.path.join(output_folder, "ckpts/model_best.pth")
            df = evaluate(eval_configs, model_configs, output_folder, c_mean=c_mean)
            n_records = df.shape[0]
            df.insert(0, column="ctrl_attrs", value=[str(attrs)]*n_records)
            df_list.append(df)

    if df_list:
        df_finetune = pd.concat(df_list, ignore_index=True)
    else:
        df_finetune = pd.DataFrame()
    path = os.path.join(output_folder, "results_finetune.csv")
    df_finetune.to_csv(path)
    return df_finetune


##### Arguments #####
parser = argparse.ArgumentParser(description="TSE Finetune")
parser.add_argument("--pretrained_dir", type=str, default="save/synthetic/pretrain") # the directory of pretrained model
parser.add_argument("--model_config_path", type=str, default="model_configs.yaml") # the config file path of model
parser.add_argument("--pretrained_model_path", type=str, default="ckpts/model_best.pth") # the path of pretrained model weights
parser.add_argument("--finetune_config_path", type=str, default="configs/synthetic/fintune.yaml") # the config file path of training
parser.add_argument("--evaluate_config_path", type=str, default="configs/synthetic/evaluate.yaml") # the config file path of evaluation

parser.add_argument("--data_type", type=str, default="synthetic") # the data type for training

parser.add_argument("--data_folder", type=str, default="datasets/synthetic") # the path of dataset
parser.add_argument("--save_folder", type=str, default="./save") # the saving path for log/ckpts/results
parser.add_argument("--n_runs", type=int, default=1)  # the number of runs
parser.add_argument("--only_evaluate", type=str, default="false") # if only evaluate without training, true or false

###### Some args to tune #####
# model
parser.add_argument("--bootstrap_ratio", type=float, default=0.5)  # the bootstrap ratio when choosing the top-k samples [0,1]
# trainer
parser.add_argument("--lr", type=float, default=1e-5)  # learning rate, should be smaller than pretrain lr
parser.add_argument("--epochs", type=int, default=50)  # training epochs
parser.add_argument("--include_self", type=int, default=-1)  # true or false
parser.add_argument("--freeze-backbone-for-strength", type=int, default=-1)
parser.add_argument("--instruction-text-dropout-prob", type=float, default=-1.0)
parser.add_argument("--strength-lr-scale", type=float, default=1.0)
parser.add_argument("--strength-diagnostics", type=int, default=0)
parser.add_argument("--strength-diagnostics-interval", type=int, default=50)
parser.add_argument("--trainable-scope", type=str, default="")
parser.add_argument("--edit-region-loss-weight", type=float, default=-1.0)
parser.add_argument("--background-loss-weight", type=float, default=-1.0)
parser.add_argument("--monotonic-loss-weight", type=float, default=-1.0)
parser.add_argument("--monotonic-margin", type=float, default=-1.0)
parser.add_argument("--late-output-coupling-enabled", type=int, default=-1)
parser.add_argument("--late-output-coupling-gain", type=float, default=-1.0)
parser.add_argument("--seed-list", type=str, default="")
parser.add_argument("--pretrained-run", type=int, default=-1)

args = parser.parse_args()

###
save_folder = args.save_folder
os.makedirs(save_folder, exist_ok=True)
print("All files will be saved to '{}'".format(save_folder))

finetune_configs = yaml.safe_load(open(args.finetune_config_path))
eval_configs = yaml.safe_load(open(args.evaluate_config_path))
pretrained_repo_root = Path(args.pretrained_dir).resolve().parents[2]
tedit_data_anchor = pretrained_repo_root / "TEdit-main"


finetune_configs["train"]["lr"] = args.lr
finetune_configs["train"]["epochs"] = args.epochs
if args.include_self >= 0:
    finetune_configs["train"]["include_self"] = bool(args.include_self)
if args.freeze_backbone_for_strength >= 0:
    finetune_configs["train"]["freeze_backbone_for_strength"] = bool(args.freeze_backbone_for_strength)
if args.trainable_scope.strip():
    finetune_configs["train"]["trainable_scope"] = args.trainable_scope.strip()
if args.instruction_text_dropout_prob >= 0.0:
    finetune_configs["train"]["instruction_text_dropout_prob"] = args.instruction_text_dropout_prob
finetune_configs["train"]["strength_lr_scale"] = args.strength_lr_scale
finetune_configs["train"]["strength_diagnostics_enabled"] = bool(args.strength_diagnostics)
finetune_configs["train"]["strength_diagnostics_interval"] = args.strength_diagnostics_interval
finetune_configs["data"]["name"] = args.data_type

eval_configs["data"]["name"] = args.data_type

###
print("Constucting dataset...")
pretrain_folder = eval_configs["data"]["folder"].split("/")[:-1] + ["pretrain"]
pretrain_folder = "/".join(pretrain_folder)
pretrain_folder = resolve_existing_path(pretrain_folder, anchors=[tedit_data_anchor])
data_configs = {
    "name": "synthetic_pretrain",
    "folder": pretrain_folder
}
dataset = EditDataset(data_configs)

###
print("Obtaining stats of the pretraining data...")
ctap_folder = resolve_existing_path(eval_configs["eval"]["ctap_folder"], anchors=[tedit_data_anchor])
pretrain_stat = PretrainStat(ctap_folder)
c_mean = pretrain_stat.get_concept_mean(dataset, split="train", batch_size=256)

###
print("Started training...")
seed_list = [1, 7, 42]
if args.seed_list.strip():
    seed_list = [int(v.strip()) for v in args.seed_list.split(",") if v.strip()]
df_list = []
eval_record_folder = eval_configs["data"]["folder"]
for n in range(args.n_runs):
    fix_seed = seed_list[n]
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    print(f"\nRun: {n}")

    pretrained_run = args.pretrained_run if args.pretrained_run >= 0 else n
    model_configs = yaml.safe_load(open(fr"{args.pretrained_dir}/{pretrained_run}/{args.model_config_path}"))
    model_configs["diffusion"]["bootstrap_ratio"] = args.bootstrap_ratio
    strength_cfg = model_configs["diffusion"].setdefault("strength_control", {})
    if args.edit_region_loss_weight >= 0.0:
        strength_cfg["edit_region_loss_weight"] = args.edit_region_loss_weight
    if args.background_loss_weight >= 0.0:
        strength_cfg["background_loss_weight"] = args.background_loss_weight
    if args.monotonic_loss_weight >= 0.0:
        strength_cfg["monotonic_loss_weight"] = args.monotonic_loss_weight
    if args.monotonic_margin >= 0.0:
        strength_cfg["monotonic_margin"] = args.monotonic_margin
    late_coupling_cfg = strength_cfg.setdefault("late_output_coupling", {})
    if args.late_output_coupling_enabled >= 0:
        late_coupling_cfg["enabled"] = bool(args.late_output_coupling_enabled)
    if args.late_output_coupling_gain >= 0.0:
        late_coupling_cfg["gain"] = args.late_output_coupling_gain
    finetune_configs["train"]["model_path"] = fr"{args.pretrained_dir}/{pretrained_run}/{args.pretrained_model_path}"

    output_folder = os.path.join(save_folder, str(n))
    os.makedirs(output_folder, exist_ok=True)
    eval_configs["eval"]["model_path"] = ""
    eval_configs["data"]["folder"] = eval_record_folder
    df = run(finetune_configs, eval_configs, model_configs, output_folder_all=output_folder, data_folder=args.data_folder, only_evaluate=args.only_evaluate, c_mean=c_mean)

    n_records = df.shape[0]
    df.insert(0, column="run", value=[n]*n_records)
    df_list.append(df)

df = pd.concat(df_list, ignore_index=True)
path = os.path.join(save_folder, "results_finetune.csv")
df.to_csv(path)

##### Statistics #####
print("\n**************************************")
print("*****", "Simple Statistics")
print("**************************************")
if not df.empty and "ctrl_attrs" in df.columns:
    df_stat = df.groupby(["ctrl_attrs", "mode", "sampler", "steps", "n_samples"], as_index=False).agg(["mean", "std"])
    df_stat.to_csv(os.path.join(save_folder, fr"results_finetune_stat.csv"))
else:
    print("No final evaluation records to summarize.")
