import os
import yaml
import json
import argparse
import pandas as pd
import torch
import numpy as np
import random

from data import EditDataset
from models.conditional_generator import ConditionalGenerator
from train.pretrainer import PreTrainer
from evaluation.base_evaluator import BaseEvaluator
from evaluation.pretrain_stat import PretrainStat


def save_configs(configs, path):
    print(json.dumps(configs, indent=4))
    with open(path, "w") as f:
        yaml.dump(configs, f, yaml.SafeDumper)


def pretrain(pretrain_configs, model_configs, eval_configs,  output_folder, c_mean=None):
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
    pretrainer = PreTrainer(pretrain_configs["train"], eval_configs, dataset, model, c_mean)
    pretrainer.train()

def evaluate(eval_configs, model_configs, output_folder, c_mean):
    eval_configs["eval"]["model_path"] = os.path.join(output_folder, "ckpts/model_best.pth")

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
    df_edit = _evaluate_edit(evaluator)
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


def _evaluate_edit(evaluator, sampler="ddim", n_samples=1):
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


def run(pretrain_configs, eval_configs, model_configs, output_folder, data_folder="", only_evaluate="false", c_mean=None):
    ### pretrain ###
    if only_evaluate == "false":
        pretrain(pretrain_configs, model_configs, eval_configs, output_folder, c_mean)
    
    ### eval ###
    ctrl_attrs = pretrain_configs["train"]["ctrl_attrs"]

    df_list = []
    for attrs in ctrl_attrs:
        print("\n**************************************")
        print("*****", attrs)
        print("**************************************")
        eval_configs["data"]["folder"] = os.path.join(data_folder, "_".join(attrs))  ######
        df = evaluate(eval_configs, model_configs, output_folder, c_mean=c_mean)
    
        n_records = df.shape[0]
        df.insert(0, column="ctrl_attrs", value=[str(attrs)]*n_records)
        df_list.append(df)

    df = pd.concat(df_list, ignore_index=True)
    path = os.path.join(output_folder, "results.csv")
    df.to_csv(path)
    return df


##### Arguments #####
parser = argparse.ArgumentParser(description="TSE")
parser.add_argument("--model_config_path", type=str, default="configs/synthetic/model_multi.yaml") # the config file path of model
parser.add_argument("--train_config_path", type=str, default="configs/synthetic/pretrain.yaml") # the config file path of training
parser.add_argument("--evaluate_config_path", type=str, default="configs/synthetic/evaluate.yaml") # the config file path of evaluation
parser.add_argument("--data_folder", type=str, default="datasets/synthetic") # the path of dataset
parser.add_argument("--save_folder", type=str, default="./save") # the saving path for log/ckpts/results
parser.add_argument("--ctap_folder", type=str, default="") # the path of ctap model weights
parser.add_argument("--n_runs", type=int, default=3)  # the number of runs

parser.add_argument("--only_evaluate", type=str, default="false") # if only evaluate without training, true or false
parser.add_argument("--evaluate_data_type", type=str, default="synthetic") # the data type of evaluation

parser.add_argument("--multipatch_num", type=int, default=3) # multipatch layers number
parser.add_argument("--L_patch_len", type=int, default=3) # multipatch patch length
parser.add_argument("--load_model_path", type=str, default="") # for resuming training

parser.add_argument("--lr", type=float, default=1e-3) # learning rate
parser.add_argument("--epochs", type=int, default=200) # training epochs
args = parser.parse_args()

###
save_folder = args.save_folder
os.makedirs(save_folder, exist_ok=True)
print("All files will be saved to '{}'".format(save_folder))

pretrain_configs = yaml.safe_load(open(args.train_config_path))
eval_configs = yaml.safe_load(open(args.evaluate_config_path))
model_configs = yaml.safe_load(open(args.model_config_path))

pretrain_configs["train"]["lr"] = args.lr
pretrain_configs["train"]["epochs"] = args.epochs

if args.load_model_path != "":
    pretrain_configs["train"]["model_path"] = args.load_model_path

model_configs["diffusion"]["multipatch_num"] = args.multipatch_num
model_configs["diffusion"]["L_patch_len"] = args.L_patch_len
if args.ctap_folder != "":
    eval_configs["eval"]["ctap_folder"] = args.ctap_folder

###
print("Constucting dataset...")
eval_configs["data"]["name"] = args.evaluate_data_type
pretrain_folder = eval_configs["data"]["folder"].split("/")[:-1] + ["pretrain"]
pretrain_folder = "/".join(pretrain_folder)
data_configs = {
    "name": "synthetic_pretrain",
    "folder": pretrain_folder
}
dataset = EditDataset(data_configs)

###
print("Obtaining stats of the pretraining data...")
pretrain_stat = PretrainStat(eval_configs["eval"]["ctap_folder"])
c_mean = pretrain_stat.get_concept_mean(dataset, split="train", batch_size=256)

###
print("Started training...")
seed_list = [1, 7, 42]
df_list = []
eval_record_folder = eval_configs["data"]["folder"]
for n in range(args.n_runs):
    fix_seed = seed_list[n]
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    print(f"\nRun: {n}")
    output_folder = os.path.join(save_folder, str(n))
    os.makedirs(output_folder, exist_ok=True)
    eval_configs["eval"]["model_path"] = ""
    eval_configs["data"]["folder"] = eval_record_folder
    df = run(pretrain_configs, eval_configs, model_configs, output_folder, 
             data_folder=args.data_folder, only_evaluate=args.only_evaluate, c_mean=c_mean)

    n_records = df.shape[0]
    df.insert(0, column="run", value=[n]*n_records)
    df_list.append(df)

df = pd.concat(df_list, ignore_index=True)
path = os.path.join(save_folder, "results.csv")
df.to_csv(path)

##### Statistics #####
print("\n**************************************")
print("*****", "Simple Statistics")
print("**************************************")
df_stat = df.groupby(["ctrl_attrs", "mode", "sampler", "steps", "n_samples"], as_index=False).agg(["mean", "std"])
df_stat.to_csv(os.path.join(save_folder, "results_stat.csv"))