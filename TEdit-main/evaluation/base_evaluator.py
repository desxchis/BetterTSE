import os
import time
import torch
import numpy as np
from .scorer import Scorer
from .energy_model import CTAP


class BaseEvaluator:
    def __init__(self, configs, dataset, model):
        self._init_cfgs(configs)
        self._init_model(model)
        self._init_data(dataset)
        self._init_scorers()
        self._init_ctap(configs["ctap_folder"])

    def _init_ctap(self, folder):
        self.ctap = CTAP(folder)

    def _init_cfgs(self, configs):
        self.configs = configs

        self.batch_size = self.configs["batch_size"]
        self.n_samples = self.configs["n_samples"]

        self.display_epoch_interval = self.configs["display_interval"]

        self.model_path = self.configs["model_path"]  # resume training

        self.output_folder = configs["output_folder"]
        os.makedirs(self.output_folder, exist_ok=True)

    def _init_model(self, model):
        self.model = model
        if self.model_path != "":
            print("Laoding pretrained model from {}".format(self.model_path))
            self.model.load_state_dict(torch.load(self.model_path))

    def _init_data(self, dataset):
        self.dataset = dataset
        self.test_loader = dataset.get_loader(split="test", batch_size=self.batch_size, shuffle=False, include_self=False)

    def _init_scorers(self):
        self.mse = Scorer("MSE")
        self.mae = Scorer("MAE")

    def _reset_eval(self, mode, sampler):
        self.curr_save_root = os.path.join(self.output_folder, f"{mode}_{sampler}")
        print(f"Results will be saved to {self.curr_save_root}.")
        os.makedirs(self.curr_save_root, exist_ok=True)

        # eval metrics
        self.mse.reset()
        self.mae.reset()
        self.ctap.reset()

        # data
        self._pred_list, self._tgt_list, self._src_list = [], [], []
        self._mse_list, self._mae_list = [], []
        self._src_attr_list, self._tgt_attr_list = [], []

    def _record_data(self, batch, pred, mse, mae):
        self._pred_list.append(pred.detach().cpu().numpy())
        self._tgt_list.append(batch["tgt_x"].detach().cpu().numpy())
        self._src_list.append(batch["src_x"].detach().cpu().numpy())
        self._mse_list.append(mse)
        self._mae_list.append(mae)
        self._src_attr_list.append(batch["src_attrs"].detach().cpu().numpy())
        self._tgt_attr_list.append(batch["tgt_attrs"].detach().cpu().numpy())

    """
    Evaluate.
    """
    def evaluate(self, mode="cond_gen", sampler="ddpm", save_pred=False, c_mean=None):
        """
        Args:
            mode: cond_gen or edit.
            sampler: ddpm or ddim.
            c_mean: the class mean of different attributes
        """
        print("\n-------------------------------")
        print(f"Evaluating the model with mode={mode} and sampler={sampler}")
        self._reset_eval(mode, sampler)
        self.model.eval()
        for batch_no, batch in enumerate(self.test_loader):
            start_time = time.time()
            pred = self.model.generate(batch, self.n_samples, mode, sampler)  # (n_samples,B,V,L)
            pred = pred.permute(0,1,3,2)  # (n_smples,B,L,V)
            pred = pred.median(dim=0).values
            tgt_x = batch["tgt_x"]  # (B,L,V)

            mse = self.mse(pred=pred, gt=tgt_x)
            mae = self.mae(pred=pred, gt=tgt_x)

            self.ctap.evaluate(batch["src_x"], pred, batch["tgt_attrs"], batch["src_attrs"])

            if save_pred:
                self._record_data(batch, pred, mse, mae)

            end_time = time.time()

            if (batch_no+1)%self.display_epoch_interval == 0:
                print("Batch", batch_no, 
                      "Current MSE {:.4f}".format(mse.mean()),
                      "Current MAE {:.4f}".format(mae.mean()),
                      "Batch Time {:.2f}s".format(end_time-start_time))

        print("Done!")
        print("MSE: ", self.mse.mean)
        print("MAE: ", self.mae.mean)

        rats, rats_abs = self.ctap.calc_rats()
        print("RATS: ", rats)
        print("RATS ABS: ", rats_abs)

        ctap = self.ctap.calc_ts2ts_attr_sim(c_mean)
        print("CTAP: ", ctap)

        ctap_dict = self.scores_list2dict(ctap, "ctap")
        rats_dict = self.scores_list2dict(rats, "rats")
        rats_abs_dict = self.scores_list2dict(rats_abs, "rats_abs")

        res_dict = {
            "mse": self.mse.mean,
            "mae": self.mae.mean,
        }
        res_dict.update(ctap_dict)
        res_dict.update(rats_dict)
        res_dict.update(rats_abs_dict)

        self.save_results()
        if save_pred:
            self.save_pred()
        return res_dict

    def scores_list2dict(self, scores, name="ctap"):
        res_dict = {}
        for i, v in enumerate(scores):
            res_dict[f"{name}_{i}"] = v
        return res_dict

    """
    Save.
    """
    def save_results(self):
        path = os.path.join(self.curr_save_root, "results.txt")
        with open(path, "w") as f:
            f.writelines("MSE: " + str(self.mse.mean) + "\n")
            f.writelines("MAE: " + str(self.mae.mean) + "\n")

    def save_pred(self):
        np.save(os.path.join(self.curr_save_root, f"pred.npy"), 
                np.concatenate(self._pred_list, axis=0))
        np.save(os.path.join(self.curr_save_root, f"tgt.npy"), 
                np.concatenate(self._tgt_list, axis=0))
        np.save(os.path.join(self.curr_save_root, f"src.npy"),
                np.concatenate(self._src_list, axis=0))

        np.save(os.path.join(self.curr_save_root, f"tgt_attrs.npy"), 
                np.concatenate(self._tgt_attr_list, axis=0))
        np.save(os.path.join(self.curr_save_root, f"src_attrs.npy"), 
                np.concatenate(self._src_attr_list, axis=0))

        np.save(os.path.join(self.curr_save_root, f"mse.npy"), 
                np.concatenate(self._mse_list, axis=0))
        np.save(os.path.join(self.curr_save_root, f"mae.npy"),
                np.concatenate(self._mae_list, axis=0))
