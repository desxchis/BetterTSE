import os
import yaml

import torch
import numpy as np
import torch.nn.functional as F

from models.energy.energy_model import EnergyModel


class CTAP:
    def __init__(self, folder):
        model_configs = yaml.safe_load(open(os.path.join(folder, "model_configs.yaml")))
        self.model = EnergyModel(model_configs)
        state_dict = torch.load(os.path.join(folder, "energy_model_best.pth"))
        self.model.load_state_dict(state_dict, strict=False)
        self.device = self.model.device
        self.model.to(self.device)

        self.reset()

    def reset(self):
        self.cos_list = []
        self.rats_list = []
        self.x_emb_list = []
        self.label_list = []

    @torch.no_grad()
    def evaluate(self, src_x, pred, attrs_tgt, attrs_src):
        src_x = src_x.to(self.device).float().permute(0,2,1)
        pred = pred.to(self.device).float().permute(0,2,1)
        attrs_tgt = attrs_tgt.to(self.device).long()
        attrs_src = attrs_src.to(self.device).long()

        B = pred.shape[0]
        ids = torch.arange(B, device=pred.device).long()

        cos_score_list = []
        rats_list = []

        cos_list_tgt, x_embs = self.model.get_ts2all_attr_scores_embs(pred)
        cos_list_src, _ = self.model.get_ts2all_attr_scores_embs(src_x)
        x_embs = x_embs.detach().cpu().numpy() 

        self.x_emb_list.append(x_embs)  # (B,N_attr,emb)
        self.label_list.append(attrs_tgt.detach().cpu().numpy())  # (B,N_attr)

        for i in range(len(cos_list_tgt)):
            # rats
            cos = cos_list_tgt[i]
            pred_probs = F.softmax(cos, dim=-1)   # (B,ops)
            pred_probs_tgt = pred_probs[ids, attrs_tgt[:,i]]  # (B)

            cos_src = cos_list_src[i]
            pred_probs_src = F.softmax(cos_src, dim=-1)   # (B,ops)
            pred_probs_src = pred_probs_src[ids, attrs_tgt[:,i]]  # (B)

            rats = torch.log(pred_probs_tgt+1e-10) - torch.log(pred_probs_src + 1e-10)
            rats = rats.detach().cpu().numpy()
            rats_list.append(rats)

            rows = np.arange(len(cos))
            cos_score = cos[rows, attrs_tgt[:, i]].detach().cpu().numpy()
            cos_score_list.append(cos_score)

        cos_scores = np.stack(cos_score_list, axis=1)  # (B,N_attr)
        self.cos_list.append(cos_scores)

        rats_list = np.stack(rats_list, axis=1)
        self.rats_list.append(rats_list)

    def calc_rats(self):
        rats = np.concatenate(self.rats_list, axis=0)  # (N,N_attr)
        rats_abs = np.abs(rats)
        rats = rats.mean(axis=0)
        rats_abs = rats_abs.mean(axis=0)
        return rats.tolist(), rats_abs.tolist()

    # concept similarity
    def calc_ts2ts_attr_sim(self, concept_mean_list):
        embs = np.concatenate(self.x_emb_list, axis=0)  # (N,N_attr,emb)
        labels = np.concatenate(self.label_list, axis=0)  # (N,N_attr)

        cos_list = []
        for i in range(self.model.n_attrs):
            l = labels[:,i]  # (N)
            c_emb = concept_mean_list[i][l]  # (N,emb)
            x_emb = embs[:,i]  # (N,emb)
            dot = np.sum(x_emb*c_emb, axis=-1)
            dot = np.mean(dot)

            x_ = x_emb/np.linalg.norm(x_emb, axis=-1, keepdims=True)
            c_ = c_emb/np.linalg.norm(c_emb, axis=-1, keepdims=True)
            cos = np.sum(x_*c_, axis=-1)
            cos = np.mean(cos)
            cos_list.append(cos)
        return cos_list
