import time
import torch
import numpy as np
from .energy_model import CTAP


class PretrainStat:
    def __init__(self, ctap_folder):
        self.ctap = CTAP(folder=ctap_folder)
        self.device = self.ctap.device
        self.n_attr_ops = self.ctap.model.n_attr_ops  # the number of ops for each attr. list

    @torch.no_grad()
    def get_concept_mean(self, dataset, split="train", batch_size=256):
        print("\n-------------------------------")
        print("Calculating concept mean...")
        self.data_loader = dataset.get_loader(split=split, batch_size=batch_size, shuffle=False, include_self=True)

        x_emb_list = []
        attr_list = []
        for batch_no, batch in enumerate(self.data_loader):
            start_time = time.time()

            x = batch["x"].permute(0,2,1).float().to(self.device)
            attrs = batch["attrs"].to(self.device)  # (B,N_attr)
            attr_list.append(attrs.cpu().numpy())

            x_embs, _ = self.ctap.model(x, attrs)  # (B,N_attr,emb)
            x_embs = x_embs.cpu().numpy()
            x_emb_list.append(x_embs)
            
            end_time = time.time()
            print("Batch", batch_no, "Batch Time {:.2f}s".format(end_time-start_time))
        
        x_embs = np.concatenate(x_emb_list, axis=0)  # (N,N_attr,emb)
        attrs = np.concatenate(attr_list, axis=0)  # (N,N_attr)
        concept_mean_list = []
        for i in range(attrs.shape[1]):
            c_mean_list = []
            atrs = attrs[:,i]  # (N)
            for j in range(self.n_attr_ops[i]):
                ids = atrs==j
                c_mean = x_embs[ids][:,i]  # (Nk, emb)
                c_mean = np.mean(c_mean, axis=0)  # (emb)
                c_mean_list.append(c_mean)
            c_mean = np.stack(c_mean_list, axis=0)  # (N_ops,emb)
            concept_mean_list.append(c_mean)
        return concept_mean_list
