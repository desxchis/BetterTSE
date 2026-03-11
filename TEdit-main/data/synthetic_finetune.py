import os
import json
import numpy as np
from torch.utils.data import Dataset


class SyntheticDataset:
    def __init__(self, folder, **kwargs):
        super().__init__()
        self.folder = folder
        self._load_meta()

    def _load_meta(self):
        self.meta = json.load(open(os.path.join(self.folder, "meta.json")))
        self.attr_list = self.meta["attr_list"]
        n_attr = len(self.attr_list)
        self.attr_ids = np.arange(n_attr)

        # the attrs to be kept must contain controlled attributes
        self.ctrl_attr_ids = set(self.meta["control_attr_ids"])
        self.side_attr_ids = set(self.attr_ids) - self.ctrl_attr_ids

        self.ctrl_attr_ids = sorted(self.ctrl_attr_ids)
        self.side_attr_ids = sorted(self.side_attr_ids)

        self.attr_n_ops = np.array(self.meta["attr_n_ops"])
        self.ctrl_attr_ops = self.attr_n_ops[self.ctrl_attr_ids]
        self.side_attr_ops = self.attr_n_ops[self.side_attr_ids]

    def get_split(self, split, include_self=False):
        return SyntheticSplit(self.folder, self.ctrl_attr_ids, split, include_self)


class SyntheticSplit(Dataset):
    def __init__(self, folder, ctrl_attr_ids, split="train", include_self=False, threshold=0.00001):
        super().__init__()
        assert split in ("train", "valid", "test"), "Please specify a valid split."
        self.split = split            
        self.folder = folder

        self.ctrl_attr_ids = ctrl_attr_ids
        self.threshold = threshold
        self.include_self = include_self  # whether include pairs that src==tgt

        self._load_data()

        print(f"Split: {self.split}, include self pairs: {self.include_self}, total samples after filtering {self.n_samples}.")

    def _load_data(self):
        ts = np.load(os.path.join(self.folder, self.split+"_ts.npy"))     # [n_samples, 2, n_steps]
        attrs = np.load(os.path.join(self.folder, self.split+"_attrs_idx.npy"))  # [n_samples, 2, n_attrs]

        if not self.include_self:
            ts, attrs = self._filter_self(ts, attrs)
        
        self.ts, self.attrs = ts, attrs

        self.n_samples = self.ts.shape[0]
        self.n_steps = self.ts.shape[2]
        self.n_attrs = self.attrs.shape[2]
        self.time_point = np.arange(self.n_steps)
    
    def _filter_self(self, ts, attrs):
        valid_ids = []
        for i in range(len(ts)):
            src_attrs, tgt_attrs = attrs[i]
            diff = np.abs(src_attrs[self.ctrl_attr_ids] - tgt_attrs[self.ctrl_attr_ids])
            if np.sum(diff) > self.threshold:
                valid_ids.append(i)
        return ts[valid_ids], attrs[valid_ids]

    def __getitem__(self, idx):
        src_x, tgt_x = self.ts[idx]
        src_attrs, tgt_attrs = self.attrs[idx]
        return {"src_x": src_x[...,np.newaxis], # (n_steps,1)
                "src_attrs": src_attrs,
                "tp": self.time_point,
                "tgt_x": tgt_x[...,np.newaxis],  
                "tgt_attrs": tgt_attrs}

    def __len__(self):
        return self.n_samples
