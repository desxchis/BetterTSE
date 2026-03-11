import os
import json
import numpy as np
from torch.utils.data import Dataset


class SyntheticPretrainDataset:
    def __init__(self, folder, **kwargs):
        super().__init__()
        self.folder = folder
        self._load_meta()

    def _load_meta(self):
        self.meta = json.load(open(os.path.join(self.folder, "meta.json")))
        self.attr_list = self.meta["attr_list"]
        n_attr = len(self.attr_list)
        self.attr_ids = np.arange(n_attr)
        self.attr_n_ops = np.array(self.meta["attr_n_ops"])

    def get_split(self, split, *args):
        return SyntheticPretrainSplit(self.folder, split)


class SyntheticPretrainSplit(Dataset):
    def __init__(self, folder, split="train"):
        super().__init__()
        assert split in ("train", "valid", "test", "all"), "Please specify a valid split."
        self.split = split            
        self.folder = folder

        self._load_data()

        print(f"Split: {self.split}, total samples {self.n_samples}.")

    def _load_data(self):
        if self.split == "all":
            ts_list = []
            attrs_list = []
            for split in ["train","valid","test"]:
                ts_list.append(np.load(os.path.join(self.folder, split+"_ts.npy")))
                attrs_list.append(np.load(os.path.join(self.folder, split+"_attrs_idx.npy")))
            ts = np.concatenate(ts_list, axis=0)
            attrs = np.concatenate(attrs_list, axis=0)
        else:
            ts = np.load(os.path.join(self.folder, self.split+"_ts.npy"))     # [n_samples, n_steps]
            attrs = np.load(os.path.join(self.folder, self.split+"_attrs_idx.npy"))  # [n_samples, n_attrs]
        
        self.ts, self.attrs = ts, attrs

        self.n_samples = self.ts.shape[0]
        self.n_steps = self.ts.shape[1]
        self.n_attrs = self.attrs.shape[1]
        self.time_point = np.arange(self.n_steps)

    def __getitem__(self, idx):
        return {"x": self.ts[idx][...,np.newaxis], # (n_steps,1)
                "attrs": self.attrs[idx],
                "tp": self.time_point}

    def __len__(self):
        return self.n_samples