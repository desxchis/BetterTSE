import os
import json
import numpy as np
from torch.utils.data import Dataset


TASK_NAME_TO_ID = {
    "trend_up": 0,
    "trend_down": 1,
    "seasonality_neutral": 2,
    "volatility_neutral": 3,
    "mixed_neutral": 4,
}

STRENGTH_ID_TO_TEXT = {
    0: "weak",
    1: "medium",
    2: "strong",
}


def _infer_strength_label(src_attrs, tgt_attrs, ctrl_attr_ids, attr_n_ops):
    src = np.asarray(src_attrs, dtype=np.int64)
    tgt = np.asarray(tgt_attrs, dtype=np.int64)
    diffs = np.abs(src[ctrl_attr_ids] - tgt[ctrl_attr_ids]).astype(np.float64)
    if diffs.size == 0:
        return 1
    max_ops = np.maximum(np.asarray(attr_n_ops, dtype=np.float64)[ctrl_attr_ids] - 1.0, 1.0)
    norm = diffs / max_ops
    score = float(np.mean(norm) + 0.35 * np.max(norm))
    if score < 0.34:
        return 0
    if score < 0.67:
        return 1
    return 2


def _infer_task_id(src_attrs, tgt_attrs, ctrl_attr_ids, attr_list):
    src = np.asarray(src_attrs, dtype=np.int64)
    tgt = np.asarray(tgt_attrs, dtype=np.int64)
    changed_attr_names = [
        str(attr_list[attr_id])
        for attr_id in ctrl_attr_ids
        if int(src[attr_id]) != int(tgt[attr_id])
    ]
    if not changed_attr_names:
        return TASK_NAME_TO_ID["mixed_neutral"]

    if "trend_types" in changed_attr_names or "trend_directions" in changed_attr_names:
        direction_idx = int(tgt[1]) if len(tgt) > 1 else 1
        return TASK_NAME_TO_ID["trend_up" if direction_idx > 0 else "trend_down"]
    if "season_cycles" in changed_attr_names:
        return TASK_NAME_TO_ID["seasonality_neutral"]
    return TASK_NAME_TO_ID["mixed_neutral"]


def _build_instruction_text(src_attrs, tgt_attrs, attr_list, strength_label):
    strength_text = STRENGTH_ID_TO_TEXT.get(int(strength_label), "medium")
    change_tokens = []
    src = np.asarray(src_attrs, dtype=np.int64)
    tgt = np.asarray(tgt_attrs, dtype=np.int64)
    for attr_id, attr_name in enumerate(attr_list):
        if int(src[attr_id]) == int(tgt[attr_id]):
            continue
        change_tokens.append(
            f"change {attr_name} from {int(src[attr_id])} to {int(tgt[attr_id])}"
        )
    if not change_tokens:
        change_tokens.append("preserve attributes")
    return f"apply {strength_text} edit and " + " and ".join(change_tokens)


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
        return SyntheticSplit(
            self.folder,
            self.ctrl_attr_ids,
            self.attr_list,
            self.attr_n_ops,
            split,
            include_self,
        )


class SyntheticSplit(Dataset):
    def __init__(
        self,
        folder,
        ctrl_attr_ids,
        attr_list,
        attr_n_ops,
        split="train",
        include_self=False,
        threshold=0.00001,
    ):
        super().__init__()
        assert split in ("train", "valid", "test"), "Please specify a valid split."
        self.split = split            
        self.folder = folder

        self.ctrl_attr_ids = ctrl_attr_ids
        self.attr_list = list(attr_list)
        self.attr_n_ops = np.asarray(attr_n_ops)
        self.threshold = threshold
        self.include_self = include_self  # whether include pairs that src==tgt

        self._load_data()

        print(f"Split: {self.split}, include self pairs: {self.include_self}, total samples after filtering {self.n_samples}.")

    def _load_data(self):
        ts = np.load(os.path.join(self.folder, self.split+"_ts.npy"))     # [n_samples, 2, n_steps]
        attrs = np.load(os.path.join(self.folder, self.split+"_attrs_idx.npy"))  # [n_samples, 2, n_attrs]
        strength_path = os.path.join(self.folder, self.split+"_strength.npy")
        task_path = os.path.join(self.folder, self.split+"_task_id.npy")

        strength_labels = np.load(strength_path) if os.path.exists(strength_path) else None
        task_ids = np.load(task_path) if os.path.exists(task_path) else None

        if not self.include_self:
            ts, attrs, strength_labels, task_ids = self._filter_self(ts, attrs, strength_labels, task_ids)
        
        self.ts, self.attrs = ts, attrs
        if strength_labels is None:
            strength_labels = np.asarray(
                [
                    _infer_strength_label(src_attrs, tgt_attrs, self.ctrl_attr_ids, self.attr_n_ops)
                    for src_attrs, tgt_attrs in attrs
                ],
                dtype=np.int64,
            )
        if task_ids is None:
            task_ids = np.asarray(
                [
                    _infer_task_id(src_attrs, tgt_attrs, self.ctrl_attr_ids, self.attr_list)
                    for src_attrs, tgt_attrs in attrs
                ],
                dtype=np.int64,
            )
        self.strength_labels = np.asarray(strength_labels, dtype=np.int64)
        self.task_ids = np.asarray(task_ids, dtype=np.int64)

        self.n_samples = self.ts.shape[0]
        self.n_steps = self.ts.shape[2]
        self.n_attrs = self.attrs.shape[2]
        self.time_point = np.arange(self.n_steps)
    
    def _filter_self(self, ts, attrs, strength_labels=None, task_ids=None):
        valid_ids = []
        for i in range(len(ts)):
            src_attrs, tgt_attrs = attrs[i]
            diff = np.abs(src_attrs[self.ctrl_attr_ids] - tgt_attrs[self.ctrl_attr_ids])
            if np.sum(diff) > self.threshold:
                valid_ids.append(i)
        valid_ids = np.asarray(valid_ids, dtype=np.int64)
        filtered_strength = None if strength_labels is None else strength_labels[valid_ids]
        filtered_task = None if task_ids is None else task_ids[valid_ids]
        return ts[valid_ids], attrs[valid_ids], filtered_strength, filtered_task

    def __getitem__(self, idx):
        src_x, tgt_x = self.ts[idx]
        src_attrs, tgt_attrs = self.attrs[idx]
        strength_label = int(self.strength_labels[idx])
        task_id = int(self.task_ids[idx])
        instruction_text = _build_instruction_text(
            src_attrs,
            tgt_attrs,
            self.attr_list,
            strength_label,
        )
        return {"src_x": src_x[...,np.newaxis], # (n_steps,1)
                "src_attrs": src_attrs,
                "tp": self.time_point,
                "tgt_x": tgt_x[...,np.newaxis],  
                "tgt_attrs": tgt_attrs,
                "strength_label": strength_label,
                "task_id": task_id,
                "instruction_text": instruction_text}

    def __len__(self):
        return self.n_samples
