from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


STRENGTH_ORDER = {"weak": 0, "medium": 1, "strong": 2}


def _trend_attrs(direction: str) -> tuple[np.ndarray, np.ndarray]:
    src_attrs = np.asarray([0, 0, 0], dtype=np.int64)
    if str(direction) in {"down", "downward"}:
        tgt_attrs = np.asarray([1, 0, 1], dtype=np.int64)
    else:
        tgt_attrs = np.asarray([1, 1, 1], dtype=np.int64)
    return src_attrs, tgt_attrs


class DiscreteStrengthFamilyDataset:
    def __init__(self, folder: str, **kwargs: Any):
        self.folder = folder
        self._load_meta()

    def _load_meta(self) -> None:
        meta_path = Path(self.folder) / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"Discrete strength family meta not found: {meta_path}")
        self.meta = json.loads(meta_path.read_text(encoding="utf-8"))
        self.attr_list = list(self.meta["attr_list"])
        self.attr_n_ops = np.asarray(self.meta["attr_n_ops"], dtype=np.int64)
        self.ctrl_attr_ids = list(self.meta["control_attr_ids"])
        self.side_attr_ids = [idx for idx in range(len(self.attr_list)) if idx not in self.ctrl_attr_ids]
        self.ctrl_attr_ops = self.attr_n_ops[self.ctrl_attr_ids]
        self.side_attr_ops = self.attr_n_ops[self.side_attr_ids]

    def get_split(self, split: str, include_self: bool = False):
        return DiscreteStrengthFamilySplit(self.folder, split=split, meta=self.meta)

    def get_loader(self, split: str, batch_size: int, shuffle: bool = True, num_workers: int = 1, include_self: bool = False):
        dataset = self.get_split(split, include_self=include_self)
        return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate_discrete_strength_families,
        )


class DiscreteStrengthFamilySplit(Dataset):
    def __init__(self, folder: str, split: str, meta: Dict[str, Any]):
        self.folder = folder
        self.split = split
        self.meta = meta
        split_path = Path(folder) / f"{split}.json"
        if not split_path.exists():
            raise FileNotFoundError(f"Discrete strength family split not found: {split_path}")
        payload = json.loads(split_path.read_text(encoding="utf-8"))
        self.families = list(payload.get("families", []))
        self.attr_list = list(meta["attr_list"])
        self.n_attrs = len(self.attr_list)

    def __len__(self) -> int:
        return len(self.families)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        family = dict(self.families[idx])
        samples = sorted(
            [dict(sample) for sample in family.get("samples", [])],
            key=lambda sample: STRENGTH_ORDER[str(sample["strength_text"])],
        )
        if [str(sample["strength_text"]) for sample in samples] != ["weak", "medium", "strong"]:
            raise ValueError(f"Family {family.get('family_id')} does not contain ordered weak/medium/strong samples.")

        direction = str(samples[0].get("direction", family.get("direction", "up")))
        src_attrs, tgt_attrs = _trend_attrs(direction)
        src_x = np.asarray(samples[0]["source_ts"], dtype=np.float32)
        tp = np.arange(src_x.shape[0], dtype=np.float32)

        records: List[Dict[str, Any]] = []
        for sample in samples:
            records.append(
                {
                    "family_id": str(family["family_id"]),
                    "strength_text": str(sample["strength_text"]),
                    "strength_label": int(sample["strength_label"]),
                    "instruction_text": str(sample["instruction_text"]),
                    "src_x": src_x[..., np.newaxis],
                    "tgt_x": np.asarray(sample["target_ts"], dtype=np.float32)[..., np.newaxis],
                    "mask_gt": np.asarray(sample["mask_gt"], dtype=np.float32)[..., np.newaxis],
                    "tp": tp,
                    "src_attrs": src_attrs,
                    "tgt_attrs": tgt_attrs,
                }
            )
        return {"family_id": str(family["family_id"]), "samples": records}


def collate_discrete_strength_families(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    flat_samples: List[Dict[str, Any]] = []
    family_sizes: List[int] = []
    family_ids: List[str] = []
    for family in batch:
        samples = list(family["samples"])
        family_sizes.append(len(samples))
        family_ids.append(str(family["family_id"]))
        flat_samples.extend(samples)

    def _stack_array(key: str, dtype: torch.dtype) -> torch.Tensor:
        arr = np.stack([np.asarray(sample[key]) for sample in flat_samples], axis=0)
        return torch.as_tensor(arr, dtype=dtype)

    instruction_text = [str(sample["instruction_text"]) for sample in flat_samples]
    strength_text = [str(sample["strength_text"]) for sample in flat_samples]

    return {
        "src_x": _stack_array("src_x", torch.float32),
        "tgt_x": _stack_array("tgt_x", torch.float32),
        "mask_gt": _stack_array("mask_gt", torch.float32),
        "tp": _stack_array("tp", torch.float32),
        "src_attrs": _stack_array("src_attrs", torch.long),
        "tgt_attrs": _stack_array("tgt_attrs", torch.long),
        "strength_label": _stack_array("strength_label", torch.long),
        "instruction_text": instruction_text,
        "strength_text": strength_text,
        "family_sizes": torch.as_tensor(family_sizes, dtype=torch.long),
        "family_ids": family_ids,
    }
