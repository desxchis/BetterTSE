from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


STRENGTH_ORDER = {"weak": 0, "medium": 1, "strong": 2}
LEGACY_STRENGTH_TO_SCALAR = {"weak": 0.0, "medium": 0.5, "strong": 1.0}
FLOAT_TOL = 1.0e-6


def _mean_abs_edit_gain(source_ts: np.ndarray, target_ts: np.ndarray, mask_gt: np.ndarray) -> float:
    edit_mask = np.asarray(mask_gt, dtype=bool)
    if not np.any(edit_mask):
        raise ValueError("edit mask must be non-empty")
    src = np.asarray(source_ts, dtype=np.float32).reshape(-1)
    tgt = np.asarray(target_ts, dtype=np.float32).reshape(-1)
    return float(np.mean(np.abs(tgt[edit_mask] - src[edit_mask])))


def _background_leakage(source_ts: np.ndarray, target_ts: np.ndarray, mask_gt: np.ndarray) -> float:
    edit_mask = np.asarray(mask_gt, dtype=bool)
    bg_mask = ~edit_mask
    if not np.any(bg_mask):
        return 0.0
    src = np.asarray(source_ts, dtype=np.float32).reshape(-1)
    tgt = np.asarray(target_ts, dtype=np.float32).reshape(-1)
    return float(np.max(np.abs(tgt[bg_mask] - src[bg_mask])))


def _family_metadata_signature(sample: Dict[str, Any]) -> Tuple[Any, ...]:
    return (
        sample.get("direction"),
        sample.get("tool_name"),
        sample.get("duration_bucket"),
        sample.get("region_start"),
        sample.get("region_end"),
        sample.get("series_length"),
    )


def _validate_family(family: Dict[str, Any]) -> List[Dict[str, Any]]:
    family_id = str(family.get("family_id"))
    samples = [dict(sample) for sample in family.get("samples", [])]
    if len(samples) < 2:
        raise ValueError(f"Family {family_id} must contain at least 2 samples, got {len(samples)}")

    for sample in samples:
        strength_text = str(sample.get("strength_text", ""))
        strength_scalar = sample.get("strength_scalar")
        if strength_scalar is None:
            strength_scalar = LEGACY_STRENGTH_TO_SCALAR.get(strength_text)
        if strength_scalar is None:
            raise ValueError(f"Family {family_id} sample is missing strength_scalar and has unknown strength_text={strength_text}")
        sample["strength_scalar"] = float(strength_scalar)
        if sample.get("strength_label") is None and strength_text in STRENGTH_ORDER:
            sample["strength_label"] = int(STRENGTH_ORDER[strength_text])

    samples = sorted(samples, key=lambda sample: float(sample["strength_scalar"]))
    strength_scalars = [float(sample["strength_scalar"]) for sample in samples]
    if any((right - left) <= FLOAT_TOL for left, right in zip(strength_scalars[:-1], strength_scalars[1:])):
        raise ValueError(f"Family {family_id} strength_scalar must be strictly increasing, got {strength_scalars}")

    base_source = np.asarray(samples[0]["source_ts"], dtype=np.float32).reshape(-1)
    base_mask = np.asarray(samples[0]["mask_gt"], dtype=np.float32).reshape(-1)
    if not np.any(base_mask > 0.5):
        raise ValueError(f"Family {family_id} has empty edit mask")
    base_signature = _family_metadata_signature(samples[0])
    target_gains: List[float] = []

    for sample in samples:
        source_ts = np.asarray(sample["source_ts"], dtype=np.float32).reshape(-1)
        target_ts = np.asarray(sample["target_ts"], dtype=np.float32).reshape(-1)
        mask_gt = np.asarray(sample["mask_gt"], dtype=np.float32).reshape(-1)
        if source_ts.shape != base_source.shape:
            raise ValueError(f"Family {family_id} source length mismatch across strengths")
        if target_ts.shape != base_source.shape or mask_gt.shape != base_source.shape:
            raise ValueError(f"Family {family_id} target/mask length mismatch across strengths")
        if not np.allclose(source_ts, base_source, atol=FLOAT_TOL, rtol=0.0):
            raise ValueError(f"Family {family_id} source_ts differs across strengths")
        if not np.allclose(mask_gt, base_mask, atol=FLOAT_TOL, rtol=0.0):
            raise ValueError(f"Family {family_id} mask_gt differs across strengths")
        if _family_metadata_signature(sample) != base_signature:
            raise ValueError(f"Family {family_id} metadata differs across strengths")
        if _background_leakage(source_ts, target_ts, mask_gt) > 1.0e-5:
            raise ValueError(f"Family {family_id} target background leakage exceeds tolerance")
        target_gains.append(_mean_abs_edit_gain(source_ts, target_ts, mask_gt))

    if any((right + FLOAT_TOL) < left for left, right in zip(target_gains[:-1], target_gains[1:])):
        raise ValueError(f"Family {family_id} target gains are not monotonic in strength_scalar order: {target_gains}")
    return samples


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
        self.validated_samples: List[List[Dict[str, Any]]] = [_validate_family(family) for family in self.families]

    def __len__(self) -> int:
        return len(self.families)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        family = dict(self.families[idx])
        samples = [dict(sample) for sample in self.validated_samples[idx]]

        direction = str(samples[0].get("direction", family.get("direction", "up")))
        src_attrs, tgt_attrs = _trend_attrs(direction)
        src_x = np.asarray(samples[0]["source_ts"], dtype=np.float32)
        tp = np.arange(src_x.shape[0], dtype=np.float32)

        records: List[Dict[str, Any]] = []
        for sample in samples:
            strength_label = sample.get("strength_label")
            records.append(
                {
                    "family_id": str(family["family_id"]),
                    "strength_text": str(sample["strength_text"]),
                    "strength_label": -1 if strength_label is None else int(strength_label),
                    "strength_scalar": float(sample["strength_scalar"]),
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
    strength_scalar = [float(sample["strength_scalar"]) for sample in flat_samples]

    family_valid = all(size >= 2 for size in family_sizes)
    family_order_valid = all(
        all(
            (strength_scalar[start + idx + 1] - strength_scalar[start + idx]) > FLOAT_TOL
            for idx in range(max(0, size - 1))
        )
        for start, size in zip(np.cumsum([0] + family_sizes[:-1]).tolist(), family_sizes)
    )

    return {
        "src_x": _stack_array("src_x", torch.float32),
        "tgt_x": _stack_array("tgt_x", torch.float32),
        "mask_gt": _stack_array("mask_gt", torch.float32),
        "tp": _stack_array("tp", torch.float32),
        "src_attrs": _stack_array("src_attrs", torch.long),
        "tgt_attrs": _stack_array("tgt_attrs", torch.long),
        "strength_label": _stack_array("strength_label", torch.long),
        "strength_scalar": torch.as_tensor(strength_scalar, dtype=torch.float32),
        "instruction_text": instruction_text,
        "strength_text": strength_text,
        "family_sizes": torch.as_tensor(family_sizes, dtype=torch.long),
        "family_ids": family_ids,
        "family_valid": bool(family_valid),
        "family_order_valid": bool(family_order_valid),
    }
