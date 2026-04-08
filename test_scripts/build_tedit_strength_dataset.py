from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


TASK_NAME_TO_ID = {
    "trend_up": 0,
    "trend_down": 1,
    "seasonality_neutral": 2,
    "volatility_neutral": 3,
    "mixed_neutral": 4,
}


def infer_strength_label(src_attrs: np.ndarray, tgt_attrs: np.ndarray, ctrl_attr_ids: list[int], attr_n_ops: list[int]) -> int:
    diffs = np.abs(src_attrs[ctrl_attr_ids] - tgt_attrs[ctrl_attr_ids]).astype(np.float64)
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


def infer_task_id(src_attrs: np.ndarray, tgt_attrs: np.ndarray, ctrl_attr_ids: list[int], attr_list: list[str]) -> int:
    changed_attr_names = [
        str(attr_list[attr_id])
        for attr_id in ctrl_attr_ids
        if int(src_attrs[attr_id]) != int(tgt_attrs[attr_id])
    ]
    if not changed_attr_names:
        return TASK_NAME_TO_ID["mixed_neutral"]
    if "trend_types" in changed_attr_names or "trend_directions" in changed_attr_names:
        direction_idx = int(tgt_attrs[1]) if len(tgt_attrs) > 1 else 1
        return TASK_NAME_TO_ID["trend_up" if direction_idx > 0 else "trend_down"]
    if "season_cycles" in changed_attr_names:
        return TASK_NAME_TO_ID["seasonality_neutral"]
    return TASK_NAME_TO_ID["mixed_neutral"]


def augment_dataset(folder: Path) -> dict[str, object]:
    meta_path = folder / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    attr_list = list(meta["attr_list"])
    ctrl_attr_ids = list(meta["control_attr_ids"])
    attr_n_ops = list(meta["attr_n_ops"])

    split_counts: dict[str, int] = {}
    for split in ("train", "valid", "test"):
        attrs_path = folder / f"{split}_attrs_idx.npy"
        if not attrs_path.exists():
            continue
        attrs = np.load(attrs_path)
        strength_labels = np.asarray(
            [
                infer_strength_label(src_attrs, tgt_attrs, ctrl_attr_ids, attr_n_ops)
                for src_attrs, tgt_attrs in attrs
            ],
            dtype=np.int64,
        )
        task_ids = np.asarray(
            [
                infer_task_id(src_attrs, tgt_attrs, ctrl_attr_ids, attr_list)
                for src_attrs, tgt_attrs in attrs
            ],
            dtype=np.int64,
        )
        np.save(folder / f"{split}_strength.npy", strength_labels)
        np.save(folder / f"{split}_task_id.npy", task_ids)
        split_counts[split] = int(len(attrs))

    meta["strength_control"] = {
        "strength_bins": ["weak", "medium", "strong"],
        "task_name_to_id": TASK_NAME_TO_ID,
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"folder": str(folder), "splits": split_counts}


def main() -> None:
    parser = argparse.ArgumentParser(description="Augment a TEdit synthetic dataset with strength/task labels.")
    parser.add_argument("--dataset-folder", required=True)
    args = parser.parse_args()

    result = augment_dataset(Path(args.dataset_folder))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
