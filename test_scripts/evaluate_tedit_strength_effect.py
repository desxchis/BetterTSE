from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

import sys

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
_TEDIT_ROOT = _ROOT / "TEdit-main"
if str(_TEDIT_ROOT) not in sys.path:
    sys.path.insert(0, str(_TEDIT_ROOT))

from data.synthetic_finetune import STRENGTH_ID_TO_TEXT, SyntheticDataset
from tool.tedit_wrapper import TEditWrapper


def _select_indices(split, max_samples: int) -> list[int]:
    indices: list[int] = []
    for idx, (src_attrs, tgt_attrs) in enumerate(split.attrs):
        if np.array_equal(src_attrs, tgt_attrs):
            continue
        indices.append(idx)
        if len(indices) >= max_samples:
            break
    return indices


def _build_instruction_text(src_attrs: np.ndarray, tgt_attrs: np.ndarray, attr_list: list[str], strength_label: int) -> str:
    strength_text = STRENGTH_ID_TO_TEXT.get(int(strength_label), "medium")
    change_tokens = []
    for attr_id, attr_name in enumerate(attr_list):
        if int(src_attrs[attr_id]) == int(tgt_attrs[attr_id]):
            continue
        change_tokens.append(
            f"change {attr_name} from {int(src_attrs[attr_id])} to {int(tgt_attrs[attr_id])}"
        )
    if not change_tokens:
        change_tokens.append("preserve attributes")
    return f"apply {strength_text} edit and " + " and ".join(change_tokens)


def _compute_region_metrics(
    base: np.ndarray,
    target: np.ndarray,
    edited: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float | None]:
    base = np.asarray(base, dtype=np.float32).reshape(-1)
    target = np.asarray(target, dtype=np.float32).reshape(-1)
    edited = np.asarray(edited, dtype=np.float32).reshape(-1)
    mask = np.asarray(mask, dtype=bool).reshape(-1)

    edit_mask = mask.astype(bool)
    bg_mask = ~edit_mask

    abs_delta = np.abs(edited - base)
    abs_target_err = np.abs(edited - target)

    metrics: dict[str, float | None] = {
        "edit_gain": None,
        "target_mae_edit_region": None,
        "bg_mae": None,
    }
    if np.any(edit_mask):
        metrics["edit_gain"] = float(np.mean(abs_delta[edit_mask]))
        metrics["target_mae_edit_region"] = float(np.mean(abs_target_err[edit_mask]))
    if np.any(bg_mask):
        metrics["bg_mae"] = float(np.mean(abs_delta[bg_mask]))
    return metrics


def _aggregate_mean(values: list[float | None]) -> float | None:
    filtered = [float(v) for v in values if v is not None]
    if not filtered:
        return None
    return float(np.mean(filtered))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate weak/medium/strong strength control on TEdit synthetic pairs.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--dataset-folder", required=True)
    parser.add_argument("--split", default="test", choices=["train", "valid", "test"])
    parser.add_argument("--max-samples", type=int, default=10)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--edit-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--mask-threshold", type=float, default=1e-6)
    parser.add_argument("--bg-drift-threshold", type=float, default=0.05)
    parser.add_argument("--task-id", type=int, default=-1, help="use -1 to omit task_id for phase-1 strength+text evaluation")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    dataset = SyntheticDataset(args.dataset_folder)
    split = dataset.get_split(args.split, include_self=True)
    selected_indices = _select_indices(split, max_samples=args.max_samples)
    if not selected_indices:
        raise ValueError("No non-self synthetic samples were found for evaluation.")

    wrapper = TEditWrapper(model_path=args.model_path, config_path=args.config_path, device=args.device)

    strengths = [0, 1, 2]
    strength_rows: dict[int, list[dict[str, Any]]] = {s: [] for s in strengths}
    pairwise_rows: list[dict[str, Any]] = []

    for eval_idx, sample_idx in enumerate(selected_indices):
        base = split.ts[sample_idx, 0].astype(np.float32)
        target = split.ts[sample_idx, 1].astype(np.float32)
        src_attrs = split.attrs[sample_idx, 0].astype(np.int64)
        tgt_attrs = split.attrs[sample_idx, 1].astype(np.int64)
        edit_mask = np.abs(target - base) > float(args.mask_threshold)

        per_strength: dict[int, dict[str, Any]] = {}
        for strength_label in strengths:
            instruction_text = _build_instruction_text(
                src_attrs=src_attrs,
                tgt_attrs=tgt_attrs,
                attr_list=dataset.attr_list,
                strength_label=strength_label,
            )
            torch.manual_seed(args.seed + eval_idx)
            np.random.seed(args.seed + eval_idx)
            edited = wrapper.edit_time_series(
                ts=base,
                src_attrs=src_attrs,
                tgt_attrs=tgt_attrs,
                n_samples=1,
                sampler="ddim",
                edit_steps=args.edit_steps,
                strength_label=strength_label,
                task_id=None if args.task_id < 0 else args.task_id,
                instruction_text=instruction_text,
            )[0]

            metrics = _compute_region_metrics(base=base, target=target, edited=edited, mask=edit_mask)
            row = {
                "sample_idx": int(sample_idx),
                "strength_label": int(strength_label),
                "strength_text": STRENGTH_ID_TO_TEXT[int(strength_label)],
                "instruction_text": instruction_text,
                "edit_region_fraction": float(np.mean(edit_mask.astype(np.float32))),
                "edit_gain": metrics["edit_gain"],
                "bg_mae": metrics["bg_mae"],
                "target_mae_edit_region": metrics["target_mae_edit_region"],
            }
            strength_rows[strength_label].append(row)
            per_strength[strength_label] = row

        g0 = per_strength[0]["edit_gain"]
        g1 = per_strength[1]["edit_gain"]
        g2 = per_strength[2]["edit_gain"]
        monotonic = bool(g0 is not None and g1 is not None and g2 is not None and g0 < g1 < g2)
        pairwise_rows.append(
            {
                "sample_idx": int(sample_idx),
                "edit_region_fraction": float(np.mean(edit_mask.astype(np.float32))),
                "weak_edit_gain": g0,
                "medium_edit_gain": g1,
                "strong_edit_gain": g2,
                "weak_bg_mae": per_strength[0]["bg_mae"],
                "medium_bg_mae": per_strength[1]["bg_mae"],
                "strong_bg_mae": per_strength[2]["bg_mae"],
                "strong_minus_weak_edit_gain": None if g0 is None or g2 is None else float(g2 - g0),
                "monotonic_hit": monotonic,
            }
        )

    summary = {
        "n_samples": len(selected_indices),
        "strength_labels": {str(k): STRENGTH_ID_TO_TEXT[k] for k in strengths},
        "edit_gain_mean": {
            STRENGTH_ID_TO_TEXT[k]: _aggregate_mean([row["edit_gain"] for row in strength_rows[k]])
            for k in strengths
        },
        "bg_mae_mean": {
            STRENGTH_ID_TO_TEXT[k]: _aggregate_mean([row["bg_mae"] for row in strength_rows[k]])
            for k in strengths
        },
        "target_mae_edit_region_mean": {
            STRENGTH_ID_TO_TEXT[k]: _aggregate_mean([row["target_mae_edit_region"] for row in strength_rows[k]])
            for k in strengths
        },
        "monotonic_hit_rate": float(np.mean([float(row["monotonic_hit"]) for row in pairwise_rows])),
        "strong_minus_weak_edit_gain_mean": _aggregate_mean(
            [row["strong_minus_weak_edit_gain"] for row in pairwise_rows]
        ),
        "bg_mae_strong_minus_weak": None
        if _aggregate_mean([row["strong_bg_mae"] for row in pairwise_rows]) is None
        or _aggregate_mean([row["weak_bg_mae"] for row in pairwise_rows]) is None
        else float(
            _aggregate_mean([row["strong_bg_mae"] for row in pairwise_rows])
            - _aggregate_mean([row["weak_bg_mae"] for row in pairwise_rows])
        ),
    }
    bg_drift = summary["bg_mae_strong_minus_weak"]
    summary["preservation_pass"] = bool(
        bg_drift is not None and float(bg_drift) <= float(args.bg_drift_threshold)
    )

    payload = {
        "config": {
            "model_path": args.model_path,
            "config_path": args.config_path,
            "dataset_folder": args.dataset_folder,
            "split": args.split,
            "max_samples": int(args.max_samples),
            "edit_steps": int(args.edit_steps),
            "seed": int(args.seed),
            "mask_threshold": float(args.mask_threshold),
            "bg_drift_threshold": float(args.bg_drift_threshold),
            "task_id": None if args.task_id < 0 else int(args.task_id),
        },
        "summary": summary,
        "per_sample": pairwise_rows,
        "per_strength_rows": strength_rows,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
