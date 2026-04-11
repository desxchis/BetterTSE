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


def _load_eval_records(dataset_folder: str, split: str, max_samples: int) -> tuple[list[dict[str, Any]], list[str]]:
    root = Path(dataset_folder)
    meta_path = root / "meta.json"
    if meta_path.exists():
        from data.discrete_strength_family import DiscreteStrengthFamilyDataset

        dataset = DiscreteStrengthFamilyDataset(str(root))
        split_ds = dataset.get_split(split, include_self=True)
        records: list[dict[str, Any]] = []
        for family_idx in range(min(len(split_ds), max_samples)):
            family = split_ds[family_idx]
            per_strength = sorted(family["samples"], key=lambda row: int(row["strength_label"]))
            base = np.asarray(per_strength[0]["src_x"], dtype=np.float32).squeeze(-1)
            target = np.asarray(per_strength[2]["tgt_x"], dtype=np.float32).squeeze(-1)
            src_attrs = np.asarray(per_strength[0]["src_attrs"], dtype=np.int64)
            tgt_attrs = np.asarray(per_strength[0]["tgt_attrs"], dtype=np.int64)
            instruction_by_strength = {
                int(row["strength_label"]): str(row.get("instruction_text")) if row.get("instruction_text") is not None else None
                for row in per_strength
            }
            edit_mask = np.asarray(per_strength[0]["mask_gt"], dtype=np.float32).squeeze(-1) > 0.5
            records.append(
                {
                    "sample_idx": int(family_idx),
                    "base": base,
                    "target": target,
                    "src_attrs": src_attrs,
                    "tgt_attrs": tgt_attrs,
                    "instruction_by_strength": instruction_by_strength,
                    "edit_mask": edit_mask,
                }
            )
        return records, list(dataset.attr_list)

    dataset = SyntheticDataset(dataset_folder)
    split_ds = dataset.get_split(split, include_self=True)
    indices: list[int] = []
    for idx, (src_attrs, tgt_attrs) in enumerate(split_ds.attrs):
        if np.array_equal(src_attrs, tgt_attrs):
            continue
        indices.append(idx)
        if len(indices) >= max_samples:
            break
    records = []
    for idx in indices:
        records.append(
            {
                "sample_idx": int(idx),
                "base": split_ds.ts[idx, 0].astype(np.float32),
                "target": split_ds.ts[idx, 1].astype(np.float32),
                "src_attrs": split_ds.attrs[idx, 0].astype(np.int64),
                "tgt_attrs": split_ds.attrs[idx, 1].astype(np.int64),
                "instruction_by_strength": {},
                "edit_mask": np.abs(split_ds.ts[idx, 1].astype(np.float32) - split_ds.ts[idx, 0].astype(np.float32)) > 0,
            }
        )
    return records, list(dataset.attr_list)


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


def _aggregate_metric(rows: list[dict[str, Any]], key: str) -> float | None:
    return _aggregate_mean([row.get(key) for row in rows])


def _safe_diff(high: float | None, low: float | None) -> float | None:
    if high is None or low is None:
        return None
    return float(high - low)


def _extract_projector_pairwise(diag: dict[str, Any]) -> dict[str, float]:
    projector = diag.get("projector") or []
    latest = projector[-1] if projector else {}
    pairwise = latest.get("projector_output_pairwise_l2") or {}
    return {
        str(key): float(value)
        for key, value in pairwise.items()
        if isinstance(value, (int, float))
    }


def _extract_generator_metrics(diag: dict[str, Any]) -> dict[str, float | None]:
    generator = diag.get("generator") or []
    latest = generator[-1] if generator else {}
    return {
        "raw_edit_region_mean_abs_delta": latest.get("raw_edit_region_mean_abs_delta"),
        "final_edit_region_mean_abs_delta": latest.get("final_edit_region_mean_abs_delta"),
        "raw_background_mean_abs_delta": latest.get("raw_background_mean_abs_delta"),
        "final_background_mean_abs_delta": latest.get("final_background_mean_abs_delta"),
        "blend_gap_edit_region_mean_abs": latest.get("blend_gap_edit_region_mean_abs"),
        "blend_gap_background_mean_abs": latest.get("blend_gap_background_mean_abs"),
    }


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
    parser.add_argument("--condition-mode", default="both", choices=["both", "label_only", "text_only"], help="ablate whether strength comes from numeric label, text, or both")
    parser.add_argument("--enable-strength-diagnostics", type=int, default=1)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    eval_records, attr_list = _load_eval_records(args.dataset_folder, args.split, args.max_samples)
    if not eval_records:
        raise ValueError("No evaluation records were found for the requested split.")

    wrapper = TEditWrapper(model_path=args.model_path, config_path=args.config_path, device=args.device)

    strengths = [0, 1, 2]
    strength_rows: dict[int, list[dict[str, Any]]] = {s: [] for s in strengths}
    pairwise_rows: list[dict[str, Any]] = []
    raw_final_rows: list[dict[str, Any]] = []

    for eval_idx, record in enumerate(eval_records):
        sample_idx = int(record["sample_idx"])
        base = np.asarray(record["base"], dtype=np.float32)
        target = np.asarray(record["target"], dtype=np.float32)
        src_attrs = np.asarray(record["src_attrs"], dtype=np.int64)
        tgt_attrs = np.asarray(record["tgt_attrs"], dtype=np.int64)
        instruction_by_strength = dict(record.get("instruction_by_strength") or {})
        edit_mask = np.asarray(record["edit_mask"], dtype=bool)
        if not np.any(edit_mask):
            edit_mask = np.abs(target - base) > float(args.mask_threshold)


        per_strength: dict[int, dict[str, Any]] = {}
        for strength_label in strengths:
            instruction_text = instruction_by_strength.get(int(strength_label)) or _build_instruction_text(
                src_attrs=src_attrs,
                tgt_attrs=tgt_attrs,
                attr_list=attr_list,
                strength_label=strength_label,
            )
            if args.condition_mode == "label_only":
                instruction_text = None
            run_strength_label = int(strength_label)
            if args.condition_mode == "text_only":
                run_strength_label = 0
            torch.manual_seed(args.seed + eval_idx)
            np.random.seed(args.seed + eval_idx)
            edited, diagnostics = wrapper.edit_time_series(
                ts=base,
                src_attrs=src_attrs,
                tgt_attrs=tgt_attrs,
                n_samples=1,
                sampler="ddim",
                edit_steps=args.edit_steps,
                strength_label=run_strength_label,
                task_id=None if args.task_id < 0 else args.task_id,
                instruction_text=instruction_text,
                return_diagnostics=True,
                enable_strength_diagnostics=bool(args.enable_strength_diagnostics),
            )
            edited = edited[0]

            metrics = _compute_region_metrics(base=base, target=target, edited=edited, mask=edit_mask)
            generator_diag = _extract_generator_metrics(diagnostics)
            projector_pairwise = _extract_projector_pairwise(diagnostics)
            row = {
                "sample_idx": int(sample_idx),
                "strength_label": int(strength_label),
                "runtime_strength_label": int(run_strength_label),
                "strength_text": STRENGTH_ID_TO_TEXT[int(strength_label)],
                "instruction_text": instruction_text,
                "edit_region_fraction": float(np.mean(edit_mask.astype(np.float32))),
                "edit_gain": metrics["edit_gain"],
                "bg_mae": metrics["bg_mae"],
                "target_mae_edit_region": metrics["target_mae_edit_region"],
                "raw_edit_region_mean_abs_delta": generator_diag.get("raw_edit_region_mean_abs_delta"),
                "final_edit_region_mean_abs_delta": generator_diag.get("final_edit_region_mean_abs_delta"),
                "raw_background_mean_abs_delta": generator_diag.get("raw_background_mean_abs_delta"),
                "final_background_mean_abs_delta": generator_diag.get("final_background_mean_abs_delta"),
                "blend_gap_edit_region_mean_abs": generator_diag.get("blend_gap_edit_region_mean_abs"),
                "blend_gap_background_mean_abs": generator_diag.get("blend_gap_background_mean_abs"),
                "projector_pairwise_l2": projector_pairwise,
            }
            row["raw_final_gap_edit_region"] = _safe_diff(
                row["raw_edit_region_mean_abs_delta"],
                row["final_edit_region_mean_abs_delta"],
            )
            row["raw_final_gap_background"] = _safe_diff(
                row["raw_background_mean_abs_delta"],
                row["final_background_mean_abs_delta"],
            )
            row["preservation_attenuation_ratio"] = None
            if row["raw_edit_region_mean_abs_delta"] not in (None, 0.0) and row["final_edit_region_mean_abs_delta"] is not None:
                row["preservation_attenuation_ratio"] = float(
                    row["final_edit_region_mean_abs_delta"] / row["raw_edit_region_mean_abs_delta"]
                )
            row["projector_pairwise_0_1"] = projector_pairwise.get("0_1")
            row["projector_pairwise_1_2"] = projector_pairwise.get("1_2")
            row["projector_pairwise_0_2"] = projector_pairwise.get("0_2")
            row["projector_pairwise_0_1_mean_abs"] = projector_pairwise.get("0_1_mean_abs")
            row["projector_pairwise_1_2_mean_abs"] = projector_pairwise.get("1_2_mean_abs")
            row["projector_pairwise_0_2_mean_abs"] = projector_pairwise.get("0_2_mean_abs")
            row["raw_monotonic_local"] = None
            row["final_monotonic_local"] = None
            row["attenuation_suspected"] = None
            raw_final_rows.append(
                {
                    "sample_idx": int(sample_idx),
                    "strength_label": int(strength_label),
                    "strength_text": STRENGTH_ID_TO_TEXT[int(strength_label)],
                    "raw_edit_region_mean_abs_delta": generator_diag.get("raw_edit_region_mean_abs_delta"),
                    "final_edit_region_mean_abs_delta": generator_diag.get("final_edit_region_mean_abs_delta"),
                    "raw_background_mean_abs_delta": generator_diag.get("raw_background_mean_abs_delta"),
                    "final_background_mean_abs_delta": generator_diag.get("final_background_mean_abs_delta"),
                    "blend_gap_edit_region_mean_abs": generator_diag.get("blend_gap_edit_region_mean_abs"),
                    "blend_gap_background_mean_abs": generator_diag.get("blend_gap_background_mean_abs"),
                }
            )
            strength_rows[strength_label].append(row)
            per_strength[strength_label] = row

        g0 = per_strength[0]["edit_gain"]
        g1 = per_strength[1]["edit_gain"]
        g2 = per_strength[2]["edit_gain"]
        monotonic = bool(g0 is not None and g1 is not None and g2 is not None and g0 < g1 < g2)
        raw0 = per_strength[0]["raw_edit_region_mean_abs_delta"]
        raw1 = per_strength[1]["raw_edit_region_mean_abs_delta"]
        raw2 = per_strength[2]["raw_edit_region_mean_abs_delta"]
        final0 = per_strength[0]["final_edit_region_mean_abs_delta"]
        final1 = per_strength[1]["final_edit_region_mean_abs_delta"]
        final2 = per_strength[2]["final_edit_region_mean_abs_delta"]
        raw_monotonic = bool(raw0 is not None and raw1 is not None and raw2 is not None and raw0 < raw1 < raw2)
        final_monotonic = bool(final0 is not None and final1 is not None and final2 is not None and final0 < final1 < final2)
        attenuation_suspected = bool(raw_monotonic and not final_monotonic)
        for label, local_value in zip([0, 1, 2], [raw_monotonic, raw_monotonic, raw_monotonic]):
            per_strength[label]["raw_monotonic_local"] = raw_monotonic
        for label, local_value in zip([0, 1, 2], [final_monotonic, final_monotonic, final_monotonic]):
            per_strength[label]["final_monotonic_local"] = final_monotonic
            per_strength[label]["attenuation_suspected"] = attenuation_suspected
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
                "weak_raw_edit_region_mean_abs_delta": raw0,
                "medium_raw_edit_region_mean_abs_delta": raw1,
                "strong_raw_edit_region_mean_abs_delta": raw2,
                "weak_final_edit_region_mean_abs_delta": final0,
                "medium_final_edit_region_mean_abs_delta": final1,
                "strong_final_edit_region_mean_abs_delta": final2,
                "strong_minus_weak_edit_gain": None if g0 is None or g2 is None else float(g2 - g0),
                "raw_monotonic_hit": raw_monotonic,
                "final_monotonic_hit": final_monotonic,
                "attenuation_suspected": attenuation_suspected,
                "monotonic_hit": monotonic,
            }
        )

    summary = {
        "condition_mode": args.condition_mode,
        "n_samples": len(eval_records),
        "strength_labels": {str(k): STRENGTH_ID_TO_TEXT[k] for k in strengths},
        "edit_gain_mean": {
            STRENGTH_ID_TO_TEXT[k]: _aggregate_metric(strength_rows[k], "edit_gain")
            for k in strengths
        },
        "bg_mae_mean": {
            STRENGTH_ID_TO_TEXT[k]: _aggregate_metric(strength_rows[k], "bg_mae")
            for k in strengths
        },
        "target_mae_edit_region_mean": {
            STRENGTH_ID_TO_TEXT[k]: _aggregate_metric(strength_rows[k], "target_mae_edit_region")
            for k in strengths
        },
        "raw_edit_region_mean_abs_delta": {
            STRENGTH_ID_TO_TEXT[k]: _aggregate_metric(strength_rows[k], "raw_edit_region_mean_abs_delta")
            for k in strengths
        },
        "final_edit_region_mean_abs_delta": {
            STRENGTH_ID_TO_TEXT[k]: _aggregate_metric(strength_rows[k], "final_edit_region_mean_abs_delta")
            for k in strengths
        },
        "blend_gap_edit_region_mean_abs": {
            STRENGTH_ID_TO_TEXT[k]: _aggregate_metric(strength_rows[k], "blend_gap_edit_region_mean_abs")
            for k in strengths
        },
        "raw_background_mean_abs_delta": {
            STRENGTH_ID_TO_TEXT[k]: _aggregate_metric(strength_rows[k], "raw_background_mean_abs_delta")
            for k in strengths
        },
        "final_background_mean_abs_delta": {
            STRENGTH_ID_TO_TEXT[k]: _aggregate_metric(strength_rows[k], "final_background_mean_abs_delta")
            for k in strengths
        },
        "blend_gap_background_mean_abs": {
            STRENGTH_ID_TO_TEXT[k]: _aggregate_metric(strength_rows[k], "blend_gap_background_mean_abs")
            for k in strengths
        },
        "raw_final_gap_edit_region_mean": {
            STRENGTH_ID_TO_TEXT[k]: _aggregate_metric(strength_rows[k], "raw_final_gap_edit_region")
            for k in strengths
        },
        "preservation_attenuation_ratio_mean": {
            STRENGTH_ID_TO_TEXT[k]: _aggregate_metric(strength_rows[k], "preservation_attenuation_ratio")
            for k in strengths
        },
        "projector_pairwise_l2_mean": {
            "0_1": _aggregate_mean([row.get("projector_pairwise_0_1") for rows in strength_rows.values() for row in rows]),
            "1_2": _aggregate_mean([row.get("projector_pairwise_1_2") for rows in strength_rows.values() for row in rows]),
            "0_2": _aggregate_mean([row.get("projector_pairwise_0_2") for rows in strength_rows.values() for row in rows]),
        },
        "monotonic_hit_rate": float(np.mean([float(row["monotonic_hit"]) for row in pairwise_rows])),
        "raw_monotonic_hit_rate": float(np.mean([float(row["raw_monotonic_hit"]) for row in pairwise_rows])),
        "final_monotonic_hit_rate": float(np.mean([float(row["final_monotonic_hit"]) for row in pairwise_rows])),
        "attenuation_suspected_rate": float(np.mean([float(row["attenuation_suspected"]) for row in pairwise_rows])),
        "strong_minus_weak_edit_gain_mean": _aggregate_mean(
            [row["strong_minus_weak_edit_gain"] for row in pairwise_rows]
        ),
        "raw_strong_minus_weak_mean": _safe_diff(
            _aggregate_mean([row["strong_raw_edit_region_mean_abs_delta"] for row in pairwise_rows]),
            _aggregate_mean([row["weak_raw_edit_region_mean_abs_delta"] for row in pairwise_rows]),
        ),
        "final_strong_minus_weak_mean": _safe_diff(
            _aggregate_mean([row["strong_final_edit_region_mean_abs_delta"] for row in pairwise_rows]),
            _aggregate_mean([row["weak_final_edit_region_mean_abs_delta"] for row in pairwise_rows]),
        ),
        "bg_mae_strong_minus_weak": _safe_diff(
            _aggregate_mean([row["strong_bg_mae"] for row in pairwise_rows]),
            _aggregate_mean([row["weak_bg_mae"] for row in pairwise_rows]),
        ),
    }
    summary["raw_to_final_monotonic_drop"] = _safe_diff(
        summary["raw_monotonic_hit_rate"],
        summary["final_monotonic_hit_rate"],
    )
    summary["preservation_flattens_strength"] = bool(
        summary["raw_monotonic_hit_rate"] is not None
        and summary["final_monotonic_hit_rate"] is not None
        and summary["raw_monotonic_hit_rate"] > summary["final_monotonic_hit_rate"]
    )
    summary["strength_visible_in_final"] = bool(
        summary["final_monotonic_hit_rate"] > 0.0
        and summary["final_strong_minus_weak_mean"] is not None
        and summary["final_strong_minus_weak_mean"] > 0.0
    )
    summary["runtime_condition_interpretation"] = {
        "both": "numeric label and instruction text both active" if args.condition_mode == "both" else False,
        "label_only": "instruction text removed; numeric label active" if args.condition_mode == "label_only" else False,
        "text_only": "numeric label fixed to weak slot; text varies by requested strength" if args.condition_mode == "text_only" else False,
    }
    summary["output_path_diagnosis"] = {
        "raw_separable": bool(summary["raw_monotonic_hit_rate"] > 0.0),
        "final_separable": bool(summary["final_monotonic_hit_rate"] > 0.0),
        "attenuation_suspected": bool(summary["attenuation_suspected_rate"] > 0.0),
    }
    summary["projector_signal_present"] = bool(
        summary["projector_pairwise_l2_mean"]["0_2"] is not None
        and summary["projector_pairwise_l2_mean"]["0_2"] > 0.0
    )
    summary["modulation_or_preservation_priority"] = (
        "preservation" if summary["preservation_flattens_strength"] else "modulation_or_conditioning"
    )
    summary["raw_vs_final_summary"] = {
        "raw_strong_minus_weak_mean": summary["raw_strong_minus_weak_mean"],
        "final_strong_minus_weak_mean": summary["final_strong_minus_weak_mean"],
        "raw_to_final_monotonic_drop": summary["raw_to_final_monotonic_drop"],
    }
    summary["preservation_pass"] = False
    bg_drift = summary["bg_mae_strong_minus_weak"]
    summary["preservation_pass"] = bool(
        bg_drift is not None and float(bg_drift) <= float(args.bg_drift_threshold)
    )
    summary["audit_focus"] = {
        "projector": summary["projector_pairwise_l2_mean"],
        "raw_final": summary["raw_vs_final_summary"],
    }
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
        "raw_final_rows": raw_final_rows,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
