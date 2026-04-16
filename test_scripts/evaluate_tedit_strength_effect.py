from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REQUIRED_STRENGTH_SCALARS = [0.0, 0.5, 1.0]

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



def _resolve_runtime_config_path(model_path: str, config_path: str) -> str:
    model_file = Path(model_path).resolve()
    requested = Path(config_path).resolve()
    candidate = model_file.parent.parent / "model_configs.yaml"
    if candidate.exists():
        return str(candidate)
    return str(requested)


DEFAULT_STRENGTH_CONTROLS = [
    {"strength_label": 0, "strength_scalar": 0.0, "strength_text": "weak"},
    {"strength_label": 1, "strength_scalar": 0.5, "strength_text": "medium"},
    {"strength_label": 2, "strength_scalar": 1.0, "strength_text": "strong"},
]


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
            per_strength = sorted(family["samples"], key=lambda row: float(row["strength_scalar"]))
            base = np.asarray(per_strength[0]["src_x"], dtype=np.float32).squeeze(-1)
            src_attrs = np.asarray(per_strength[0]["src_attrs"], dtype=np.int64)
            tgt_attrs = np.asarray(per_strength[0]["tgt_attrs"], dtype=np.int64)
            edit_mask = np.asarray(per_strength[0]["mask_gt"], dtype=np.float32).squeeze(-1) > 0.5
            controls = []
            for row in per_strength:
                controls.append(
                    {
                        "strength_label": None if row.get("strength_label") is None else int(row["strength_label"]),
                        "strength_scalar": float(row["strength_scalar"]),
                        "strength_text": str(row.get("strength_text", STRENGTH_ID_TO_TEXT.get(int(row.get("strength_label", 1)), "medium"))),
                        "instruction_text": str(row.get("instruction_text")) if row.get("instruction_text") is not None else None,
                        "target": np.asarray(row["tgt_x"], dtype=np.float32).squeeze(-1),
                    }
                )
            records.append(
                {
                    "sample_idx": int(family_idx),
                    "base": base,
                    "src_attrs": src_attrs,
                    "tgt_attrs": tgt_attrs,
                    "edit_mask": edit_mask,
                    "controls": controls,
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
        controls = []
        for control in DEFAULT_STRENGTH_CONTROLS:
            controls.append(
                {
                    **control,
                    "instruction_text": None,
                    "target": split_ds.ts[idx, 1].astype(np.float32),
                }
            )
        records.append(
            {
                "sample_idx": int(idx),
                "base": split_ds.ts[idx, 0].astype(np.float32),
                "src_attrs": split_ds.attrs[idx, 0].astype(np.int64),
                "tgt_attrs": split_ds.attrs[idx, 1].astype(np.int64),
                "edit_mask": np.abs(split_ds.ts[idx, 1].astype(np.float32) - split_ds.ts[idx, 0].astype(np.float32)) > 0,
                "controls": controls,
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


def _aggregate_nested_numeric_dict(records: list[dict[str, Any]], key_candidates: list[str]) -> dict[str, float]:
    bucket: dict[str, list[float]] = {}
    for record in records:
        nested = None
        for key in key_candidates:
            value = record.get(key)
            if isinstance(value, dict):
                nested = value
                break
        if not isinstance(nested, dict):
            continue
        for sub_key, sub_value in nested.items():
            if isinstance(sub_value, (int, float)):
                bucket.setdefault(str(sub_key), []).append(float(sub_value))
    return {
        key: float(np.mean(values))
        for key, values in bucket.items() if values
    }


def _aggregate_nested_metric_by_strength(records: list[dict[str, Any]], key_candidates: list[str]) -> dict[str, dict[str, float]]:
    bucket: dict[str, dict[str, list[float]]] = {}
    for record in records:
        nested = None
        for key in key_candidates:
            value = record.get(key)
            if isinstance(value, dict):
                nested = value
                break
        if not isinstance(nested, dict):
            continue
        for strength_key, payload in nested.items():
            if not isinstance(payload, dict):
                continue
            for metric_name, metric_value in payload.items():
                if isinstance(metric_value, (int, float)):
                    bucket.setdefault(str(strength_key), {}).setdefault(str(metric_name), []).append(float(metric_value))
    return {
        strength_key: {
            metric_name: float(np.mean(values))
            for metric_name, values in metric_dict.items() if values
        }
        for strength_key, metric_dict in bucket.items()
    }


def _average_tied_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < len(sorted_values):
        end = start + 1
        while end < len(sorted_values) and sorted_values[end] == sorted_values[start]:
            end += 1
        avg_rank = 0.5 * (start + end - 1)
        ranks[order[start:end]] = avg_rank
        start = end
    return ranks


def _spearman_rho(x_values: list[float], y_values: list[float]) -> float | None:
    if len(x_values) < 2 or len(y_values) < 2 or len(x_values) != len(y_values):
        return None
    x = np.asarray(x_values, dtype=np.float64)
    y = np.asarray(y_values, dtype=np.float64)
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return None
    x_rank = _average_tied_ranks(x)
    y_rank = _average_tied_ranks(y)
    x_rank -= x_rank.mean()
    y_rank -= y_rank.mean()
    denom = float(np.sqrt(np.sum(x_rank * x_rank) * np.sum(y_rank * y_rank)))
    if denom <= 1.0e-12:
        return None
    return float(np.sum(x_rank * y_rank) / denom)


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

    runtime_config_path = _resolve_runtime_config_path(args.model_path, args.config_path)
    eval_records, attr_list = _load_eval_records(args.dataset_folder, args.split, args.max_samples)
    if not eval_records:
        raise ValueError("No evaluation records were found for the requested split.")

    wrapper = TEditWrapper(model_path=args.model_path, config_path=runtime_config_path, device=args.device)

    strength_rows: dict[str, list[dict[str, Any]]] = {}
    pairwise_rows: list[dict[str, Any]] = []
    raw_final_rows: list[dict[str, Any]] = []

    for eval_idx, record in enumerate(eval_records):
        sample_idx = int(record["sample_idx"])
        base = np.asarray(record["base"], dtype=np.float32)
        src_attrs = np.asarray(record["src_attrs"], dtype=np.int64)
        tgt_attrs = np.asarray(record["tgt_attrs"], dtype=np.int64)
        edit_mask = np.asarray(record["edit_mask"], dtype=bool)
        controls = list(record.get("controls") or [])
        if not np.any(edit_mask):
            strongest_target = np.asarray(controls[-1]["target"], dtype=np.float32)
            edit_mask = np.abs(strongest_target - base) > float(args.mask_threshold)

        controls = sorted(controls, key=lambda row: float(row["strength_scalar"]))
        if not controls:
            raise ValueError(f"Evaluation record {sample_idx} does not contain controls.")

        batch_size = len(controls)
        numeric_strength_scalars = [float(control["strength_scalar"]) for control in controls]
        numeric_strength_labels = [control.get("strength_label") for control in controls]
        weakest_scalar = float(numeric_strength_scalars[0])
        weakest_label = next((int(value) for value in numeric_strength_labels if value is not None), 0)
        instruction_texts = []
        for control in controls:
            label_for_prompt = int(control["strength_label"]) if control.get("strength_label") is not None else int(round(float(control["strength_scalar"]) * 2))
            instruction_texts.append(
                control.get("instruction_text") or _build_instruction_text(
                    src_attrs=src_attrs,
                    tgt_attrs=tgt_attrs,
                    attr_list=attr_list,
                    strength_label=label_for_prompt,
                )
            )
        if args.condition_mode == "label_only":
            instruction_texts = None
            run_strength_scalars = numeric_strength_scalars
            run_strength_labels = None if any(value is None for value in numeric_strength_labels) else [int(value) for value in numeric_strength_labels]
        elif args.condition_mode == "text_only":
            run_strength_scalars = [weakest_scalar] * batch_size
            run_strength_labels = [weakest_label] * batch_size
        else:
            run_strength_scalars = numeric_strength_scalars
            run_strength_labels = None if any(value is None for value in numeric_strength_labels) else [int(value) for value in numeric_strength_labels]

        torch.manual_seed(args.seed + eval_idx)
        np.random.seed(args.seed + eval_idx)
        try:
            edited_batch, diagnostics = wrapper.edit_time_series(
                ts=np.repeat(base.reshape(1, -1), batch_size, axis=0),
                src_attrs=src_attrs,
                tgt_attrs=tgt_attrs,
                n_samples=1,
                sampler="ddim",
                edit_steps=args.edit_steps,
                strength_label=run_strength_labels,
                strength_scalar=run_strength_scalars,
                task_id=None if args.task_id < 0 else [int(args.task_id)] * batch_size,
                instruction_text=instruction_texts,
                return_diagnostics=True,
                enable_strength_diagnostics=bool(args.enable_strength_diagnostics),
            )
        except Exception as exc:
            raise RuntimeError(
                f"Strength eval failed at sample_idx={sample_idx}, eval_idx={eval_idx}, condition_mode={args.condition_mode}, model_path={args.model_path}, config_path={runtime_config_path}"
            ) from exc
        model_diag = diagnostics["model"][0] if diagnostics.get("model") else {}
        raw_batch = np.asarray(model_diag.get("raw_reverse_output"), dtype=np.float32).squeeze(1)
        projector_pairwise = _aggregate_nested_numeric_dict(
            diagnostics.get("projector") or [],
            ["projector_output_pairwise_l2_by_scalar", "projector_output_pairwise_l2"],
        )
        projector_mean_norm_by_scalar = _aggregate_nested_metric_by_strength(
            diagnostics.get("projector") or [],
            ["projector_output_mean_norm_by_scalar", "projector_output_mean_norm_by_strength"],
        )
        modulation_gamma_pairwise = _aggregate_nested_numeric_dict(
            (diagnostics.get("modulation_base") or []) + (diagnostics.get("modulation_weaver") or []),
            ["delta_gamma_pairwise_l2_by_scalar", "delta_gamma_pairwise_l2"],
        )
        modulation_beta_pairwise = _aggregate_nested_numeric_dict(
            (diagnostics.get("modulation_base") or []) + (diagnostics.get("modulation_weaver") or []),
            ["delta_beta_pairwise_l2_by_scalar", "delta_beta_pairwise_l2"],
        )

        per_strength: dict[str, dict[str, Any]] = {}
        for control_idx, control in enumerate(controls):
            scalar_key = f"{float(control['strength_scalar']):.4f}"
            strength_rows.setdefault(scalar_key, [])
            edited = np.asarray(edited_batch[control_idx], dtype=np.float32)
            raw_output = np.asarray(raw_batch[control_idx], dtype=np.float32)
            target = np.asarray(control["target"], dtype=np.float32)

            metrics = _compute_region_metrics(base=base, target=target, edited=edited, mask=edit_mask)
            raw_metrics = _compute_region_metrics(base=base, target=target, edited=raw_output, mask=edit_mask)
            row = {
                "sample_idx": int(sample_idx),
                "strength_label": None if control.get("strength_label") is None else int(control["strength_label"]),
                "runtime_strength_label": None if run_strength_labels is None else int(run_strength_labels[control_idx]),
                "strength_scalar": float(control["strength_scalar"]),
                "runtime_strength_scalar": float(run_strength_scalars[control_idx]),
                "strength_text": str(control["strength_text"]),
                "instruction_text": None if instruction_texts is None else instruction_texts[control_idx],
                "edit_region_fraction": float(np.mean(edit_mask.astype(np.float32))),
                "edit_gain": metrics["edit_gain"],
                "bg_mae": metrics["bg_mae"],
                "target_mae_edit_region": metrics["target_mae_edit_region"],
                "target_edit_gain": _compute_region_metrics(base=base, target=target, edited=target, mask=edit_mask)["edit_gain"],
                "gain_gap_abs": None if metrics["edit_gain"] is None else float(abs(metrics["edit_gain"] - _compute_region_metrics(base=base, target=target, edited=target, mask=edit_mask)["edit_gain"])),
                "raw_edit_region_mean_abs_delta": raw_metrics["edit_gain"],
                "final_edit_region_mean_abs_delta": metrics["edit_gain"],
                "raw_background_mean_abs_delta": raw_metrics["bg_mae"],
                "final_background_mean_abs_delta": metrics["bg_mae"],
                "blend_gap_edit_region_mean_abs": None if metrics["edit_gain"] is None or raw_metrics["edit_gain"] is None else float(abs(metrics["edit_gain"] - raw_metrics["edit_gain"])),
                "blend_gap_background_mean_abs": None if metrics["bg_mae"] is None or raw_metrics["bg_mae"] is None else float(abs(metrics["bg_mae"] - raw_metrics["bg_mae"])),
                "projector_pairwise_l2": projector_pairwise,
                "projector_mean_norm_by_scalar": projector_mean_norm_by_scalar,
                "modulation_delta_gamma_pairwise_l2": modulation_gamma_pairwise,
                "modulation_delta_beta_pairwise_l2": modulation_beta_pairwise,
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
            row["raw_monotonic_local"] = None
            row["final_monotonic_local"] = None
            row["attenuation_suspected"] = None
            raw_final_rows.append(
                {
                    "sample_idx": int(sample_idx),
                    "strength_scalar": float(control["strength_scalar"]),
                    "strength_text": str(control["strength_text"]),
                    "raw_edit_region_mean_abs_delta": row["raw_edit_region_mean_abs_delta"],
                    "final_edit_region_mean_abs_delta": row["final_edit_region_mean_abs_delta"],
                    "raw_background_mean_abs_delta": row["raw_background_mean_abs_delta"],
                    "final_background_mean_abs_delta": row["final_background_mean_abs_delta"],
                    "blend_gap_edit_region_mean_abs": row["blend_gap_edit_region_mean_abs"],
                    "blend_gap_background_mean_abs": row["blend_gap_background_mean_abs"],
                }
            )
            strength_rows[scalar_key].append(row)
            per_strength[scalar_key] = row

        ordered_scalars = [f"{float(control['strength_scalar']):.4f}" for control in controls]
        ordered_numeric_scalars = [float(control["strength_scalar"]) for control in controls]
        edit_gains = [per_strength[key]["edit_gain"] for key in ordered_scalars]
        raw_gains = [per_strength[key]["raw_edit_region_mean_abs_delta"] for key in ordered_scalars]
        final_gains = [per_strength[key]["final_edit_region_mean_abs_delta"] for key in ordered_scalars]
        monotonic = bool(all(edit_gains[idx] is not None and edit_gains[idx + 1] is not None and edit_gains[idx] < edit_gains[idx + 1] for idx in range(len(edit_gains) - 1)))
        raw_monotonic = bool(all(raw_gains[idx] is not None and raw_gains[idx + 1] is not None and raw_gains[idx] < raw_gains[idx + 1] for idx in range(len(raw_gains) - 1)))
        final_monotonic = bool(all(final_gains[idx] is not None and final_gains[idx + 1] is not None and final_gains[idx] < final_gains[idx + 1] for idx in range(len(final_gains) - 1)))
        attenuation_suspected = bool(raw_monotonic and not final_monotonic)
        for key in ordered_scalars:
            per_strength[key]["raw_monotonic_local"] = raw_monotonic
            per_strength[key]["final_monotonic_local"] = final_monotonic
            per_strength[key]["attenuation_suspected"] = attenuation_suspected
        family_spearman = _spearman_rho(
            ordered_numeric_scalars,
            [float(value) for value in edit_gains if value is not None],
        ) if all(value is not None for value in edit_gains) else None
        pairwise_rows.append(
            {
                "sample_idx": int(sample_idx),
                "edit_region_fraction": float(np.mean(edit_mask.astype(np.float32))),
                "min_strength_scalar": float(ordered_numeric_scalars[0]),
                "max_strength_scalar": float(ordered_numeric_scalars[-1]),
                "weak_edit_gain": edit_gains[0],
                "medium_edit_gain": edit_gains[len(edit_gains) // 2],
                "strong_edit_gain": edit_gains[-1],
                "weak_bg_mae": per_strength[ordered_scalars[0]]["bg_mae"],
                "medium_bg_mae": per_strength[ordered_scalars[len(ordered_scalars) // 2]]["bg_mae"],
                "strong_bg_mae": per_strength[ordered_scalars[-1]]["bg_mae"],
                "weak_raw_edit_region_mean_abs_delta": raw_gains[0],
                "medium_raw_edit_region_mean_abs_delta": raw_gains[len(raw_gains) // 2],
                "strong_raw_edit_region_mean_abs_delta": raw_gains[-1],
                "weak_final_edit_region_mean_abs_delta": final_gains[0],
                "medium_final_edit_region_mean_abs_delta": final_gains[len(final_gains) // 2],
                "strong_final_edit_region_mean_abs_delta": final_gains[-1],
                "strong_minus_weak_edit_gain": None if edit_gains[0] is None or edit_gains[-1] is None else float(edit_gains[-1] - edit_gains[0]),
                "gain_range": None if edit_gains[0] is None or edit_gains[-1] is None else float(edit_gains[-1] - edit_gains[0]),
                "family_spearman_rho_strength_gain": family_spearman,
                "gain_calibration_mae": _aggregate_mean([row["gain_gap_abs"] for row in per_strength.values()]),
                "raw_monotonic_hit": raw_monotonic,
                "final_monotonic_hit": final_monotonic,
                "attenuation_suspected": attenuation_suspected,
                "monotonic_hit": monotonic,
            }
        )

    scalar_keys = sorted(strength_rows.keys(), key=float)
    summary = {
        "condition_mode": args.condition_mode,
        "n_samples": len(eval_records),
        "strength_scalars": scalar_keys,
        "edit_gain_mean": {
            key: _aggregate_metric(strength_rows[key], "edit_gain")
            for key in scalar_keys
        },
        "bg_mae_mean": {
            key: _aggregate_metric(strength_rows[key], "bg_mae")
            for key in scalar_keys
        },
        "target_mae_edit_region_mean": {
            key: _aggregate_metric(strength_rows[key], "target_mae_edit_region")
            for key in scalar_keys
        },
        "target_edit_gain_mean": {
            key: _aggregate_metric(strength_rows[key], "target_edit_gain")
            for key in scalar_keys
        },
        "raw_edit_region_mean_abs_delta": {
            key: _aggregate_metric(strength_rows[key], "raw_edit_region_mean_abs_delta")
            for key in scalar_keys
        },
        "final_edit_region_mean_abs_delta": {
            key: _aggregate_metric(strength_rows[key], "final_edit_region_mean_abs_delta")
            for key in scalar_keys
        },
        "blend_gap_edit_region_mean_abs": {
            key: _aggregate_metric(strength_rows[key], "blend_gap_edit_region_mean_abs")
            for key in scalar_keys
        },
        "raw_background_mean_abs_delta": {
            key: _aggregate_metric(strength_rows[key], "raw_background_mean_abs_delta")
            for key in scalar_keys
        },
        "final_background_mean_abs_delta": {
            key: _aggregate_metric(strength_rows[key], "final_background_mean_abs_delta")
            for key in scalar_keys
        },
        "blend_gap_background_mean_abs": {
            key: _aggregate_metric(strength_rows[key], "blend_gap_background_mean_abs")
            for key in scalar_keys
        },
        "raw_final_gap_edit_region_mean": {
            key: _aggregate_metric(strength_rows[key], "raw_final_gap_edit_region")
            for key in scalar_keys
        },
        "preservation_attenuation_ratio_mean": {
            key: _aggregate_metric(strength_rows[key], "preservation_attenuation_ratio")
            for key in scalar_keys
        },
        "projector_pairwise_l2_mean": _aggregate_nested_numeric_dict(
            [row for rows in strength_rows.values() for row in rows],
            ["projector_pairwise_l2"],
        ),
        "projector_mean_norm_by_scalar": _aggregate_nested_metric_by_strength(
            [row for rows in strength_rows.values() for row in rows],
            ["projector_mean_norm_by_scalar"],
        ),
        "modulation_delta_gamma_pairwise_l2_mean": _aggregate_nested_numeric_dict(
            [row for rows in strength_rows.values() for row in rows],
            ["modulation_delta_gamma_pairwise_l2"],
        ),
        "modulation_delta_beta_pairwise_l2_mean": _aggregate_nested_numeric_dict(
            [row for rows in strength_rows.values() for row in rows],
            ["modulation_delta_beta_pairwise_l2"],
        ),
        "monotonic_hit_rate": float(np.mean([float(row["monotonic_hit"]) for row in pairwise_rows])),
        "raw_monotonic_hit_rate": float(np.mean([float(row["raw_monotonic_hit"]) for row in pairwise_rows])),
        "final_monotonic_hit_rate": float(np.mean([float(row["final_monotonic_hit"]) for row in pairwise_rows])),
        "attenuation_suspected_rate": float(np.mean([float(row["attenuation_suspected"]) for row in pairwise_rows])),
        "strong_minus_weak_edit_gain_mean": _aggregate_mean(
            [row["strong_minus_weak_edit_gain"] for row in pairwise_rows]
        ),
        "family_spearman_rho_strength_gain_mean": _aggregate_mean(
            [row["family_spearman_rho_strength_gain"] for row in pairwise_rows]
        ),
        "gain_range_mean": _aggregate_mean([row["gain_range"] for row in pairwise_rows]),
        "gain_calibration_mae_mean": _aggregate_mean([row["gain_calibration_mae"] for row in pairwise_rows]),
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
        "both": "numeric scalar control and instruction text both active" if args.condition_mode == "both" else False,
        "label_only": "instruction text removed; numeric scalar control active" if args.condition_mode == "label_only" else False,
        "text_only": "numeric control fixed to weakest anchor; text varies by requested strength" if args.condition_mode == "text_only" else False,
    }
    summary["output_path_diagnosis"] = {
        "raw_separable": bool(summary["raw_monotonic_hit_rate"] > 0.0),
        "final_separable": bool(summary["final_monotonic_hit_rate"] > 0.0),
        "attenuation_suspected": bool(summary["attenuation_suspected_rate"] > 0.0),
    }
    summary["projector_signal_present"] = bool(
        len(summary["projector_pairwise_l2_mean"]) > 0
        and max(float(value) for value in summary["projector_pairwise_l2_mean"].values()) > 0.0
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
    required_strength_keys = {f"{value:.4f}" for value in REQUIRED_STRENGTH_SCALARS}
    if len(pairwise_rows) != len(eval_records):
        raise RuntimeError(f"Expected {len(eval_records)} per-sample rows, got {len(pairwise_rows)}")
    if summary["n_samples"] != len(eval_records):
        raise RuntimeError(f"Summary n_samples={summary['n_samples']} expected {len(eval_records)}")
    if not required_strength_keys.issubset(set(strength_rows.keys())):
        raise RuntimeError(f"Missing strength rows for {sorted(required_strength_keys - set(strength_rows.keys()))}")
    for key in required_strength_keys:
        if len(strength_rows[key]) != len(eval_records):
            raise RuntimeError(f"Strength row count mismatch for {key}: expected {len(eval_records)} got {len(strength_rows[key])}")
    for metric_key in ("monotonic_hit_rate", "raw_monotonic_hit_rate", "final_monotonic_hit_rate"):
        if summary.get(metric_key) is None:
            raise RuntimeError(f"Missing summary metric {metric_key}")

    payload = {
        "status": {
            "ok": True,
            "stage": "evaluate_tedit_strength_effect",
            "condition_mode": args.condition_mode,
        },
        "config": {
            "model_path": args.model_path,
            "config_path": runtime_config_path,
            "requested_config_path": args.config_path,
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
