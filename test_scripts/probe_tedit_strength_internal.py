from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

import numpy as np
import torch

import sys

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
_TEDIT_ROOT = _ROOT / "TEdit-main"
if str(_TEDIT_ROOT) not in sys.path:
    sys.path.insert(0, str(_TEDIT_ROOT))

from tool.tedit_wrapper import TEditWrapper


DEFAULT_STRENGTH_CONTROLS = [
    {"strength_label": 0, "strength_scalar": 0.0, "strength_text": "weak"},
    {"strength_label": 1, "strength_scalar": 0.5, "strength_text": "medium"},
    {"strength_label": 2, "strength_scalar": 1.0, "strength_text": "strong"},
]


def _resolve_runtime_config_path(model_path: str, config_path: str) -> str:
    requested = Path(config_path).resolve()
    if requested.exists():
        return str(requested)
    model_file = Path(model_path).resolve()
    for candidate_name in ("resolved_runtime_config.json", "model_configs.yaml"):
        candidate = model_file.parent.parent / candidate_name
        if candidate.exists():
            return str(candidate)
    return str(requested)


def _load_wrapper_config(config_path: str) -> dict[str, Any]:
    config_file = Path(config_path).resolve()
    text = config_file.read_text(encoding="utf-8")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload = yaml.safe_load(text)
    if isinstance(payload, dict):
        resolved_model = payload.get("resolved_configs", {}).get("model")
        if isinstance(resolved_model, dict) and all(key in resolved_model for key in ("attrs", "side", "diffusion")):
            return resolved_model
    if isinstance(payload, dict) and all(key in payload for key in ("attrs", "side", "diffusion")):
        return payload
    raise ValueError(f"Unsupported TEdit wrapper config structure: {config_file}")


def _build_wrapper(model_path: str, config_path: str, device: str, output_path: str) -> TEditWrapper:
    runtime_config_path = _resolve_runtime_config_path(model_path, config_path)
    wrapper_config = _load_wrapper_config(runtime_config_path)
    wrapper = TEditWrapper(model_path=None, config_path=None, device=device)
    wrapper_config_path = Path(output_path).resolve().parent / "_wrapper_model_config.yaml"
    wrapper_config_path.write_text(yaml.safe_dump(wrapper_config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    wrapper.load_model(model_path=model_path, config_path=str(wrapper_config_path))
    return wrapper




def _to_jsonable(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def _aggregate_mean(values):
    filtered = [float(v) for v in values if v is not None]
    if not filtered:
        return None
    return float(np.mean(filtered))


def _aggregate_nested_numeric_dict(records, key_candidates):
    bucket = {}
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


def _aggregate_nested_metric_by_strength(records, key_candidates):
    bucket = {}
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


def _extract_scalar_metrics(diagnostics: dict[str, object]) -> dict[str, object]:
    projector = diagnostics.get("projector") or []
    modulation_records = list(diagnostics.get("modulation_base") or []) + list(diagnostics.get("modulation_weaver") or [])
    stage_records = [
        record for record in modulation_records
        if isinstance(record, dict) and isinstance(record.get("stage_name"), str)
    ]
    modulation_core_records = [
        record for record in modulation_records
        if isinstance(record, dict) and record.get("stage_name") is None
    ]

    stage_summary = {}
    for record in stage_records:
        stage_name = str(record.get("stage_name"))
        stage_summary[stage_name] = {
            "mean_by_scalar": record.get("stage_mean_by_scalar") if isinstance(record.get("stage_mean_by_scalar"), dict) else None,
            "pairwise_l2_by_scalar": record.get("stage_pairwise_l2_by_scalar") if isinstance(record.get("stage_pairwise_l2_by_scalar"), dict) else None,
            "stage_norm_mean": record.get("stage_norm_mean"),
            "stage_abs_mean": record.get("stage_abs_mean"),
            "stage_feature_std_mean": record.get("stage_feature_std_mean"),
        }

    modulation_by_scalar = {}
    for tensor_name in ["delta_gamma", "delta_beta", "gamma_orig", "beta_orig", "gamma_final", "beta_final", "strength_cond"]:
        mean_key = f"{tensor_name}_mean_by_scalar"
        pairwise_key = f"{tensor_name}_pairwise_l2_by_scalar"
        modulation_by_scalar[tensor_name] = {
            "mean_by_scalar": _aggregate_nested_metric_by_strength(modulation_core_records, [mean_key]),
            "pairwise_l2_by_scalar": _aggregate_nested_numeric_dict(modulation_core_records, [pairwise_key]),
            "norm_mean": _aggregate_mean([record.get(f"{tensor_name}_norm_mean") for record in modulation_core_records]),
            "abs_mean": _aggregate_mean([record.get(f"{tensor_name}_abs_mean") for record in modulation_core_records]),
        }

    final_output_gain_gate_mean_by_scalar = _aggregate_nested_metric_by_strength(
        modulation_core_records,
        ["final_output_gain_gate_mean_by_scalar"],
    )
    final_output_gain_gate_min_by_scalar = _aggregate_nested_metric_by_strength(
        modulation_core_records,
        ["final_output_gain_gate_min_by_scalar"],
    )
    final_output_gain_gate_max_by_scalar = _aggregate_nested_metric_by_strength(
        modulation_core_records,
        ["final_output_gain_gate_max_by_scalar"],
    )
    residual_content_abs_mean_by_scalar = _aggregate_nested_metric_by_strength(
        modulation_core_records,
        ["residual_content_abs_mean_by_scalar"],
    )
    skip_branch_abs_mean_by_scalar = _aggregate_nested_metric_by_strength(
        modulation_core_records,
        ["skip_branch_abs_mean_by_scalar"],
    )
    residual_carrier_restored_abs_mean_by_scalar = _aggregate_nested_metric_by_strength(
        modulation_core_records,
        ["residual_carrier_restored_abs_mean_by_scalar"],
    )
    residual_restored_to_skip_abs_ratio_by_scalar = _aggregate_nested_metric_by_strength(
        modulation_core_records,
        ["residual_restored_to_skip_abs_ratio_by_scalar"],
    )
    final_output_strength_scale_by_scalar = _aggregate_nested_metric_by_strength(
        modulation_core_records,
        ["final_output_strength_scale_by_scalar"],
    )
    final_output_strength_mapping_by_scalar = _aggregate_nested_metric_by_strength(
        modulation_core_records,
        ["final_output_strength_mapping_by_scalar"],
    )

    return {
        "projector_pairwise_l2": _aggregate_nested_numeric_dict(
            projector,
            ["projector_output_pairwise_l2_by_scalar", "projector_output_pairwise_l2"],
        ),
        "projector_strength_mean_norms": {
            str(k): v.get("norm") if isinstance(v, dict) else None
            for k, v in _aggregate_nested_metric_by_strength(
                projector,
                ["projector_output_mean_norm_by_scalar", "projector_output_mean_norm_by_strength"],
            ).items()
        },
        "modulation_delta_gamma_pairwise_l2": _aggregate_nested_numeric_dict(
            modulation_core_records,
            ["delta_gamma_pairwise_l2_by_scalar", "delta_gamma_pairwise_l2"],
        ),
        "modulation_delta_beta_pairwise_l2": _aggregate_nested_numeric_dict(
            modulation_core_records,
            ["delta_beta_pairwise_l2_by_scalar", "delta_beta_pairwise_l2"],
        ),
        "modulation_delta_gamma_abs_mean": _aggregate_mean([record.get("delta_gamma_abs_mean") for record in modulation_core_records]),
        "modulation_delta_beta_abs_mean": _aggregate_mean([record.get("delta_beta_abs_mean") for record in modulation_core_records]),
        "modulation_delta_gamma_over_base_mean": _aggregate_mean([record.get("delta_gamma_over_base_mean") for record in modulation_core_records]),
        "modulation_delta_beta_over_base_mean": _aggregate_mean([record.get("delta_beta_over_base_mean") for record in modulation_core_records]),
        "final_output_gain_gate_mean": _aggregate_mean([record.get("final_output_gain_gate_mean") for record in modulation_core_records]),
        "final_output_gain_gate_min": _aggregate_mean([record.get("final_output_gain_gate_min") for record in modulation_core_records]),
        "final_output_gain_gate_max": _aggregate_mean([record.get("final_output_gain_gate_max") for record in modulation_core_records]),
        "final_output_gain_gate_mean_by_scalar": {
            str(k): v.get("mean") if isinstance(v, dict) else None
            for k, v in final_output_gain_gate_mean_by_scalar.items()
        },
        "final_output_gain_gate_min_by_scalar": {
            str(k): v.get("mean") if isinstance(v, dict) else None
            for k, v in final_output_gain_gate_min_by_scalar.items()
        },
        "final_output_gain_gate_max_by_scalar": {
            str(k): v.get("mean") if isinstance(v, dict) else None
            for k, v in final_output_gain_gate_max_by_scalar.items()
        },
        "final_output_strength_scale_mean": _aggregate_mean([record.get("final_output_strength_scale_mean") for record in modulation_core_records]),
        "final_output_strength_scale_min": _aggregate_mean([record.get("final_output_strength_scale_min") for record in modulation_core_records]),
        "final_output_strength_scale_max": _aggregate_mean([record.get("final_output_strength_scale_max") for record in modulation_core_records]),
        "final_output_strength_scale_by_scalar": {
            str(k): v.get("mean") if isinstance(v, dict) else None
            for k, v in final_output_strength_scale_by_scalar.items()
        },
        "final_output_strength_mapping_mean": _aggregate_mean([record.get("final_output_strength_mapping_mean") for record in modulation_core_records]),
        "final_output_strength_mapping_min": _aggregate_mean([record.get("final_output_strength_mapping_min") for record in modulation_core_records]),
        "final_output_strength_mapping_max": _aggregate_mean([record.get("final_output_strength_mapping_max") for record in modulation_core_records]),
        "final_output_strength_mapping_by_scalar": {
            str(k): v.get("mean") if isinstance(v, dict) else None
            for k, v in final_output_strength_mapping_by_scalar.items()
        },
        "output_branch_carrier_enabled": _aggregate_mean([record.get("output_branch_carrier_enabled") for record in modulation_core_records]),
        "output_branch_carrier_skip_scale": _aggregate_mean([record.get("output_branch_carrier_skip_scale") for record in modulation_core_records]),
        "residual_content_abs_mean": _aggregate_mean([record.get("residual_content_abs_mean") for record in modulation_core_records]),
        "skip_branch_abs_mean": _aggregate_mean([record.get("skip_branch_abs_mean") for record in modulation_core_records]),
        "residual_carrier_source_abs_mean": _aggregate_mean([record.get("residual_carrier_source_abs_mean") for record in modulation_core_records]),
        "residual_carrier_restored_abs_mean": _aggregate_mean([record.get("residual_carrier_restored_abs_mean") for record in modulation_core_records]),
        "residual_content_to_skip_abs_ratio": _aggregate_mean([record.get("residual_content_to_skip_abs_ratio") for record in modulation_core_records]),
        "residual_restored_to_skip_abs_ratio": _aggregate_mean([record.get("residual_restored_to_skip_abs_ratio") for record in modulation_core_records]),
        "residual_content_abs_mean_by_scalar": {
            str(k): v.get("mean") if isinstance(v, dict) else None
            for k, v in residual_content_abs_mean_by_scalar.items()
        },
        "skip_branch_abs_mean_by_scalar": {
            str(k): v.get("mean") if isinstance(v, dict) else None
            for k, v in skip_branch_abs_mean_by_scalar.items()
        },
        "residual_carrier_restored_abs_mean_by_scalar": {
            str(k): v.get("mean") if isinstance(v, dict) else None
            for k, v in residual_carrier_restored_abs_mean_by_scalar.items()
        },
        "residual_restored_to_skip_abs_ratio_by_scalar": {
            str(k): v.get("mean") if isinstance(v, dict) else None
            for k, v in residual_restored_to_skip_abs_ratio_by_scalar.items()
        },
        "modulation_by_scalar": modulation_by_scalar,
        "stage_by_scalar": stage_summary,
    }


def _linf_diff(left: dict[str, object], right: dict[str, object], key: str) -> float | None:
    left_value = left.get(key)
    right_value = right.get(key)
    if left_value is None or right_value is None:
        return None
    return float(abs(float(left_value) - float(right_value)))


def _extract_stage_scalar_metric(stage_summary: dict[str, object], stage_name: str, scalar_key: str, metric_key: str) -> float | None:
    stage_payload = stage_summary.get(stage_name)
    if not isinstance(stage_payload, dict):
        return None
    mean_by_scalar = stage_payload.get("mean_by_scalar")
    if not isinstance(mean_by_scalar, dict):
        return None
    scalar_payload = mean_by_scalar.get(scalar_key)
    if not isinstance(scalar_payload, dict):
        return None
    value = scalar_payload.get(metric_key)
    if value is None:
        return None
    return float(value)


def _extract_stage_pairwise_metric(stage_summary: dict[str, object], stage_name: str, left_key: str, right_key: str, suffix: str = "") -> float | None:
    stage_payload = stage_summary.get(stage_name)
    if not isinstance(stage_payload, dict):
        return None
    pairwise = stage_payload.get("pairwise_l2_by_scalar")
    if not isinstance(pairwise, dict):
        return None
    metric_key = f"{left_key}_{right_key}{suffix}"
    value = pairwise.get(metric_key)
    if value is None:
        return None
    return float(value)


def _build_stage_transition_summary(stage_summary: dict[str, object], scalar_keys: list[str]) -> dict[str, object]:
    focus_stages = [
        "post_modulation",
        "post_mid_projection",
        "post_side_add",
        "mid_gate_logits",
        "mid_filter_logits",
        "mid_gate_activation",
        "mid_filter_activation",
        "mid_gated_output",
        "post_output_projection",
        "residual_content_branch",
        "skip_branch",
        "residual_carrier_source_branch",
        "residual_carrier_restored_branch",
        "residual_amplitude_branch",
        "residual_gain_gated_branch",
        "residual_merge",
        "skip_aggregate",
        "final_head_projection",
        "final_head_relu",
        "patch_decoder_concat",
        "final_multipatch_output",
        "final_strength_mapped_output",
        "final_strength_scaled_output",
    ]
    transitions = {}
    if len(scalar_keys) < 2:
        return transitions
    weakest_key = scalar_keys[0]
    strongest_key = scalar_keys[-1]
    for stage_name in focus_stages:
        weak_abs = _extract_stage_scalar_metric(stage_summary, stage_name, weakest_key, "mean_abs")
        strong_abs = _extract_stage_scalar_metric(stage_summary, stage_name, strongest_key, "mean_abs")
        weak_norm = _extract_stage_scalar_metric(stage_summary, stage_name, weakest_key, "norm")
        strong_norm = _extract_stage_scalar_metric(stage_summary, stage_name, strongest_key, "norm")
        pairwise_l2 = _extract_stage_pairwise_metric(stage_summary, stage_name, weakest_key, strongest_key)
        pairwise_mean_abs = _extract_stage_pairwise_metric(stage_summary, stage_name, weakest_key, strongest_key, "_mean_abs")
        if weak_abs is None and strong_abs is None and weak_norm is None and strong_norm is None and pairwise_l2 is None and pairwise_mean_abs is None:
            continue
        transitions[stage_name] = {
            "weak_abs": weak_abs,
            "strong_abs": strong_abs,
            "strong_minus_weak_abs": None if weak_abs is None or strong_abs is None else float(strong_abs - weak_abs),
            "weak_norm": weak_norm,
            "strong_norm": strong_norm,
            "strong_minus_weak_norm": None if weak_norm is None or strong_norm is None else float(strong_norm - weak_norm),
            "weak_strong_pairwise_l2": pairwise_l2,
            "weak_strong_pairwise_mean_abs": pairwise_mean_abs,
        }
    return transitions


def _region_mean_abs_delta(base: np.ndarray, edited: np.ndarray, mask: np.ndarray) -> float | None:
    edit_mask = np.asarray(mask, dtype=bool).reshape(-1)
    if not np.any(edit_mask):
        return None
    base = np.asarray(base, dtype=np.float32).reshape(-1)
    edited = np.asarray(edited, dtype=np.float32).reshape(-1)
    return float(np.mean(np.abs(edited[edit_mask] - base[edit_mask])))


def _region_stats(values: np.ndarray, mask: np.ndarray) -> dict[str, float | None]:
    region_mask = np.asarray(mask, dtype=bool).reshape(-1)
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    if not np.any(region_mask):
        return {
            "mean": None,
            "abs_mean": None,
            "min": None,
            "max": None,
            "floor_distance": None,
        }
    region = arr[region_mask]
    return {
        "mean": float(np.mean(region)),
        "abs_mean": float(np.mean(np.abs(region))),
        "min": float(np.min(region)),
        "max": float(np.max(region)),
        "floor_distance": float(np.mean(np.abs(region))),
    }


def _gap_sequence(values: list[float | None]) -> list[float | None]:
    gaps: list[float | None] = []
    for left, right in zip(values[:-1], values[1:]):
        if left is None or right is None:
            gaps.append(None)
        else:
            gaps.append(float(right - left))
    return gaps


def _safe_spearman(values_x: list[float], values_y: list[float]) -> float | None:
    if len(values_x) < 2 or len(values_x) != len(values_y):
        return None
    x = np.asarray(values_x, dtype=np.float64)
    y = np.asarray(values_y, dtype=np.float64)
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return None
    x_rank = np.argsort(np.argsort(x, kind="mergesort"), kind="mergesort").astype(np.float64)
    y_rank = np.argsort(np.argsort(y, kind="mergesort"), kind="mergesort").astype(np.float64)
    x_rank -= np.mean(x_rank)
    y_rank -= np.mean(y_rank)
    denom = np.sqrt(np.sum(x_rank**2) * np.sum(y_rank**2))
    if float(denom) <= 1.0e-12:
        return None
    return float(np.sum(x_rank * y_rank) / denom)


def _json_number_or_none(value: Any) -> float | None:
    return None if value is None else float(value)


def resolve_dataset_folder(dataset_folder: str, selector: str | None = None) -> Path:
    root = Path(dataset_folder)
    if selector:
        candidate = root / selector
        if (candidate / "meta.json").exists():
            return candidate
        if (root / "meta.json").exists() and root.name == selector:
            return root
        return candidate
    if (root / "meta.json").exists():
        return root
    return root


def _load_probe_sample(dataset_folder: Path, split: str, requested_idx: int) -> dict[str, object]:
    meta_path = dataset_folder / "meta.json"
    if meta_path.exists():
        from data.discrete_strength_family import DiscreteStrengthFamilyDataset

        dataset = DiscreteStrengthFamilyDataset(str(dataset_folder))
        split_ds = dataset.get_split(split, include_self=True)
        idx = int(requested_idx) if requested_idx >= 0 else 0
        family = split_ds[idx]
        sample = family["samples"][0]
        controls = []
        for row in sorted(family["samples"], key=lambda value: float(value["strength_scalar"])):
            controls.append(
                {
                    "strength_label": None if row.get("strength_label") is None else int(row["strength_label"]),
                    "strength_scalar": float(row["strength_scalar"]),
                    "instruction_text": str(row.get("instruction_text")) if row.get("instruction_text") is not None else None,
                    "target": np.asarray(row["tgt_x"], dtype=np.float32).squeeze(-1),
                    "strength_text": str(row.get("strength_text", "medium")),
                }
            )
        edit_mask = np.asarray(sample["mask_gt"], dtype=np.float32).squeeze(-1) > 0.5
        edit_indices = np.flatnonzero(edit_mask)
        region_start = int(edit_indices[0]) if edit_indices.size > 0 else None
        region_end = int(edit_indices[-1] + 1) if edit_indices.size > 0 else None
        base_ts = np.asarray(sample["src_x"], dtype=np.float32).squeeze(-1)
        return {
            "probe_idx": idx,
            "family_id": str(sample.get("family_id", idx)),
            "tool_name": str(sample.get("tool_name", "unknown")),
            "family_semantic_tag": str(sample.get("family_semantic_tag", "unknown")),
            "task_id": sample.get("task_id"),
            "base": base_ts,
            "src_attrs": np.asarray(sample["src_attrs"], dtype=np.int64),
            "tgt_attrs": np.asarray(sample["tgt_attrs"], dtype=np.int64),
            "instruction_text": str(sample.get("instruction_text")) if sample.get("instruction_text") is not None else None,
            "mask_gt": edit_mask,
            "region_start": region_start,
            "region_end": region_end,
            "region_len": None if region_start is None or region_end is None else int(region_end - region_start),
            "series_length": int(base_ts.shape[0]),
            "controls": controls,
        }

    ts = np.load(dataset_folder / f"{split}_ts.npy")
    attrs = np.load(dataset_folder / f"{split}_attrs_idx.npy")
    idx = int(requested_idx) if requested_idx >= 0 else 0
    if requested_idx < 0:
        for cand_idx in range(len(attrs)):
            if not np.array_equal(attrs[cand_idx, 0], attrs[cand_idx, 1]):
                idx = cand_idx
                break
    instruction_text = None
    try:
        from data.synthetic_finetune import SyntheticDataset

        dataset = SyntheticDataset(str(dataset_folder))
        sample = dataset.get_split(split, include_self=True)[idx]
        if sample.get("instruction_text") is not None:
            instruction_text = str(sample.get("instruction_text"))
    except Exception:
        instruction_text = None
    return {
        "probe_idx": idx,
        "base": ts[idx, 0].astype(np.float32),
        "src_attrs": attrs[idx, 0].astype(np.int64),
        "tgt_attrs": attrs[idx, 1].astype(np.int64),
        "instruction_text": instruction_text,
        "mask_gt": np.abs(ts[idx, 1].astype(np.float32) - ts[idx, 0].astype(np.float32)) > 0,
        "controls": [
            {
                **control,
                "instruction_text": instruction_text,
                "target": ts[idx, 1].astype(np.float32),
            }
            for control in DEFAULT_STRENGTH_CONTROLS
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe whether TEdit strength injection changes diffusion outputs.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--dataset-folder", required=True)
    parser.add_argument("--split", default="train", choices=["train", "valid", "test"])
    parser.add_argument("--sample-idx", "--probe-idx", dest="sample_idx", type=int, default=-1)
    parser.add_argument("--edit-steps", type=int, default=10)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--task-id", type=int, default=-1, help="use -1 to omit task_id and probe strength-only path")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--use-dataset-instruction", type=int, default=1)
    parser.add_argument("--condition-mode", default="both", choices=["both", "label_only", "text_only"], help="ablate whether strength comes from numeric scalar/label, text, or both")
    parser.add_argument("--enable-strength-diagnostics", type=int, default=1)
    parser.add_argument("--flip-beta-sign", type=int, default=0, choices=[0, 1])
    parser.add_argument("--selector", default="")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    dataset_root = resolve_dataset_folder(args.dataset_folder, args.selector.strip() or None)
    sample_payload = _load_probe_sample(dataset_root, args.split, args.sample_idx)
    idx = int(sample_payload["probe_idx"])

    base = np.asarray(sample_payload["base"], dtype=np.float32)
    src_attrs = np.asarray(sample_payload["src_attrs"], dtype=np.int64)
    tgt_attrs = np.asarray(sample_payload["tgt_attrs"], dtype=np.int64)
    edit_mask = np.asarray(sample_payload["mask_gt"], dtype=bool)
    controls = sorted(list(sample_payload.get("controls") or []), key=lambda row: float(row["strength_scalar"]))
    if not controls:
        raise ValueError("probe sample does not contain any strength controls")

    instruction_texts = []
    for control in controls:
        text = control.get("instruction_text")
        if not bool(args.use_dataset_instruction):
            text = sample_payload.get("instruction_text")
        instruction_texts.append(text)

    numeric_strength_scalars = [float(control["strength_scalar"]) for control in controls]
    numeric_strength_labels = [control.get("strength_label") for control in controls]
    weakest_scalar = numeric_strength_scalars[0]
    weakest_label = next((int(value) for value in numeric_strength_labels if value is not None), 0)

    if args.condition_mode == "label_only":
        run_instruction_texts = None
        run_strength_scalars = numeric_strength_scalars
        run_strength_labels = None if any(value is None for value in numeric_strength_labels) else [int(value) for value in numeric_strength_labels]
    elif args.condition_mode == "text_only":
        run_instruction_texts = instruction_texts
        run_strength_scalars = [weakest_scalar] * len(controls)
        run_strength_labels = [weakest_label] * len(controls)
    else:
        run_instruction_texts = instruction_texts
        run_strength_scalars = numeric_strength_scalars
        run_strength_labels = None if any(value is None for value in numeric_strength_labels) else [int(value) for value in numeric_strength_labels]

    wrapper = _build_wrapper(args.model_path, args.config_path, args.device, args.output)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    edited_batch, diagnostics = wrapper.edit_time_series(
        ts=np.repeat(base.reshape(1, -1), len(controls), axis=0),
        src_attrs=src_attrs,
        tgt_attrs=tgt_attrs,
        n_samples=1,
        sampler="ddim",
        edit_steps=args.edit_steps,
        strength_label=run_strength_labels,
        strength_scalar=run_strength_scalars,
        task_id=None if args.task_id < 0 else [int(args.task_id)] * len(controls),
        instruction_text=run_instruction_texts,
        edit_mask=edit_mask.astype(np.float32),
        return_diagnostics=True,
        enable_strength_diagnostics=bool(args.enable_strength_diagnostics),
        flip_beta_sign_inference=bool(args.flip_beta_sign),
    )
    model_diag = diagnostics["model"][0] if diagnostics.get("model") else {}
    raw_reverse_output = model_diag.get("raw_reverse_output")
    if raw_reverse_output is None:
        raw_batch = np.zeros_like(edited_batch, dtype=np.float32)
    else:
        if hasattr(raw_reverse_output, "detach"):
            raw_reverse_output = raw_reverse_output.detach().cpu().numpy()
        raw_batch = np.asarray(raw_reverse_output, dtype=np.float32).squeeze(1)

    rows = []
    outputs = []
    target_gain_seq: list[float | None] = []
    pred_gain_seq: list[float | None] = []
    pred_floor_distance_seq: list[float | None] = []
    target_floor_distance_seq: list[float | None] = []
    for control_idx, control in enumerate(controls):
        edited = np.asarray(edited_batch[control_idx], dtype=np.float32)
        raw_output = np.asarray(raw_batch[control_idx], dtype=np.float32)
        target = np.asarray(control["target"], dtype=np.float32)
        outputs.append(edited)
        source_stats = _region_stats(base, edit_mask)
        target_stats = _region_stats(target, edit_mask)
        pred_stats = _region_stats(edited, edit_mask)
        target_gain = _region_mean_abs_delta(base, target, edit_mask)
        pred_gain = _region_mean_abs_delta(base, edited, edit_mask)
        target_gain_seq.append(target_gain)
        pred_gain_seq.append(pred_gain)
        target_floor_distance_seq.append(target_stats["floor_distance"])
        pred_floor_distance_seq.append(pred_stats["floor_distance"])
        rows.append(
            {
                "strength_label": None if control.get("strength_label") is None else int(control["strength_label"]),
                "runtime_strength_label": None if run_strength_labels is None else int(run_strength_labels[control_idx]),
                "strength_scalar": float(control["strength_scalar"]),
                "runtime_strength_scalar": float(run_strength_scalars[control_idx]),
                "strength_text": str(control.get("strength_text", "medium")),
                "peak_abs_delta": float(np.max(np.abs(edited - base))),
                "mean_abs_delta": float(np.mean(np.abs(edited - base))),
                "sum_delta": float(np.sum(edited - base)),
                "raw_edit_region_mean_abs_delta": _region_mean_abs_delta(base, raw_output, edit_mask),
                "final_edit_region_mean_abs_delta": pred_gain,
                "blend_gap_edit_region_mean_abs": _region_mean_abs_delta(raw_output, edited, edit_mask),
                "target_edit_region_mean_abs_delta": target_gain,
                "gain_error": None if pred_gain is None or target_gain is None else float(pred_gain - target_gain),
                "source_edit_stats": source_stats,
                "target_edit_stats": target_stats,
                "pred_edit_stats": pred_stats,
                "target_floor_distance": target_stats["floor_distance"],
                "pred_floor_distance": pred_stats["floor_distance"],
            }
        )

    target_gap_seq = _gap_sequence(target_gain_seq)
    pred_gap_seq = _gap_sequence(pred_gain_seq)
    gain_error_seq = [
        None if pred is None or target is None else float(pred - target)
        for pred, target in zip(pred_gain_seq, target_gain_seq)
    ]
    gap_error_seq = [
        None if pred is None or target is None else float(pred - target)
        for pred, target in zip(pred_gap_seq, target_gap_seq)
    ]
    strength_scalar_seq = [float(row["strength_scalar"]) for row in rows]
    family_spearman = _safe_spearman(
        strength_scalar_seq,
        [float(value) for value in pred_gain_seq if value is not None],
    ) if all(value is not None for value in pred_gain_seq) else None

    family_profile = {
        "family_id": sample_payload.get("family_id"),
        "tool_name": sample_payload.get("tool_name"),
        "family_semantic_tag": sample_payload.get("family_semantic_tag"),
        "task_id": sample_payload.get("task_id"),
        "region_start": sample_payload.get("region_start"),
        "region_end": sample_payload.get("region_end"),
        "region_len": sample_payload.get("region_len"),
        "series_length": sample_payload.get("series_length"),
        "edit_region_fraction": float(np.mean(edit_mask.astype(np.float32))),
        "strength_scalar_seq": strength_scalar_seq,
        "target_gain_seq": [_json_number_or_none(value) for value in target_gain_seq],
        "pred_gain_seq": [_json_number_or_none(value) for value in pred_gain_seq],
        "pred_minus_target_seq": [_json_number_or_none(value) for value in gain_error_seq],
        "target_gap_seq": [_json_number_or_none(value) for value in target_gap_seq],
        "pred_gap_seq": [_json_number_or_none(value) for value in pred_gap_seq],
        "pred_gap_minus_target_gap_seq": [_json_number_or_none(value) for value in gap_error_seq],
        "source_edit_stats": _region_stats(base, edit_mask),
        "target_floor_distance_seq": [_json_number_or_none(value) for value in target_floor_distance_seq],
        "pred_floor_distance_seq": [_json_number_or_none(value) for value in pred_floor_distance_seq],
        "family_spearman": family_spearman,
        "family_monotonic_hit": bool(
            all(pred_gain_seq[idx] is not None and pred_gain_seq[idx + 1] is not None and pred_gain_seq[idx] < pred_gain_seq[idx + 1] for idx in range(len(pred_gain_seq) - 1))
        ),
        "worst_gain_error": None if not gain_error_seq else max((abs(value) for value in gain_error_seq if value is not None), default=None),
        "worst_gap_error": None if not gap_error_seq else max((abs(value) for value in gap_error_seq if value is not None), default=None),
    }

    sample_profile = {
        "family_id": sample_payload.get("family_id"),
        "probe_idx": int(idx),
        "tool_name": sample_payload.get("tool_name"),
        "family_semantic_tag": sample_payload.get("family_semantic_tag"),
        "task_id": sample_payload.get("task_id"),
        "region_start": sample_payload.get("region_start"),
        "region_end": sample_payload.get("region_end"),
        "region_len": sample_payload.get("region_len"),
        "series_length": sample_payload.get("series_length"),
        "edit_region_fraction": float(np.mean(edit_mask.astype(np.float32))),
        "source_edit_stats": _region_stats(base, edit_mask),
    }

    hard_case_diagnostics = {
        "sample_profile": sample_profile,
        "family_profile": family_profile,
    }

    scalar_diagnostics = _extract_scalar_metrics(diagnostics)
    scalar_keys = [f"{float(row['strength_scalar']):.4f}" for row in rows]
    stage_transition_summary = _build_stage_transition_summary(
        scalar_diagnostics.get("stage_by_scalar") or {},
        scalar_keys,
    )
    final_edit_delta_linf = {}
    for left_idx in range(len(rows)):
        for right_idx in range(left_idx + 1, len(rows)):
            final_edit_delta_linf[f"{scalar_keys[left_idx]}_{scalar_keys[right_idx]}"] = _linf_diff(
                rows[left_idx], rows[right_idx], "final_edit_region_mean_abs_delta"
            )
    raw_values = [row["raw_edit_region_mean_abs_delta"] for row in rows]
    final_values = [row["final_edit_region_mean_abs_delta"] for row in rows]
    raw_edit_monotonic = bool(
        all(raw_values[idx] is not None and raw_values[idx + 1] is not None and raw_values[idx] < raw_values[idx + 1] for idx in range(len(raw_values) - 1))
    )
    final_edit_monotonic = bool(
        all(final_values[idx] is not None and final_values[idx + 1] is not None and final_values[idx] < final_values[idx + 1] for idx in range(len(final_values) - 1))
    )
    summary = {
        "condition_mode": args.condition_mode,
        "flip_beta_sign": bool(args.flip_beta_sign),
        "raw_edit_region_mean_abs_delta": {
            scalar_keys[row_idx]: row["raw_edit_region_mean_abs_delta"]
            for row_idx, row in enumerate(rows)
        },
        "final_edit_region_mean_abs_delta": {
            scalar_keys[row_idx]: row["final_edit_region_mean_abs_delta"]
            for row_idx, row in enumerate(rows)
        },
        "blend_gap_edit_region_mean_abs": {
            scalar_keys[row_idx]: row["blend_gap_edit_region_mean_abs"]
            for row_idx, row in enumerate(rows)
        },
        "projector_pairwise_l2": scalar_diagnostics.get("projector_pairwise_l2"),
        "projector_strength_mean_norms": scalar_diagnostics.get("projector_strength_mean_norms"),
        "modulation_delta_gamma_pairwise_l2": scalar_diagnostics.get("modulation_delta_gamma_pairwise_l2"),
        "modulation_delta_beta_pairwise_l2": scalar_diagnostics.get("modulation_delta_beta_pairwise_l2"),
        "modulation_delta_gamma_abs_mean": scalar_diagnostics.get("modulation_delta_gamma_abs_mean"),
        "modulation_delta_beta_abs_mean": scalar_diagnostics.get("modulation_delta_beta_abs_mean"),
        "modulation_delta_gamma_over_base_mean": scalar_diagnostics.get("modulation_delta_gamma_over_base_mean"),
        "modulation_delta_beta_over_base_mean": scalar_diagnostics.get("modulation_delta_beta_over_base_mean"),
        "final_output_gain_gate_mean": scalar_diagnostics.get("final_output_gain_gate_mean"),
        "final_output_gain_gate_min": scalar_diagnostics.get("final_output_gain_gate_min"),
        "final_output_gain_gate_max": scalar_diagnostics.get("final_output_gain_gate_max"),
        "final_output_gain_gate_mean_by_scalar": scalar_diagnostics.get("final_output_gain_gate_mean_by_scalar"),
        "final_output_gain_gate_min_by_scalar": scalar_diagnostics.get("final_output_gain_gate_min_by_scalar"),
        "final_output_gain_gate_max_by_scalar": scalar_diagnostics.get("final_output_gain_gate_max_by_scalar"),
        "final_output_strength_scale_mean": scalar_diagnostics.get("final_output_strength_scale_mean"),
        "final_output_strength_scale_min": scalar_diagnostics.get("final_output_strength_scale_min"),
        "final_output_strength_scale_max": scalar_diagnostics.get("final_output_strength_scale_max"),
        "final_output_strength_scale_by_scalar": scalar_diagnostics.get("final_output_strength_scale_by_scalar"),
        "final_output_strength_mapping_mean": scalar_diagnostics.get("final_output_strength_mapping_mean"),
        "final_output_strength_mapping_min": scalar_diagnostics.get("final_output_strength_mapping_min"),
        "final_output_strength_mapping_max": scalar_diagnostics.get("final_output_strength_mapping_max"),
        "final_output_strength_mapping_by_scalar": scalar_diagnostics.get("final_output_strength_mapping_by_scalar"),
        "output_branch_carrier_enabled": scalar_diagnostics.get("output_branch_carrier_enabled"),
        "output_branch_carrier_skip_scale": scalar_diagnostics.get("output_branch_carrier_skip_scale"),
        "residual_content_abs_mean": scalar_diagnostics.get("residual_content_abs_mean"),
        "skip_branch_abs_mean": scalar_diagnostics.get("skip_branch_abs_mean"),
        "residual_carrier_source_abs_mean": scalar_diagnostics.get("residual_carrier_source_abs_mean"),
        "residual_carrier_restored_abs_mean": scalar_diagnostics.get("residual_carrier_restored_abs_mean"),
        "residual_content_to_skip_abs_ratio": scalar_diagnostics.get("residual_content_to_skip_abs_ratio"),
        "residual_restored_to_skip_abs_ratio": scalar_diagnostics.get("residual_restored_to_skip_abs_ratio"),
        "residual_content_abs_mean_by_scalar": scalar_diagnostics.get("residual_content_abs_mean_by_scalar"),
        "skip_branch_abs_mean_by_scalar": scalar_diagnostics.get("skip_branch_abs_mean_by_scalar"),
        "residual_carrier_restored_abs_mean_by_scalar": scalar_diagnostics.get("residual_carrier_restored_abs_mean_by_scalar"),
        "residual_restored_to_skip_abs_ratio_by_scalar": scalar_diagnostics.get("residual_restored_to_skip_abs_ratio_by_scalar"),
        "modulation_by_scalar": scalar_diagnostics.get("modulation_by_scalar"),
        "stage_by_scalar": scalar_diagnostics.get("stage_by_scalar"),
        "stage_transition_summary": stage_transition_summary,
        "final_edit_delta_linf": final_edit_delta_linf,
        "raw_edit_monotonic": raw_edit_monotonic,
        "final_edit_monotonic": final_edit_monotonic,
    }
    payload = {
        "dataset_folder": str(dataset_root),
        "split": args.split,
        "probe_idx": int(idx),
        "family_id": sample_payload.get("family_id"),
        "src_attrs": src_attrs.tolist(),
        "tgt_attrs": tgt_attrs.tolist(),
        "instruction_text": sample_payload.get("instruction_text"),
        "condition_mode": args.condition_mode,
        "flip_beta_sign": bool(args.flip_beta_sign),
        "seed": int(args.seed),
        "task_id": None if args.task_id < 0 else int(args.task_id),
        "rows": rows,
        "summary": summary,
        "hard_case_diagnostics": hard_case_diagnostics,
        "scalar_diagnostics": scalar_diagnostics,
        "output_pairwise_linf": {
            f"{scalar_keys[left_idx]}_{scalar_keys[right_idx]}": float(np.max(np.abs(outputs[left_idx] - outputs[right_idx])))
            for left_idx in range(len(outputs))
            for right_idx in range(left_idx + 1, len(outputs))
        },
        "diagnostics": _to_jsonable(diagnostics),
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
