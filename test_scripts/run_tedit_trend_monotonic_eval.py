from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import yaml

import sys

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tool.tedit_wrapper import TEditWrapper


DEFAULT_STRENGTH_TO_SCALAR = {
    "weak": 0.0,
    "medium": 0.5,
    "strong": 1.0,
}
LEGACY_0_1_2_STRENGTH_TO_SCALAR = {
    "weak": 0.0,
    "medium": 1.0,
    "strong": 2.0,
}
DEFAULT_SWEEP = [0.0, 0.25, 0.5, 0.75, 1.0]
LEGACY_SWEEP = [0.0, 0.5, 1.0, 1.5, 2.0]
FLOAT_TOL = 1.0e-6


def _compute_metrics(source_ts: np.ndarray, target_ts: np.ndarray, edited_ts: np.ndarray, mask_gt: np.ndarray) -> Dict[str, float | None]:
    edit_mask = mask_gt.astype(bool)
    bg_mask = ~edit_mask
    metrics: Dict[str, float | None] = {
        "edit_gain": None,
        "bg_mae": None,
        "background_leak_max": None,
        "target_mae_edit_region": None,
    }
    if np.any(edit_mask):
        metrics["edit_gain"] = float(np.mean(np.abs(edited_ts[edit_mask] - source_ts[edit_mask])))
        metrics["target_mae_edit_region"] = float(np.mean(np.abs(edited_ts[edit_mask] - target_ts[edit_mask])))
    if np.any(bg_mask):
        bg_abs = np.abs(edited_ts[bg_mask] - source_ts[bg_mask])
        metrics["bg_mae"] = float(np.mean(bg_abs))
        metrics["background_leak_max"] = float(np.max(bg_abs))
    return metrics


def _aggregate(values: List[float | None]) -> float | None:
    filtered = [float(v) for v in values if v is not None]
    if not filtered:
        return None
    return float(np.mean(filtered))


def _safe_diff(high: float | None, low: float | None) -> float | None:
    if high is None or low is None:
        return None
    return float(high - low)


def _collapse_indicator(gap_a: float | None, gap_b: float | None, tol: float = FLOAT_TOL) -> float | None:
    valid = [float(gap) for gap in (gap_a, gap_b) if gap is not None]
    if not valid:
        return None
    return float(min(valid) <= tol)


def _gap_balance_ratio(gap_a: float | None, gap_b: float | None, tol: float = FLOAT_TOL) -> float | None:
    if gap_a is None or gap_b is None:
        return None
    denom = max(abs(float(gap_a)), abs(float(gap_b)), tol)
    return float(min(abs(float(gap_a)), abs(float(gap_b))) / denom)


def _to_jsonable(value: Any) -> Any:
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


def _trend_attrs(direction: str) -> tuple[np.ndarray, np.ndarray]:
    src_attrs = np.array([0, 0, 0], dtype=np.int64)
    if direction == "down":
        tgt_attrs = np.array([1, 0, 1], dtype=np.int64)
    else:
        tgt_attrs = np.array([1, 1, 1], dtype=np.int64)
    return src_attrs, tgt_attrs


def _family_direction(family: Dict[str, Any]) -> str:
    samples = family.get("samples", [])
    for sample in samples:
        direction = str(sample.get("direction", ""))
        if direction in {"up", "down"}:
            return direction
        if direction in {"upward", "downward"}:
            return "up" if direction == "upward" else "down"
    return "up"


def _resolve_strength_scalar(sample: Dict[str, Any]) -> float:
    scalar = sample.get("strength_scalar")
    if scalar is not None:
        return float(scalar)
    strength_text = str(sample.get("strength_text", ""))
    if strength_text in DEFAULT_STRENGTH_TO_SCALAR:
        return float(DEFAULT_STRENGTH_TO_SCALAR[strength_text])
    raise ValueError(f"Sample missing strength_scalar and unknown strength_text={strength_text}")


def _resolve_scalar_scheme(benchmark: Dict[str, Any]) -> str:
    scalar_scheme = str(benchmark.get("scalar_scheme", "")).strip()
    if scalar_scheme:
        return scalar_scheme
    strength_axis = benchmark.get("strength_axis")
    if isinstance(strength_axis, dict):
        anchor_mapping = strength_axis.get("anchor_mapping")
        if isinstance(anchor_mapping, dict):
            medium = anchor_mapping.get("medium")
            strong = anchor_mapping.get("strong")
            if medium == 1.0 and strong == 2.0:
                return "legacy_0_1_2"
    return "default_0_0p5_1"


def _required_anchor_scalars_for_scheme(scalar_scheme: str) -> set[float]:
    if scalar_scheme == "legacy_0_1_2":
        return {0.0, 1.0, 2.0}
    return {0.0, 0.5, 1.0}


def _default_sweep_for_scheme(scalar_scheme: str) -> List[float]:
    if scalar_scheme == "legacy_0_1_2":
        return list(LEGACY_SWEEP)
    return list(DEFAULT_SWEEP)


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


def _spearman_rho(x: List[float], y: List[float]) -> float | None:
    if len(x) < 2 or len(y) < 2 or len(x) != len(y):
        return None
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    if np.allclose(x_arr, x_arr[0]) or np.allclose(y_arr, y_arr[0]):
        return None
    x_rank = _average_tied_ranks(x_arr)
    y_rank = _average_tied_ranks(y_arr)
    x_rank -= x_rank.mean()
    y_rank -= y_rank.mean()
    denom = float(np.sqrt(np.sum(x_rank * x_rank) * np.sum(y_rank * y_rank)))
    if denom <= 1.0e-12:
        return None
    return float(np.sum(x_rank * y_rank) / denom)


def _parse_sweep_values(raw: str) -> List[float]:
    if not raw.strip():
        return list(DEFAULT_SWEEP)
    values = sorted({float(token.strip()) for token in raw.split(",") if token.strip()})
    if not values:
        return list(DEFAULT_SWEEP)
    return values


def _family_primary_sample(family: Dict[str, Any]) -> Dict[str, Any]:
    samples = family.get("samples", [])
    if not isinstance(samples, list) or not samples:
        raise ValueError(f"Family {family.get('family_id')} has no samples")
    midpoint = len(samples) // 2
    return dict(samples[midpoint])


def _select_family_subset(
    families: List[Dict[str, Any]],
    *,
    diagnosis_mode: bool,
    diagnosis_short_count: int,
    diagnosis_medium_count: int,
    diagnosis_long_count: int,
    diagnosis_family_ids: List[str] | None,
) -> List[Dict[str, Any]]:
    if diagnosis_family_ids:
        requested = {str(value) for value in diagnosis_family_ids}
        selected = [family for family in families if str(family.get("family_id")) in requested]
        if len(selected) != len(requested):
            found = {str(family.get("family_id")) for family in selected}
            missing = sorted(requested - found)
            raise ValueError(f"Requested diagnosis family_id not found: {missing}")
        return selected
    if not diagnosis_mode:
        return families

    buckets: Dict[str, List[Dict[str, Any]]] = {"short": [], "medium": [], "long": []}
    for family in families:
        primary = _family_primary_sample(family)
        bucket = str(primary.get("duration_bucket", family.get("duration_bucket", "unknown")))
        if bucket in buckets:
            buckets[bucket].append(family)

    selected: List[Dict[str, Any]] = []
    selected.extend(buckets["short"][: max(0, diagnosis_short_count)])
    selected.extend(buckets["medium"][: max(0, diagnosis_medium_count)])
    selected.extend(buckets["long"][: max(0, diagnosis_long_count)])
    if not selected:
        raise ValueError("Diagnosis family selection produced zero families")
    return selected


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


def _apply_final_mapping_overrides(
    wrapper_config: dict[str, Any],
    *,
    scalar_transform_scale: float | None = None,
    scalar_transform_offset: float | None = None,
    scalar_transform_name: str | None = None,
    scalar_prior_scale: float | None = None,
    gain_order_direction: str | None = None,
    scope: str | None = None,
) -> dict[str, Any]:
    diffusion_cfg = wrapper_config.setdefault("diffusion", {})
    strength_cfg = diffusion_cfg.setdefault("strength_control", {})
    mapping_cfg = strength_cfg.setdefault("final_output_strength_mapping", {})
    if scalar_prior_scale is not None:
        mapping_cfg["scalar_prior_scale"] = float(scalar_prior_scale)
    if gain_order_direction:
        mapping_cfg["gain_order_direction"] = str(gain_order_direction)
    if scope:
        mapping_cfg["scope"] = str(scope)
    if scalar_transform_scale is not None or scalar_transform_offset is not None or scalar_transform_name:
        transform_cfg = mapping_cfg.setdefault("scalar_transform", {})
        transform_cfg["enabled"] = True
        if scalar_transform_scale is not None:
            transform_cfg["scale"] = float(scalar_transform_scale)
        if scalar_transform_offset is not None:
            transform_cfg["offset"] = float(scalar_transform_offset)
        if scalar_transform_name:
            transform_cfg["name"] = str(scalar_transform_name)
    return wrapper_config


def _resolve_wrapper_config_path(
    config_path: str,
    output_path: Path,
    *,
    scalar_transform_scale: float | None = None,
    scalar_transform_offset: float | None = None,
    scalar_transform_name: str | None = None,
    scalar_prior_scale: float | None = None,
    gain_order_direction: str | None = None,
    scope: str | None = None,
) -> str:
    if (
        scalar_transform_scale is None
        and scalar_transform_offset is None
        and not scalar_transform_name
        and scalar_prior_scale is None
        and not gain_order_direction
        and not scope
    ):
        return config_path
    wrapper_config = _load_wrapper_config(config_path)
    wrapper_config = _apply_final_mapping_overrides(
        wrapper_config,
        scalar_transform_scale=scalar_transform_scale,
        scalar_transform_offset=scalar_transform_offset,
        scalar_transform_name=scalar_transform_name,
        scalar_prior_scale=scalar_prior_scale,
        gain_order_direction=gain_order_direction,
        scope=scope,
    )
    wrapper_config_path = output_path.resolve().parent / "_wrapper_model_config.yaml"
    wrapper_config_path.parent.mkdir(parents=True, exist_ok=True)
    wrapper_config_path.write_text(yaml.safe_dump(wrapper_config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return str(wrapper_config_path)


def run_eval(
    benchmark_path: Path,
    model_path: str,
    config_path: str,
    output_path: Path,
    *,
    max_families: int,
    edit_steps: int,
    sampler: str,
    seed: int,
    device: str,
    smooth_radius: float,
    bg_drift_threshold: float,
    probe_json: str | None,
    sweep_values: List[float],
    generation_route: str = "soft_region",
    eval_mask_routed: bool = False,
    final_mapping_scalar_transform_scale: float | None = None,
    final_mapping_scalar_transform_offset: float | None = None,
    final_mapping_scalar_transform_name: str | None = None,
    final_mapping_scalar_prior_scale: float | None = None,
    final_mapping_gain_order_direction: str | None = None,
    final_mapping_scope: str | None = None,
    diagnosis_mode: bool = False,
    diagnosis_short_count: int = 1,
    diagnosis_medium_count: int = 1,
    diagnosis_long_count: int = 1,
    diagnosis_family_ids: List[str] | None = None,
    diagnosis_dump_series: bool = False,
) -> Dict[str, Any]:
    if not benchmark_path.exists() or not benchmark_path.is_file():
        raise ValueError(f"Benchmark file not found: {benchmark_path}")
    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
    scalar_scheme = _resolve_scalar_scheme(benchmark)
    all_families = [fam for fam in benchmark.get("families", []) if str(fam.get("tool_name")) == "trend_injection"]
    families = _select_family_subset(
        all_families[:max_families],
        diagnosis_mode=bool(diagnosis_mode),
        diagnosis_short_count=int(diagnosis_short_count),
        diagnosis_medium_count=int(diagnosis_medium_count),
        diagnosis_long_count=int(diagnosis_long_count),
        diagnosis_family_ids=diagnosis_family_ids,
    )
    if not families:
        raise ValueError("No trend_injection families found in benchmark.")
    if not model_path or not config_path:
        raise ValueError("model_path and config_path are required")

    required_keys = {"source_ts", "target_ts", "mask_gt", "region", "instruction_text"}
    required_anchor_scalars = _required_anchor_scalars_for_scheme(scalar_scheme)
    for family in families:
        samples = family.get("samples", [])
        if not isinstance(samples, list) or not samples:
            raise ValueError(f"Family {family.get('family_id')} has no samples")
        anchor_scalars = set()
        for sample in samples:
            missing_keys = sorted(required_keys - set(sample.keys()))
            if missing_keys:
                raise ValueError(f"Family {family.get('family_id')} sample missing keys {missing_keys}")
            anchor_scalars.add(float(_resolve_strength_scalar(sample)))
        if not required_anchor_scalars.issubset(anchor_scalars):
            raise ValueError(f"Family {family.get('family_id')} missing required anchors {sorted(required_anchor_scalars - anchor_scalars)}")

    if generation_route not in {"soft_region", "standard"}:
        raise ValueError(f"Unsupported generation_route={generation_route}")
    wrapper_config_path = _resolve_wrapper_config_path(
        config_path,
        output_path,
        scalar_transform_scale=final_mapping_scalar_transform_scale,
        scalar_transform_offset=final_mapping_scalar_transform_offset,
        scalar_transform_name=final_mapping_scalar_transform_name,
        scalar_prior_scale=final_mapping_scalar_prior_scale,
        gain_order_direction=final_mapping_gain_order_direction,
        scope=final_mapping_scope,
    )
    wrapper = TEditWrapper(model_path=model_path, config_path=wrapper_config_path, device=device)
    wrapper.set_edit_steps(edit_steps)

    family_rows: List[Dict[str, Any]] = []
    per_sweep_rows: Dict[str, List[Dict[str, Any]]] = {f"{value:.4f}": [] for value in sweep_values}

    for family_idx, family in enumerate(families):
        direction = _family_direction(family)
        src_attrs, tgt_attrs = _trend_attrs(direction)
        samples = [dict(sample) for sample in family.get("samples", [])]
        for sample in samples:
            sample["strength_scalar"] = _resolve_strength_scalar(sample)
        samples.sort(key=lambda sample: float(sample["strength_scalar"]))
        anchor_by_scalar = {float(sample["strength_scalar"]): sample for sample in samples}
        nearest_anchor = [min(anchor_by_scalar.keys(), key=lambda anchor: abs(anchor - value)) for value in sweep_values]

        source_ts = np.asarray(samples[0]["source_ts"], dtype=np.float32)
        mask_gt = np.asarray(samples[0]["mask_gt"], dtype=np.int64)
        region = samples[0]["region"]
        if not isinstance(region, (list, tuple)) or len(region) != 2:
            raise ValueError(f"Family {family.get('family_id')} has invalid region={region}")
        start_idx, end_idx = [int(v) for v in region]
        if not (0 <= start_idx < end_idx <= len(source_ts)):
            raise ValueError(f"Family {family.get('family_id')} has out-of-range region={region}")
        instruction_text = str(samples[len(samples) // 2]["instruction_text"])

        row: Dict[str, Any] = {
            "family_id": family.get("family_id"),
            "tool_name": family.get("tool_name"),
            "direction": direction,
            "duration_bucket": str(samples[0].get("duration_bucket", family.get("duration_bucket", "unknown"))),
            "region": family.get("region"),
            "anchor_strength_scalar": [float(sample["strength_scalar"]) for sample in samples],
            "scalar_scheme": scalar_scheme,
            "anchor_target_edit_gain": [
                float(np.mean(np.abs(np.asarray(sample["target_ts"], dtype=np.float32)[mask_gt.astype(bool)] - source_ts[mask_gt.astype(bool)])))
                for sample in samples
            ],
            "sweep": [],
        }
        diagnosis_route = {
            "enabled": bool(diagnosis_mode),
            "route_name": str(generation_route),
            "family_reason": None,
        }
        if diagnosis_mode:
            bucket = str(row["duration_bucket"])
            if bucket == "short":
                diagnosis_route["family_reason"] = "diagnosis_short"
            elif bucket == "medium":
                diagnosis_route["family_reason"] = "diagnosis_medium"
            elif bucket == "long":
                diagnosis_route["family_reason"] = "diagnosis_long"
            else:
                diagnosis_route["family_reason"] = "diagnosis_selected"
            row["diagnosis"] = diagnosis_route
            row["series_length"] = int(source_ts.shape[0])
            row["edit_region_length"] = int(max(0, end_idx - start_idx))
            row["mask_coverage_ratio"] = float(np.mean(mask_gt.astype(np.float32)))
            row["source_preview"] = source_ts.tolist() if diagnosis_dump_series else None
            row["target_preview_by_anchor"] = {
                f"{float(sample['strength_scalar']):.4f}": np.asarray(sample["target_ts"], dtype=np.float32).tolist()
                for sample in samples
            } if diagnosis_dump_series else None
            row["effect_route"] = []

        sweep_gains: List[float] = []
        sweep_scalars: List[float] = []
        sweep_bg: List[float] = []
        target_mae_values: List[float] = []

        for sweep_idx, (strength_scalar, anchor_scalar) in enumerate(zip(sweep_values, nearest_anchor)):
            anchor_sample = anchor_by_scalar[anchor_scalar]
            target_ts = np.asarray(anchor_sample["target_ts"], dtype=np.float32)

            torch.manual_seed(seed + family_idx * 97 + sweep_idx)
            np.random.seed(seed + family_idx * 97 + sweep_idx)
            route_diagnostics = None
            if generation_route == "standard":
                edited_ts = wrapper.edit_time_series(
                    ts=source_ts,
                    src_attrs=src_attrs,
                    tgt_attrs=tgt_attrs,
                    n_samples=1,
                    sampler=sampler,
                    edit_steps=edit_steps,
                    strength_scalar=float(strength_scalar),
                    task_id=None,
                    instruction_text=instruction_text,
                    edit_mask=mask_gt.astype(np.float32) if eval_mask_routed else None,
                )
            else:
                if diagnosis_mode:
                    edited_ts, route_diagnostics = wrapper.edit_region_soft(
                        ts=source_ts,
                        start_idx=start_idx,
                        end_idx=end_idx,
                        src_attrs=src_attrs,
                        tgt_attrs=tgt_attrs,
                        n_samples=1,
                        sampler=sampler,
                        smooth_radius=smooth_radius,
                        strength_scalar=float(strength_scalar),
                        task_id=None,
                        instruction_text=instruction_text,
                        return_diagnostics=True,
                        enable_strength_diagnostics=True,
                    )
                else:
                    edited_ts = wrapper.edit_region_soft(
                        ts=source_ts,
                        start_idx=start_idx,
                        end_idx=end_idx,
                        src_attrs=src_attrs,
                        tgt_attrs=tgt_attrs,
                        n_samples=1,
                        sampler=sampler,
                        smooth_radius=smooth_radius,
                        strength_scalar=float(strength_scalar),
                        task_id=None,
                        instruction_text=instruction_text,
                    )
            edited_ts = np.asarray(edited_ts, dtype=np.float32).reshape(-1)
            effect_record = None
            if diagnosis_mode:
                effect_output, effect_diagnostics = wrapper.edit_time_series(
                    ts=source_ts,
                    src_attrs=src_attrs,
                    tgt_attrs=tgt_attrs,
                    n_samples=1,
                    sampler=sampler,
                    edit_steps=edit_steps,
                    strength_scalar=float(strength_scalar),
                    task_id=None,
                    instruction_text=instruction_text,
                    edit_mask=mask_gt.astype(np.float32),
                    return_diagnostics=True,
                    enable_strength_diagnostics=True,
                )
                effect_output = np.asarray(effect_output, dtype=np.float32).reshape(-1)
                effect_metrics = _compute_metrics(source_ts, target_ts, effect_output, mask_gt)
                effect_record = {
                    "strength_scalar": float(strength_scalar),
                    "nearest_anchor_scalar": float(anchor_scalar),
                    "route": "effect",
                    **effect_metrics,
                    "series": effect_output.tolist() if diagnosis_dump_series else None,
                    "diagnostics": _to_jsonable(effect_diagnostics),
                }
                row["effect_route"].append(effect_record)

            route_diag_json = _to_jsonable(route_diagnostics) if route_diagnostics is not None else None
            route_series = edited_ts.tolist() if diagnosis_dump_series else None
            metrics = _compute_metrics(source_ts, target_ts, edited_ts, mask_gt)
            effect_edit_gain = None if effect_record is None else effect_record.get("edit_gain")
            effect_bg_mae = None if effect_record is None else effect_record.get("bg_mae")
            effect_target_mae = None if effect_record is None else effect_record.get("target_mae_edit_region")
            effect_minus_route_gain = None
            if effect_edit_gain is not None and metrics["edit_gain"] is not None:
                effect_minus_route_gain = float(effect_edit_gain - float(metrics["edit_gain"]))
            effect_minus_route_bg = None
            if effect_bg_mae is not None and metrics["bg_mae"] is not None:
                effect_minus_route_bg = float(effect_bg_mae - float(metrics["bg_mae"]))
            effect_minus_route_target = None
            if effect_target_mae is not None and metrics["target_mae_edit_region"] is not None:
                effect_minus_route_target = float(effect_target_mae - float(metrics["target_mae_edit_region"]))

            if diagnosis_mode:
                edit_mask_bool = mask_gt.astype(bool)
                route_delta = np.abs(edited_ts - source_ts)
                bg_mask = ~edit_mask_bool
                record_compare = {
                    "effect_minus_route_edit_gain": effect_minus_route_gain,
                    "effect_minus_route_bg_mae": effect_minus_route_bg,
                    "effect_minus_route_target_mae_edit_region": effect_minus_route_target,
                    "route_edit_region_abs_mean": float(np.mean(route_delta[edit_mask_bool])) if np.any(edit_mask_bool) else None,
                    "route_background_abs_mean": float(np.mean(route_delta[bg_mask])) if np.any(bg_mask) else None,
                }
            else:
                record_compare = None

            if edited_ts.shape != source_ts.shape:
                raise RuntimeError(f"Family {family.get('family_id')} output shape {edited_ts.shape} != input shape {source_ts.shape}")
            if metrics["edit_gain"] is not None:
                sweep_scalars.append(float(strength_scalar))
                sweep_gains.append(float(metrics["edit_gain"]))
            if metrics["bg_mae"] is not None:
                sweep_bg.append(float(metrics["bg_mae"]))
            if metrics["target_mae_edit_region"] is not None:
                target_mae_values.append(float(metrics["target_mae_edit_region"]))
            record = {
                "family_id": family.get("family_id"),
                "strength_scalar": float(strength_scalar),
                "nearest_anchor_scalar": float(anchor_scalar),
                "instruction_text": instruction_text,
                **metrics,
            }
            if diagnosis_mode:
                record["diagnostics"] = {
                    "route": route_diag_json,
                    "compare": record_compare,
                }
                record["series"] = route_series
            per_sweep_rows[f"{strength_scalar:.4f}"].append(record)
            row["sweep"].append(record)

        adjacent_monotonic = bool(
            len(sweep_gains) >= 2
            and all((sweep_gains[idx + 1] + FLOAT_TOL) >= sweep_gains[idx] for idx in range(len(sweep_gains) - 1))
        )
        rho = _spearman_rho(sweep_scalars, sweep_gains)
        gain_range = None if len(sweep_gains) < 2 else float(sweep_gains[-1] - sweep_gains[0])
        off_anchor_values = [value for value in sweep_values if all(abs(value - anchor) > FLOAT_TOL for anchor in anchor_by_scalar.keys())]
        off_anchor_records = [record for record in row["sweep"] if any(abs(record["strength_scalar"] - value) <= FLOAT_TOL for value in off_anchor_values)]
        off_anchor_monotonic = bool(
            len(off_anchor_records) < 2
            or all(
                (off_anchor_records[idx + 1]["edit_gain"] is not None)
                and (off_anchor_records[idx]["edit_gain"] is not None)
                and (off_anchor_records[idx + 1]["edit_gain"] + FLOAT_TOL) >= off_anchor_records[idx]["edit_gain"]
                for idx in range(len(off_anchor_records) - 1)
            )
        )

        medium_minus_weak = None if len(sweep_gains) < 2 else _safe_diff(sweep_gains[1], sweep_gains[0])
        strong_minus_medium = None if len(sweep_gains) < 3 else _safe_diff(sweep_gains[2], sweep_gains[1])
        strong_minus_weak = None if len(sweep_gains) < 2 else _safe_diff(sweep_gains[-1], sweep_gains[0])
        weak_le_medium_pass = bool(len(sweep_gains) >= 2 and (sweep_gains[1] + FLOAT_TOL) >= sweep_gains[0])
        medium_le_strong_pass = bool(len(sweep_gains) >= 3 and (sweep_gains[2] + FLOAT_TOL) >= sweep_gains[1])
        min_adjacent_gap = None if len(sweep_gains) < 2 else min(
            gap for gap in (medium_minus_weak, strong_minus_medium) if gap is not None
        )
        gap_balance_ratio = _gap_balance_ratio(medium_minus_weak, strong_minus_medium)
        adjacent_gap_collapse = _collapse_indicator(medium_minus_weak, strong_minus_medium)

        row["adjacent_monotonic_pass"] = adjacent_monotonic
        row["weak_le_medium_pass"] = weak_le_medium_pass
        row["medium_le_strong_pass"] = medium_le_strong_pass
        row["medium_minus_weak"] = medium_minus_weak
        row["strong_minus_medium"] = strong_minus_medium
        row["strong_minus_weak"] = strong_minus_weak
        row["min_adjacent_gap"] = min_adjacent_gap
        row["gap_balance_ratio"] = gap_balance_ratio
        row["adjacent_gap_collapse"] = adjacent_gap_collapse
        row["spacing_primary_route"] = str(generation_route)
        row["local_path_definition"] = "edit_time_series + edit_mask + final_output_strength_mapping.scope=edit_region"
        row["local_path_route_active"] = bool(generation_route == "standard" and eval_mask_routed and final_mapping_scope == "edit_region")
        row["local_path_route_exclusions"] = [
            "edit_region_soft latent blending",
            "unmasked global standard route",
        ]
        row["spacing_primary_metric"] = True
        row["spearman_rho_strength_gain"] = rho
        row["gain_range"] = gain_range
        row["bg_mae_strong_minus_weak"] = None if len(sweep_bg) < 2 else _safe_diff(sweep_bg[-1], sweep_bg[0])
        row["bg_mae_mean"] = _aggregate(sweep_bg)
        row["target_mae_edit_region_mean"] = _aggregate(target_mae_values)
        row["preservation_pass"] = bool((row["bg_mae_mean"] is not None) and (row["bg_mae_mean"] <= bg_drift_threshold))
        row["edit_minus_bg_gain_delta"] = None if row["strong_minus_weak"] is None or row["bg_mae_strong_minus_weak"] is None else float(row["strong_minus_weak"] - row["bg_mae_strong_minus_weak"])
        row["off_anchor_monotonic_pass"] = off_anchor_monotonic
        family_rows.append(row)

    probe_gate = None
    if probe_json:
        probe = json.loads(Path(probe_json).read_text(encoding="utf-8"))
        probe_gate = {
            "probe_path": probe_json,
            "diff_0_2_linf": float(probe.get("diff_0_2_linf", 0.0)),
            "pass": bool(float(probe.get("diff_0_2_linf", 0.0)) > 0.0),
        }

    duration_bucket_rows: Dict[str, List[Dict[str, Any]]] = {}
    for row in family_rows:
        duration_bucket_rows.setdefault(str(row.get("duration_bucket", "unknown")), []).append(row)

    final_mapping_overrides = {
        "scalar_transform": {
            "scale": None if final_mapping_scalar_transform_scale is None else float(final_mapping_scalar_transform_scale),
            "offset": None if final_mapping_scalar_transform_offset is None else float(final_mapping_scalar_transform_offset),
            "name": None if not final_mapping_scalar_transform_name else str(final_mapping_scalar_transform_name),
        },
        "scalar_prior_scale": None if final_mapping_scalar_prior_scale is None else float(final_mapping_scalar_prior_scale),
        "gain_order_direction": None if not final_mapping_gain_order_direction else str(final_mapping_gain_order_direction),
        "scope": None if not final_mapping_scope else str(final_mapping_scope),
    }
    route_metadata = {
        "route_statement": "edit_time_series + edit_mask + final_output_strength_mapping.scope=edit_region",
        "generation_route": str(generation_route),
        "generation_route_label": "standard_edit_time_series" if generation_route == "standard" else str(generation_route),
        "acceptance_route": "local_path_mask_routed" if generation_route == "standard" and eval_mask_routed else None,
        "eval_mask_routed": bool(eval_mask_routed),
        "local_path_route_active": bool(generation_route == "standard" and eval_mask_routed and final_mapping_scope == "edit_region"),
        "local_path_route_exclusions": [
            "edit_region_soft latent blending",
            "unmasked global standard route",
        ],
        "final_mapping_overrides": final_mapping_overrides,
    }

    summary = {
        "num_families": len(family_rows),
        "family_filter": "trend_injection",
        "probe_gate": probe_gate,
        "scalar_scheme": scalar_scheme,
        "sweep_values": sweep_values,
        "edit_gain_mean": {
            strength_scalar: _aggregate([row["edit_gain"] for row in per_sweep_rows[strength_scalar]])
            for strength_scalar in sorted(per_sweep_rows.keys())
        },
        "bg_mae_mean": {
            strength_scalar: _aggregate([row["bg_mae"] for row in per_sweep_rows[strength_scalar]])
            for strength_scalar in sorted(per_sweep_rows.keys())
        },
        "target_mae_edit_region_mean": {
            strength_scalar: _aggregate([row["target_mae_edit_region"] for row in per_sweep_rows[strength_scalar]])
            for strength_scalar in sorted(per_sweep_rows.keys())
        },
        "adjacent_monotonic_pass_rate": float(np.mean([float(row["adjacent_monotonic_pass"]) for row in family_rows])),
        "weak_le_medium_pass_rate": float(np.mean([float(row["weak_le_medium_pass"]) for row in family_rows])),
        "medium_le_strong_pass_rate": float(np.mean([float(row["medium_le_strong_pass"]) for row in family_rows])),
        "off_anchor_monotonic_pass_rate": float(np.mean([float(row["off_anchor_monotonic_pass"]) for row in family_rows])),
        "medium_minus_weak_mean": _aggregate([row["medium_minus_weak"] for row in family_rows]),
        "strong_minus_medium_mean": _aggregate([row["strong_minus_medium"] for row in family_rows]),
        "strong_minus_weak_mean": _aggregate([row["strong_minus_weak"] for row in family_rows]),
        "min_adjacent_gap_mean": _aggregate([row["min_adjacent_gap"] for row in family_rows]),
        "gap_balance_ratio_mean": _aggregate([row["gap_balance_ratio"] for row in family_rows]),
        "adjacent_gap_collapse_rate": float(np.mean([float(row["adjacent_gap_collapse"]) for row in family_rows])),
        "spacing_metrics_primary": True,
        "local_path_default_definition": route_metadata["route_statement"],
        "generation_route_label": route_metadata["generation_route_label"],
        "acceptance_route": route_metadata["acceptance_route"],
        "local_path_route_active": route_metadata["local_path_route_active"],
        "local_path_route_exclusions": route_metadata["local_path_route_exclusions"],
        "final_mapping_overrides": route_metadata["final_mapping_overrides"],
        "gain_range_mean": _aggregate([row["gain_range"] for row in family_rows]),
        "family_spearman_rho_mean": _aggregate([row["spearman_rho_strength_gain"] for row in family_rows]),
        "preservation_pass_rate": float(np.mean([float(row["preservation_pass"]) for row in family_rows])),
        "bg_mae_strong_minus_weak_mean": _aggregate([row["bg_mae_strong_minus_weak"] for row in family_rows]),
        "edit_minus_bg_gain_delta_mean": _aggregate([row["edit_minus_bg_gain_delta"] for row in family_rows]),
        "topline_acceptance_contract": {
            "adjacent_spacing": ["medium_minus_weak_mean", "strong_minus_medium_mean", "min_adjacent_gap_mean", "adjacent_gap_collapse_rate"],
            "adjacent_ordering": ["weak_le_medium_pass_rate", "medium_le_strong_pass_rate", "adjacent_monotonic_pass_rate"],
            "spacing_quality": ["gap_balance_ratio_mean"],
            "locality_safety": ["bg_mae_strong_minus_weak_mean", "preservation_pass_rate", "edit_minus_bg_gain_delta_mean"],
            "duration_stability": ["duration_bucket_summary.short", "duration_bucket_summary.medium", "duration_bucket_summary.long"],
            "raw_vs_final_diagnosis": ["probe_gate"],
        },
        "duration_bucket_summary": {
            bucket: {
                "num_families": len(bucket_rows),
                "adjacent_monotonic_pass_rate": float(np.mean([float(item["adjacent_monotonic_pass"]) for item in bucket_rows])),
                "weak_le_medium_pass_rate": float(np.mean([float(item["weak_le_medium_pass"]) for item in bucket_rows])),
                "medium_le_strong_pass_rate": float(np.mean([float(item["medium_le_strong_pass"]) for item in bucket_rows])),
                "off_anchor_monotonic_pass_rate": float(np.mean([float(item["off_anchor_monotonic_pass"]) for item in bucket_rows])),
                "medium_minus_weak_mean": _aggregate([item["medium_minus_weak"] for item in bucket_rows]),
                "strong_minus_medium_mean": _aggregate([item["strong_minus_medium"] for item in bucket_rows]),
                "strong_minus_weak_mean": _aggregate([item["strong_minus_weak"] for item in bucket_rows]),
                "min_adjacent_gap_mean": _aggregate([item["min_adjacent_gap"] for item in bucket_rows]),
                "gap_balance_ratio_mean": _aggregate([item["gap_balance_ratio"] for item in bucket_rows]),
                "adjacent_gap_collapse_rate": float(np.mean([float(item["adjacent_gap_collapse"]) for item in bucket_rows])),
                "gain_range_mean": _aggregate([item["gain_range"] for item in bucket_rows]),
                "family_spearman_rho_mean": _aggregate([item["spearman_rho_strength_gain"] for item in bucket_rows]),
                "bg_mae_mean": _aggregate([item["bg_mae_mean"] for item in bucket_rows]),
                "bg_mae_strong_minus_weak_mean": _aggregate([item["bg_mae_strong_minus_weak"] for item in bucket_rows]),
                "edit_minus_bg_gain_delta_mean": _aggregate([item["edit_minus_bg_gain_delta"] for item in bucket_rows]),
                "preservation_pass_rate": float(np.mean([float(item["preservation_pass"]) for item in bucket_rows])),
            }
            for bucket, bucket_rows in sorted(duration_bucket_rows.items())
        },
    }
    summary["minimum_usable_gain_range_mean"] = summary["gain_range_mean"]
    summary["family_spearman_rho_strength_gain_mean"] = summary["family_spearman_rho_mean"]

    if len(family_rows) != int(summary["num_families"]):
        raise RuntimeError(f"Family count mismatch: rows={len(family_rows)} summary={summary['num_families']}")
    if int(summary["num_families"]) <= 0:
        raise RuntimeError("Monotonic eval produced zero families")
    for metric_key in ("adjacent_monotonic_pass_rate", "off_anchor_monotonic_pass_rate", "preservation_pass_rate"):
        if summary.get(metric_key) is None:
            raise RuntimeError(f"Missing summary metric {metric_key}")

    payload = {
        "status": {
            "ok": True,
            "stage": "run_tedit_trend_monotonic_eval",
        },
        "config": {
            "benchmark_path": str(benchmark_path),
            "model_path": model_path,
            "config_path": wrapper_config_path,
            "requested_config_path": config_path,
            "max_families": max_families,
            "edit_steps": edit_steps,
            "sampler": sampler,
            "seed": seed,
            "device": device,
            "smooth_radius": smooth_radius,
            "bg_drift_threshold": bg_drift_threshold,
            "sweep_values": sweep_values,
            "scalar_scheme": scalar_scheme,
            "generation_route": generation_route,
            "generation_route_label": route_metadata["generation_route_label"],
            "acceptance_route": route_metadata["acceptance_route"],
            "eval_mask_routed": bool(eval_mask_routed),
            "final_mapping_scalar_transform_scale": final_mapping_scalar_transform_scale,
            "final_mapping_scalar_transform_offset": final_mapping_scalar_transform_offset,
            "final_mapping_scalar_transform_name": final_mapping_scalar_transform_name,
            "final_mapping_scalar_prior_scale": final_mapping_scalar_prior_scale,
            "final_mapping_gain_order_direction": final_mapping_gain_order_direction,
            "final_mapping_scope": final_mapping_scope,
            "final_mapping_overrides": final_mapping_overrides,
            "diagnosis_mode": bool(diagnosis_mode),
            "diagnosis_short_count": int(diagnosis_short_count),
            "diagnosis_medium_count": int(diagnosis_medium_count),
            "diagnosis_long_count": int(diagnosis_long_count),
            "diagnosis_family_ids": diagnosis_family_ids,
            "diagnosis_dump_series": bool(diagnosis_dump_series),
        },
        "summary": summary,
        "families": family_rows,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path = output_path.with_suffix(".md")
    lines = [
        "# Trend Monotonic Eval",
        "",
        f"- benchmark: `{benchmark_path}`",
        f"- num_families: {summary['num_families']}",
        f"- spacing_metrics_primary: {summary['spacing_metrics_primary']}",
        f"- local_path_default_definition: {summary['local_path_default_definition']}",
        f"- generation_route_label: {summary['generation_route_label']}",
        f"- acceptance_route: {summary['acceptance_route']}",
        f"- final_mapping_overrides: {json.dumps(summary['final_mapping_overrides'], ensure_ascii=False, sort_keys=True)}",
        f"- local_path_route_active: {summary['local_path_route_active']}",
        f"- local_path_route_exclusions: {', '.join(summary['local_path_route_exclusions'])}",
        f"- weak_le_medium_pass_rate: {summary['weak_le_medium_pass_rate']:.4f}",
        f"- medium_le_strong_pass_rate: {summary['medium_le_strong_pass_rate']:.4f}",
        f"- medium_minus_weak_mean: {summary['medium_minus_weak_mean']}",
        f"- strong_minus_medium_mean: {summary['strong_minus_medium_mean']}",
        f"- strong_minus_weak_mean: {summary['strong_minus_weak_mean']}",
        f"- min_adjacent_gap_mean: {summary['min_adjacent_gap_mean']}",
        f"- gap_balance_ratio_mean: {summary['gap_balance_ratio_mean']}",
        f"- adjacent_gap_collapse_rate: {summary['adjacent_gap_collapse_rate']}",
        f"- adjacent_monotonic_pass_rate: {summary['adjacent_monotonic_pass_rate']:.4f}",
        f"- preservation_pass_rate: {summary['preservation_pass_rate']:.4f}",
        f"- bg_mae_strong_minus_weak_mean: {summary['bg_mae_strong_minus_weak_mean']}",
        f"- edit_minus_bg_gain_delta_mean: {summary['edit_minus_bg_gain_delta_mean']}",
        f"- short_bucket_min_adjacent_gap_mean: {summary.get('duration_bucket_summary', {}).get('short', {}).get('min_adjacent_gap_mean')}",
        f"- short_bucket_adjacent_gap_collapse_rate: {summary.get('duration_bucket_summary', {}).get('short', {}).get('adjacent_gap_collapse_rate')}",
        f"- short_bucket_preservation_pass_rate: {summary.get('duration_bucket_summary', {}).get('short', {}).get('preservation_pass_rate')}",
        f"- medium_bucket_min_adjacent_gap_mean: {summary.get('duration_bucket_summary', {}).get('medium', {}).get('min_adjacent_gap_mean')}",
        f"- medium_bucket_adjacent_gap_collapse_rate: {summary.get('duration_bucket_summary', {}).get('medium', {}).get('adjacent_gap_collapse_rate')}",
        f"- medium_bucket_preservation_pass_rate: {summary.get('duration_bucket_summary', {}).get('medium', {}).get('preservation_pass_rate')}",
        f"- long_bucket_min_adjacent_gap_mean: {summary.get('duration_bucket_summary', {}).get('long', {}).get('min_adjacent_gap_mean')}",
        f"- long_bucket_adjacent_gap_collapse_rate: {summary.get('duration_bucket_summary', {}).get('long', {}).get('adjacent_gap_collapse_rate')}",
        f"- long_bucket_preservation_pass_rate: {summary.get('duration_bucket_summary', {}).get('long', {}).get('preservation_pass_rate')}",
        f"- off_anchor_monotonic_pass_rate: {summary['off_anchor_monotonic_pass_rate']:.4f}",
        f"- gain_range_mean: {summary['gain_range_mean']}",
        f"- family_spearman_rho_mean: {summary['family_spearman_rho_mean']}",
    ]
    if probe_gate is not None:
        lines.append(f"- probe_gate_pass: {probe_gate['pass']}")
        lines.append(f"- probe_diff_0_2_linf: {probe_gate['diff_0_2_linf']}")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Run scalar trend-family monotonic evaluation for strength-conditioned TEdit.")
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-families", type=int, default=18)
    parser.add_argument("--edit-steps", type=int, default=10)
    parser.add_argument("--sampler", default="ddim")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--smooth-radius", type=float, default=3.0)
    parser.add_argument("--bg-drift-threshold", type=float, default=0.05)
    parser.add_argument("--probe-json", default=None)
    parser.add_argument("--sweep-values", default="")
    parser.add_argument("--generation-route", default="soft_region", choices=["soft_region", "standard"])
    parser.add_argument("--eval-mask-routed", type=int, default=0, choices=[0, 1])
    parser.add_argument("--final-mapping-scalar-transform-scale", type=float, default=None)
    parser.add_argument("--final-mapping-scalar-transform-offset", type=float, default=None)
    parser.add_argument("--final-mapping-scalar-transform-name", default="")
    parser.add_argument("--final-mapping-scalar-prior-scale", type=float, default=None)
    parser.add_argument("--final-mapping-gain-order-direction", default="", choices=["", "increasing", "decreasing"])
    parser.add_argument("--final-mapping-scope", default="", choices=["", "global", "edit_region"])
    parser.add_argument("--diagnosis-mode", type=int, default=0, choices=[0, 1])
    parser.add_argument("--diagnosis-short-count", type=int, default=1)
    parser.add_argument("--diagnosis-medium-count", type=int, default=1)
    parser.add_argument("--diagnosis-long-count", type=int, default=1)
    parser.add_argument("--diagnosis-family-ids", default="")
    parser.add_argument("--diagnosis-dump-series", type=int, default=0, choices=[0, 1])
    args = parser.parse_args()

    diagnosis_family_ids = [token.strip() for token in str(args.diagnosis_family_ids).split(",") if token.strip()] or None

    payload = run_eval(
        benchmark_path=Path(args.benchmark),
        model_path=args.model_path,
        config_path=args.config_path,
        output_path=Path(args.output),
        max_families=int(args.max_families),
        edit_steps=int(args.edit_steps),
        sampler=str(args.sampler),
        seed=int(args.seed),
        device=str(args.device),
        smooth_radius=float(args.smooth_radius),
        bg_drift_threshold=float(args.bg_drift_threshold),
        probe_json=args.probe_json,
        sweep_values=_parse_sweep_values(args.sweep_values) if str(args.sweep_values).strip() else _default_sweep_for_scheme(_resolve_scalar_scheme(json.loads(Path(args.benchmark).read_text(encoding="utf-8")))),
        generation_route=str(args.generation_route),
        eval_mask_routed=bool(args.eval_mask_routed),
        final_mapping_scalar_transform_scale=args.final_mapping_scalar_transform_scale,
        final_mapping_scalar_transform_offset=args.final_mapping_scalar_transform_offset,
        final_mapping_scalar_transform_name=args.final_mapping_scalar_transform_name.strip() or None,
        final_mapping_scalar_prior_scale=args.final_mapping_scalar_prior_scale,
        final_mapping_gain_order_direction=args.final_mapping_gain_order_direction.strip() or None,
        final_mapping_scope=args.final_mapping_scope.strip() or None,
        diagnosis_mode=bool(args.diagnosis_mode),
        diagnosis_short_count=int(args.diagnosis_short_count),
        diagnosis_medium_count=int(args.diagnosis_medium_count),
        diagnosis_long_count=int(args.diagnosis_long_count),
        diagnosis_family_ids=diagnosis_family_ids,
        diagnosis_dump_series=bool(args.diagnosis_dump_series),
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
