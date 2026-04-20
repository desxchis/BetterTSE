from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np


LOCAL_BURST_HINTS = (
    "局部短时", "短时片段", "突发", "骤然", "瞬时", "burst", "爆发式", "突发式",
)
GLOBAL_SCALE_HINTS = (
    "整体", "全段", "全程", "普遍", "总体", "整体波动", "整体更乱", "整体更噪", "整体不稳定",
)
MONOTONIC_UP_HINTS = (
    "逐渐加剧", "越来越乱", "逐步失真", "持续恶化", "波动逐步放大", "后面更乱",
)
MONOTONIC_DOWN_HINTS = (
    "逐渐恢复", "逐步恢复", "逐步平稳", "慢慢恢复", "波动逐步减弱", "前面更乱",
)
PREVIEW_NON_MONOTONIC_HINTS = (
    "先高后低再高", "先低后高再低", "反复起伏", "双峰", "多段变化", "忽强忽弱", "来回波动",
)

VALID_VOLATILITY_SUBTYPES = {
    "global_scale",
    "local_burst",
    "envelope_monotonic",
    "preview_non_monotonic",
}


@dataclass
class VolatilityAuditResult:
    operator_name: str
    params: Dict[str, Any]
    objective: float
    metrics: Dict[str, float]
    edited_ts: List[float]
    search_space_size: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "operator_name": self.operator_name,
            "params": dict(self.params),
            "objective": float(self.objective),
            "metrics": dict(self.metrics),
            "edited_ts": list(self.edited_ts),
            "search_space_size": int(self.search_space_size),
        }


def infer_volatility_subtype_from_text(text: str) -> str:
    normalized = str(text or "")
    if any(token in normalized for token in PREVIEW_NON_MONOTONIC_HINTS):
        return "preview_non_monotonic"
    if any(token in normalized for token in LOCAL_BURST_HINTS):
        return "local_burst"
    if any(token in normalized for token in MONOTONIC_UP_HINTS) or any(token in normalized for token in MONOTONIC_DOWN_HINTS):
        return "envelope_monotonic"
    if any(token in normalized for token in GLOBAL_SCALE_HINTS):
        return "global_scale"
    return "global_scale"


def _normalize_subtype(value: Any) -> str:
    subtype = str(value or "").strip().lower()
    if subtype in VALID_VOLATILITY_SUBTYPES:
        return subtype
    return ""


def _is_short_region(region: List[int] | Tuple[int, int] | None, ts_length: int | None) -> bool:
    if not region or ts_length is None or ts_length <= 0 or len(region) != 2:
        return False
    start = int(region[0])
    end = int(region[1])
    width = max(1, end - start)
    return (width / max(ts_length, 1)) <= 0.22


def _is_long_region(region: List[int] | Tuple[int, int] | None, ts_length: int | None) -> bool:
    if not region or ts_length is None or ts_length <= 0 or len(region) != 2:
        return False
    start = int(region[0])
    end = int(region[1])
    width = max(1, end - start)
    return (width / max(ts_length, 1)) >= 0.35


def _default_params_for_subtype(subtype: str, text: str) -> Dict[str, Any]:
    if subtype == "local_burst":
        return {
            "background_scale": 0.5,
            "burst_center": 0.5,
            "burst_width": 0.25,
            "burst_amplitude": 2.4,
            "burst_envelope_sharpness": 0.8,
            "baseline_offset_ratio": 0.05,
        }
    if subtype == "envelope_monotonic":
        if any(token in str(text or "") for token in MONOTONIC_DOWN_HINTS):
            start_scale, end_scale = 2.0, 0.3
        else:
            start_scale, end_scale = 0.3, 2.0
        return {
            "base_noise_scale": 1.0,
            "start_scale": start_scale,
            "end_scale": end_scale,
            "baseline_offset_ratio": 0.05,
            "trend_preserve": 0.0,
        }
    return {
        "base_noise_scale": 1.0,
        "local_std_target_ratio": 2.0,
        "baseline_offset_ratio": 0.05,
        "trend_preserve": 0.0,
    }


def resolve_volatility_subtype_route(
    *,
    text: str,
    proposed_subtype: str | None = None,
    region: List[int] | Tuple[int, int] | None = None,
    ts_length: int | None = None,
) -> Dict[str, Any]:
    normalized = str(text or "")
    proposed = _normalize_subtype(proposed_subtype) or infer_volatility_subtype_from_text(normalized)
    guarded = proposed
    guard_reason = "planner_subtype"

    if any(token in normalized for token in PREVIEW_NON_MONOTONIC_HINTS):
        guarded = "preview_non_monotonic"
        guard_reason = "preview_non_monotonic_text_guard"
    elif any(token in normalized for token in LOCAL_BURST_HINTS):
        if guarded not in {"preview_non_monotonic", "envelope_monotonic"}:
            guarded = "local_burst"
            guard_reason = "local_burst_text_guard"
    elif any(token in normalized for token in MONOTONIC_UP_HINTS) or any(token in normalized for token in MONOTONIC_DOWN_HINTS):
        if guarded != "preview_non_monotonic":
            guarded = "envelope_monotonic"
            guard_reason = "monotonic_envelope_text_guard"
    elif any(token in normalized for token in GLOBAL_SCALE_HINTS):
        if guarded != "preview_non_monotonic":
            guarded = "global_scale"
            guard_reason = "global_scale_text_guard"
    elif _is_short_region(region, ts_length):
        if guarded not in {"preview_non_monotonic", "envelope_monotonic"}:
            guarded = "local_burst"
            guard_reason = "short_region_burst_guard"
    elif _is_long_region(region, ts_length) and any(token in normalized for token in ("逐渐", "持续", "越来越", "慢慢")):
        if guarded != "preview_non_monotonic":
            guarded = "envelope_monotonic"
            guard_reason = "long_region_monotonic_guard"

    final_subtype = guarded
    is_preview = final_subtype == "preview_non_monotonic"
    routed_subtype = "global_scale" if is_preview else final_subtype
    tool_name = {
        "global_scale": "volatility_global_scale",
        "local_burst": "volatility_local_burst",
        "envelope_monotonic": "volatility_envelope_monotonic",
    }[routed_subtype]
    canonical_tool = tool_name
    return {
        "proposed_subtype": proposed,
        "guarded_subtype": guarded,
        "final_subtype": final_subtype,
        "guard_reason": guard_reason,
        "is_preview": is_preview,
        "tool_name": tool_name,
        "canonical_tool": canonical_tool,
        "parameters": _default_params_for_subtype(routed_subtype, normalized),
    }


def infer_volatility_subtool_from_text(text: str) -> Dict[str, Any]:
    routed = resolve_volatility_subtype_route(text=text)
    return {
        "tool_name": routed["tool_name"],
        "canonical_tool": routed["canonical_tool"],
        "parameters": routed["parameters"],
        "volatility_subtype": routed["final_subtype"],
        "volatility_routing": {
            "proposed_subtype": routed["proposed_subtype"],
            "guarded_subtype": routed["guarded_subtype"],
            "final_subtype": routed["final_subtype"],
            "guard_reason": routed["guard_reason"],
            "is_preview": routed["is_preview"],
        },
    }


def _safe_region(region: List[int], length: int) -> Tuple[int, int]:
    start = max(0, min(int(region[0]), length - 1))
    end = max(start + 1, min(int(region[1]), length))
    return start, end


def _window_splits(start: int, end: int, num_windows: int = 4) -> List[Tuple[int, int]]:
    edges = np.linspace(start, end, num_windows + 1).astype(int)
    return [(int(edges[i]), max(int(edges[i + 1]), int(edges[i]) + 1)) for i in range(num_windows)]


def local_std_error(target_region: np.ndarray, edited_region: np.ndarray) -> float:
    return float(abs(np.std(edited_region) - np.std(target_region)))


def roughness_error(target_region: np.ndarray, edited_region: np.ndarray) -> float:
    target_rough = float(np.mean(np.abs(np.diff(target_region)))) if len(target_region) > 1 else 0.0
    edited_rough = float(np.mean(np.abs(np.diff(edited_region)))) if len(edited_region) > 1 else 0.0
    return float(abs(edited_rough - target_rough))


def windowed_energy_profile_error(target_region: np.ndarray, edited_region: np.ndarray, num_windows: int = 4) -> float:
    length = len(target_region)
    windows = _window_splits(0, length, num_windows=num_windows)
    errors = []
    for start, end in windows:
        target_energy = float(np.mean(np.square(np.diff(target_region[start:end])))) if end - start > 1 else 0.0
        edited_energy = float(np.mean(np.square(np.diff(edited_region[start:end])))) if end - start > 1 else 0.0
        errors.append(abs(edited_energy - target_energy))
    return float(np.mean(errors)) if errors else 0.0


def classify_volatility_pattern(target_region: np.ndarray, base_region: np.ndarray, num_windows: int = 4) -> str:
    windows = _window_splits(0, len(target_region), num_windows=num_windows)
    energies = []
    for start, end in windows:
        target_energy = float(np.mean(np.square(np.diff(target_region[start:end])))) if end - start > 1 else 0.0
        base_energy = float(np.mean(np.square(np.diff(base_region[start:end])))) if end - start > 1 else 0.0
        energies.append(max(target_energy - base_energy, 0.0))
    arr = np.asarray(energies, dtype=np.float64)
    total = float(np.sum(arr))
    if total <= 1e-8:
        return "uniform_variance"
    ratios = arr / total
    nonzero = arr[arr > 1e-8]
    if nonzero.size > 0 and float(np.max(ratios)) >= 0.55:
        return "local_burst"
    if nonzero.size > 0 and float(np.max(nonzero) / max(np.min(nonzero), 1e-8)) <= 1.8:
        return "uniform_variance"
    return "time_varying_envelope"


def classify_volatility_subpattern(target_region: np.ndarray, base_region: np.ndarray, num_windows: int = 4) -> str:
    pattern = classify_volatility_pattern(target_region, base_region, num_windows=num_windows)
    if pattern != "time_varying_envelope":
        return pattern
    windows = _window_splits(0, len(target_region), num_windows=num_windows)
    energies = []
    for start, end in windows:
        target_energy = float(np.mean(np.square(np.diff(target_region[start:end])))) if end - start > 1 else 0.0
        base_energy = float(np.mean(np.square(np.diff(base_region[start:end])))) if end - start > 1 else 0.0
        energies.append(max(target_energy - base_energy, 0.0))
    arr = np.asarray(energies, dtype=np.float64)
    if arr.size < 3:
        return "monotonic_envelope"
    diffs = np.diff(arr)
    signs = np.sign(diffs[np.abs(diffs) > 1e-8])
    if signs.size <= 1:
        return "monotonic_envelope"
    if np.all(signs >= 0) or np.all(signs <= 0):
        return "monotonic_envelope"
    return "non_monotonic_envelope"


def compute_volatility_audit_metrics(
    *,
    base_ts: np.ndarray,
    target_ts: np.ndarray,
    edited_ts: np.ndarray,
    region: List[int],
) -> Dict[str, float]:
    base = np.asarray(base_ts, dtype=np.float64).flatten()
    target = np.asarray(target_ts, dtype=np.float64).flatten()
    edited = np.asarray(edited_ts, dtype=np.float64).flatten()
    start, end = _safe_region(region, len(base))
    target_region = target[start:end]
    edited_region = edited[start:end]
    mask = np.zeros(len(base), dtype=bool)
    mask[start:end] = True
    outside = ~mask
    return {
        "mae_vs_target": float(np.mean(np.abs(edited - target))),
        "mse_vs_target": float(np.mean((edited - target) ** 2)),
        "preservation_mae": float(np.mean(np.abs(edited[outside] - base[outside]))) if np.any(outside) else 0.0,
        "local_std_error": local_std_error(target_region, edited_region),
        "roughness_error": roughness_error(target_region, edited_region),
        "windowed_energy_profile_error": windowed_energy_profile_error(target_region, edited_region),
    }


def _audit_objective(metrics: Dict[str, float]) -> float:
    return (
        metrics["mae_vs_target"]
        + 0.02 * metrics["mse_vs_target"]
        + 0.15 * metrics["preservation_mae"]
        + 0.35 * metrics["local_std_error"]
        + 0.15 * metrics["roughness_error"]
        + 0.10 * metrics["windowed_energy_profile_error"]
    )


def _global_scale_objective(metrics: Dict[str, float]) -> float:
    return (
        0.38 * metrics["mae_vs_target"]
        + 0.01 * metrics["mse_vs_target"]
        + 0.32 * metrics["local_std_error"]
        + 0.05 * metrics["roughness_error"]
        + 0.10 * metrics["windowed_energy_profile_error"]
    )


def _local_burst_objective(metrics: Dict[str, float]) -> float:
    return (
        0.42 * metrics["mae_vs_target"]
        + 0.01 * metrics["mse_vs_target"]
        + 0.10 * metrics["local_std_error"]
        + 0.17 * metrics["roughness_error"]
        + 0.30 * metrics["windowed_energy_profile_error"]
    )


def _envelope_monotonic_objective(metrics: Dict[str, float]) -> float:
    return (
        0.55 * metrics["mae_vs_target"]
        + 0.01 * metrics["mse_vs_target"]
        + 0.15 * metrics["local_std_error"]
        + 0.20 * metrics["roughness_error"]
        + 0.20 * metrics["windowed_energy_profile_error"]
    )


def _objective_for_variant(objective_variant: str, metrics: Dict[str, float]) -> float:
    if objective_variant == "default":
        return _audit_objective(metrics)
    if objective_variant == "global_scale":
        return _global_scale_objective(metrics)
    if objective_variant == "local_burst":
        return _local_burst_objective(metrics)
    if objective_variant == "envelope_monotonic":
        return _envelope_monotonic_objective(metrics)
    raise ValueError(f"unknown objective_variant: {objective_variant}")


def _base_trend_and_residual(base_ts: np.ndarray, start: int, end: int) -> Tuple[np.ndarray, np.ndarray]:
    region = np.asarray(base_ts[start:end], dtype=np.float64)
    x = np.arange(len(region), dtype=np.float64)
    if len(region) <= 1:
        return region.copy(), np.zeros_like(region)
    coeffs = np.polyfit(x, region, 1)
    trend = np.polyval(coeffs, x)
    residual = region - trend
    return trend.astype(np.float64), residual.astype(np.float64)


def _global_series_stats(base_ts: np.ndarray) -> Tuple[float, float, float]:
    arr = np.asarray(base_ts, dtype=np.float64)
    data_min = float(np.min(arr))
    data_range = max(float(np.max(arr) - data_min), 1e-3)
    data_std = max(float(np.std(arr)), 1e-3)
    return data_min, data_range, data_std


def volatility_global_subwindow(
    base_ts: np.ndarray,
    region: List[int],
    amplify_factor: float,
    active_len_ratio: float,
    center_ratio: float,
    envelope_strength: float,
) -> np.ndarray:
    base = np.asarray(base_ts, dtype=np.float64).copy()
    start, end = _safe_region(region, len(base))
    trend, residual = _base_trend_and_residual(base, start, end)
    region_len = end - start
    active_len = max(3, min(region_len, int(round(region_len * active_len_ratio))))
    center = start + int(round(center_ratio * max(region_len - 1, 1)))
    active_start = max(start, min(center - active_len // 2, end - active_len))
    active_end = min(end, active_start + active_len)
    envelope = np.zeros(region_len, dtype=np.float64)
    rel_start = active_start - start
    rel_end = active_end - start
    if rel_end > rel_start:
        xs = np.linspace(-1.0, 1.0, rel_end - rel_start)
        shape = np.exp(-0.5 * np.square(xs / max(envelope_strength, 0.2)))
        envelope[rel_start:rel_end] = shape
    envelope = np.clip(envelope, 0.0, 1.0)
    new_region = trend + residual * (1.0 + (amplify_factor - 1.0) * envelope)
    base[start:end] = new_region
    return base.astype(np.float32)


def volatility_global_scale(
    base_ts: np.ndarray,
    region: List[int],
    base_noise_scale: float,
    local_std_target_ratio: float,
    baseline_offset_ratio: float,
    trend_preserve: float,
    seed: int = 41,
) -> np.ndarray:
    base = np.asarray(base_ts, dtype=np.float64).copy()
    start, end = _safe_region(region, len(base))
    trend, _ = _base_trend_and_residual(base, start, end)
    region_len = end - start
    data_min, data_range, data_std = _global_series_stats(base)
    baseline = data_min + data_range * baseline_offset_ratio
    rng = np.random.RandomState(seed + start + end + int(round(local_std_target_ratio * 10)))
    noise_std = data_std * local_std_target_ratio
    region_noise = rng.normal(loc=0.0, scale=noise_std * base_noise_scale, size=region_len)
    trend_component = trend_preserve * (trend - float(np.mean(trend)))
    new_region = baseline + trend_component + region_noise
    base[start:end] = new_region
    return base.astype(np.float32)


def volatility_burst_local(
    base_ts: np.ndarray,
    region: List[int],
    background_scale: float,
    burst_center: float,
    burst_width: float,
    burst_amplitude: float,
    burst_envelope_sharpness: float,
    baseline_offset_ratio: float,
    seed: int = 7,
) -> np.ndarray:
    base = np.asarray(base_ts, dtype=np.float64).copy()
    start, end = _safe_region(region, len(base))
    _, residual = _base_trend_and_residual(base, start, end)
    region_len = end - start
    data_min, data_range, data_std = _global_series_stats(base)
    baseline = data_min + data_range * baseline_offset_ratio
    xs = np.linspace(0.0, 1.0, region_len, dtype=np.float64)
    width = max(float(burst_width), 0.08)
    center = float(np.clip(burst_center, 0.0, 1.0))
    envelope = np.exp(-0.5 * np.square((xs - center) / max(width * burst_envelope_sharpness, 0.05)))
    envelope = np.clip(envelope, 0.0, 1.0)
    rng = np.random.RandomState(seed + start + end + int(round(burst_amplitude * 10)))
    background_noise = rng.normal(loc=0.0, scale=data_std * background_scale, size=region_len)
    burst_noise = rng.normal(loc=0.0, scale=max(float(np.std(residual)), 1e-3) * burst_amplitude, size=region_len)
    new_region = baseline + background_noise + burst_noise * envelope
    base[start:end] = new_region
    return base.astype(np.float32)


def volatility_envelope_noise(
    base_ts: np.ndarray,
    region: List[int],
    noise_scale: float,
    envelope_strength: float,
    center_ratio: float,
    bias_ratio: float,
    seed: int = 13,
) -> np.ndarray:
    base = np.asarray(base_ts, dtype=np.float64).copy()
    start, end = _safe_region(region, len(base))
    trend, residual = _base_trend_and_residual(base, start, end)
    region_len = end - start
    xs = np.linspace(0.0, 1.0, region_len)
    center = np.clip(center_ratio, 0.0, 1.0)
    envelope = np.exp(-0.5 * np.square((xs - center) / max(envelope_strength, 0.15)))
    rng = np.random.RandomState(seed + start + end + int(round(noise_scale * 10)))
    residual_std = max(float(np.std(residual)), 1e-3)
    full_noise = rng.normal(loc=0.0, scale=residual_std * noise_scale, size=region_len)
    bias = bias_ratio * residual_std * envelope
    new_region = trend + residual + full_noise * envelope + bias
    base[start:end] = new_region
    return base.astype(np.float32)


def volatility_piecewise_envelope_noise(
    base_ts: np.ndarray,
    region: List[int],
    noise_scale: float,
    base_scale: float,
    knot_1_pos: float,
    knot_2_pos: float,
    knot_1_scale: float,
    knot_2_scale: float,
    tail_scale: float,
    seed: int = 23,
) -> np.ndarray:
    base = np.asarray(base_ts, dtype=np.float64).copy()
    start, end = _safe_region(region, len(base))
    trend, residual = _base_trend_and_residual(base, start, end)
    region_len = end - start
    xs = np.linspace(0.0, 1.0, region_len)
    k1 = float(np.clip(knot_1_pos, 0.05, 0.80))
    k2 = float(np.clip(knot_2_pos, k1 + 0.05, 0.95))
    control_x = np.asarray([0.0, k1, k2, 1.0], dtype=np.float64)
    control_y = np.asarray([base_scale, knot_1_scale, knot_2_scale, tail_scale], dtype=np.float64)
    envelope = np.interp(xs, control_x, control_y)
    envelope = np.clip(envelope, 0.0, 4.0)
    rng = np.random.RandomState(seed + start + end + int(round(noise_scale * 10)))
    residual_std = max(float(np.std(residual)), 1e-3)
    raw_noise = rng.normal(loc=0.0, scale=residual_std * noise_scale, size=region_len)
    new_region = trend + residual + raw_noise * envelope
    base[start:end] = new_region
    return base.astype(np.float32)


def volatility_envelope_monotonic(
    base_ts: np.ndarray,
    region: List[int],
    base_noise_scale: float,
    start_scale: float,
    end_scale: float,
    baseline_offset_ratio: float,
    trend_preserve: float,
    seed: int = 31,
) -> np.ndarray:
    base = np.asarray(base_ts, dtype=np.float64).copy()
    start, end = _safe_region(region, len(base))
    trend, _ = _base_trend_and_residual(base, start, end)
    region_len = end - start
    envelope = np.linspace(start_scale, end_scale, region_len, dtype=np.float64)
    envelope = np.clip(envelope, 0.0, 4.0)
    data_min, data_range, data_std = _global_series_stats(base)
    baseline = data_min + data_range * baseline_offset_ratio
    rng = np.random.RandomState(seed + start + end + int(round(base_noise_scale * 10)))
    raw_noise = rng.normal(loc=0.0, scale=data_std * base_noise_scale, size=region_len)
    trend_component = trend_preserve * (trend - float(np.mean(trend)))
    new_region = baseline + trend_component + raw_noise * envelope
    base[start:end] = new_region
    return base.astype(np.float32)


def heuristic_volatility_operator(base_ts: np.ndarray, region: List[int]) -> np.ndarray:
    return volatility_global_subwindow(
        base_ts=np.asarray(base_ts, dtype=np.float32),
        region=region,
        amplify_factor=2.0,
        active_len_ratio=1.0,
        center_ratio=0.5,
        envelope_strength=1.0,
    )


def build_noise_calibration_baselines(
    *,
    base_ts: np.ndarray,
    region: List[int],
    subtype: str,
) -> Dict[str, np.ndarray]:
    base = np.asarray(base_ts, dtype=np.float32)
    selected_subtype = str(subtype or "uniform_variance").strip().lower()
    baselines: Dict[str, np.ndarray] = {
        "generic_noise_family": heuristic_volatility_operator(base, region),
        "volatility_global_scale": volatility_global_scale(
            base,
            region,
            base_noise_scale=1.0,
            local_std_target_ratio=2.0,
            baseline_offset_ratio=0.05,
            trend_preserve=0.0,
        ),
        "volatility_local_burst": volatility_burst_local(
            base,
            region,
            background_scale=0.5,
            burst_center=0.5,
            burst_width=0.25,
            burst_amplitude=2.4,
            burst_envelope_sharpness=0.8,
            baseline_offset_ratio=0.05,
        ),
    }
    if selected_subtype == "monotonic_envelope":
        baselines["volatility_envelope_monotonic"] = volatility_envelope_monotonic(
            base,
            region,
            base_noise_scale=1.0,
            start_scale=0.3,
            end_scale=2.0,
            baseline_offset_ratio=0.05,
            trend_preserve=0.0,
        )
    return baselines


def _candidate_grid(operator_name: str) -> List[Dict[str, Any]]:
    if operator_name == "volatility_global_scale":
        grid = []
        for base_noise_scale in (0.8, 1.0, 1.2):
            for std_ratio in (1.2, 1.6, 2.0, 2.6, 3.2):
                for baseline_offset in (0.02, 0.05, 0.08, 0.12):
                    for trend_preserve in (0.0, 0.1, 0.2):
                        grid.append(
                            {
                                "base_noise_scale": float(base_noise_scale),
                                "local_std_target_ratio": float(std_ratio),
                                "baseline_offset_ratio": float(baseline_offset),
                                "trend_preserve": float(trend_preserve),
                            }
                        )
        return grid
    if operator_name == "global_subwindow":
        grid = []
        for amplify in (1.0, 1.2, 1.5, 2.0, 2.5, 3.0):
            for active_len in (0.35, 0.55, 0.75, 1.0):
                for center in (0.2, 0.5, 0.8):
                    for envelope in (0.35, 0.6, 1.0):
                        grid.append(
                            {
                                "amplify_factor": float(amplify),
                                "active_len_ratio": float(active_len),
                                "center_ratio": float(center),
                                "envelope_strength": float(envelope),
                            }
                        )
        return grid
    if operator_name == "burst_local":
        grid = []
        for background_scale in (0.2, 0.5, 0.8):
            for center in (0.25, 0.5, 0.75):
                for width in (0.15, 0.25, 0.4):
                    for amplitude in (1.2, 1.8, 2.4, 3.0):
                        for sharpness in (0.5, 0.8, 1.1):
                            for baseline_offset in (0.02, 0.05, 0.08):
                                grid.append(
                                    {
                                        "background_scale": float(background_scale),
                                        "burst_center": float(center),
                                        "burst_width": float(width),
                                        "burst_amplitude": float(amplitude),
                                        "burst_envelope_sharpness": float(sharpness),
                                        "baseline_offset_ratio": float(baseline_offset),
                                    }
                                )
        return grid
    if operator_name == "envelope_noise":
        grid = []
        for scale in (0.8, 1.2, 1.6, 2.0, 2.6):
            for envelope in (0.18, 0.3, 0.5, 0.8):
                for center in (0.25, 0.5, 0.75):
                    for bias in (-0.4, -0.2, 0.0, 0.2):
                        grid.append(
                            {
                                "noise_scale": float(scale),
                                "envelope_strength": float(envelope),
                                "center_ratio": float(center),
                                "bias_ratio": float(bias),
                            }
                        )
        return grid
    if operator_name == "piecewise_envelope_noise":
        grid = []
        for noise_scale in (0.8, 1.2, 1.6, 2.0, 2.6):
            for base_scale in (0.2, 0.5, 0.8):
                for k1 in (0.2, 0.35, 0.5):
                    for k2 in (0.55, 0.7, 0.85):
                        if k2 <= k1:
                            continue
                        for s1 in (0.6, 1.2, 2.0, 3.0):
                            for s2 in (0.6, 1.2, 2.0, 3.0):
                                for tail in (0.2, 0.8, 1.6):
                                    grid.append(
                                        {
                                            "noise_scale": float(noise_scale),
                                            "base_scale": float(base_scale),
                                            "knot_1_pos": float(k1),
                                            "knot_2_pos": float(k2),
                                            "knot_1_scale": float(s1),
                                            "knot_2_scale": float(s2),
                                            "tail_scale": float(tail),
                                        }
                                    )
        return grid
    if operator_name == "volatility_envelope_monotonic":
        grid = []
        for base_noise_scale in (0.8, 1.0, 1.2):
            for start_scale in (0.2, 0.5, 0.8, 1.2):
                for end_scale in (0.8, 1.2, 2.0, 3.0):
                    for baseline_offset in (0.02, 0.05, 0.08, 0.12):
                        for trend_preserve in (0.0, 0.08, 0.16):
                            if abs(end_scale - start_scale) < 0.2:
                                continue
                        grid.append(
                            {
                                "base_noise_scale": float(base_noise_scale),
                                "start_scale": float(start_scale),
                                "end_scale": float(end_scale),
                                "baseline_offset_ratio": float(baseline_offset),
                                "trend_preserve": float(trend_preserve),
                            }
                        )
        return grid
    raise ValueError(f"unknown volatility operator: {operator_name}")


def _apply_operator(operator_name: str, base_ts: np.ndarray, region: List[int], params: Dict[str, Any]) -> np.ndarray:
    if operator_name == "volatility_global_scale":
        return volatility_global_scale(base_ts, region, **params)
    if operator_name == "global_subwindow":
        return volatility_global_subwindow(base_ts, region, **params)
    if operator_name == "burst_local":
        return volatility_burst_local(base_ts, region, **params)
    if operator_name == "envelope_noise":
        return volatility_envelope_noise(base_ts, region, **params)
    if operator_name == "piecewise_envelope_noise":
        return volatility_piecewise_envelope_noise(base_ts, region, **params)
    if operator_name == "volatility_envelope_monotonic":
        return volatility_envelope_monotonic(base_ts, region, **params)
    raise ValueError(f"unknown volatility operator: {operator_name}")


def search_best_volatility_operator(
    *,
    operator_name: str,
    base_ts: np.ndarray,
    target_ts: np.ndarray,
    region: List[int],
    objective_variant: str = "default",
) -> VolatilityAuditResult:
    base = np.asarray(base_ts, dtype=np.float32)
    target = np.asarray(target_ts, dtype=np.float32)
    candidates = _candidate_grid(operator_name)
    best_params = None
    best_metrics = None
    best_score = float("inf")
    best_sequence = None
    for params in candidates:
        edited = _apply_operator(operator_name, base, region, params)
        metrics = compute_volatility_audit_metrics(
            base_ts=base,
            target_ts=target,
            edited_ts=edited,
            region=region,
        )
        score = _objective_for_variant(objective_variant, metrics)
        if score < best_score:
            best_score = score
            best_params = dict(params)
            best_metrics = metrics
            best_sequence = np.asarray(edited, dtype=np.float32)
    if best_params is None or best_metrics is None or best_sequence is None:
        raise RuntimeError("no volatility candidate evaluated")
    return VolatilityAuditResult(
        operator_name=operator_name,
        params=best_params,
        objective=float(best_score),
        metrics=best_metrics,
        edited_ts=np.asarray(best_sequence, dtype=float).tolist(),
        search_space_size=len(candidates),
    )
