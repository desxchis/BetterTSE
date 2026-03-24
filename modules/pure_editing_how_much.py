from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np

from modules.pure_editing_volatility import search_best_volatility_operator
from tool.tedit_wrapper import TEditWrapper
from tool.ts_editors import (
    hybrid_down_soft,
    hybrid_up_soft,
    spike_inject,
    step_shift,
    volatility_increase,
)


@dataclass
class PureEditingTeacherResult:
    tool_name: str
    region: List[int]
    params: Dict[str, Any]
    objective: float
    metrics: Dict[str, float]
    teacher_sequence: List[float]
    search_space_size: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "region": list(self.region),
            "params": dict(self.params),
            "objective": float(self.objective),
            "metrics": dict(self.metrics),
            "teacher_sequence": list(self.teacher_sequence),
            "search_space_size": int(self.search_space_size),
        }


def _safe_region(region: List[int], length: int) -> Tuple[int, int]:
    start = max(0, min(int(region[0]), length - 1))
    end = max(start + 1, min(int(region[1]), length))
    return start, end


def compute_pure_editing_parameter_metrics(
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
    mask = np.zeros(len(base), dtype=bool)
    mask[start:end] = True

    outside = ~mask
    target_delta = target - base
    edited_delta = edited - base

    region_len = max(1, end - start)
    target_peak = float(np.max(np.abs(target_delta[start:end]))) if region_len else 0.0
    edited_peak = float(np.max(np.abs(edited_delta[start:end]))) if region_len else 0.0
    area_target = float(np.sum(target_delta[start:end])) if region_len else 0.0
    area_edited = float(np.sum(edited_delta[start:end])) if region_len else 0.0

    return {
        "mae_vs_target": float(np.mean(np.abs(edited - target))),
        "mse_vs_target": float(np.mean((edited - target) ** 2)),
        "preservation_mae": float(np.mean(np.abs(edited[outside] - base[outside]))) if np.any(outside) else 0.0,
        "peak_delta_error": float(abs(edited_peak - target_peak)),
        "signed_area_error": float(abs(area_edited - area_target)),
        "duration_error": 0.0,
    }


def _objective(metrics: Dict[str, float]) -> float:
    return (
        metrics["mae_vs_target"]
        + 0.02 * metrics["mse_vs_target"]
        + 0.20 * metrics["preservation_mae"]
        + 0.02 * metrics["peak_delta_error"]
        + 0.002 * metrics["signed_area_error"]
    )


def _local_scale(base_ts: np.ndarray, region: List[int]) -> float:
    start, end = _safe_region(region, len(base_ts))
    region_arr = np.asarray(base_ts[start:end], dtype=np.float64)
    data_range = float(np.max(base_ts) - np.min(base_ts))
    return max(float(np.std(region_arr)), data_range * 0.05, 1e-3)


def _apply_tool_candidate(
    *,
    tool_name: str,
    base_ts: np.ndarray,
    region: List[int],
    params: Dict[str, Any],
    tedit: TEditWrapper | None,
) -> np.ndarray:
    start, end = _safe_region(region, len(base_ts))
    series = np.asarray(base_ts, dtype=np.float32)

    if tool_name == "spike_inject":
        return spike_inject(
            ts=series,
            start_idx=start,
            end_idx=end,
            center=int(params["center"]),
            amplitude=float(params["amplitude"]),
            width=float(params["width"]),
        )
    if tool_name == "step_shift":
        return step_shift(
            ts=series,
            start_idx=start,
            end_idx=end,
            level_shift=float(params["level_shift"]),
            left_ramp_steps=int(params["left_ramp_steps"]),
            right_ramp_steps=int(params["right_ramp_steps"]),
        )
    if tool_name == "volatility_increase":
        return volatility_increase(
            ts=series,
            start_idx=start,
            end_idx=end,
            amplify_factor=float(params["amplify_factor"]),
        )
    if tool_name == "hybrid_up":
        if tedit is None:
            raise ValueError("tedit is required for hybrid_up teacher search")
        return hybrid_up_soft(
            ts=series,
            start_idx=start,
            end_idx=end,
            math_shift=float(params["math_shift"]),
            tedit=tedit,
        )
    if tool_name == "hybrid_down":
        if tedit is None:
            raise ValueError("tedit is required for hybrid_down teacher search")
        return hybrid_down_soft(
            ts=series,
            start_idx=start,
            end_idx=end,
            math_shift=float(params["math_shift"]),
            tedit=tedit,
        )
    raise ValueError(f"unsupported pure editing teacher tool: {tool_name}")


def _spike_candidates(base_ts: np.ndarray, region: List[int], direction: str) -> List[Dict[str, Any]]:
    start, end = _safe_region(region, len(base_ts))
    region_len = max(1, end - start)
    center_base = (start + end) // 2
    scale = _local_scale(base_ts, region)
    sign = -1.0 if str(direction).lower() in {"down", "downward", "negative"} else 1.0
    centers = [center_base + offset for offset in (-region_len // 6, 0, region_len // 6)]
    widths = [max(1.5, region_len / 10.0), max(2.0, region_len / 6.0), max(3.0, region_len / 4.0)]
    amplitudes = [sign * scale * factor for factor in (0.75, 1.25, 1.75, 2.5, 3.25)]
    candidates = []
    for center in centers:
        clamped_center = max(start, min(end - 1, int(center)))
        for width in widths:
            for amplitude in amplitudes:
                candidates.append(
                    {
                        "center": clamped_center,
                        "width": float(width),
                        "amplitude": float(amplitude),
                    }
                )
    return candidates


def _step_candidates(base_ts: np.ndarray, region: List[int], direction: str) -> List[Dict[str, Any]]:
    start, end = _safe_region(region, len(base_ts))
    region_len = max(1, end - start)
    scale = _local_scale(base_ts, region)
    sign = -1.0 if str(direction).lower() in {"down", "downward", "negative"} else 1.0
    left_options = sorted({1, max(1, region_len // 10), max(1, region_len // 6)})
    right_options = sorted({1, max(2, region_len // 5), max(3, region_len // 3)})
    level_shifts = [sign * scale * factor for factor in (0.75, 1.25, 1.75, 2.5, 3.25)]
    candidates = []
    for level_shift in level_shifts:
        for left_ramp in left_options:
            for right_ramp in right_options:
                candidates.append(
                    {
                        "level_shift": float(level_shift),
                        "left_ramp_steps": int(left_ramp),
                        "right_ramp_steps": int(right_ramp),
                    }
                )
    return candidates


def _volatility_candidates() -> List[Dict[str, Any]]:
    return [{"amplify_factor": float(factor)} for factor in (0.85, 1.0, 1.1, 1.35, 1.6, 2.0, 2.5, 3.0)]


def _hybrid_candidates(base_ts: np.ndarray, region: List[int], direction: str) -> List[Dict[str, Any]]:
    scale = _local_scale(base_ts, region)
    sign = -1.0 if str(direction).lower() in {"down", "downward", "negative"} else 1.0
    return [{"math_shift": float(sign * scale * factor)} for factor in (0.75, 1.25, 1.75, 2.25, 3.0, 4.0)]


def teacher_search_pure_editing_params(
    *,
    tool_name: str,
    base_ts: np.ndarray,
    target_ts: np.ndarray,
    region: List[int],
    direction: str = "up",
    tedit: TEditWrapper | None = None,
) -> PureEditingTeacherResult:
    base = np.asarray(base_ts, dtype=np.float32).flatten()
    target = np.asarray(target_ts, dtype=np.float32).flatten()
    safe_region = list(_safe_region(region, len(base)))

    if tool_name == "volatility_global_scale":
        audit = search_best_volatility_operator(
            operator_name="volatility_global_scale",
            base_ts=base,
            target_ts=target,
            region=safe_region,
            objective_variant="global_scale",
        )
        return PureEditingTeacherResult(
            tool_name=tool_name,
            region=safe_region,
            params=dict(audit.params),
            objective=float(audit.objective),
            metrics={
                "mae_vs_target": float(audit.metrics["mae_vs_target"]),
                "mse_vs_target": float(audit.metrics["mse_vs_target"]),
                "preservation_mae": float(audit.metrics["preservation_mae"]),
                "peak_delta_error": 0.0,
                "signed_area_error": 0.0,
                "duration_error": 0.0,
            },
            teacher_sequence=list(audit.edited_ts),
            search_space_size=int(audit.search_space_size),
        )
    if tool_name == "volatility_local_burst":
        audit = search_best_volatility_operator(
            operator_name="burst_local",
            base_ts=base,
            target_ts=target,
            region=safe_region,
            objective_variant="local_burst",
        )
        return PureEditingTeacherResult(
            tool_name=tool_name,
            region=safe_region,
            params=dict(audit.params),
            objective=float(audit.objective),
            metrics={
                "mae_vs_target": float(audit.metrics["mae_vs_target"]),
                "mse_vs_target": float(audit.metrics["mse_vs_target"]),
                "preservation_mae": float(audit.metrics["preservation_mae"]),
                "peak_delta_error": 0.0,
                "signed_area_error": 0.0,
                "duration_error": 0.0,
            },
            teacher_sequence=list(audit.edited_ts),
            search_space_size=int(audit.search_space_size),
        )
    if tool_name == "volatility_envelope_monotonic":
        audit = search_best_volatility_operator(
            operator_name="volatility_envelope_monotonic",
            base_ts=base,
            target_ts=target,
            region=safe_region,
            objective_variant="envelope_monotonic",
        )
        return PureEditingTeacherResult(
            tool_name=tool_name,
            region=safe_region,
            params=dict(audit.params),
            objective=float(audit.objective),
            metrics={
                "mae_vs_target": float(audit.metrics["mae_vs_target"]),
                "mse_vs_target": float(audit.metrics["mse_vs_target"]),
                "preservation_mae": float(audit.metrics["preservation_mae"]),
                "peak_delta_error": 0.0,
                "signed_area_error": 0.0,
                "duration_error": 0.0,
            },
            teacher_sequence=list(audit.edited_ts),
            search_space_size=int(audit.search_space_size),
        )

    if tool_name == "spike_inject":
        candidates = _spike_candidates(base, safe_region, direction)
    elif tool_name == "step_shift":
        candidates = _step_candidates(base, safe_region, direction)
    elif tool_name == "volatility_increase":
        candidates = _volatility_candidates()
    elif tool_name in {"hybrid_up", "hybrid_down"}:
        candidates = _hybrid_candidates(base, safe_region, direction)
    else:
        raise ValueError(f"unsupported teacher-search tool: {tool_name}")

    best_params: Dict[str, Any] | None = None
    best_metrics: Dict[str, float] | None = None
    best_score = float("inf")
    best_sequence: np.ndarray | None = None

    for params in candidates:
        edited = _apply_tool_candidate(
            tool_name=tool_name,
            base_ts=base,
            region=safe_region,
            params=params,
            tedit=tedit,
        )
        metrics = compute_pure_editing_parameter_metrics(
            base_ts=base,
            target_ts=target,
            edited_ts=edited,
            region=safe_region,
        )
        score = _objective(metrics)
        if score < best_score:
            best_score = score
            best_params = dict(params)
            best_metrics = metrics
            best_sequence = np.asarray(edited, dtype=np.float32)

    if best_params is None or best_metrics is None or best_sequence is None:
        raise RuntimeError("teacher search failed to produce any candidate")

    return PureEditingTeacherResult(
        tool_name=tool_name,
        region=safe_region,
        params=best_params,
        objective=float(best_score),
        metrics=best_metrics,
        teacher_sequence=np.asarray(best_sequence, dtype=float).tolist(),
        search_space_size=len(candidates),
    )
