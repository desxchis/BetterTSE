from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np


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


def _base_trend_and_residual(base_ts: np.ndarray, start: int, end: int) -> Tuple[np.ndarray, np.ndarray]:
    region = np.asarray(base_ts[start:end], dtype=np.float64)
    x = np.arange(len(region), dtype=np.float64)
    if len(region) <= 1:
        return region.copy(), np.zeros_like(region)
    coeffs = np.polyfit(x, region, 1)
    trend = np.polyval(coeffs, x)
    residual = region - trend
    return trend.astype(np.float64), residual.astype(np.float64)


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


def volatility_burst_local(
    base_ts: np.ndarray,
    region: List[int],
    noise_scale: float,
    active_len_ratio: float,
    center_ratio: float,
    envelope_strength: float,
    seed: int = 7,
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
    xs = np.linspace(-1.0, 1.0, rel_end - rel_start)
    envelope[rel_start:rel_end] = np.exp(-0.5 * np.square(xs / max(envelope_strength, 0.2)))
    rng = np.random.RandomState(seed + start + end + int(round(noise_scale * 10)))
    residual_std = max(float(np.std(residual)), 1e-3)
    burst_noise = rng.normal(loc=0.0, scale=residual_std * noise_scale, size=region_len)
    new_region = trend + residual + burst_noise * envelope
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


def heuristic_volatility_operator(base_ts: np.ndarray, region: List[int]) -> np.ndarray:
    return volatility_global_subwindow(
        base_ts=np.asarray(base_ts, dtype=np.float32),
        region=region,
        amplify_factor=2.0,
        active_len_ratio=1.0,
        center_ratio=0.5,
        envelope_strength=1.0,
    )


def _candidate_grid(operator_name: str) -> List[Dict[str, Any]]:
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
        for scale in (0.8, 1.2, 1.6, 2.0, 2.6, 3.2):
            for active_len in (0.2, 0.35, 0.5, 0.7):
                for center in (0.2, 0.5, 0.8):
                    for envelope in (0.25, 0.45, 0.7):
                        grid.append(
                            {
                                "noise_scale": float(scale),
                                "active_len_ratio": float(active_len),
                                "center_ratio": float(center),
                                "envelope_strength": float(envelope),
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
    raise ValueError(f"unknown volatility operator: {operator_name}")


def _apply_operator(operator_name: str, base_ts: np.ndarray, region: List[int], params: Dict[str, Any]) -> np.ndarray:
    if operator_name == "global_subwindow":
        return volatility_global_subwindow(base_ts, region, **params)
    if operator_name == "burst_local":
        return volatility_burst_local(base_ts, region, **params)
    if operator_name == "envelope_noise":
        return volatility_envelope_noise(base_ts, region, **params)
    raise ValueError(f"unknown volatility operator: {operator_name}")


def search_best_volatility_operator(
    *,
    operator_name: str,
    base_ts: np.ndarray,
    target_ts: np.ndarray,
    region: List[int],
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
        score = _audit_objective(metrics)
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
