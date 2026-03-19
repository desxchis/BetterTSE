from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

from modules.forecast_revision import apply_revision_profile, calibrate_revision


def _effective_region(region: List[int], horizon: int) -> Tuple[int, int]:
    start = max(0, min(int(region[0]), horizon))
    end = max(start, min(int(region[1]), horizon))
    return start, end


def _strength_scale(strength: str) -> float:
    return {
        "none": 0.0,
        "weak": 0.8,
        "medium": 1.0,
        "strong": 1.35,
    }.get(str(strength or "medium"), 1.0)


def anchor_forecast_to_history(history_ts: np.ndarray, base_forecast: np.ndarray) -> tuple[np.ndarray, Dict[str, float]]:
    history = np.asarray(history_ts, dtype=np.float64).flatten()
    forecast = np.asarray(base_forecast, dtype=np.float64).flatten()
    if history.size == 0 or forecast.size == 0:
        return forecast.copy(), {"anchor_shift": 0.0, "history_last": 0.0, "forecast_first_before": 0.0}
    shift = float(history[-1] - forecast[0])
    anchored = forecast + shift
    return anchored.astype(np.float64), {
        "anchor_shift": shift,
        "history_last": float(history[-1]),
        "forecast_first_before": float(forecast[0]),
    }


def apply_physical_revision_injection(
    base_forecast: np.ndarray,
    intent: Dict[str, Any],
    region: List[int],
    *,
    seed: int = 7,
    params: Dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    base = np.asarray(base_forecast, dtype=np.float64).flatten()
    target = base.copy()
    delta = np.zeros_like(base)
    params = dict(params or {})

    start, end = _effective_region(region, len(base))
    if end <= start:
        return target, delta, {"injection_type": "none", "region": [start, end]}

    local = base[start:end]
    local_std = max(float(np.std(local)) if local.size else 0.0, float(np.std(base)) if base.size else 0.0, 1e-3)
    data_min = float(np.min(base)) if base.size else 0.0
    data_max = float(np.max(base)) if base.size else 0.0
    data_range = max(data_max - data_min, local_std * 6.0, 1e-3)
    strength = _strength_scale(str(intent.get("strength", "medium")))
    shape = str(intent.get("shape", "none"))
    direction = str(intent.get("direction", "neutral"))
    sign = -1.0 if direction == "down" else 1.0
    region_len = end - start
    rng = np.random.RandomState(seed)

    amplitude = float(params.get("amplitude", local_std * strength))
    amplitude = max(abs(amplitude), local_std * 0.35)

    if shape == "hump":
        hump_t = np.linspace(0.0, 2.0 * np.pi, region_len, dtype=np.float64)
        hump = (1.0 - np.cos(hump_t)) / 2.0
        offset = sign * max(amplitude, data_range * 0.12 * strength) * hump
        target[start:end] = base[start:end] + offset
        delta[start:end] = offset
        return target, delta, {
            "injection_type": "trend_injection",
            "direction": "upward" if sign > 0 else "downward",
            "amplitude": float(np.max(np.abs(offset))) if offset.size else 0.0,
            "region": [start, end],
        }

    if shape in {"step", "plateau"}:
        magnitude = sign * max(amplitude, data_range * 0.10 * strength)
        offset = np.ones(region_len, dtype=np.float64) * magnitude
        if end < len(base):
            ramp_out = min(max(3, region_len // 3), 20)
            offset[-ramp_out:] = np.linspace(magnitude, 0.0, ramp_out, dtype=np.float64)
        target[start:end] = base[start:end] + offset
        delta[start:end] = offset
        return target, delta, {
            "injection_type": "step_change",
            "direction": "up" if sign > 0 else "down",
            "magnitude": float(magnitude),
            "region": [start, end],
        }

    if shape == "flatline":
        floor_value = float(params.get("floor_value", np.percentile(base, 3) if base.size else 0.0))
        ramp = min(5, max(1, region_len // 6))
        target[start:end] = floor_value
        entry_val = float(base[start])
        target[start:start + ramp] = np.linspace(entry_val, floor_value, ramp, dtype=np.float64)
        if end < len(base):
            exit_val = float(base[end])
            target[end - ramp:end] = np.linspace(floor_value, exit_val, ramp, dtype=np.float64)
        delta[start:end] = target[start:end] - base[start:end]
        return target, delta, {
            "injection_type": "hard_zero",
            "floor_value": floor_value,
            "ramp": ramp,
            "region": [start, end],
        }

    if shape == "irregular_noise":
        baseline = float(data_min + data_range * 0.05)
        noise_std = float(max(local_std * (1.2 + 0.8 * strength), amplitude))
        noise = rng.normal(loc=baseline, scale=noise_std, size=region_len)
        target[start:end] = noise.astype(np.float64)
        delta[start:end] = target[start:end] - base[start:end]
        return target, delta, {
            "injection_type": "noise_injection",
            "baseline": baseline,
            "noise_std": noise_std,
            "region": [start, end],
        }

    return target, delta, {"injection_type": "none", "region": [start, end]}


def estimate_projection_metrics(
    base_forecast: np.ndarray,
    future_gt: np.ndarray,
    revision_target: np.ndarray,
) -> Dict[str, float]:
    base = np.asarray(base_forecast, dtype=np.float64).flatten()
    future = np.asarray(future_gt, dtype=np.float64).flatten()
    target = np.asarray(revision_target, dtype=np.float64).flatten()
    residual = future - target
    base_delta = future - base
    denom = float(np.mean(np.abs(base_delta))) if base_delta.size else 0.0
    explained = 0.0
    if base_delta.size and np.mean(np.abs(base_delta)) > 1e-8:
        explained = 1.0 - float(np.mean(np.abs(residual)) / np.mean(np.abs(base_delta)))
    return {
        "mae_target_vs_future_gt": float(np.mean(np.abs(target - future))) if future.size else 0.0,
        "rmse_target_vs_future_gt": float(np.sqrt(np.mean((target - future) ** 2))) if future.size else 0.0,
        "explained_delta_ratio": float(max(-1.0, min(1.0, explained))),
        "base_mae_vs_future_gt": float(np.mean(np.abs(base - future))) if future.size else 0.0,
        "residual_mae": float(np.mean(np.abs(residual))) if residual.size else 0.0,
        "delta_mae_scale": denom,
    }


def project_revision_target_from_future(
    *,
    history_ts: np.ndarray,
    base_forecast: np.ndarray,
    future_gt: np.ndarray,
    intent: Dict[str, Any],
    region: List[int],
    context_text: str = "",
    strategy: str = "rule_local_stats",
    seed: int = 7,
) -> tuple[np.ndarray, np.ndarray, Dict[str, Any], Dict[str, Any]]:
    base = np.asarray(base_forecast, dtype=np.float64).flatten()
    future = np.asarray(future_gt, dtype=np.float64).flatten()
    if future.size != base.size:
        raise ValueError("project_revision_target_from_future requires base_forecast and future_gt to share the same length.")
    if str(intent.get("shape", "none")) in {"none", ""}:
        residual = future - base
        metrics = estimate_projection_metrics(base, future, base)
        metadata = {
            "projection_family": "none",
            "projection_grid_size": 0,
            "best_loss": metrics["mae_target_vs_future_gt"],
            "seed": int(seed),
        }
        return base.copy(), residual, metrics, metadata

    region = list(region)
    base_params = calibrate_revision(
        intent=intent,
        region=region,
        history_ts=np.asarray(history_ts, dtype=np.float64),
        base_forecast=base,
        context_text=context_text,
        strategy=strategy,
    )
    start, end = _effective_region(region, len(base))
    total_len = max(1, end - start)
    duration_candidates = sorted({max(1, int(round(total_len * ratio))) for ratio in (0.5, 0.75, 1.0)})
    recovery_candidates = [0.0, 0.15, 0.35, 0.55, 0.8]
    amplitude_candidates = [0.5, 0.8, 1.0, 1.25, 1.6, 2.0]
    if str(intent.get("shape")) == "flatline":
        recovery_candidates = [0.0, 0.15, 0.35]
    if str(intent.get("shape")) == "irregular_noise":
        recovery_candidates = [0.0]

    best_target = base.copy()
    best_loss = float(np.mean((best_target - future) ** 2))
    best_params = dict(base_params)
    grid_size = 0
    for duration in duration_candidates:
        for recovery in recovery_candidates:
            for amp_mul in amplitude_candidates:
                params = dict(base_params)
                params["duration"] = int(duration)
                params["recovery_rate"] = float(recovery)
                params["amplitude"] = float(max(1e-6, abs(float(base_params.get("amplitude", 0.0))) * amp_mul))
                candidate, _ = apply_revision_profile(base, intent, region, params, seed=seed)
                loss = float(np.mean((candidate - future) ** 2))
                grid_size += 1
                if loss < best_loss:
                    best_loss = loss
                    best_target = candidate.astype(np.float64)
                    best_params = params

    residual = future - best_target
    metrics = estimate_projection_metrics(base, future, best_target)
    metadata = {
        "projection_family": str(intent.get("shape", "none")),
        "projection_grid_size": int(grid_size),
        "best_loss": float(best_loss),
        "seed": int(seed),
        "best_params": best_params,
    }
    return best_target.astype(np.float64), residual.astype(np.float64), metrics, metadata
