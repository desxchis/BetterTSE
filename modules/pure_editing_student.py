from __future__ import annotations

import json
from pathlib import Path
import random
from typing import Any, Dict, List

import numpy as np

from modules.pure_editing_volatility import classify_volatility_subpattern


SUPPORTED_STUDENT_TOOLS = (
    "spike_inject",
    "step_shift",
    "hybrid_up",
    "hybrid_down",
    "volatility_global_scale",
    "volatility_local_burst",
    "volatility_envelope_monotonic",
)

TOOL_PARAM_KEYS: dict[str, tuple[str, ...]] = {
    "spike_inject": ("amplitude", "width", "center"),
    "step_shift": ("level_shift", "left_ramp_steps", "right_ramp_steps"),
    "hybrid_up": ("math_shift",),
    "hybrid_down": ("math_shift",),
    "volatility_global_scale": ("base_noise_scale", "local_std_target_ratio", "baseline_offset_ratio", "trend_preserve"),
    "volatility_local_burst": (
        "background_scale",
        "burst_center",
        "burst_width",
        "burst_amplitude",
        "burst_envelope_sharpness",
        "baseline_offset_ratio",
    ),
    "volatility_envelope_monotonic": ("base_noise_scale", "start_scale", "end_scale", "baseline_offset_ratio", "trend_preserve"),
}

TOOL_TARGET_BOUNDS: dict[str, dict[str, tuple[float, float]]] = {
    "spike_inject": {
        "amplitude": (-8.0, 8.0),
        "width": (0.05, 0.60),
        "center": (0.0, 1.0),
    },
    "step_shift": {
        "level_shift": (-8.0, 8.0),
        "left_ramp_steps": (0.0, 0.6),
        "right_ramp_steps": (0.0, 0.8),
    },
    "hybrid_up": {"math_shift": (0.0, 8.0)},
    "hybrid_down": {"math_shift": (-8.0, 0.0)},
    "volatility_global_scale": {
        "base_noise_scale": (0.4, 2.0),
        "local_std_target_ratio": (1.0, 4.0),
        "baseline_offset_ratio": (0.0, 0.2),
        "trend_preserve": (0.0, 0.4),
    },
    "volatility_local_burst": {
        "background_scale": (0.0, 1.5),
        "burst_center": (0.0, 1.0),
        "burst_width": (0.05, 0.6),
        "burst_amplitude": (0.5, 4.5),
        "burst_envelope_sharpness": (0.2, 1.6),
        "baseline_offset_ratio": (0.0, 0.2),
    },
    "volatility_envelope_monotonic": {
        "base_noise_scale": (0.4, 2.0),
        "start_scale": (0.0, 4.0),
        "end_scale": (0.0, 4.0),
        "baseline_offset_ratio": (0.0, 0.2),
        "trend_preserve": (0.0, 0.4),
    },
}

SHAPE_VALUES = ("none", "hump", "step", "flatline", "irregular_noise", "plateau")
DURATION_VALUES = ("none", "short", "medium", "long", "full", "medium_or_long")
STRENGTH_VALUES = ("none", "weak", "medium", "strong")
ONSET_VALUES = ("none", "abrupt", "gradual")
RECOVERY_VALUES = ("none", "persistent", "recovering", "gradual")
EFFECT_VALUES = ("none", "impulse", "level", "trend", "shutdown", "volatility")

GLOBAL_HINTS = ("整体", "全段", "整体更噪", "整体不稳定")
BURST_HINTS = ("局部", "短时", "突发", "骤然", "burst", "那一小段")
MONOTONIC_UP_HINTS = ("逐渐加剧", "越来越乱", "后面更乱", "持续加剧")
MONOTONIC_DOWN_HINTS = ("逐渐恢复", "慢慢恢复", "前面更乱", "逐步平稳")

VALID_STUDENT_MODEL_KINDS = ("linear", "quadratic", "mixed_capacity")
VALID_PREDICTION_VARIANTS = ("v1", "clip", "clip_guard", "clip_softguard")


def _safe_region(region: List[int], length: int) -> tuple[int, int]:
    start = max(0, min(int(region[0]), length - 1))
    end = max(start + 1, min(int(region[1]), length))
    return start, end


def _one_hot(value: str, vocab: tuple[str, ...]) -> list[float]:
    return [1.0 if value == item else 0.0 for item in vocab]


def _strength_scalar(value: str) -> float:
    return {"weak": 0.8, "medium": 1.0, "strong": 1.3}.get(str(value or "medium"), 1.0)


def _estimate_slope(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size < 2:
        return 0.0
    x = np.arange(arr.size, dtype=np.float64)
    A = np.column_stack([x, np.ones_like(x)])
    slope, _ = np.linalg.lstsq(A, arr, rcond=None)[0]
    return float(slope)


def _stats(values: np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def _tool_name_from_sample(sample: dict[str, Any]) -> str | None:
    operator = str(sample.get("injection_operator", ""))
    if operator == "trend_injection":
        return "spike_inject"
    if operator == "step_change":
        return "step_shift"
    if operator == "multiplier":
        return "hybrid_up"
    if operator == "hard_zero":
        return "hybrid_down"
    if operator == "noise_injection":
        subtype = (
            sample.get("volatility_subtype_gt")
            or (sample.get("stress_metadata") or {}).get("volatility_subtype_gt")
            or (sample.get("edit_intent_gt") or {}).get("volatility_subtype")
        )
        subtype = str(subtype or "").strip()
        if not subtype:
            base = np.asarray(sample.get("base_ts", []), dtype=np.float64)
            target = np.asarray(sample.get("target_ts", []), dtype=np.float64)
            start, end = _safe_region([sample.get("gt_start", 0), sample.get("gt_end", 0)], len(base))
            subtype = classify_volatility_subpattern(target[start:end], base[start:end])
        if subtype == "global_scale":
            return "volatility_global_scale"
        if subtype == "local_burst":
            return "volatility_local_burst"
        if subtype == "envelope_monotonic" or subtype == "monotonic_envelope":
            return "volatility_envelope_monotonic"
        return None
    return None


def _tool_guard_config(tool_name: str) -> dict[str, float]:
    if tool_name in {"step_shift", "hybrid_down"}:
        return {
            "support_mult": 1.8,
            "clip_delta_tol": 1.2,
            "softguard_gain": 0.15,
            "quality_floor": 0.55,
            "quality_gain": 0.35,
            "semantic_gain": 0.35,
            "heuristic_ratio_tol": 2.5,
        }
    if tool_name in {"spike_inject", "hybrid_up", "volatility_local_burst"}:
        return {
            "support_mult": 1.35,
            "clip_delta_tol": 0.7,
            "softguard_gain": 0.45,
            "quality_floor": 0.45,
            "quality_gain": 0.50,
            "semantic_gain": 0.55,
            "heuristic_ratio_tol": 2.0,
        }
    if tool_name in {"volatility_global_scale", "volatility_envelope_monotonic"}:
        return {
            "support_mult": 1.10,
            "clip_delta_tol": 0.35,
            "softguard_gain": 0.85,
            "quality_floor": 0.35,
            "quality_gain": 0.70,
            "semantic_gain": 0.45,
            "heuristic_ratio_tol": 1.5,
        }
    return {
        "support_mult": 1.25,
        "clip_delta_tol": 0.5,
        "softguard_gain": 0.5,
        "quality_floor": 0.45,
        "quality_gain": 0.5,
        "semantic_gain": 0.45,
        "heuristic_ratio_tol": 2.0,
    }


def build_heuristic_params_for_tool(
    *,
    tool_name: str,
    base_ts: np.ndarray,
    region: List[int],
    prompt_text: str = "",
) -> Dict[str, Any]:
    start, end = int(region[0]), int(region[1])
    local = np.asarray(base_ts[start:end], dtype=np.float64)
    local_std = max(float(np.std(local)), float(np.std(base_ts)) * 0.25, 1e-3)
    region_len = max(1, end - start)
    prompt = str(prompt_text or "")
    if tool_name == "spike_inject":
        sign = -1.0 if any(token in prompt for token in ("回落", "下降", "下滑", "更低")) else 1.0
        return {
            "amplitude": sign * local_std * 2.0,
            "width": max(2.0, region_len / 6.0),
            "center": int((start + end) // 2),
        }
    if tool_name == "step_shift":
        sign = -1.0 if any(token in prompt for token in ("回落", "下降", "下滑", "更低")) else 1.0
        return {
            "level_shift": sign * local_std * 2.0,
            "left_ramp_steps": max(1, region_len // 10),
            "right_ramp_steps": max(1, region_len // 5),
        }
    if tool_name == "hybrid_up":
        return {"math_shift": local_std * 2.0}
    if tool_name == "hybrid_down":
        return {"math_shift": -local_std * 2.0}
    if tool_name == "volatility_global_scale":
        return {
            "base_noise_scale": 1.0,
            "local_std_target_ratio": 2.0,
            "baseline_offset_ratio": 0.05,
            "trend_preserve": 0.0,
        }
    if tool_name == "volatility_local_burst":
        return {
            "background_scale": 0.5,
            "burst_center": 0.5,
            "burst_width": 0.25,
            "burst_amplitude": 2.4,
            "burst_envelope_sharpness": 0.8,
            "baseline_offset_ratio": 0.05,
        }
    if tool_name == "volatility_envelope_monotonic":
        descending = any(token in prompt for token in MONOTONIC_DOWN_HINTS)
        return {
            "base_noise_scale": 1.0,
            "start_scale": 2.0 if descending else 0.3,
            "end_scale": 0.3 if descending else 2.0,
            "baseline_offset_ratio": 0.05,
            "trend_preserve": 0.0,
        }
    raise ValueError(f"unsupported tool_name: {tool_name}")


def derive_student_tool_and_region(sample: dict[str, Any]) -> dict[str, Any] | None:
    tool_name = _tool_name_from_sample(sample)
    if tool_name not in SUPPORTED_STUDENT_TOOLS:
        return None
    base_ts = np.asarray(sample.get("base_ts", []), dtype=np.float32)
    region = list(_safe_region([sample.get("gt_start", 0), sample.get("gt_end", len(base_ts))], len(base_ts)))
    prompt = str(sample.get("vague_prompt") or "")
    if not prompt:
        event_prompts = sample.get("event_prompts") or []
        if event_prompts:
            prompt = str(event_prompts[0].get("prompt", ""))
    return {
        "tool_name": tool_name,
        "region": region,
        "prompt_text": prompt,
        "intent": dict(sample.get("edit_intent_gt") or {}),
    }


def build_student_feature_vector(
    *,
    tool_name: str,
    base_ts: np.ndarray,
    region: List[int],
    prompt_text: str,
    intent: Dict[str, Any],
) -> np.ndarray:
    base = np.asarray(base_ts, dtype=np.float64).reshape(-1)
    start, end = _safe_region(region, len(base))
    local = base[start:end]
    local_stats = _stats(local)
    global_stats = _stats(base)
    local_std = max(local_stats["std"], global_stats["std"] * 0.25, 1e-3)
    region_len = max(1, end - start)
    length = max(1, len(base))
    text = str(prompt_text or "")
    diffs = np.diff(local) if len(local) > 1 else np.asarray([0.0], dtype=np.float64)
    roughness = float(np.mean(np.abs(diffs))) if diffs.size else 0.0
    diff_energy = float(np.mean(np.square(diffs))) if diffs.size else 0.0

    features: list[float] = [
        1.0,
        start / length,
        end / length,
        region_len / length,
        ((start + end) / 2.0) / length,
        local_stats["mean"],
        local_stats["std"],
        local_std,
        (local_stats["max"] - local_stats["min"]) / max(local_std, 1e-6),
        _estimate_slope(local) / max(local_std, 1e-6),
        (local_stats["mean"] - global_stats["mean"]) / max(local_std, 1e-6),
        _strength_scalar(str(intent.get("strength", "medium"))),
        1.0 if any(token in text for token in GLOBAL_HINTS) else 0.0,
        1.0 if any(token in text for token in BURST_HINTS) else 0.0,
        1.0 if any(token in text for token in MONOTONIC_UP_HINTS) else 0.0,
        1.0 if any(token in text for token in MONOTONIC_DOWN_HINTS) else 0.0,
    ]
    if tool_name == "spike_inject":
        second_diffs = np.diff(local, n=2) if len(local) > 2 else np.asarray([0.0], dtype=np.float64)
        features.extend(
            [
                float(np.max(np.abs(local - np.mean(local)))) / max(local_std, 1e-6),
                roughness / max(local_std, 1e-6),
                float(np.mean(np.abs(second_diffs))) / max(local_std, 1e-6),
            ]
        )
    elif tool_name == "step_shift":
        half = max(1, len(local) // 2)
        left = local[:half]
        right = local[half:] if len(local[half:]) else local[-half:]
        features.extend(
            [
                (float(np.mean(right)) - float(np.mean(left))) / max(local_std, 1e-6),
                float(np.std(left)) / max(local_std, 1e-6),
                float(np.std(right)) / max(local_std, 1e-6),
            ]
        )
    elif tool_name in {"hybrid_up", "hybrid_down"}:
        start_val = float(local[0]) if len(local) else 0.0
        end_val = float(local[-1]) if len(local) else 0.0
        features.extend(
            [
                (end_val - start_val) / max(local_std, 1e-6),
                _estimate_slope(local),
                float(np.mean(local)) / max(local_std, 1e-6),
            ]
        )
    elif tool_name.startswith("volatility_"):
        windows = np.array_split(local, 4) if len(local) >= 4 else [local]
        window_stds = [float(np.std(window)) for window in windows]
        window_energies = [
            float(np.mean(np.square(np.diff(window)))) if len(window) > 1 else 0.0
            for window in windows
        ]
        features.extend(
            [
                roughness / max(local_std, 1e-6),
                diff_energy / max(local_std ** 2, 1e-6),
                float(max(window_stds, default=0.0)) / max(local_std, 1e-6),
                float(min(window_stds, default=0.0)) / max(local_std, 1e-6),
                float(max(window_energies, default=0.0)) / max(diff_energy, 1e-6),
                float(min(window_energies, default=0.0)) / max(diff_energy, 1e-6),
            ]
        )
    features.extend(_one_hot(str(intent.get("effect_family", "none")), EFFECT_VALUES))
    features.extend(_one_hot(str(intent.get("shape", "none")), SHAPE_VALUES))
    features.extend(_one_hot(str(intent.get("duration", "none")), DURATION_VALUES))
    features.extend(_one_hot(str(intent.get("strength", "none")), STRENGTH_VALUES))
    features.extend(_one_hot(str(intent.get("onset", "none")), ONSET_VALUES))
    features.extend(_one_hot(str(intent.get("recovery", "none")), RECOVERY_VALUES))
    return np.asarray(features, dtype=np.float64)


def _local_scale(base_ts: np.ndarray, region: List[int]) -> float:
    start, end = _safe_region(region, len(base_ts))
    local = np.asarray(base_ts[start:end], dtype=np.float64)
    global_std = float(np.std(base_ts)) if len(base_ts) else 1.0
    return max(float(np.std(local)), global_std * 0.25, 1e-3)


def params_to_target_vector(
    *,
    tool_name: str,
    params: Dict[str, Any],
    base_ts: np.ndarray,
    region: List[int],
) -> np.ndarray:
    start, end = _safe_region(region, len(base_ts))
    region_len = max(1, end - start)
    scale = _local_scale(base_ts, region)
    if tool_name == "spike_inject":
        return np.asarray([
            float(params.get("amplitude", 0.0)) / scale,
            float(params.get("width", max(2.0, region_len / 6.0))) / region_len,
            (float(params.get("center", (start + end) / 2.0)) - start) / max(region_len - 1, 1),
        ], dtype=np.float64)
    if tool_name == "step_shift":
        return np.asarray([
            float(params.get("level_shift", 0.0)) / scale,
            float(params.get("left_ramp_steps", 1.0)) / region_len,
            float(params.get("right_ramp_steps", 1.0)) / region_len,
        ], dtype=np.float64)
    if tool_name in {"hybrid_up", "hybrid_down"}:
        return np.asarray([float(params.get("math_shift", 0.0)) / scale], dtype=np.float64)
    keys = TOOL_PARAM_KEYS[tool_name]
    return np.asarray([float(params.get(key, 0.0)) for key in keys], dtype=np.float64)


def target_vector_to_params(
    *,
    tool_name: str,
    target: np.ndarray,
    base_ts: np.ndarray,
    region: List[int],
) -> Dict[str, Any]:
    values = np.asarray(target, dtype=np.float64).reshape(-1)
    start, end = _safe_region(region, len(base_ts))
    region_len = max(1, end - start)
    scale = _local_scale(base_ts, region)
    bounded = []
    for idx, key in enumerate(TOOL_PARAM_KEYS[tool_name]):
        low, high = TOOL_TARGET_BOUNDS[tool_name][key]
        raw_value = float(values[idx]) if idx < len(values) else 0.0
        bounded.append(float(np.clip(raw_value, low, high)))

    if tool_name == "spike_inject":
        return {
            "amplitude": bounded[0] * scale,
            "width": max(2.0, bounded[1] * region_len),
            "center": int(round(start + bounded[2] * max(region_len - 1, 1))),
        }
    if tool_name == "step_shift":
        return {
            "level_shift": bounded[0] * scale,
            "left_ramp_steps": int(round(max(1.0, bounded[1] * region_len))),
            "right_ramp_steps": int(round(max(1.0, bounded[2] * region_len))),
        }
    if tool_name in {"hybrid_up", "hybrid_down"}:
        return {"math_shift": bounded[0] * scale}
    return {key: bounded[idx] for idx, key in enumerate(TOOL_PARAM_KEYS[tool_name])}


def fit_tool_conditioned_student(
    samples: List[Dict[str, Any]],
    *,
    alpha: float = 1.0,
    model_kind: str = "linear",
    seed: int = 7,
) -> Dict[str, Any]:
    if model_kind not in VALID_STUDENT_MODEL_KINDS:
        raise ValueError(f"unsupported model_kind: {model_kind}")
    grouped: dict[str, list[dict[str, Any]]] = {}
    for sample in samples:
        tool_name = str(sample.get("tool_name", ""))
        if tool_name not in SUPPORTED_STUDENT_TOOLS or "teacher_params" not in sample:
            continue
        grouped.setdefault(tool_name, []).append(sample)
    if not grouped:
        raise ValueError("No supported samples for pure-editing student training")

    model: Dict[str, Any] = {
        "model_type": "tool_conditioned_pure_editing_student_v1",
        "alpha": float(alpha),
        "model_kind": model_kind,
        "tool_heads": {},
    }
    for tool_name, rows in grouped.items():
        X = []
        Y = []
        for row in rows:
            base_ts = np.asarray(row["base_ts"], dtype=np.float64)
            region = list(row["region"])
            X.append(build_student_feature_vector(
                tool_name=tool_name,
                base_ts=base_ts,
                region=region,
                prompt_text=str(row.get("prompt_text", "")),
                intent=dict(row.get("intent") or {}),
            ))
            Y.append(params_to_target_vector(
                tool_name=tool_name,
                params=dict(row["teacher_params"]),
                base_ts=base_ts,
                region=region,
            ))
        X_arr = np.asarray(X, dtype=np.float64)
        Y_arr = np.asarray(Y, dtype=np.float64)
        feat_mean = X_arr.mean(axis=0)
        feat_std = X_arr.std(axis=0)
        feat_std = np.where(feat_std < 1e-6, 1.0, feat_std)
        X_norm = (X_arr - feat_mean) / feat_std
        head_kind = _resolve_head_kind(model_kind, tool_name)
        if head_kind in {"linear", "quadratic"}:
            X_aug = _expand_design_matrix(X_norm, head_kind)
            if len(rows) == 1:
                weights = np.zeros((X_aug.shape[1], Y_arr.shape[1]), dtype=np.float64)
                weights[-1, :] = Y_arr[0]
            else:
                reg = alpha * np.eye(X_aug.shape[1], dtype=np.float64)
                reg[-1, -1] = 0.0
                weights = np.linalg.solve(X_aug.T @ X_aug + reg, X_aug.T @ Y_arr)
            head_payload = {
                "head_kind": head_kind,
                "weights": weights.tolist(),
            }
        else:
            head_payload = _fit_mlp_head(X_norm, Y_arr, alpha=alpha, seed=seed + len(rows) + len(tool_name))
        cv_quality = _estimate_head_cv_quality(
            tool_name=tool_name,
            rows=rows,
            alpha=alpha,
            model_kind=model_kind,
            seed=seed,
        )
        model["tool_heads"][tool_name] = {
            "head_kind": head_kind,
            "feature_mean": feat_mean.tolist(),
            "feature_std": feat_std.tolist(),
            "target_keys": list(TOOL_PARAM_KEYS[tool_name]),
            "train_count": len(rows),
            "target_p05": np.quantile(Y_arr, 0.05, axis=0).tolist(),
            "target_p95": np.quantile(Y_arr, 0.95, axis=0).tolist(),
            "feature_abs_z_p95": float(np.quantile(np.max(np.abs(X_norm), axis=1), 0.95)),
            "cv_student_mae": float(cv_quality["student_mae"]),
            "cv_heuristic_mae": float(cv_quality["heuristic_mae"]),
            "cv_teacher_gap_closed": float(cv_quality["teacher_gap_closed"]),
            "cv_student_better_rate_vs_heuristic": float(cv_quality["student_better_rate_vs_heuristic"]),
            **head_payload,
        }
    return model


def predict_tool_conditioned_params(
    *,
    model: Dict[str, Any],
    tool_name: str,
    base_ts: np.ndarray,
    region: List[int],
    prompt_text: str,
    intent: Dict[str, Any],
    prediction_variant: str = "v1",
    return_metadata: bool = False,
) -> Dict[str, Any] | None:
    if prediction_variant not in VALID_PREDICTION_VARIANTS:
        raise ValueError(f"unsupported prediction_variant: {prediction_variant}")
    head = (model.get("tool_heads") or {}).get(tool_name)
    if not isinstance(head, dict):
        return None
    features = build_student_feature_vector(
        tool_name=tool_name,
        base_ts=np.asarray(base_ts, dtype=np.float64),
        region=region,
        prompt_text=prompt_text,
        intent=intent,
    )
    feat_mean = np.asarray(head.get("feature_mean", []), dtype=np.float64)
    feat_std = np.asarray(head.get("feature_std", []), dtype=np.float64)
    feat_std = np.where(feat_std < 1e-6, 1.0, feat_std)
    head_kind = str(head.get("head_kind", "linear"))
    if features.size != feat_mean.size:
        raise ValueError(f"feature size mismatch for {tool_name}: got {features.size}, expected {feat_mean.size}")
    x_norm = (features - feat_mean) / feat_std
    support_score = float(np.max(np.abs(x_norm))) if x_norm.size else 0.0
    if head_kind in {"linear", "quadratic"}:
        weights = np.asarray(head.get("weights", []), dtype=np.float64)
        x_aug = _expand_feature_vector(x_norm, head_kind)
        raw_prediction = x_aug @ weights
    elif head_kind == "mlp":
        raw_prediction = _predict_mlp_head(x_norm, head)
    else:
        raise ValueError(f"unsupported head_kind: {head_kind}")
    clipped_prediction = np.asarray(raw_prediction, dtype=np.float64).copy()
    clipped = False
    if prediction_variant in {"clip", "clip_guard", "clip_softguard"}:
        p05 = np.asarray(head.get("target_p05", []), dtype=np.float64)
        p95 = np.asarray(head.get("target_p95", []), dtype=np.float64)
        if p05.size == clipped_prediction.size == p95.size:
            clipped_prediction = np.clip(clipped_prediction, p05, p95)
            clipped = bool(np.any(np.abs(clipped_prediction - raw_prediction) > 1e-8))
    guard_triggered = False
    guard_reason = None
    guard_weight = 0.0
    config = _tool_guard_config(tool_name)
    if prediction_variant == "clip_guard":
        support_limit = float(head.get("feature_abs_z_p95", 4.0)) * config["support_mult"]
        if support_score > support_limit:
            guard_triggered = True
            guard_reason = "low_support"
        elif clipped:
            delta = float(np.max(np.abs(clipped_prediction - raw_prediction)))
            if delta > config["clip_delta_tol"]:
                guard_triggered = True
                guard_reason = "aggressive_clip"
    elif prediction_variant == "clip_softguard":
        support_limit = float(head.get("feature_abs_z_p95", 4.0)) * config["support_mult"]
        support_risk = max(0.0, (support_score - support_limit) / max(support_limit, 1e-6))
        clip_risk = 0.0
        if clipped:
            delta = float(np.max(np.abs(clipped_prediction - raw_prediction)))
            clip_risk = max(0.0, (delta - config["clip_delta_tol"]) / max(config["clip_delta_tol"], 1e-6))
        if np.any(np.isnan(raw_prediction)) or np.any(np.isinf(raw_prediction)):
            support_risk = max(support_risk, 1.0)
            clip_risk = max(clip_risk, 1.0)
        cv_gap_closed = float(head.get("cv_teacher_gap_closed", 1.0))
        quality_risk = max(0.0, (config["quality_floor"] - cv_gap_closed) / max(config["quality_floor"], 1e-6))
        quality_weight = float(np.clip(config["quality_gain"] * quality_risk, 0.0, 1.0))
        heuristic_params = build_heuristic_params_for_tool(
            tool_name=tool_name,
            base_ts=np.asarray(base_ts, dtype=np.float64),
            region=region,
            prompt_text=prompt_text,
        )
        heuristic_target = params_to_target_vector(
            tool_name=tool_name,
            params=heuristic_params,
            base_ts=np.asarray(base_ts, dtype=np.float64),
            region=region,
        )
        candidate_params = target_vector_to_params(
            tool_name=tool_name,
            target=clipped_prediction,
            base_ts=np.asarray(base_ts, dtype=np.float64),
            region=region,
        )
        semantic_risk = _semantic_risk(
            tool_name=tool_name,
            candidate_params=candidate_params,
            heuristic_params=heuristic_params,
            intent=intent,
            ratio_tol=config["heuristic_ratio_tol"],
        )
        semantic_weight = float(np.clip(config["semantic_gain"] * semantic_risk, 0.0, 1.0))
        guard_weight = float(np.clip(max(config["softguard_gain"] * max(support_risk, clip_risk), quality_weight, semantic_weight), 0.0, 1.0))
        if guard_weight >= 0.999:
            guard_triggered = True
            guard_reason = "softguard_full_fallback"
    if guard_triggered:
        fallback = build_heuristic_params_for_tool(
            tool_name=tool_name,
            base_ts=np.asarray(base_ts, dtype=np.float64),
            region=region,
            prompt_text=prompt_text,
        )
        if return_metadata:
            return {
                "params": fallback,
                "source": "heuristic_fallback",
                "support_score": support_score,
                "guard_reason": guard_reason,
                "clipped": clipped,
                "guard_weight": 1.0,
                "cv_teacher_gap_closed": float(head.get("cv_teacher_gap_closed", 1.0)),
            }
        return fallback
    prediction = clipped_prediction if prediction_variant in {"clip", "clip_guard"} else raw_prediction
    if prediction_variant == "clip_softguard":
        prediction = (1.0 - guard_weight) * clipped_prediction + guard_weight * heuristic_target
    params = target_vector_to_params(
        tool_name=tool_name,
        target=prediction,
        base_ts=np.asarray(base_ts, dtype=np.float64),
        region=region,
    )
    if return_metadata:
        return {
            "params": params,
            "source": "softguard_blend" if prediction_variant == "clip_softguard" and guard_weight > 1e-6 else "student",
            "support_score": support_score,
            "guard_reason": guard_reason,
            "clipped": clipped,
            "guard_weight": guard_weight,
            "cv_teacher_gap_closed": float(head.get("cv_teacher_gap_closed", 1.0)),
        }
    return target_vector_to_params(
        tool_name=tool_name,
        target=prediction,
        base_ts=np.asarray(base_ts, dtype=np.float64),
        region=region,
    )


def build_student_runtime_override(
    *,
    model: Dict[str, Any],
    plan: Dict[str, Any],
    base_ts: np.ndarray,
    prompt_text: str,
    prediction_variant: str = "v1",
) -> Dict[str, Any] | None:
    routing = dict(plan.get("volatility_routing") or {})
    if str(routing.get("final_subtype", "")) == "preview_non_monotonic":
        return None
    tool_name = str(plan.get("tool_name") or "")
    if tool_name not in SUPPORTED_STUDENT_TOOLS:
        return None
    region = list((plan.get("parameters") or {}).get("region") or [])
    if len(region) != 2:
        return None
    intent = dict(plan.get("intent") or {})
    predicted = predict_tool_conditioned_params(
        model=model,
        tool_name=tool_name,
        base_ts=np.asarray(base_ts, dtype=np.float64),
        region=region,
        prompt_text=prompt_text,
        intent=intent,
        prediction_variant=prediction_variant,
        return_metadata=True,
    )
    if predicted is None:
        return None
    override = dict(predicted["params"])
    override["region"] = region
    return {
        "parameters": override,
        "source": str(predicted.get("source", "student")),
        "guard_reason": predicted.get("guard_reason"),
        "support_score": float(predicted.get("support_score", 0.0)),
        "clipped": bool(predicted.get("clipped", False)),
        "guard_weight": float(predicted.get("guard_weight", 0.0)),
    }


def save_student_model(model: Dict[str, Any], path: str) -> None:
    Path(path).write_text(json.dumps(model, ensure_ascii=False, indent=2), encoding="utf-8")


def load_student_model(path: str) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _resolve_head_kind(model_kind: str, tool_name: str) -> str:
    if model_kind == "linear":
        return "linear"
    if model_kind == "quadratic":
        return "quadratic"
    if model_kind == "mixed_capacity":
        if tool_name in {"volatility_global_scale", "volatility_envelope_monotonic"}:
            return "mlp"
        if tool_name in {"spike_inject", "hybrid_up", "volatility_local_burst"}:
            return "quadratic"
        return "linear"
    raise ValueError(f"unsupported model_kind: {model_kind}")


def _expand_design_matrix(x_norm: np.ndarray, head_kind: str) -> np.ndarray:
    if head_kind == "linear":
        return np.column_stack([x_norm, np.ones(len(x_norm), dtype=np.float64)])
    if head_kind == "quadratic":
        return np.column_stack([x_norm, np.square(x_norm), np.ones(len(x_norm), dtype=np.float64)])
    raise ValueError(f"unsupported head_kind: {head_kind}")


def _expand_feature_vector(x_norm: np.ndarray, head_kind: str) -> np.ndarray:
    if head_kind == "linear":
        return np.concatenate([x_norm, np.ones(1, dtype=np.float64)])
    if head_kind == "quadratic":
        return np.concatenate([x_norm, np.square(x_norm), np.ones(1, dtype=np.float64)])
    raise ValueError(f"unsupported head_kind: {head_kind}")


def _fit_mlp_head(x_norm: np.ndarray, targets: np.ndarray, *, alpha: float, seed: int) -> Dict[str, Any]:
    rng = np.random.RandomState(seed)
    input_dim = x_norm.shape[1]
    output_dim = targets.shape[1]
    hidden_dim = int(min(16, max(6, input_dim // 3)))
    w1 = rng.normal(scale=0.12, size=(input_dim, hidden_dim))
    b1 = np.zeros(hidden_dim, dtype=np.float64)
    w2 = rng.normal(scale=0.12, size=(hidden_dim, output_dim))
    b2 = np.zeros(output_dim, dtype=np.float64)
    lr = 0.03
    steps = 600
    n = max(1, x_norm.shape[0])
    for _ in range(steps):
        hidden_pre = x_norm @ w1 + b1
        hidden = np.tanh(hidden_pre)
        pred = hidden @ w2 + b2
        diff = (pred - targets) / n
        grad_w2 = hidden.T @ diff + alpha * w2
        grad_b2 = np.sum(diff, axis=0)
        hidden_grad = (diff @ w2.T) * (1.0 - np.square(hidden))
        grad_w1 = x_norm.T @ hidden_grad + alpha * w1
        grad_b1 = np.sum(hidden_grad, axis=0)
        w1 -= lr * grad_w1
        b1 -= lr * grad_b1
        w2 -= lr * grad_w2
        b2 -= lr * grad_b2
    return {
        "head_kind": "mlp",
        "hidden_dim": hidden_dim,
        "w1": w1.tolist(),
        "b1": b1.tolist(),
        "w2": w2.tolist(),
        "b2": b2.tolist(),
    }


def _predict_mlp_head(x_norm: np.ndarray, head: Dict[str, Any]) -> np.ndarray:
    w1 = np.asarray(head.get("w1", []), dtype=np.float64)
    b1 = np.asarray(head.get("b1", []), dtype=np.float64)
    w2 = np.asarray(head.get("w2", []), dtype=np.float64)
    b2 = np.asarray(head.get("b2", []), dtype=np.float64)
    hidden = np.tanh(x_norm @ w1 + b1)
    return hidden @ w2 + b2


def _semantic_risk(
    *,
    tool_name: str,
    candidate_params: dict[str, Any],
    heuristic_params: dict[str, Any],
    intent: dict[str, Any],
    ratio_tol: float,
) -> float:
    direction = str(intent.get("direction", "") or "")
    if tool_name == "spike_inject":
        cand = abs(float(candidate_params.get("amplitude", 0.0)))
        heur = max(abs(float(heuristic_params.get("amplitude", 0.0))), 1e-6)
        mismatch = 1.0 if ((direction == "up" and float(candidate_params.get("amplitude", 0.0)) < 0.0) or (direction == "down" and float(candidate_params.get("amplitude", 0.0)) > 0.0)) else 0.0
        ratio_risk = max(0.0, cand / (heur * ratio_tol) - 1.0)
        return max(mismatch, min(1.0, ratio_risk))
    if tool_name == "step_shift":
        cand = abs(float(candidate_params.get("level_shift", 0.0)))
        heur = max(abs(float(heuristic_params.get("level_shift", 0.0))), 1e-6)
        mismatch = 1.0 if ((direction == "up" and float(candidate_params.get("level_shift", 0.0)) < 0.0) or (direction == "down" and float(candidate_params.get("level_shift", 0.0)) > 0.0)) else 0.0
        ratio_risk = max(0.0, cand / (heur * ratio_tol) - 1.0)
        return max(mismatch, min(1.0, ratio_risk))
    if tool_name in {"hybrid_up", "hybrid_down"}:
        cand = abs(float(candidate_params.get("math_shift", 0.0)))
        heur = max(abs(float(heuristic_params.get("math_shift", 0.0))), 1e-6)
        ratio_risk = max(0.0, cand / (heur * ratio_tol) - 1.0)
        return min(1.0, ratio_risk)
    if tool_name == "volatility_global_scale":
        cand = float(candidate_params.get("local_std_target_ratio", 1.0))
        heur = max(float(heuristic_params.get("local_std_target_ratio", 1.0)), 1e-6)
        return min(1.0, max(0.0, cand / (heur * ratio_tol) - 1.0))
    if tool_name == "volatility_local_burst":
        cand = float(candidate_params.get("burst_amplitude", 1.0))
        heur = max(float(heuristic_params.get("burst_amplitude", 1.0)), 1e-6)
        return min(1.0, max(0.0, cand / (heur * ratio_tol) - 1.0))
    if tool_name == "volatility_envelope_monotonic":
        cand = max(float(candidate_params.get("start_scale", 0.0)), float(candidate_params.get("end_scale", 0.0)))
        heur = max(float(heuristic_params.get("start_scale", 0.0)), float(heuristic_params.get("end_scale", 0.0)), 1e-6)
        return min(1.0, max(0.0, cand / (heur * ratio_tol) - 1.0))
    return 0.0


def _fit_head_from_xy(x_norm: np.ndarray, y_arr: np.ndarray, *, head_kind: str, alpha: float, seed: int) -> Dict[str, Any]:
    if head_kind in {"linear", "quadratic"}:
        x_aug = _expand_design_matrix(x_norm, head_kind)
        if len(y_arr) == 1:
            weights = np.zeros((x_aug.shape[1], y_arr.shape[1]), dtype=np.float64)
            weights[-1, :] = y_arr[0]
        else:
            reg = alpha * np.eye(x_aug.shape[1], dtype=np.float64)
            reg[-1, -1] = 0.0
            weights = np.linalg.solve(x_aug.T @ x_aug + reg, x_aug.T @ y_arr)
        return {"head_kind": head_kind, "weights": weights.tolist()}
    if head_kind == "mlp":
        return _fit_mlp_head(x_norm, y_arr, alpha=alpha, seed=seed)
    raise ValueError(f"unsupported head_kind: {head_kind}")


def _predict_head_from_xy(x_norm: np.ndarray, head: Dict[str, Any]) -> np.ndarray:
    head_kind = str(head.get("head_kind", "linear"))
    if head_kind in {"linear", "quadratic"}:
        x_aug = _expand_feature_vector(x_norm, head_kind)
        weights = np.asarray(head.get("weights", []), dtype=np.float64)
        return x_aug @ weights
    if head_kind == "mlp":
        return _predict_mlp_head(x_norm, head)
    raise ValueError(f"unsupported head_kind: {head_kind}")


def _estimate_head_cv_quality(
    *,
    tool_name: str,
    rows: list[dict[str, Any]],
    alpha: float,
    model_kind: str,
    seed: int,
) -> dict[str, float]:
    if len(rows) <= 1:
        return {
            "student_mae": 0.0,
            "heuristic_mae": 0.0,
            "teacher_gap_closed": 1.0,
            "student_better_rate_vs_heuristic": 1.0,
        }
    idxs = list(range(len(rows)))
    rng = random.Random(seed + len(rows) + len(tool_name))
    rng.shuffle(idxs)
    if len(rows) <= 8:
        folds = [[i] for i in idxs]
    else:
        fold_count = min(3, len(rows))
        folds = [idxs[i::fold_count] for i in range(fold_count)]
    student_errs: list[float] = []
    heuristic_errs: list[float] = []
    better_flags: list[float] = []
    head_kind = _resolve_head_kind(model_kind, tool_name)
    for fold_id, heldout in enumerate(folds):
        train = [rows[i] for i in idxs if i not in heldout]
        if not train:
            continue
        x_train = []
        y_train = []
        for row in train:
            base_ts = np.asarray(row["base_ts"], dtype=np.float64)
            region = list(row["region"])
            x_train.append(build_student_feature_vector(
                tool_name=tool_name,
                base_ts=base_ts,
                region=region,
                prompt_text=str(row.get("prompt_text", "")),
                intent=dict(row.get("intent") or {}),
            ))
            y_train.append(params_to_target_vector(
                tool_name=tool_name,
                params=dict(row["teacher_params"]),
                base_ts=base_ts,
                region=region,
            ))
        x_train_arr = np.asarray(x_train, dtype=np.float64)
        y_train_arr = np.asarray(y_train, dtype=np.float64)
        feat_mean = x_train_arr.mean(axis=0)
        feat_std = x_train_arr.std(axis=0)
        feat_std = np.where(feat_std < 1e-6, 1.0, feat_std)
        x_train_norm = (x_train_arr - feat_mean) / feat_std
        head = _fit_head_from_xy(x_train_norm, y_train_arr, head_kind=head_kind, alpha=alpha, seed=seed + fold_id)
        for i in heldout:
            row = rows[i]
            base_ts = np.asarray(row["base_ts"], dtype=np.float64)
            region = list(row["region"])
            teacher_target = params_to_target_vector(
                tool_name=tool_name,
                params=dict(row["teacher_params"]),
                base_ts=base_ts,
                region=region,
            )
            heuristic_target = params_to_target_vector(
                tool_name=tool_name,
                params=build_heuristic_params_for_tool(
                    tool_name=tool_name,
                    base_ts=base_ts,
                    region=region,
                    prompt_text=str(row.get("prompt_text", "")),
                ),
                base_ts=base_ts,
                region=region,
            )
            x = build_student_feature_vector(
                tool_name=tool_name,
                base_ts=base_ts,
                region=region,
                prompt_text=str(row.get("prompt_text", "")),
                intent=dict(row.get("intent") or {}),
            )
            x_norm = (x - feat_mean) / feat_std
            pred = _predict_head_from_xy(x_norm, head)
            student_err = float(np.mean(np.abs(pred - teacher_target)))
            heuristic_err = float(np.mean(np.abs(heuristic_target - teacher_target)))
            student_errs.append(student_err)
            heuristic_errs.append(heuristic_err)
            better_flags.append(1.0 if student_err < heuristic_err else 0.0)
    student_mae = float(np.mean(student_errs)) if student_errs else 0.0
    heuristic_mae = float(np.mean(heuristic_errs)) if heuristic_errs else 0.0
    gap = heuristic_mae - 0.0
    gap_closed = 0.0
    if heuristic_mae > 1e-8:
        gap_closed = float((heuristic_mae - student_mae) / heuristic_mae)
    return {
        "student_mae": student_mae,
        "heuristic_mae": heuristic_mae,
        "teacher_gap_closed": gap_closed,
        "student_better_rate_vs_heuristic": float(np.mean(better_flags)) if better_flags else 0.0,
    }
