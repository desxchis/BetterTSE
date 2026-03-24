from __future__ import annotations

import json
from pathlib import Path
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
) -> Dict[str, Any]:
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
        X_aug = np.column_stack([X_norm, np.ones(len(X_norm), dtype=np.float64)])
        if len(rows) == 1:
            weights = np.zeros((X_aug.shape[1], Y_arr.shape[1]), dtype=np.float64)
            weights[-1, :] = Y_arr[0]
        else:
            reg = alpha * np.eye(X_aug.shape[1], dtype=np.float64)
            reg[-1, -1] = 0.0
            weights = np.linalg.solve(X_aug.T @ X_aug + reg, X_aug.T @ Y_arr)
        model["tool_heads"][tool_name] = {
            "feature_mean": feat_mean.tolist(),
            "feature_std": feat_std.tolist(),
            "weights": weights.tolist(),
            "target_keys": list(TOOL_PARAM_KEYS[tool_name]),
            "train_count": len(rows),
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
) -> Dict[str, Any] | None:
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
    weights = np.asarray(head.get("weights", []), dtype=np.float64)
    if features.size != feat_mean.size:
        raise ValueError(f"feature size mismatch for {tool_name}: got {features.size}, expected {feat_mean.size}")
    x_aug = np.concatenate([(features - feat_mean) / feat_std, np.ones(1, dtype=np.float64)])
    prediction = x_aug @ weights
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
    )
    if predicted is None:
        return None
    override = dict(predicted)
    override["region"] = region
    return override


def save_student_model(model: Dict[str, Any], path: str) -> None:
    Path(path).write_text(json.dumps(model, ensure_ascii=False, indent=2), encoding="utf-8")


def load_student_model(path: str) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
