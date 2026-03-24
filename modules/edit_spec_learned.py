from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

SPEC_KEYS = (
    "delta_level_z",
    "slope_ratio",
    "amp_ratio",
    "vol_ratio",
    "duration_ratio",
    "recovery_ratio",
    "floor_ratio",
)

SHAPE_VALUES = ("none", "hump", "step", "flatline", "irregular_noise", "plateau")
DIRECTION_VALUES = ("neutral", "up", "down")
DURATION_VALUES = ("none", "short", "medium", "long", "full_horizon")
STRENGTH_VALUES = ("none", "weak", "medium", "strong")
EFFECT_VALUES = ("none", "impulse", "level", "shutdown", "volatility")


def _fit_affine_1d(x: np.ndarray, y: np.ndarray, alpha: float = 1.0) -> tuple[float, float]:
    x_arr = np.asarray(x, dtype=np.float64).reshape(-1)
    y_arr = np.asarray(y, dtype=np.float64).reshape(-1)
    if x_arr.size == 0 or y_arr.size == 0:
        return 1.0, 0.0
    A = np.column_stack([x_arr, np.ones_like(x_arr)])
    reg = alpha * np.eye(2, dtype=np.float64)
    reg[-1, -1] = 0.0
    coeff = np.linalg.solve(A.T @ A + reg, A.T @ y_arr)
    return float(coeff[0]), float(coeff[1])


def _group_key(intent: dict[str, Any], model_type: str) -> str:
    effect = str(intent.get("effect_family", "none"))
    if model_type == "family_affine_edit_spec_calibrator":
        return effect
    if model_type == "family_duration_affine_edit_spec_calibrator":
        duration = str(intent.get("duration", "none"))
        return f"{effect}::{duration}"
    return "__global__"


def _safe_div(num: float, den: float, fallback: float = 0.0) -> float:
    if abs(den) < 1e-8:
        return float(fallback)
    return float(num / den)


def _estimate_slope(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size < 2:
        return 0.0
    x = np.arange(arr.size, dtype=np.float64)
    A = np.column_stack([x, np.ones_like(x)])
    slope, _ = np.linalg.lstsq(A, arr, rcond=None)[0]
    return float(slope)


def _stats(arr: np.ndarray) -> dict[str, float]:
    arr = np.asarray(arr, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
    }


def _text_intensity_modifier(text: str) -> float:
    normalized = text or ""
    modifier = 1.0
    if any(token in normalized for token in ("略微", "轻微", "稍微", "小幅", "温和")):
        modifier *= 0.75
    if any(token in normalized for token in ("明显", "显著", "大幅", "剧烈", "强烈")):
        modifier *= 1.25
    if any(token in normalized for token in ("迅速", "快速", "短时冲高", "短时下探")):
        modifier *= 1.1
    return float(modifier)


def _duration_ratio_from_bucket(bucket: str) -> float:
    return {
        "none": 0.0,
        "short": 0.55,
        "medium": 0.8,
        "long": 1.0,
        "full_horizon": 1.0,
    }.get(bucket, 0.8)


def _strength_scalar(strength: str) -> float:
    return {
        "none": 0.0,
        "weak": 0.8,
        "medium": 1.0,
        "strong": 1.45,
    }.get(strength, 1.0)


def _effective_region(region: list[int], horizon: int) -> tuple[int, int]:
    start = max(0, min(int(region[0]), horizon))
    end = max(start, min(int(region[1]), horizon))
    return start, end


def build_feature_vector(
    intent: dict[str, Any],
    region: list[int],
    history_ts: np.ndarray,
    base_forecast: np.ndarray,
    context_text: str,
) -> np.ndarray:
    history_arr = np.asarray(history_ts, dtype=np.float64)
    base_arr = np.asarray(base_forecast, dtype=np.float64)
    start, end = _effective_region(region, len(base_arr))
    local = base_arr[start:end]
    if local.size == 0:
        local = base_arr
    hist_stats = _stats(history_arr)
    base_stats = _stats(base_arr)
    local_stats = _stats(local)
    scale = max(hist_stats["std"], base_stats["std"], local_stats["std"], 1e-3)
    region_len = max(1, end - start)
    horizon = max(1, len(base_arr))
    local_trend = _estimate_slope(local)

    features: list[float] = [
        _text_intensity_modifier(context_text),
        _strength_scalar(str(intent.get("strength", "medium"))),
        _duration_ratio_from_bucket(str(intent.get("duration", "medium"))),
        start / horizon,
        end / horizon,
        region_len / horizon,
        hist_stats["mean"],
        hist_stats["std"],
        base_stats["mean"],
        base_stats["std"],
        local_stats["mean"],
        local_stats["std"],
        scale,
        local_trend,
        abs(local_trend) / max(scale, 1e-6),
        _safe_div(local_stats["mean"] - base_stats["mean"], scale, 0.0),
        _safe_div(local_stats["mean"] - hist_stats["mean"], scale, 0.0),
    ]

    def _one_hot(value: str, vocab: tuple[str, ...]) -> list[float]:
        return [1.0 if value == item else 0.0 for item in vocab]

    features.extend(_one_hot(str(intent.get("shape", "none")), SHAPE_VALUES))
    features.extend(_one_hot(str(intent.get("direction", "neutral")), DIRECTION_VALUES))
    features.extend(_one_hot(str(intent.get("duration", "none")), DURATION_VALUES))
    features.extend(_one_hot(str(intent.get("strength", "none")), STRENGTH_VALUES))
    features.extend(_one_hot(str(intent.get("effect_family", "none")), EFFECT_VALUES))
    return np.asarray(features, dtype=np.float64)


def _postprocess_spec(intent: dict[str, Any], values: np.ndarray) -> dict[str, Any]:
    shape = str(intent.get("shape", "none"))
    spec = {key: float(value) for key, value in zip(SPEC_KEYS, values.tolist())}
    spec["delta_level_z"] = float(np.clip(spec["delta_level_z"], -4.0, 4.0))
    spec["slope_ratio"] = float(np.clip(spec["slope_ratio"], 0.0, 4.0))
    spec["amp_ratio"] = float(np.clip(spec["amp_ratio"], 0.0, 4.0))
    spec["vol_ratio"] = float(np.clip(spec["vol_ratio"], 0.0, 4.0))
    spec["duration_ratio"] = float(np.clip(spec["duration_ratio"], 0.0, 1.0))
    spec["recovery_ratio"] = float(np.clip(spec["recovery_ratio"], 0.0, 1.0))
    spec["floor_ratio"] = float(np.clip(spec["floor_ratio"], 0.0, 4.0))

    if shape in {"step", "plateau", "hump"}:
        spec["vol_ratio"] = 0.0
        spec["floor_ratio"] = 0.0
    elif shape == "flatline":
        spec["slope_ratio"] = 0.0
        spec["vol_ratio"] = 0.0
        spec["amp_ratio"] = 0.0
        spec["delta_level_z"] = float(np.clip(spec["delta_level_z"], -4.0, 0.0))
    elif shape == "irregular_noise":
        spec["delta_level_z"] = 0.0
        spec["amp_ratio"] = 1.0
        spec["recovery_ratio"] = 0.0
        spec["floor_ratio"] = 0.0
    return spec


def _apply_group_affine(pred: np.ndarray, *, intent: dict[str, Any], model: dict[str, Any]) -> np.ndarray:
    model_type = str(model.get("model_type", "linear_edit_spec_calibrator"))
    if model_type == "linear_edit_spec_calibrator":
        return pred
    group_affine = model.get("group_affine") or {}
    params = group_affine.get(_group_key(intent, model_type))
    if not params:
        return pred
    adjusted = np.asarray(pred, dtype=np.float64).copy()
    scales = params.get("scales", {})
    biases = params.get("biases", {})
    for idx, key in enumerate(SPEC_KEYS):
        adjusted[idx] = float(scales.get(key, 1.0)) * adjusted[idx] + float(biases.get(key, 0.0))
    return adjusted


def fit_linear_calibrator(
    samples: list[dict[str, Any]],
    alpha: float = 1.0,
    model_kind: str = "linear",
) -> dict[str, Any]:
    X = []
    Y = []
    intents = []
    for sample in samples:
        if "intent" not in sample or "region" not in sample or "edit_spec_gt" not in sample:
            continue
        intents.append(dict(sample["intent"]))
        X.append(build_feature_vector(
            intent=sample["intent"],
            region=sample["region"],
            history_ts=np.asarray(sample["history_ts"], dtype=np.float64),
            base_forecast=np.asarray(sample["base_forecast"], dtype=np.float64),
            context_text=str(sample.get("context_text", "")),
        ))
        Y.append([float(sample["edit_spec_gt"].get(key, 0.0)) for key in SPEC_KEYS])
    if not X:
        raise ValueError("No valid samples for calibrator training")
    X_arr = np.asarray(X, dtype=np.float64)
    Y_arr = np.asarray(Y, dtype=np.float64)
    feat_mean = X_arr.mean(axis=0)
    feat_std = X_arr.std(axis=0)
    feat_std = np.where(feat_std < 1e-6, 1.0, feat_std)
    X_norm = (X_arr - feat_mean) / feat_std
    X_aug = np.column_stack([X_norm, np.ones(len(X_norm), dtype=np.float64)])
    reg = alpha * np.eye(X_aug.shape[1], dtype=np.float64)
    reg[-1, -1] = 0.0
    weights = np.linalg.solve(X_aug.T @ X_aug + reg, X_aug.T @ Y_arr)
    model_type = {
        "linear": "linear_edit_spec_calibrator",
        "family_affine": "family_affine_edit_spec_calibrator",
        "family_duration_affine": "family_duration_affine_edit_spec_calibrator",
    }.get(model_kind)
    if model_type is None:
        raise ValueError(f"Unsupported model_kind: {model_kind}")
    model = {
        "model_type": model_type,
        "alpha": float(alpha),
        "spec_keys": list(SPEC_KEYS),
        "feature_mean": feat_mean.tolist(),
        "feature_std": feat_std.tolist(),
        "weights": weights.tolist(),
    }
    if model_type != "linear_edit_spec_calibrator":
        base_pred = X_aug @ weights
        grouped: dict[str, dict[str, list[float]]] = {}
        for idx, intent in enumerate(intents):
            key = _group_key(intent, model_type)
            bucket = grouped.setdefault(key, {})
            for spec_idx, spec_key in enumerate(SPEC_KEYS):
                bucket.setdefault(f"{spec_key}_x", []).append(float(base_pred[idx, spec_idx]))
                bucket.setdefault(f"{spec_key}_y", []).append(float(Y_arr[idx, spec_idx]))
        group_affine: dict[str, dict[str, Any]] = {}
        for key, bucket in grouped.items():
            count = len(bucket.get(f"{SPEC_KEYS[0]}_x", []))
            if count < 4:
                continue
            scales: dict[str, float] = {}
            biases: dict[str, float] = {}
            for spec_key in SPEC_KEYS:
                scale, bias = _fit_affine_1d(
                    np.asarray(bucket[f"{spec_key}_x"], dtype=np.float64),
                    np.asarray(bucket[f"{spec_key}_y"], dtype=np.float64),
                    alpha=max(1e-6, alpha * 0.25),
                )
                scales[spec_key] = scale
                biases[spec_key] = bias
            group_affine[key] = {
                "count": count,
                "scales": scales,
                "biases": biases,
            }
        model["group_affine"] = group_affine
        model["grouping"] = "effect_family" if model_type == "family_affine_edit_spec_calibrator" else "effect_family_duration"
    return model


def save_model(model: dict[str, Any], path: str) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(model, ensure_ascii=False, indent=2), encoding="utf-8")


def load_model(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def predict_with_model(
    model: dict[str, Any],
    intent: dict[str, Any],
    region: list[int],
    history_ts: np.ndarray,
    base_forecast: np.ndarray,
    context_text: str,
) -> dict[str, Any]:
    x = build_feature_vector(intent, region, history_ts, base_forecast, context_text)
    mean_arr = np.asarray(model["feature_mean"], dtype=np.float64)
    std_arr = np.asarray(model["feature_std"], dtype=np.float64)
    weights = np.asarray(model["weights"], dtype=np.float64)
    x_norm = (x - mean_arr) / std_arr
    x_aug = np.concatenate([x_norm, np.array([1.0], dtype=np.float64)])
    pred = x_aug @ weights
    pred = _apply_group_affine(np.asarray(pred, dtype=np.float64), intent=intent, model=model)
    spec = _postprocess_spec(intent, np.asarray(pred, dtype=np.float64))
    spec["strategy"] = {
        "linear_edit_spec_calibrator": "learned_linear",
        "family_affine_edit_spec_calibrator": "learned_family_affine",
        "family_duration_affine_edit_spec_calibrator": "learned_family_duration_affine",
    }.get(str(model.get("model_type", "")), "learned_linear")
    return spec
