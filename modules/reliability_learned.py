from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from modules.edit_spec_learned import build_feature_vector


TOOL_VALUES = ("none", "oracle", "step_shift", "spike_inject", "volatility_increase", "hybrid_up", "hybrid_down")


def build_reliability_feature_vector(
    intent: dict[str, Any],
    region: list[int],
    history_ts: np.ndarray,
    base_forecast: np.ndarray,
    context_text: str,
    plan_confidence: float,
    tool_name: str,
    disagreement: dict[str, float],
) -> np.ndarray:
    base_features = build_feature_vector(
        intent=intent,
        region=region,
        history_ts=history_ts,
        base_forecast=base_forecast,
        context_text=context_text,
    )
    extras = [
        float(plan_confidence),
        float(disagreement.get("delta_gap", 0.0)),
        float(disagreement.get("duration_gap", 0.0)),
        float(disagreement.get("amp_gap", 0.0)),
        float(disagreement.get("slope_gap", 0.0)),
        float(disagreement.get("recovery_gap", 0.0)),
    ]
    tool_one_hot = [1.0 if tool_name == item else 0.0 for item in TOOL_VALUES]
    return np.asarray(list(base_features.tolist()) + extras + tool_one_hot, dtype=np.float64)


def fit_linear_reliability_model(samples: list[dict[str, Any]], alpha: float = 1.0) -> dict[str, Any]:
    X = []
    Y = []
    for sample in samples:
        if "intent" not in sample or "region" not in sample or "reliability_target" not in sample:
            continue
        X.append(build_reliability_feature_vector(
            intent=sample["intent"],
            region=sample["region"],
            history_ts=np.asarray(sample["history_ts"], dtype=np.float64),
            base_forecast=np.asarray(sample["base_forecast"], dtype=np.float64),
            context_text=str(sample.get("context_text", "")),
            plan_confidence=float(sample.get("plan_confidence", 0.75)),
            tool_name=str(sample.get("tool_name", "none")),
            disagreement=dict(sample.get("disagreement_features", {})),
        ))
        Y.append(float(sample["reliability_target"]))
    if not X:
        raise ValueError("No valid samples for reliability model training")

    X_arr = np.asarray(X, dtype=np.float64)
    y_arr = np.asarray(Y, dtype=np.float64)
    feat_mean = X_arr.mean(axis=0)
    feat_std = X_arr.std(axis=0)
    feat_std = np.where(feat_std < 1e-6, 1.0, feat_std)
    X_norm = (X_arr - feat_mean) / feat_std
    X_aug = np.column_stack([X_norm, np.ones(len(X_norm), dtype=np.float64)])
    reg = alpha * np.eye(X_aug.shape[1], dtype=np.float64)
    reg[-1, -1] = 0.0
    weights = np.linalg.solve(X_aug.T @ X_aug + reg, X_aug.T @ y_arr)
    return {
        "model_type": "linear_semantic_reliability",
        "alpha": float(alpha),
        "feature_mean": feat_mean.tolist(),
        "feature_std": feat_std.tolist(),
        "weights": weights.tolist(),
    }


def predict_reliability(
    model: dict[str, Any],
    *,
    intent: dict[str, Any],
    region: list[int],
    history_ts: np.ndarray,
    base_forecast: np.ndarray,
    context_text: str,
    plan_confidence: float,
    tool_name: str,
    disagreement: dict[str, float],
) -> float:
    x = build_reliability_feature_vector(
        intent=intent,
        region=region,
        history_ts=history_ts,
        base_forecast=base_forecast,
        context_text=context_text,
        plan_confidence=plan_confidence,
        tool_name=tool_name,
        disagreement=disagreement,
    )
    mean_arr = np.asarray(model["feature_mean"], dtype=np.float64)
    std_arr = np.asarray(model["feature_std"], dtype=np.float64)
    weights = np.asarray(model["weights"], dtype=np.float64)
    x_norm = (x - mean_arr) / std_arr
    x_aug = np.concatenate([x_norm, np.array([1.0], dtype=np.float64)])
    pred = float(x_aug @ weights)
    return float(np.clip(pred, 0.0, 1.0))


def save_model(model: dict[str, Any], path: str) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(model, ensure_ascii=False, indent=2), encoding="utf-8")


def load_model(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
