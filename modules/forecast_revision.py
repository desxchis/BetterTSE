from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Tuple

import numpy as np


@dataclass
class ForecastRevisionSample:
    sample_id: str
    dataset_name: str
    history_ts: List[float]
    future_gt: List[float]
    base_forecast: List[float]
    revision_target: List[float]
    context_text: str
    forecast_horizon: int
    edit_mask_gt: List[int]
    delta_gt: List[float]
    revision_applicable_gt: bool
    edit_intent_gt: Dict[str, Any]
    effect_family_gt: str
    direction_gt: str
    shape_gt: str
    strength_bucket_gt: str
    duration_bucket_gt: str
    revision_operator_family: str
    revision_operator_params: Dict[str, Any]
    timestamp: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def summarize_stats(values: np.ndarray) -> Dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
    }


def infer_future_bucket(text: str) -> str:
    normalized = text or ""
    if any(token in normalized for token in ("整个预测窗口", "全窗口", "全时段", "full horizon", "full_horizon")):
        return "full_horizon"
    if any(token in normalized for token in ("前段", "开始阶段", "刚进入预测窗口", "最开始")):
        return "early_horizon"
    if any(token in normalized for token in ("中段", "中期", "一半左右")):
        return "mid_horizon"
    if any(token in normalized for token in ("后段", "临近结束", "末段", "快到窗口末尾")):
        return "late_horizon"
    return "mid_horizon"


def heuristic_revision_plan(context_text: str, horizon: int) -> Dict[str, Any]:
    normalized = context_text or ""
    if any(token in normalized for token in ("无新增影响", "暂无额外冲击", "维持原预测", "没有新的修正信号")):
        return {
            "revision_needed": False,
            "confidence": 0.9,
            "intent": {
                "effect_family": "none",
                "direction": "neutral",
                "shape": "none",
                "duration": "none",
                "strength": "none",
            },
            "localization": {
                "position_bucket": "none",
                "region": [0, 0],
            },
            "tool_name": "none",
        }
    bucket = infer_future_bucket(normalized)
    finance_down = any(token in normalized.lower() for token in ("bearish", "downgrade", "cut guidance", "crashing", "negative outlook"))
    finance_up = any(token in normalized.lower() for token in ("bullish", "upgrade", "positive outlook", "beats estimates", "raising guidance"))
    if any(token in normalized for token in ("flatline", "停摆", "中断", "低位运行", "极低水平")):
        shape = "flatline"
        effect_family = "shutdown"
        direction = "down"
        tool_name = "hybrid_down"
        strength = "strong"
    elif any(token in normalized for token in ("重新定价", "预期上修", "预期下修", "评级上调", "评级下调", "指引上修", "指引下修")) or finance_up or finance_down:
        shape = "step"
        effect_family = "level"
        direction = "down" if (finance_down or any(token in normalized for token in ("下修", "下调", "偏空", "利空"))) else "up"
        tool_name = "step_shift"
        strength = "medium"
    elif any(token in normalized for token in ("趋势抬升", "趋势走弱", "持续偏高", "持续偏低", "drift")):
        shape = "plateau"
        effect_family = "level"
        direction = "down" if any(token in normalized for token in ("走弱", "偏低", "下行")) else "up"
        tool_name = "hybrid_up" if direction == "up" else "hybrid_down"
        strength = "medium"
    elif any(token in normalized for token in ("切换", "跳变", "step", "新状态")):
        shape = "step"
        effect_family = "level"
        direction = "down" if any(token in normalized for token in ("低位", "更受限", "下降")) else "up"
        tool_name = "step_shift"
        strength = "strong"
    elif any(token in normalized for token in ("噪声", "失真", "无规律波动", "杂乱")):
        shape = "irregular_noise"
        effect_family = "volatility"
        direction = "neutral"
        tool_name = "volatility_increase"
        strength = "medium"
    elif any(token in normalized for token in ("高位维持", "持续偏高", "plateau", "维持高位")):
        shape = "plateau"
        effect_family = "level"
        direction = "up"
        tool_name = "hybrid_up"
        strength = "medium"
    else:
        shape = "hump"
        effect_family = "impulse"
        direction = "up" if any(token in normalized for token in ("抬升", "冲高", "上扬", "偏高")) else "down"
        tool_name = "spike_inject"
        strength = "medium"

    duration_bucket = "medium"
    if any(token in normalized for token in ("短时", "短暂", "很快")):
        duration_bucket = "short"
    elif any(token in normalized for token in ("持续", "一段时间", "较长时间")):
        duration_bucket = "long"

    region = localize_future_region(bucket, duration_bucket, horizon)
    confidence = 0.85 if any(token in normalized for token in ("预计", "将", "会", "可能")) else 0.75
    return {
        "revision_needed": True,
        "confidence": confidence,
        "intent": {
            "effect_family": effect_family,
            "direction": direction,
            "shape": shape,
            "duration": duration_bucket,
            "strength": strength,
        },
        "localization": {
            "position_bucket": bucket,
            "region": region,
        },
        "tool_name": tool_name,
    }


def localize_future_region(bucket: str, duration_bucket: str, horizon: int) -> List[int]:
    if bucket == "full_horizon":
        return [0, int(horizon)]
    if duration_bucket == "short":
        win = max(4, horizon // 6)
    elif duration_bucket == "long":
        win = max(8, horizon // 3)
    else:
        win = max(6, horizon // 4)

    if bucket == "early_horizon":
        start = 0
    elif bucket == "late_horizon":
        start = max(0, horizon - win)
    else:
        start = max(0, (horizon - win) // 2)
    end = min(horizon, start + win)
    return [int(start), int(end)]


def calibrate_revision(
    intent: Dict[str, Any],
    region: List[int],
    history_ts: np.ndarray,
    base_forecast: np.ndarray,
) -> Dict[str, Any]:
    history_stats = summarize_stats(history_ts)
    forecast_stats = summarize_stats(base_forecast)
    start, end = int(region[0]), int(region[1])
    local_forecast = np.asarray(base_forecast[max(0, start):max(0, end)], dtype=np.float64)
    local_stats = summarize_stats(local_forecast if local_forecast.size > 0 else base_forecast)
    strength = intent.get("strength", "medium")
    shape = intent.get("shape")
    region_len = max(1, int(region[1] - region[0]))
    scale = max(history_stats["std"], forecast_stats["std"], local_stats["std"], 1e-3)

    # Keep the calibrator close to benchmark operator generation for v1.
    amplitude_factor = {
        "hump": {"weak": 0.75, "medium": 1.0, "strong": 1.35},
        "step": {"weak": 0.8, "medium": 1.0, "strong": 1.6},
        "plateau": {"weak": 0.8, "medium": 1.0, "strong": 1.6},
        "flatline": {"weak": 0.8, "medium": 1.0, "strong": 1.6},
        "irregular_noise": {"weak": 0.25, "medium": 0.4, "strong": 0.55},
    }.get(shape, {"weak": 0.6, "medium": 1.0, "strong": 1.4})
    amplitude = float(scale * amplitude_factor.get(strength, amplitude_factor["medium"]))

    params = {
        "amplitude": amplitude,
        "duration": int(region_len),
        "onset_lag": 0,
        "recovery_rate": 0.35,
        "volatility_scale": 1.0,
    }
    if shape == "hump":
        params["recovery_rate"] = 0.45
    elif shape == "step":
        params["recovery_rate"] = 0.0
    elif shape == "plateau":
        params["recovery_rate"] = 0.35
    elif shape == "flatline":
        flat_scale = max(forecast_stats["std"], 1e-3)
        params["floor_value"] = float(forecast_stats["min"] - flat_scale * 1.2)
    elif shape == "irregular_noise":
        params["volatility_scale"] = {"weak": 1.15, "medium": 1.45, "strong": 1.8}.get(strength, 1.45)
    return params


def apply_revision_profile(
    base_forecast: np.ndarray,
    intent: Dict[str, Any],
    region: List[int],
    params: Dict[str, Any],
    seed: int = 7,
) -> Tuple[np.ndarray, np.ndarray]:
    edited = np.asarray(base_forecast, dtype=np.float64).copy()
    delta = np.zeros_like(edited)
    start, end = int(region[0]), int(region[1])
    start = max(0, min(start, len(edited)))
    end = max(start, min(end, len(edited)))
    if end <= start:
        return edited, delta

    length = end - start
    x = np.linspace(-1.0, 1.0, length)
    amplitude = float(params.get("amplitude", 0.0))
    direction = -1.0 if intent.get("direction") == "down" else 1.0
    shape = intent.get("shape")

    if shape in {"none", None}:
        return edited, delta
    if shape == "step":
        profile = np.ones(length, dtype=np.float64)
    elif shape == "plateau":
        ramp = np.minimum(np.linspace(0.0, 1.0, length), np.linspace(1.0, 0.0, length))
        profile = 0.6 + 0.4 * (ramp / max(np.max(ramp), 1e-6))
    elif shape == "flatline":
        floor_value = float(params.get("floor_value", np.min(edited[start:end]) - amplitude))
        delta[start:end] = floor_value - edited[start:end]
        edited[start:end] = floor_value
        return edited, delta
    elif shape == "irregular_noise":
        rng = np.random.default_rng(seed)
        noise = rng.normal(0.0, params.get("volatility_scale", 1.0) * amplitude, size=length)
        delta[start:end] = noise
        edited[start:end] = edited[start:end] + noise
        return edited, delta
    else:
        profile = np.exp(-4.0 * x * x)

    delta[start:end] = direction * amplitude * profile
    edited[start:end] = edited[start:end] + delta[start:end]
    return edited, delta


def compute_tiou(pred_region: List[int], gt_mask: np.ndarray) -> float:
    gt_indices = np.where(gt_mask > 0.5)[0]
    if gt_indices.size == 0:
        return 0.0
    gt_start, gt_end = int(gt_indices[0]), int(gt_indices[-1] + 1)
    pred_start, pred_end = int(pred_region[0]), int(pred_region[1])
    inter = max(0, min(pred_end, gt_end) - max(pred_start, gt_start))
    union = max(pred_end, gt_end) - min(pred_start, gt_start)
    return float(inter / union) if union > 0 else 0.0


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.abs(y_true) + np.abs(y_pred)
    denom = np.where(denom < 1e-8, 1.0, denom)
    return float(np.mean(2.0 * np.abs(y_pred - y_true) / denom))


def evaluate_revision_sample(
    base_forecast: np.ndarray,
    edited_forecast: np.ndarray,
    future_gt: np.ndarray,
    revision_target: np.ndarray,
    pred_region: List[int],
    gt_mask: np.ndarray,
) -> Dict[str, float]:
    base_forecast = np.asarray(base_forecast, dtype=np.float64)
    edited_forecast = np.asarray(edited_forecast, dtype=np.float64)
    future_gt = np.asarray(future_gt, dtype=np.float64)
    revision_target = np.asarray(revision_target, dtype=np.float64)
    mask = np.asarray(gt_mask, dtype=np.float64) > 0.5

    revision_gain = float(np.mean(np.abs(base_forecast - revision_target)) - np.mean(np.abs(edited_forecast - revision_target)))
    over_edit_rate = float(np.mean(np.abs(edited_forecast[~mask] - base_forecast[~mask]) > 1e-6)) if (~mask).any() else 0.0
    magnitude_error = float(np.mean(np.abs((edited_forecast - base_forecast)[mask] - (revision_target - base_forecast)[mask]))) if mask.any() else 0.0
    preservation = 1.0 - float(np.mean(np.abs(edited_forecast[~mask] - base_forecast[~mask]))) if (~mask).any() else 1.0

    return {
        "base_mae_vs_revision_target": float(np.mean(np.abs(base_forecast - revision_target))),
        "edited_mae_vs_revision_target": float(np.mean(np.abs(edited_forecast - revision_target))),
        "base_mae_vs_future_gt": float(np.mean(np.abs(base_forecast - future_gt))),
        "edited_mae_vs_future_gt": float(np.mean(np.abs(edited_forecast - future_gt))),
        "edited_mse_vs_revision_target": float(np.mean((edited_forecast - revision_target) ** 2)),
        "edited_smape_vs_revision_target": smape(revision_target, edited_forecast),
        "future_t_iou": compute_tiou(pred_region, gt_mask),
        "revision_gain": revision_gain,
        "magnitude_calibration_error": magnitude_error,
        "outside_region_preservation": preservation,
        "over_edit_rate": over_edit_rate,
    }


def compute_intent_alignment(plan: Dict[str, Any], sample: Dict[str, Any]) -> Dict[str, Any]:
    intent = plan.get("intent", {}) if isinstance(plan, dict) else {}
    revision_needed = bool(plan.get("revision_needed", False)) if isinstance(plan, dict) else False
    effect_match = intent.get("effect_family") == sample.get("effect_family_gt")
    direction_match = intent.get("direction") == sample.get("direction_gt")
    shape_match = intent.get("shape") == sample.get("shape_gt")
    duration_match = intent.get("duration") == sample.get("duration_bucket_gt")
    strength_match = intent.get("strength") == sample.get("strength_bucket_gt")
    revision_needed_match = revision_needed == bool(sample.get("revision_applicable_gt", True))
    matches = [
        effect_match,
        direction_match,
        shape_match,
        duration_match,
        strength_match,
    ]
    return {
        "revision_needed_match": revision_needed_match,
        "effect_family_match": effect_match,
        "direction_match": direction_match,
        "shape_match": shape_match,
        "duration_match": duration_match,
        "strength_match": strength_match,
        "intent_match_score": float(np.mean(matches)),
    }
