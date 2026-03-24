from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Tuple

import numpy as np

from modules.edit_spec_learned import load_model as load_edit_spec_model, predict_with_model
from modules.reliability_learned import load_model as load_reliability_bundle, predict_reliability


CALIBRATION_SPEC_KEYS = (
    "delta_level_z",
    "slope_ratio",
    "amp_ratio",
    "vol_ratio",
    "duration_ratio",
    "recovery_ratio",
    "floor_ratio",
)

TEACHER_SEARCH_SEED_STRATEGIES = (
    "rule_local_stats",
    "discrete_strength_table",
    "text_direct_numeric",
)

POSITION_BUCKET_VALUES = ("early", "mid", "late", "full", "none")
POSITION_BUCKET_ALIASES = {
    "early": "early",
    "mid": "mid",
    "middle": "mid",
    "late": "late",
    "full": "full",
    "none": "none",
    "early_horizon": "early",
    "mid_horizon": "mid",
    "middle_horizon": "mid",
    "late_horizon": "late",
    "full_horizon": "full",
}

REVISION_SAMPLE_SOURCE_OF_TRUTH_FIELDS = (
    "history_ts",
    "future_gt",
    "base_forecast",
    "revision_target",
    "edit_mask_gt",
    "edit_intent_gt",
    "revision_applicable_gt",
    "revision_operator_params",
)

REVISION_SAMPLE_DERIVED_CACHE_FIELDS = (
    "effect_family_gt",
    "direction_gt",
    "shape_gt",
    "strength_bucket_gt",
    "duration_bucket_gt",
    "revision_operator_family",
    "edit_spec_gt",
)


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
    projection_residual: List[float] | None = None
    revision_target_source: str = "unspecified"
    target_projection_metrics: Dict[str, Any] | None = None
    intent_struct: Dict[str, Any] | None = None
    edit_spec_gt: Dict[str, Any] | None = None
    timestamp: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def normalize_position_bucket(bucket: str, fallback: str = "mid") -> str:
    return POSITION_BUCKET_ALIASES.get(str(bucket or "").strip().lower(), fallback)


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


def _safe_div(num: float, den: float, fallback: float = 1.0) -> float:
    if abs(den) < 1e-8:
        return float(fallback)
    return float(num / den)


def _clamp(value: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, value)))


def _estimate_slope(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size < 2:
        return 0.0
    x = np.arange(arr.size, dtype=np.float64)
    A = np.column_stack([x, np.ones_like(x)])
    slope, _ = np.linalg.lstsq(A, arr, rcond=None)[0]
    return float(slope)


def _effective_region_bounds(region: List[int], horizon: int) -> Tuple[int, int]:
    start = max(0, min(int(region[0]), horizon))
    end = max(start, min(int(region[1]), horizon))
    return start, end


def _local_context(region: List[int], history_ts: np.ndarray, base_forecast: np.ndarray) -> Dict[str, Any]:
    forecast_arr = np.asarray(base_forecast, dtype=np.float64)
    horizon = len(forecast_arr)
    start, end = _effective_region_bounds(region, horizon)
    local_forecast = forecast_arr[start:end]
    if local_forecast.size == 0:
        local_forecast = forecast_arr
    history_stats = summarize_stats(history_ts)
    forecast_stats = summarize_stats(forecast_arr)
    local_stats = summarize_stats(local_forecast)
    scale = max(history_stats["std"], forecast_stats["std"], local_stats["std"], 1e-3)
    return {
        "start": start,
        "end": end,
        "region_len": max(1, end - start),
        "history_stats": history_stats,
        "forecast_stats": forecast_stats,
        "local_stats": local_stats,
        "local_forecast": local_forecast,
        "scale": float(scale),
    }


def _zero_edit_spec(strategy: str = "none") -> Dict[str, Any]:
    spec = {key: 0.0 for key in CALIBRATION_SPEC_KEYS}
    spec.update({"strategy": strategy})
    return spec


def _sanitize_edit_spec(intent: Dict[str, Any], spec: Dict[str, Any], strategy: str) -> Dict[str, Any]:
    shape = str(intent.get("shape", "none"))
    clean = {key: float(spec.get(key, 0.0)) for key in CALIBRATION_SPEC_KEYS}
    clean["delta_level_z"] = _clamp(clean["delta_level_z"], -4.0, 4.0)
    clean["slope_ratio"] = _clamp(clean["slope_ratio"], 0.0, 4.0)
    clean["amp_ratio"] = _clamp(clean["amp_ratio"], 0.0, 4.0)
    clean["vol_ratio"] = _clamp(clean["vol_ratio"], 0.0, 4.0)
    clean["duration_ratio"] = _clamp(clean["duration_ratio"], 0.0, 1.0)
    clean["recovery_ratio"] = _clamp(clean["recovery_ratio"], 0.0, 1.0)
    clean["floor_ratio"] = _clamp(clean["floor_ratio"], 0.0, 4.0)

    if shape in {"step", "plateau", "hump"}:
        clean["vol_ratio"] = 0.0
        clean["floor_ratio"] = 0.0
    elif shape == "flatline":
        clean["slope_ratio"] = 0.0
        clean["vol_ratio"] = 0.0
        clean["amp_ratio"] = 0.0
        clean["delta_level_z"] = _clamp(clean["delta_level_z"], -4.0, 0.0)
    elif shape == "irregular_noise":
        clean["delta_level_z"] = 0.0
        clean["amp_ratio"] = 1.0
        clean["recovery_ratio"] = 0.0
        clean["floor_ratio"] = 0.0
    clean["strategy"] = strategy
    return clean


def _duration_ratio_from_bucket(bucket: str) -> float:
    return {
        "none": 0.0,
        "short": 0.55,
        "medium": 0.8,
        "long": 1.0,
        "full_horizon": 1.0,
        "full": 1.0,
    }.get(bucket, 0.8)


def _strength_scalar(strength: str) -> float:
    return {
        "none": 0.0,
        "weak": 0.8,
        "medium": 1.0,
        "strong": 1.45,
    }.get(strength, 1.0)


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


def _activation_confidence(intent: Dict[str, Any], context_text: str, plan_confidence: float | None) -> float:
    shape = str(intent.get("shape", "none"))
    duration = str(intent.get("duration", "none"))
    strength = str(intent.get("strength", "none"))
    effect_family = str(intent.get("effect_family", "none"))
    direction = str(intent.get("direction", "neutral"))

    score = float(plan_confidence if plan_confidence is not None else 0.75)
    normalized = (context_text or "").lower()

    if effect_family == "level" and shape in {"step", "plateau"}:
        score += 0.12
    elif shape == "hump":
        score -= 0.10
    elif effect_family in {"volatility", "shutdown"}:
        score -= 0.06

    if duration in {"medium", "long", "full_horizon"}:
        score += 0.06
    elif duration == "short":
        score -= 0.08

    if strength in {"medium", "strong"}:
        score += 0.04
    elif strength == "weak":
        score -= 0.05

    if direction == "neutral":
        score -= 0.04

    if any(token in normalized for token in ("upgrade", "downgrade", "guidance", "repric", "reprice", "forecast", "outlook", "estimate")):
        score += 0.05
    if any(token in normalized for token in ("brief", "temporary", "intraday", "volatile", "noise", "uncertain")):
        score -= 0.06
    return _clamp(score, 0.0, 1.0)


def _guarded_learned_spec(
    intent: Dict[str, Any],
    region: List[int],
    history_ts: np.ndarray,
    base_forecast: np.ndarray,
    context_text: str,
    model_path: str,
) -> Dict[str, Any]:
    learned = predict_with_model(load_edit_spec_model(model_path), intent, region, history_ts, base_forecast, context_text)
    rule = predict_edit_spec(
        intent=intent,
        region=region,
        history_ts=history_ts,
        base_forecast=base_forecast,
        context_text=context_text,
        strategy="rule_local_stats",
    )

    shape = str(intent.get("shape", "none"))
    duration = str(intent.get("duration", "none"))
    strength = str(intent.get("strength", "none"))
    effect_family = str(intent.get("effect_family", "none"))

    structurally_simple = (
        effect_family == "level"
        and shape in {"step", "plateau"}
        and duration in {"medium", "long", "full_horizon"}
        and strength in {"medium", "strong"}
    )

    delta_gap = abs(float(learned.get("delta_level_z", 0.0)) - float(rule.get("delta_level_z", 0.0)))
    duration_gap = abs(float(learned.get("duration_ratio", 0.0)) - float(rule.get("duration_ratio", 0.0)))
    amp_gap = abs(float(learned.get("amp_ratio", 0.0)) - float(rule.get("amp_ratio", 0.0)))

    learned_abs_delta = abs(float(learned.get("delta_level_z", 0.0)))
    rule_abs_delta = abs(float(rule.get("delta_level_z", 0.0)))
    overconfident = learned_abs_delta > max(1.75 * max(rule_abs_delta, 0.35), rule_abs_delta + 0.8)

    if (not structurally_simple) or delta_gap > 0.9 or duration_gap > 0.25 or amp_gap > 0.75 or overconfident:
        fallback = dict(rule)
        fallback["strategy"] = "learned_rule_guarded:fallback_rule"
        return fallback

    blended = {}
    for key in CALIBRATION_SPEC_KEYS:
        blended[key] = float(0.65 * float(learned.get(key, 0.0)) + 0.35 * float(rule.get(key, 0.0)))
    blended["duration_ratio"] = _clamp(float(blended.get("duration_ratio", 0.0)), 0.0, 1.0)
    blended["recovery_ratio"] = _clamp(float(blended.get("recovery_ratio", 0.0)), 0.0, 1.0)
    blended["vol_ratio"] = max(0.0, float(blended.get("vol_ratio", 0.0)))
    blended["amp_ratio"] = max(0.0, float(blended.get("amp_ratio", 0.0)))
    blended["floor_ratio"] = max(0.0, float(blended.get("floor_ratio", 0.0)))
    blended["strategy"] = "learned_rule_guarded:blended"
    return blended


def _shrunk_learned_spec(
    intent: Dict[str, Any],
    region: List[int],
    history_ts: np.ndarray,
    base_forecast: np.ndarray,
    context_text: str,
    model_path: str,
) -> Dict[str, Any]:
    learned = predict_with_model(load_edit_spec_model(model_path), intent, region, history_ts, base_forecast, context_text)
    rule = predict_edit_spec(
        intent=intent,
        region=region,
        history_ts=history_ts,
        base_forecast=base_forecast,
        context_text=context_text,
        strategy="rule_local_stats",
    )

    shape = str(intent.get("shape", "none"))
    duration = str(intent.get("duration", "none"))
    strength = str(intent.get("strength", "none"))
    effect_family = str(intent.get("effect_family", "none"))

    if shape in {None, "none"} or effect_family == "none":
        fallback = dict(rule)
        fallback["strategy"] = "learned_rule_shrunk:rule_only"
        return fallback

    delta_gap = abs(float(learned.get("delta_level_z", 0.0)) - float(rule.get("delta_level_z", 0.0)))
    duration_gap = abs(float(learned.get("duration_ratio", 0.0)) - float(rule.get("duration_ratio", 0.0)))
    amp_gap = abs(float(learned.get("amp_ratio", 0.0)) - float(rule.get("amp_ratio", 0.0)))
    slope_gap = abs(float(learned.get("slope_ratio", 0.0)) - float(rule.get("slope_ratio", 0.0)))
    recovery_gap = abs(float(learned.get("recovery_ratio", 0.0)) - float(rule.get("recovery_ratio", 0.0)))

    rule_abs_delta = abs(float(rule.get("delta_level_z", 0.0)))
    learned_abs_delta = abs(float(learned.get("delta_level_z", 0.0)))
    relative_delta_gap = delta_gap / max(rule_abs_delta + 0.35, 0.75)
    disagreement = (
        0.45 * relative_delta_gap
        + 0.20 * (duration_gap / 0.35)
        + 0.20 * (amp_gap / 0.8)
        + 0.10 * (slope_gap / 1.25)
        + 0.05 * (recovery_gap / 0.35)
    )

    structurally_simple = effect_family == "level" and shape in {"step", "plateau"}
    if structurally_simple:
        max_weight = 0.55
    elif shape == "hump":
        max_weight = 0.22
    else:
        max_weight = 0.18

    if duration in {"short", "none"}:
        max_weight *= 0.55
    elif duration in {"medium", "long", "full_horizon"}:
        max_weight *= 1.0
    else:
        max_weight *= 0.7

    if strength == "weak":
        max_weight *= 0.7
    elif strength == "strong":
        max_weight *= 1.0
    else:
        max_weight *= 0.85

    if learned_abs_delta > max(rule_abs_delta + 1.25, 2.0 * max(rule_abs_delta, 0.25)):
        max_weight *= 0.35

    learned_weight = _clamp(max_weight / (1.0 + max(disagreement, 0.0)), 0.0, max_weight)

    blended = {}
    for key in CALIBRATION_SPEC_KEYS:
        rule_value = float(rule.get(key, 0.0))
        learned_value = float(learned.get(key, 0.0))
        blended[key] = float(rule_value + learned_weight * (learned_value - rule_value))

    blended["duration_ratio"] = _clamp(float(blended.get("duration_ratio", 0.0)), 0.0, 1.0)
    blended["recovery_ratio"] = _clamp(float(blended.get("recovery_ratio", 0.0)), 0.0, 1.0)
    blended["vol_ratio"] = max(0.0, float(blended.get("vol_ratio", 0.0)))
    blended["amp_ratio"] = max(0.0, float(blended.get("amp_ratio", 0.0)))
    blended["floor_ratio"] = max(0.0, float(blended.get("floor_ratio", 0.0)))
    blended["strategy"] = f"learned_rule_shrunk:w={learned_weight:.3f}"
    return blended


def _confidence_gated_learned_spec(
    intent: Dict[str, Any],
    region: List[int],
    history_ts: np.ndarray,
    base_forecast: np.ndarray,
    context_text: str,
    model_path: str,
    plan_confidence: float | None,
) -> Dict[str, Any]:
    learned = predict_with_model(load_edit_spec_model(model_path), intent, region, history_ts, base_forecast, context_text)
    rule = predict_edit_spec(
        intent=intent,
        region=region,
        history_ts=history_ts,
        base_forecast=base_forecast,
        context_text=context_text,
        strategy="rule_local_stats",
    )

    activation = _activation_confidence(intent, context_text, plan_confidence)
    if activation < 0.82:
        fallback = dict(rule)
        fallback["strategy"] = f"learned_confidence_gated:rule@{activation:.3f}"
        return fallback

    max_weight = 0.25 + 0.55 * ((activation - 0.82) / 0.18)
    max_weight = _clamp(max_weight, 0.25, 0.80)

    blended = {}
    for key in CALIBRATION_SPEC_KEYS:
        rule_value = float(rule.get(key, 0.0))
        learned_value = float(learned.get(key, 0.0))
        blended[key] = float(rule_value + max_weight * (learned_value - rule_value))

    blended["duration_ratio"] = _clamp(float(blended.get("duration_ratio", 0.0)), 0.0, 1.0)
    blended["recovery_ratio"] = _clamp(float(blended.get("recovery_ratio", 0.0)), 0.0, 1.0)
    blended["vol_ratio"] = max(0.0, float(blended.get("vol_ratio", 0.0)))
    blended["amp_ratio"] = max(0.0, float(blended.get("amp_ratio", 0.0)))
    blended["floor_ratio"] = max(0.0, float(blended.get("floor_ratio", 0.0)))
    blended["strategy"] = f"learned_confidence_gated:blend@{activation:.3f}:w={max_weight:.3f}"
    return blended


def _disagreement_features(rule: Dict[str, Any], learned: Dict[str, Any]) -> Dict[str, float]:
    return {
        "delta_gap": abs(float(learned.get("delta_level_z", 0.0)) - float(rule.get("delta_level_z", 0.0))),
        "duration_gap": abs(float(learned.get("duration_ratio", 0.0)) - float(rule.get("duration_ratio", 0.0))),
        "amp_gap": abs(float(learned.get("amp_ratio", 0.0)) - float(rule.get("amp_ratio", 0.0))),
        "slope_gap": abs(float(learned.get("slope_ratio", 0.0)) - float(rule.get("slope_ratio", 0.0))),
        "recovery_gap": abs(float(learned.get("recovery_ratio", 0.0)) - float(rule.get("recovery_ratio", 0.0))),
    }


def _reliability_gated_learned_spec(
    intent: Dict[str, Any],
    region: List[int],
    history_ts: np.ndarray,
    base_forecast: np.ndarray,
    context_text: str,
    model_path: str,
    plan_confidence: float | None,
    sample: Dict[str, Any] | None,
) -> Dict[str, Any]:
    bundle = load_reliability_bundle(model_path)
    learned_model_path = bundle.get("learned_calibrator_path")
    if not learned_model_path:
        raise ValueError("learned_reliability_gate bundle missing learned_calibrator_path")

    learned = predict_with_model(load_edit_spec_model(learned_model_path), intent, region, history_ts, base_forecast, context_text)
    rule = predict_edit_spec(
        intent=intent,
        region=region,
        history_ts=history_ts,
        base_forecast=base_forecast,
        context_text=context_text,
        strategy="rule_local_stats",
    )
    disagreement = _disagreement_features(rule, learned)
    tool_name = "none"
    if sample is not None:
        tool_name = str(sample.get("tool_name", "none"))
    reliability = predict_reliability(
        bundle["reliability_model"],
        intent=intent,
        region=region,
        history_ts=history_ts,
        base_forecast=base_forecast,
        context_text=context_text,
        plan_confidence=float(plan_confidence if plan_confidence is not None else 0.75),
        tool_name=tool_name,
        disagreement=disagreement,
    )
    threshold = float(bundle.get("threshold", 0.75))
    if reliability < threshold:
        fallback = dict(rule)
        fallback["strategy"] = f"learned_reliability_gated:rule@{reliability:.3f}"
        return fallback
    kept = dict(learned)
    kept["strategy"] = f"learned_reliability_gated:learned@{reliability:.3f}"
    return kept


def _teacher_candidate_scales(shape: str) -> List[Dict[str, float]]:
    if shape == "irregular_noise":
        return [
            {"duration_ratio": ds, "vol_ratio": vs}
            for ds in (0.8, 1.0, 1.2)
            for vs in (0.8, 1.0, 1.25, 1.5)
        ]
    if shape == "flatline":
        return [
            {"delta_level_z": ds, "duration_ratio": dur, "floor_ratio": fs}
            for ds in (0.8, 1.0, 1.2, 1.4)
            for dur in (0.8, 1.0, 1.2)
            for fs in (0.8, 1.0, 1.25)
        ]
    if shape == "hump":
        return [
            {"delta_level_z": ds, "duration_ratio": dur, "amp_ratio": amp, "recovery_ratio": rec}
            for ds in (0.75, 1.0, 1.2, 1.4)
            for dur in (0.8, 1.0, 1.2)
            for amp in (0.85, 1.0, 1.15)
            for rec in (0.8, 1.0, 1.2)
        ]
    if shape == "step":
        return [
            {"delta_level_z": ds, "duration_ratio": dur, "amp_ratio": amp, "recovery_ratio": rec}
            for ds in (0.75, 1.0, 1.2, 1.4)
            for dur in (0.8, 1.0, 1.2)
            for amp in (0.9, 1.0, 1.1)
            for rec in (0.0, 0.35, 0.6)
        ]
    return [
        {"delta_level_z": ds, "duration_ratio": dur, "amp_ratio": amp, "recovery_ratio": rec}
        for ds in (0.75, 1.0, 1.2, 1.4)
        for dur in (0.8, 1.0, 1.2)
        for amp in (0.9, 1.0, 1.15)
        for rec in (0.8, 1.0, 1.2)
    ]


def _apply_teacher_candidate(base_spec: Dict[str, Any], scales: Dict[str, float]) -> Dict[str, Any]:
    candidate = {key: float(base_spec.get(key, 0.0)) for key in CALIBRATION_SPEC_KEYS}
    for key, scale in scales.items():
        if key == "duration_ratio" or key == "recovery_ratio":
            candidate[key] = float(candidate.get(key, 0.0)) * float(scale)
        else:
            candidate[key] = float(candidate.get(key, 0.0)) * float(scale)
    return candidate


def search_teacher_edit_spec(
    *,
    intent: Dict[str, Any],
    region: List[int],
    history_ts: np.ndarray,
    base_forecast: np.ndarray,
    revision_target: np.ndarray,
    future_gt: np.ndarray,
    gt_mask: np.ndarray,
    context_text: str = "",
    seed_strategies: Tuple[str, ...] = TEACHER_SEARCH_SEED_STRATEGIES,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    shape = str(intent.get("shape", "none"))
    if shape in {"none", ""}:
        return _zero_edit_spec(strategy="teacher_search"), {"candidate_count": 0, "best_objective": 0.0}

    seeds: List[Tuple[str, Dict[str, Any]]] = []
    for strategy in seed_strategies:
        spec = predict_edit_spec(
            intent=intent,
            region=region,
            history_ts=history_ts,
            base_forecast=base_forecast,
            context_text=context_text,
            strategy=strategy,
        )
        seeds.append((strategy, spec))

    best_spec: Dict[str, Any] | None = None
    best_metrics: Dict[str, Any] | None = None
    best_objective: float | None = None
    candidate_count = 0

    for seed_name, seed_spec in seeds:
        sanitized_seed = _sanitize_edit_spec(intent, seed_spec, strategy=f"teacher_seed:{seed_name}")
        for scales in _teacher_candidate_scales(shape):
            candidate_count += 1
            raw_candidate = _apply_teacher_candidate(sanitized_seed, scales)
            candidate = _sanitize_edit_spec(intent, raw_candidate, strategy=f"teacher_search:{seed_name}")
            params = project_edit_spec_to_params(
                edit_spec=candidate,
                intent=intent,
                region=region,
                history_ts=history_ts,
                base_forecast=base_forecast,
            )
            edited, _ = apply_revision_profile(base_forecast, intent, region, params)
            metrics = evaluate_revision_sample(
                base_forecast=base_forecast,
                edited_forecast=edited,
                future_gt=future_gt,
                revision_target=revision_target,
                pred_region=region,
                gt_mask=gt_mask,
            )
            objective = (
                float(metrics["edited_mae_vs_revision_target"])
                + 0.12 * max(0.0, 1.0 - float(metrics["outside_region_preservation"]))
                + 0.08 * float(metrics["over_edit_rate"])
                + 0.04 * float(metrics["edited_mae_vs_future_gt"])
            )
            if best_objective is None or objective < best_objective:
                best_objective = objective
                best_spec = candidate
                best_metrics = {
                    "seed_strategy": seed_name,
                    "scales": dict(scales),
                    "objective": float(objective),
                    **metrics,
                }

    if best_spec is None or best_metrics is None or best_objective is None:
        fallback = predict_edit_spec(
            intent=intent,
            region=region,
            history_ts=history_ts,
            base_forecast=base_forecast,
            context_text=context_text,
            strategy="rule_local_stats",
        )
        return _sanitize_edit_spec(intent, fallback, strategy="teacher_search:fallback_rule"), {
            "candidate_count": candidate_count,
            "best_objective": None,
        }
    best_spec["strategy"] = f"teacher_search:{best_metrics['seed_strategy']}"
    return best_spec, {
        "candidate_count": candidate_count,
        "best_objective": float(best_objective),
        "best_metrics": best_metrics,
    }


def infer_future_bucket(text: str) -> str:
    normalized = text or ""
    if any(token in normalized for token in ("整个预测窗口", "全窗口", "全时段", "full horizon", "full_horizon")):
        return "full"
    if any(token in normalized for token in ("前段", "开始阶段", "刚进入预测窗口", "最开始", "short term", "short-term", "next 1-3 months", "in the short term")):
        return "early"
    if any(token in normalized for token in ("中段", "中期", "一半左右", "mid term", "mid-term")):
        return "mid"
    if any(token in normalized for token in ("后段", "临近结束", "末段", "快到窗口末尾", "late term", "late-term")):
        return "late"
    return "mid"


def heuristic_revision_plan(context_text: str, horizon: int) -> Dict[str, Any]:
    normalized = context_text or ""
    prompt_section = normalized.split("\n\n")[-1].strip() if "\n\n" in normalized else normalized
    lowered = normalized.lower()
    prompt_lowered = prompt_section.lower()

    structured_direction = None
    structured_shape = None
    structured_duration = None
    structured_strength = None
    structured_bucket = None
    if "direction=up" in lowered:
        structured_direction = "up"
    elif "direction=down" in lowered:
        structured_direction = "down"
    elif "direction=neutral" in lowered:
        structured_direction = "neutral"
    if "shape=plateau" in lowered:
        structured_shape = "plateau"
    elif "shape=step" in lowered:
        structured_shape = "step"
    elif "shape=hump" in lowered:
        structured_shape = "hump"
    elif "shape=flatline" in lowered:
        structured_shape = "flatline"
    elif "shape=irregular_noise" in lowered:
        structured_shape = "irregular_noise"
    elif "shape=none" in lowered:
        structured_shape = "none"
    if "duration=short" in lowered:
        structured_duration = "short"
    elif "duration=medium" in lowered:
        structured_duration = "medium"
    elif "duration=long" in lowered:
        structured_duration = "long"
    elif "duration=none" in lowered:
        structured_duration = "none"
    if "strength=weak" in lowered:
        structured_strength = "weak"
    elif "strength=medium" in lowered:
        structured_strength = "medium"
    elif "strength=strong" in lowered:
        structured_strength = "strong"
    elif "strength=none" in lowered:
        structured_strength = "none"
    if "bucket=full_horizon" in lowered or "bucket=full" in lowered:
        structured_bucket = "full"
    elif "bucket=early_horizon" in lowered or "bucket=early" in lowered:
        structured_bucket = "early"
    elif "bucket=mid_horizon" in lowered or "bucket=middle" in lowered or "bucket=mid" in lowered:
        structured_bucket = "mid"
    elif "bucket=late_horizon" in lowered or "bucket=late" in lowered:
        structured_bucket = "late"
    elif "bucket=none" in lowered:
        structured_bucket = "none"

    if structured_shape is not None and structured_direction is not None:
        if structured_shape == "none" or structured_direction == "neutral":
            return {
                "revision_needed": False,
                "confidence": 0.95,
                "intent": {
                    "effect_family": "none",
                    "direction": "neutral",
                    "shape": "none",
                    "duration": structured_duration or "none",
                    "strength": structured_strength or "none",
                },
                "localization": {
                    "position_bucket": structured_bucket or "none",
                    "region": [0, 0],
                },
                "tool_name": "none",
            }
        if structured_shape == "flatline":
            effect_family = "shutdown"
            tool_name = "hybrid_down"
        elif structured_shape == "irregular_noise":
            effect_family = "volatility"
            tool_name = "volatility_increase"
        elif structured_shape == "hump":
            effect_family = "impulse"
            tool_name = "spike_inject"
        elif structured_shape == "step":
            effect_family = "level"
            tool_name = "step_shift"
        else:
            effect_family = "level"
            tool_name = "hybrid_down" if structured_direction == "down" else "hybrid_up"
        bucket = normalize_position_bucket(structured_bucket or infer_future_bucket(normalized))
        duration_bucket = structured_duration or "medium"
        return {
            "revision_needed": True,
            "confidence": 0.95,
            "intent": {
                "effect_family": effect_family,
                "direction": structured_direction,
                "shape": structured_shape,
                "duration": duration_bucket,
                "strength": structured_strength or "medium",
            },
            "localization": {
                "position_bucket": normalize_position_bucket(bucket, fallback="mid"),
                "region": localize_future_region(bucket, duration_bucket, horizon),
            },
            "tool_name": tool_name,
        }

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
    bucket = normalize_position_bucket(infer_future_bucket(prompt_section))
    finance_down = any(token in prompt_lowered for token in ("bearish", "downgrade", "cut guidance", "negative outlook"))
    finance_up = any(token in prompt_lowered for token in ("bullish", "upgrade", "positive outlook", "beats estimates", "raising guidance"))
    if any(token in prompt_lowered for token in ("no meaningful change", "no major change", "remain unchanged", "little change expected")):
        return {
            "revision_needed": False,
            "confidence": 0.82,
            "intent": {
                "effect_family": "none",
                "direction": "neutral",
                "shape": "none",
                "duration": "none",
                "strength": "none",
            },
            "localization": {"position_bucket": "none", "region": [0, 0]},
            "tool_name": "none",
        }
    if any(token in normalized for token in ("flatline", "停摆", "中断", "低位运行", "极低水平")):
        shape = "flatline"
        effect_family = "shutdown"
        direction = "down"
        tool_name = "hybrid_down"
        strength = "strong"
    elif any(token in prompt_lowered for token in (
        "remain stable", "remain relatively stable", "stabilize", "stabilise",
        "remain stable or decrease slightly", "remain stable or increase slightly",
        "rebound or stabilize", "fluctuate around the current level", "stable to slightly lower",
        "stable to slightly higher", "slow pace", "modest pace", "continue to hover around",
        "decrease or remain stable", "increase or remain stable"
    )):
        shape = "plateau"
        effect_family = "level"
        if any(token in prompt_lowered for token in ("decrease slightly", "slight decrease", "decrease or remain stable", "decline", "fall", "widen", "lower", "downward")):
            direction = "down"
        elif any(token in prompt_lowered for token in ("increase slightly", "slight increase", "increase or remain stable", "increase", "rise", "grow", "narrow", "higher", "upward")):
            direction = "up"
        else:
            return {
                "revision_needed": False,
                "confidence": 0.7,
                "intent": {
                    "effect_family": "none",
                    "direction": "neutral",
                    "shape": "none",
                    "duration": "none",
                    "strength": "none",
                },
                "localization": {"position_bucket": "none", "region": [0, 0]},
                "tool_name": "none",
            }
        tool_name = "hybrid_down" if direction == "down" else "hybrid_up"
        strength = "weak"
    elif any(token in normalized for token in ("重新定价", "预期上修", "预期下修", "评级上调", "评级下调", "指引上修", "指引下修")) or finance_up or finance_down:
        shape = "step"
        effect_family = "level"
        direction = "down" if (finance_down or any(token in normalized for token in ("下修", "下调", "偏空", "利空"))) else "up"
        tool_name = "step_shift"
        strength = "medium"
    elif any(token in prompt_lowered for token in (
        "continue to increase", "continue to decrease", "continue to decline", "continue to rise",
        "continue to widen", "continue to narrow", "remain wide", "remain positive", "overall trend will remain positive",
        "likely that gasoline prices will continue", "likely that vehicle miles traveled will continue",
        "trade deficit will continue", "prices may rise", "prices may fall", "likely increase", "likely decrease",
        "continued increase", "continued decrease", "continue to grow", "continue growing", "continue to expand",
        "will continue to grow", "will continue to increase", "will continue to decline", "will continue to decrease",
        "likely to continue to increase", "likely to continue to decrease", "likely to remain above", "likely to remain below",
        "travel will continue", "vmt will continue", "gasoline prices are likely to continue",
        "sustained growth", "growth trend continues", "reach a new high", "vehicle miles traveled will reach",
        "vmt growth", "population growth and urbanization", "estimated increase", "continue to narrow"
    )):
        shape = "plateau"
        effect_family = "level"
        if any(token in prompt_lowered for token in ("decrease", "decline", "fall", "widen", "narrow", "lower", "below")):
            direction = "down"
        else:
            direction = "up"
        tool_name = "hybrid_down" if direction == "down" else "hybrid_up"
        strength = "medium"
    elif any(token in prompt_lowered for token in (
        "begin to decrease", "begin to increase", "start to decrease", "start to increase",
        "may decrease", "may increase", "is expected to decrease", "is expected to increase"
    )):
        shape = "plateau"
        effect_family = "level"
        direction = "down" if any(token in prompt_lowered for token in ("decrease", "lower", "narrow")) else "up"
        tool_name = "hybrid_down" if direction == "down" else "hybrid_up"
        strength = "weak"
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
    elif any(token in normalized for token in ("噪声", "失真", "无规律波动", "杂乱")) or any(token in prompt_lowered for token in ("volatile", "volatility", "erratic", "high variability")):
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
    if any(token in normalized for token in ("短时", "短暂", "很快")) or any(token in prompt_lowered for token in ("short term", "short-term", "next 1-3 months", "in the short term")):
        duration_bucket = "short"
    elif any(token in normalized for token in ("持续", "一段时间", "较长时间")) or any(token in prompt_lowered for token in ("long term", "long-term", "next 4-18 months")):
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
            "position_bucket": normalize_position_bucket(bucket, fallback="mid"),
            "region": region,
        },
        "tool_name": tool_name,
    }


def localize_future_region(bucket: str, duration_bucket: str, horizon: int) -> List[int]:
    bucket = normalize_position_bucket(bucket)
    if bucket == "full":
        return [0, int(horizon)]
    if duration_bucket == "short":
        win = max(4, horizon // 6)
    elif duration_bucket == "long":
        win = max(8, horizon // 3)
    else:
        win = max(6, horizon // 4)

    if bucket == "early":
        start = 0
    elif bucket == "late":
        start = max(0, horizon - win)
    else:
        start = max(0, (horizon - win) // 2)
    end = min(horizon, start + win)
    return [int(start), int(end)]


def extract_gt_edit_spec(
    sample: Dict[str, Any],
    history_ts: np.ndarray | None = None,
    base_forecast: np.ndarray | None = None,
) -> Dict[str, Any]:
    if sample.get("edit_spec_gt"):
        spec = dict(sample["edit_spec_gt"])
        spec.setdefault("strategy", "gt")
        return spec

    history_arr = np.asarray(history_ts if history_ts is not None else sample["history_ts"], dtype=np.float64)
    base_arr = np.asarray(base_forecast if base_forecast is not None else sample["base_forecast"], dtype=np.float64)
    target_arr = np.asarray(sample["revision_target"], dtype=np.float64)
    mask = np.asarray(sample["edit_mask_gt"], dtype=np.float64) > 0.5
    idx = np.where(mask)[0]
    if idx.size == 0:
        return _zero_edit_spec(strategy="gt")

    region = [int(idx[0]), int(idx[-1] + 1)]
    ctx = _local_context(region, history_arr, base_arr)
    start, end = ctx["start"], ctx["end"]
    scale = ctx["scale"]
    region_len = ctx["region_len"]
    shape = str(sample.get("shape_gt", "none"))

    local_base = base_arr[start:end]
    local_target = target_arr[start:end]
    local_delta = local_target - local_base
    params = (sample.get("revision_operator_params") or {}).get("params", {})

    base_slope = _estimate_slope(local_base)
    target_slope = _estimate_slope(local_target)
    tail_len = max(2, region_len // 3)
    recovery_slope = _estimate_slope(local_target[-tail_len:]) if local_target.size >= tail_len else 0.0
    amplitude = abs(float(params.get("amplitude", np.max(np.abs(local_delta)) if local_delta.size else 0.0)))

    spec = _zero_edit_spec(strategy="gt")
    spec["delta_level_z"] = _clamp(float(np.mean(local_delta) / scale), -4.0, 4.0)
    spec["duration_ratio"] = _clamp(_safe_div(float(params.get("duration", region_len)), float(region_len), fallback=1.0), 0.0, 1.0)
    spec["recovery_ratio"] = _clamp(
        float(params.get("recovery_rate", max(0.0, min(1.0, -recovery_slope / scale)))) if local_target.size else 0.0,
        0.0,
        1.0,
    )

    if shape in {"hump", "step", "plateau"}:
        spec["amp_ratio"] = _clamp(_safe_div(amplitude, scale, fallback=1.0), 0.0, 4.0)
        spec["slope_ratio"] = _clamp(abs(_safe_div(target_slope, base_slope, fallback=1.0)), 0.0, 4.0)
    elif shape == "flatline":
        floor_value = float(params.get("floor_value", np.min(local_target) if local_target.size else ctx["local_stats"]["mean"]))
        spec["delta_level_z"] = _clamp(float(np.min(local_delta) / scale) if local_delta.size else spec["delta_level_z"], -4.0, 0.0)
        spec["floor_ratio"] = _clamp(float((ctx["local_stats"]["mean"] - floor_value) / scale), 0.0, 4.0)
        spec["amp_ratio"] = 0.0
        spec["slope_ratio"] = 0.0
        spec["vol_ratio"] = 0.0
    elif shape == "irregular_noise":
        spec["amp_ratio"] = 1.0
        spec["vol_ratio"] = _clamp(float(params.get("volatility_scale", 1.0)), 0.0, 4.0)
        spec["recovery_ratio"] = 0.0
    return spec


def predict_edit_spec(
    intent: Dict[str, Any],
    region: List[int],
    history_ts: np.ndarray,
    base_forecast: np.ndarray,
    context_text: str = "",
    strategy: str = "rule_local_stats",
    sample: Dict[str, Any] | None = None,
    model_path: str | None = None,
    plan_confidence: float | None = None,
) -> Dict[str, Any]:
    shape = intent.get("shape")
    if shape in {None, "none"}:
        return _zero_edit_spec(strategy=strategy)
    if strategy == "teacher_distilled_linear":
        strategy = "learned_linear"
    elif strategy == "teacher_distilled_shrunk":
        strategy = "learned_rule_shrunk"
    if strategy == "oracle_from_sample":
        if sample is None:
            raise ValueError("sample is required for oracle_from_sample strategy")
        return extract_gt_edit_spec(sample, history_ts=history_ts, base_forecast=base_forecast)
    if strategy == "teacher_search_oracle":
        if sample is None:
            raise ValueError("sample is required for teacher_search_oracle strategy")
        spec, _ = search_teacher_edit_spec(
            intent=intent,
            region=region,
            history_ts=history_ts,
            base_forecast=base_forecast,
            revision_target=np.asarray(sample["revision_target"], dtype=np.float64),
            future_gt=np.asarray(sample["future_gt"], dtype=np.float64),
            gt_mask=np.asarray(sample["edit_mask_gt"], dtype=np.float64),
            context_text=context_text,
        )
        return spec
    if strategy == "learned_linear":
        if not model_path:
            raise ValueError("model_path is required for learned_linear strategy")
        model = load_edit_spec_model(model_path)
        return predict_with_model(model, intent, region, history_ts, base_forecast, context_text)
    if strategy == "learned_rule_guarded":
        if not model_path:
            raise ValueError("model_path is required for learned_rule_guarded strategy")
        return _guarded_learned_spec(intent, region, history_ts, base_forecast, context_text, model_path)
    if strategy == "learned_rule_shrunk":
        if not model_path:
            raise ValueError("model_path is required for learned_rule_shrunk strategy")
        return _shrunk_learned_spec(intent, region, history_ts, base_forecast, context_text, model_path)
    if strategy == "learned_confidence_gated":
        if not model_path:
            raise ValueError("model_path is required for learned_confidence_gated strategy")
        return _confidence_gated_learned_spec(intent, region, history_ts, base_forecast, context_text, model_path, plan_confidence)
    if strategy == "learned_reliability_gated":
        if not model_path:
            raise ValueError("model_path is required for learned_reliability_gated strategy")
        return _reliability_gated_learned_spec(intent, region, history_ts, base_forecast, context_text, model_path, plan_confidence, sample)

    ctx = _local_context(region, history_ts, base_forecast)
    strength = str(intent.get("strength", "medium"))
    duration_bucket = str(intent.get("duration", "medium"))
    strength_scale = _strength_scalar(strength)
    duration_ratio = _duration_ratio_from_bucket(duration_bucket)
    text_modifier = _text_intensity_modifier(context_text)
    shape = str(shape)
    structured_hint = "[revisionhint]" in str(context_text or "").lower()
    hinted_bucket = infer_future_bucket(context_text) if structured_hint else None

    spec = _zero_edit_spec(strategy=strategy)
    spec["duration_ratio"] = duration_ratio

    if strategy in {"discrete_strength_table", "text_direct_numeric"}:
        magnitude = strength_scale if strategy == "discrete_strength_table" else strength_scale * text_modifier
        if shape == "flatline":
            spec.update({
                "delta_level_z": -1.25 * magnitude,
                "amp_ratio": 0.2,
                "recovery_ratio": 0.0,
                "floor_ratio": 1.25 * magnitude,
            })
        elif shape == "irregular_noise":
            spec.update({
                "delta_level_z": 0.0,
                "amp_ratio": 1.0,
                "vol_ratio": 1.0 + 0.45 * magnitude,
                "recovery_ratio": 0.0,
            })
        elif shape == "hump":
            spec.update({
                "delta_level_z": 0.9 * magnitude,
                "amp_ratio": 1.2,
                "slope_ratio": 1.2,
                "recovery_ratio": 0.7,
            })
        elif shape == "plateau":
            spec.update({
                "delta_level_z": 1.0 * magnitude,
                "amp_ratio": 1.1,
                "slope_ratio": 1.0,
                "recovery_ratio": 0.35,
            })
        else:
            spec.update({
                "delta_level_z": 1.0 * magnitude,
                "amp_ratio": 1.0,
                "slope_ratio": 1.0,
                "recovery_ratio": 0.0,
            })
        return spec

    local_trend = _estimate_slope(ctx["local_forecast"])
    trend_scale = 1.0 + min(abs(local_trend) / max(ctx["scale"], 1e-6), 0.5)
    magnitude = strength_scale * text_modifier * trend_scale

    if shape == "flatline":
        spec.update({
            "delta_level_z": -1.35 * magnitude,
            "amp_ratio": 0.15,
            "vol_ratio": 0.5,
            "recovery_ratio": 0.0,
            "floor_ratio": 1.1 * magnitude,
        })
    elif shape == "irregular_noise":
        spec.update({
            "delta_level_z": 0.0,
            "amp_ratio": 1.0,
            "vol_ratio": 1.2 + 0.55 * magnitude,
            "recovery_ratio": 0.0,
        })
    elif shape == "hump":
        spec.update({
            "delta_level_z": 0.95 * magnitude,
            "amp_ratio": 1.25 + 0.1 * max(0.0, magnitude - 1.0),
            "slope_ratio": 1.15 + 0.1 * magnitude,
            "recovery_ratio": 0.75,
        })
    elif shape == "plateau":
        spec.update({
            "delta_level_z": 1.0 * magnitude,
            "amp_ratio": 1.1,
            "slope_ratio": 1.0 + 0.05 * magnitude,
            "recovery_ratio": 0.35,
        })
    else:
        spec.update({
            "delta_level_z": 1.05 * magnitude,
            "amp_ratio": 1.0,
            "slope_ratio": 1.0,
            "recovery_ratio": 0.0,
        })

    if structured_hint and shape in {"plateau", "step"}:
        hint_scale = {
            "weak": 0.45,
            "medium": 0.62,
            "strong": 0.8,
        }.get(strength, 0.62)
        spec["delta_level_z"] = float(spec.get("delta_level_z", 0.0)) * hint_scale
        spec["amp_ratio"] = min(float(spec.get("amp_ratio", 1.0)), 0.9 if strength == "weak" else 1.0)
        if normalize_position_bucket(hinted_bucket, fallback="mid") == "full":
            spec["duration_ratio"] = 1.0
            spec["recovery_ratio"] = 0.0
    return spec


def project_edit_spec_to_params(
    edit_spec: Dict[str, Any],
    intent: Dict[str, Any],
    region: List[int],
    history_ts: np.ndarray,
    base_forecast: np.ndarray,
    executor_family: str = "math",
) -> Dict[str, Any]:
    ctx = _local_context(region, history_ts, base_forecast)
    region_len = ctx["region_len"]
    scale = ctx["scale"]
    local_stats = ctx["local_stats"]
    forecast_stats = ctx["forecast_stats"]

    duration = int(round(max(1.0, region_len * max(float(edit_spec.get("duration_ratio", 1.0)), 1e-3))))
    duration = max(1, min(duration, region_len))
    shape = intent.get("shape")

    amplitude = abs(float(edit_spec.get("delta_level_z", 0.0))) * scale
    amplitude *= max(float(edit_spec.get("amp_ratio", 1.0)), 1e-3)
    recovery_rate = _clamp(float(edit_spec.get("recovery_ratio", 0.35)), 0.0, 1.0)
    vol_ratio = max(float(edit_spec.get("vol_ratio", 1.0)), 0.0)
    floor_ratio = max(float(edit_spec.get("floor_ratio", 0.0)), 0.0)

    params = {
        "amplitude": float(amplitude),
        "duration": int(duration),
        "onset_lag": 0,
        "recovery_rate": float(recovery_rate),
        "volatility_scale": float(max(1.0, vol_ratio)),
        "executor_family": executor_family,
    }
    if shape == "flatline":
        floor_value = float(local_stats["mean"] - max(floor_ratio, abs(float(edit_spec.get("delta_level_z", 0.0)))) * scale)
        params["floor_value"] = floor_value
    elif shape == "irregular_noise":
        params["volatility_scale"] = float(max(1.05, vol_ratio))
        params["amplitude"] = float(max(scale * 0.25, abs(float(edit_spec.get("delta_level_z", 0.0))) * scale))
    elif shape == "step":
        params["recovery_rate"] = float(recovery_rate)
    elif shape == "plateau":
        params["recovery_rate"] = float(max(0.15, recovery_rate))
    elif shape == "hump":
        params["recovery_rate"] = float(max(0.35, recovery_rate))

    params["local_scale"] = float(scale)
    params["local_mean"] = float(local_stats["mean"])
    params["forecast_min"] = float(forecast_stats["min"])
    return params


def calibrate_revision(
    intent: Dict[str, Any],
    region: List[int],
    history_ts: np.ndarray,
    base_forecast: np.ndarray,
    context_text: str = "",
    strategy: str = "rule_local_stats",
    sample: Dict[str, Any] | None = None,
    model_path: str | None = None,
    plan_confidence: float | None = None,
) -> Dict[str, Any]:
    edit_spec = predict_edit_spec(
        intent=intent,
        region=region,
        history_ts=history_ts,
        base_forecast=base_forecast,
        context_text=context_text,
        strategy=strategy,
        sample=sample,
        model_path=model_path,
        plan_confidence=plan_confidence,
    )
    return project_edit_spec_to_params(
        edit_spec=edit_spec,
        intent=intent,
        region=region,
        history_ts=history_ts,
        base_forecast=base_forecast,
    )


def apply_revision_profile(
    base_forecast: np.ndarray,
    intent: Dict[str, Any],
    region: List[int],
    params: Dict[str, Any],
    seed: int = 7,
) -> Tuple[np.ndarray, np.ndarray]:
    edited = np.asarray(base_forecast, dtype=np.float64).copy()
    delta = np.zeros_like(edited)
    start, end = _effective_region_bounds(region, len(edited))
    if end <= start:
        return edited, delta

    total_len = end - start
    active_len = int(max(1, min(total_len, round(float(params.get("duration", total_len))))))
    tail_len = max(0, total_len - active_len)
    amplitude = float(params.get("amplitude", 0.0))
    direction = -1.0 if intent.get("direction") == "down" else 1.0
    shape = intent.get("shape")
    recovery_rate = _clamp(float(params.get("recovery_rate", 0.35)), 0.0, 1.0)

    if shape in {"none", None}:
        return edited, delta

    if shape == "flatline":
        floor_value = float(params.get("floor_value", np.min(edited[start:start + active_len]) - amplitude))
        delta[start:start + active_len] = floor_value - edited[start:start + active_len]
        edited[start:start + active_len] = floor_value
        if tail_len > 0:
            tail_idx = np.arange(tail_len, dtype=np.float64)
            if recovery_rate <= 0.0:
                recovery_profile = np.zeros(tail_len, dtype=np.float64)
            else:
                recovery_profile = 1.0 - np.exp(-(tail_idx + 1.0) * recovery_rate)
                recovery_profile = np.clip(recovery_profile, 0.0, 1.0)
            target_tail = floor_value + recovery_profile * (base_forecast[start + active_len:end] - floor_value)
            delta[start + active_len:end] = target_tail - edited[start + active_len:end]
            edited[start + active_len:end] = target_tail
        return edited, delta

    if shape == "irregular_noise":
        rng = np.random.default_rng(seed)
        noise = rng.normal(0.0, params.get("volatility_scale", 1.0) * max(amplitude, 1e-6), size=active_len)
        if noise.size:
            noise[0] = 0.0
        delta[start:start + active_len] = noise
        edited[start:start + active_len] = edited[start:start + active_len] + noise
        return edited, delta

    profile = np.zeros(total_len, dtype=np.float64)
    active_x = np.linspace(0.0, 1.0, active_len)
    if shape == "step":
        ramp_len = max(2, min(active_len, active_len // 4 if active_len >= 4 else active_len))
        profile[:active_len] = 1.0
        profile[:ramp_len] = np.linspace(0.0, 1.0, ramp_len, dtype=np.float64)
    elif shape == "plateau":
        active_profile = np.ones(active_len, dtype=np.float64)
        ramp_len = max(1, active_len // 4)
        active_profile[:ramp_len] = np.linspace(0.0, 1.0, ramp_len)
        if active_len > ramp_len:
            active_profile[-ramp_len:] = np.linspace(1.0, 0.8, ramp_len)
        profile[:active_len] = active_profile
    else:
        rise = np.sin(np.pi * active_x)
        profile[:active_len] = np.clip(rise, 0.0, 1.0)

    if tail_len > 0 and recovery_rate > 0.0:
        decay = np.exp(-np.linspace(0.0, 3.0 * recovery_rate, tail_len))
        profile[active_len:] = profile[active_len - 1] * decay
    elif tail_len > 0 and shape == "step":
        profile[active_len:] = 1.0

    if profile.size:
        profile[0] = 0.0
    delta[start:end] = direction * amplitude * profile
    edited[start:end] = edited[start:end] + delta[start:end]
    if start == 0 and edited.size:
        delta[0] = 0.0
        edited[0] = base_forecast[0]
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


def evaluate_calibration(
    edit_spec_gt: Dict[str, Any],
    edit_spec_pred: Dict[str, Any],
    base_forecast: np.ndarray,
    revision_target: np.ndarray,
    edited_forecast: np.ndarray,
    region: List[int],
) -> Dict[str, float]:
    start, end = _effective_region_bounds(region, len(base_forecast))
    region_len = max(1, end - start)
    base_arr = np.asarray(base_forecast, dtype=np.float64)
    target_arr = np.asarray(revision_target, dtype=np.float64)
    edited_arr = np.asarray(edited_forecast, dtype=np.float64)

    gt_delta = target_arr[start:end] - base_arr[start:end]
    pred_delta = edited_arr[start:end] - base_arr[start:end]
    tail_len = max(2, region_len // 3)

    npe = float(np.mean([
        abs(float(edit_spec_pred.get(key, 0.0)) - float(edit_spec_gt.get(key, 0.0)))
        for key in CALIBRATION_SPEC_KEYS
    ]))
    duration_gt = int(round(region_len * max(float(edit_spec_gt.get("duration_ratio", 0.0)), 0.0)))
    duration_pred = int(round(region_len * max(float(edit_spec_pred.get("duration_ratio", 0.0)), 0.0)))

    return {
        "normalized_parameter_error": npe,
        "peak_delta_error": float(abs(np.max(np.abs(pred_delta)) - np.max(np.abs(gt_delta)))) if gt_delta.size else 0.0,
        "signed_area_error": float(abs(np.sum(pred_delta) - np.sum(gt_delta))) if gt_delta.size else 0.0,
        "duration_error": float(abs(duration_pred - duration_gt)),
        "recovery_slope_error": float(abs(_estimate_slope(pred_delta[-tail_len:]) - _estimate_slope(gt_delta[-tail_len:]))) if gt_delta.size >= tail_len else 0.0,
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
