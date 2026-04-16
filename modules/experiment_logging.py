from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from modules.region_localizer import normalize_position_bucket


PURE_EDITING_LOGGER_VERSION = "pure_editing_logger_v1"
REVISION_LOGGER_VERSION = "forecast_revision_logger_v1"
UNIFIED_SCHEMA_VERSION = "bettertse_experiment_record_v1"


_PURE_EDIT_GT_TOOL_MAP = {
    "trend_injection": ("impulse_spike", "spike_inject"),
    "step_change": ("level_step", "step_shift"),
    "noise_injection": ("volatility_increase", "volatility_increase"),
    "hard_zero": ("trend_linear_down", "hybrid_down"),
    "multiplier": ("trend_linear_up", "hybrid_up"),
    "none": ("none", "none"),
}


def region_to_bucket(start: int, end: int, length: int) -> str:
    total = max(1, int(length))
    start = max(0, min(int(start), total))
    end = max(start, min(int(end), total))
    if end <= start:
        return "none"
    if start == 0 and end >= total:
        return "full"
    center_ratio = ((start + end) / 2.0) / float(total)
    if center_ratio <= 0.34:
        return "early"
    if center_ratio >= 0.67:
        return "late"
    return "mid"


def compute_target_similarity(
    prediction: np.ndarray | List[float],
    target: np.ndarray | List[float],
) -> Dict[str, float]:
    pred_arr = np.asarray(prediction, dtype=np.float64).flatten()
    target_arr = np.asarray(target, dtype=np.float64).flatten()
    if pred_arr.size == 0 or target_arr.size == 0:
        return {"mae_vs_target": 0.0, "mse_vs_target": 0.0}
    return {
        "mae_vs_target": float(np.mean(np.abs(pred_arr - target_arr))),
        "mse_vs_target": float(np.mean((pred_arr - target_arr) ** 2)),
    }


def extract_pure_editing_gt_labels(sample: Dict[str, Any], gt_config: Dict[str, Any]) -> Dict[str, Any]:
    base_ts = sample.get("base_ts", [])
    gt_start = int(gt_config.get("start_step", sample.get("gt_start", 0)) or 0)
    gt_end = int(gt_config.get("end_step", sample.get("gt_end", 0)) or 0)
    edit_intent = gt_config.get("edit_intent_gt") if isinstance(gt_config.get("edit_intent_gt"), dict) else {}
    injection_operator = str(gt_config.get("injection_operator", sample.get("injection_operator", "none")))
    canonical_tool, hybrid_tool = _PURE_EDIT_GT_TOOL_MAP.get(injection_operator, ("none", "none"))
    injection_config = sample.get("injection_config", {}) if isinstance(sample.get("injection_config"), dict) else {}
    parameter_label = {
        "magnitude": injection_config.get("magnitude")
        or injection_config.get("multiplier")
        or injection_config.get("trend_slope")
        or injection_config.get("noise_std"),
        "duration_extent": max(0, gt_end - gt_start),
        "onset_style": "abrupt" if injection_operator in {"step_change", "hard_zero"} else "smooth",
        "recovery_style": "recovering" if injection_operator in {"trend_injection", "step_change"} else "persistent",
    }
    return {
        "task_type": str(sample.get("task_type", gt_config.get("task_type", "unknown"))),
        "series_id": str(sample.get("sample_id", "")),
        "context_text": str(sample.get("vague_prompt", "")),
        "intent_label": edit_intent,
        "localization_label": {
            "position_bucket": region_to_bucket(gt_start, gt_end, len(base_ts)),
            "region": [gt_start, gt_end],
        },
        "canonical_tool_label": canonical_tool,
        "hybrid_tool_label": hybrid_tool,
        "parameter_label": parameter_label,
    }


def build_pure_editing_record(
    *,
    sample: Dict[str, Any],
    gt_config: Dict[str, Any],
    prompt_text: str,
    mode: str,
    plan: Dict[str, Any],
    metrics: Dict[str, Any],
    intent_alignment: Dict[str, Any],
    visualization_path: str | None,
) -> Dict[str, Any]:
    labels = extract_pure_editing_gt_labels(sample, gt_config)
    pred_region = list(plan.get("parameters", {}).get("region", [0, len(sample.get("base_ts", []))]))
    prediction = {
        "intent_label": plan.get("intent", {}),
        "localization_label": {
            "position_bucket": normalize_position_bucket(
                str((plan.get("localization") or {}).get("position_bucket", region_to_bucket(pred_region[0], pred_region[1], len(sample.get("base_ts", [])))))
            ),
            "region": pred_region,
        },
        "canonical_tool_label": plan.get("canonical_tool") or (plan.get("execution") or {}).get("canonical_tool"),
        "hybrid_tool_label": plan.get("tool_name") or (plan.get("execution") or {}).get("tool_name"),
        "parameter_label": dict(plan.get("parameters", {})),
    }
    return {
        "schema_version": UNIFIED_SCHEMA_VERSION,
        "logger_version": PURE_EDITING_LOGGER_VERSION,
        "task_family": "pure_time_series_editing",
        "target_regime": "controlled_physical_injection",
        "sample_id": sample.get("sample_id"),
        "series_id": labels["series_id"],
        "dataset_name": sample.get("dataset_name"),
        "mode": mode,
        "model_name": "bettertse_pipeline",
        "task_type": labels["task_type"],
        "context_text": prompt_text,
        "ground_truth": {
            "intent_label": labels["intent_label"],
            "localization_label": labels["localization_label"],
            "canonical_tool_label": labels["canonical_tool_label"],
            "hybrid_tool_label": labels["hybrid_tool_label"],
            "parameter_label": labels["parameter_label"],
        },
        "prediction": prediction,
        "intent_alignment": intent_alignment,
        "metrics": metrics,
        "visualization_path": visualization_path,
    }


def build_revision_record(
    *,
    sample: Dict[str, Any],
    mode: str,
    plan: Dict[str, Any],
    pred_region: List[int],
    metrics: Dict[str, Any],
    calibration: Dict[str, Any],
    calibration_metrics: Dict[str, Any],
    intent_alignment: Dict[str, Any],
    visualization_path: str | None,
    calibration_strategy: str,
    revision_executor: str,
) -> Dict[str, Any]:
    gt_region = list((sample.get("revision_operator_params") or {}).get("region", [0, 0]))
    gt_bucket = normalize_position_bucket(str((sample.get("revision_operator_params") or {}).get("bucket", region_to_bucket(gt_region[0], gt_region[1], len(sample.get("base_forecast", []))))), fallback="none")
    pred_bucket = normalize_position_bucket(
        str((plan.get("localization") or {}).get("position_bucket", region_to_bucket(pred_region[0], pred_region[1], len(sample.get("base_forecast", [])))))
    )
    return {
        "schema_version": UNIFIED_SCHEMA_VERSION,
        "logger_version": REVISION_LOGGER_VERSION,
        "task_family": "forecast_revision",
        "target_regime": sample.get("target_regime", "future_guided_projected_revision"),
        "sample_id": sample.get("sample_id"),
        "series_id": sample.get("sample_id"),
        "dataset_name": sample.get("dataset_name"),
        "mode": mode,
        "model_name": sample.get("baseline_name"),
        "task_type": sample.get("shape_gt", "revision"),
        "context_text": sample.get("context_text", ""),
        "ground_truth": {
            "intent_label": sample.get("edit_intent_gt"),
            "localization_label": {"position_bucket": gt_bucket, "region": gt_region},
            "canonical_tool_label": sample.get("revision_operator_family"),
            "hybrid_tool_label": (sample.get("revision_operator_params") or {}).get("tool_name"),
            "parameter_label": (sample.get("revision_operator_params") or {}).get("params", {}),
        },
        "prediction": {
            "intent_label": plan.get("intent", {}),
            "localization_label": {"position_bucket": pred_bucket, "region": list(pred_region)},
            "canonical_tool_label": plan.get("canonical_tool") or (plan.get("execution_intent") or {}).get("shape") or plan.get("tool_name"),
            "hybrid_tool_label": plan.get("tool_name"),
            "parameter_label": calibration,
        },
        "intent_alignment": intent_alignment,
        "metrics": metrics,
        "calibration_metrics": calibration_metrics,
        "calibration_strategy": calibration_strategy,
        "revision_executor": revision_executor,
        "visualization_path": visualization_path,
    }
