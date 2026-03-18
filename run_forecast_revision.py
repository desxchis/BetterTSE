from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from modules.experiment_visualization import (
    build_visualization_dir,
    build_visualization_path,
    save_forecast_revision_visualization,
)
from modules.forecast_revision import (
    _zero_edit_spec,
    apply_revision_profile,
    calibrate_revision,
    compute_intent_alignment,
    evaluate_calibration,
    evaluate_revision_sample,
    extract_gt_edit_spec,
    heuristic_revision_plan,
    predict_edit_spec,
    project_edit_spec_to_params,
)


def _gt_region(mask: np.ndarray) -> List[int]:
    idx = np.where(mask > 0.5)[0]
    if idx.size == 0:
        return [0, 0]
    return [int(idx[0]), int(idx[-1] + 1)]


def _none_plan() -> Dict[str, Any]:
    return {
        "revision_needed": False,
        "confidence": 1.0,
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


def _oracle_plan(sample: Dict[str, Any], region: List[int]) -> Dict[str, Any]:
    if not bool(sample.get("revision_applicable_gt", True)):
        return _none_plan()
    return {
        "revision_needed": True,
        "confidence": 1.0,
        "intent": {
            "effect_family": sample["effect_family_gt"],
            "direction": sample["direction_gt"],
            "shape": sample["shape_gt"],
            "duration": sample["duration_bucket_gt"],
            "strength": sample["strength_bucket_gt"],
        },
        "localization": {
            "position_bucket": sample["revision_operator_params"].get("bucket", "mid_horizon"),
            "region": region,
        },
        "tool_name": "oracle",
    }


def _apply_tool_family_override(intent: Dict[str, Any], tool_name: str) -> Dict[str, Any]:
    overridden = dict(intent)
    if tool_name == "step_shift":
        overridden["effect_family"] = "level"
        overridden["shape"] = "step"
    elif tool_name == "spike_inject":
        overridden["effect_family"] = "impulse"
        overridden["shape"] = "hump"
    elif tool_name == "volatility_increase":
        overridden["effect_family"] = "volatility"
        overridden["shape"] = "irregular_noise"
        overridden["direction"] = "neutral"
    elif tool_name == "hybrid_up":
        overridden["effect_family"] = "level"
        overridden["shape"] = "plateau"
        overridden["direction"] = "up"
    elif tool_name == "hybrid_down":
        overridden["effect_family"] = "level"
        overridden["shape"] = "plateau"
        overridden["direction"] = "down"
    return overridden


def _predict_params(
    *,
    intent: Dict[str, Any],
    region: List[int],
    history_ts: np.ndarray,
    base_forecast: np.ndarray,
    context_text: str,
    sample: Dict[str, Any],
    strategy: str,
    calibration_model_path: str | None = None,
    plan_confidence: float | None = None,
    tool_name: str | None = None,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    model_sample = dict(sample)
    if tool_name is not None:
        model_sample["tool_name"] = tool_name
    edit_spec = predict_edit_spec(
        intent=intent,
        region=region,
        history_ts=history_ts,
        base_forecast=base_forecast,
        context_text=context_text,
        strategy=strategy,
        sample=model_sample,
        model_path=calibration_model_path,
        plan_confidence=plan_confidence,
    )
    params = project_edit_spec_to_params(
        edit_spec=edit_spec,
        intent=intent,
        region=region,
        history_ts=history_ts,
        base_forecast=base_forecast,
    )
    return edit_spec, params


def _global_revision(
    base_forecast: np.ndarray,
    history_ts: np.ndarray,
    context_text: str,
    intent: Dict[str, Any],
    sample: Dict[str, Any],
    calibration_strategy: str,
    calibration_model_path: str | None = None,
    plan_confidence: float | None = None,
    tool_name: str | None = None,
    revision_executor: str = "profile",
    tedit_model_path: str | None = None,
    tedit_config_path: str | None = None,
    tedit_device: str = "cuda:0",
) -> tuple[np.ndarray, Dict[str, Any], Dict[str, Any], List[int], Dict[str, Any]]:
    horizon = len(base_forecast)
    region = [0, horizon]
    edit_spec, params = _predict_params(
        intent=intent,
        region=region,
        history_ts=history_ts,
        base_forecast=base_forecast,
        context_text=context_text,
        sample=sample,
        strategy=calibration_strategy,
        calibration_model_path=calibration_model_path,
        plan_confidence=plan_confidence,
        tool_name=tool_name,
    )
    edited, execution_metadata = _apply_with_executor(
        revision_executor=revision_executor,
        history_ts=history_ts,
        base_forecast=base_forecast,
        intent=intent,
        region=region,
        params=params,
        tool_name=tool_name,
        tedit_model_path=tedit_model_path,
        tedit_config_path=tedit_config_path,
        tedit_device=tedit_device,
        sample_metadata=sample,
    )
    return edited, edit_spec, params, region, execution_metadata


def _apply_with_executor(
    *,
    revision_executor: str,
    history_ts: np.ndarray,
    base_forecast: np.ndarray,
    intent: Dict[str, Any],
    region: List[int],
    params: Dict[str, Any],
    tool_name: str | None,
    tedit_model_path: str | None,
    tedit_config_path: str | None,
    tedit_device: str,
    sample_metadata: Dict[str, Any] | None = None,
) -> tuple[np.ndarray, Dict[str, Any]]:
    if revision_executor == "tedit_hybrid":
        from modules.forecast_revision_executor import apply_tedit_hybrid_revision

        edited, _, execution_metadata = apply_tedit_hybrid_revision(
            history_ts=history_ts,
            base_forecast=base_forecast,
            intent=intent,
            region=region,
            params=params,
            preferred_tool_name=tool_name,
            tedit_model_path=tedit_model_path,
            tedit_config_path=tedit_config_path,
            tedit_device=tedit_device,
            sample_metadata=sample_metadata,
        )
        return edited, execution_metadata
    edited, _ = apply_revision_profile(base_forecast, intent, region, params)
    return edited, {"executor": "profile", "tool_name": tool_name}


def run_revision(
    benchmark_path: str,
    output_path: str,
    mode: str = "localized_full_revision",
    max_samples: int | None = None,
    vis_dir: str | None = None,
    save_visualizations: bool = True,
    calibration_strategy: str = "rule_local_stats",
    calibration_model_path: str | None = None,
    revision_executor: str = "profile",
    tedit_model_path: str | None = None,
    tedit_config_path: str | None = None,
    tedit_device: str = "cuda:0",
) -> Dict[str, Any]:
    with open(benchmark_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    samples = payload.get("samples", [])
    if max_samples is not None:
        samples = samples[:max_samples]

    vis_dir_obj = build_visualization_dir(output_path, vis_dir) if save_visualizations else None
    run_timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    results: List[Dict[str, Any]] = []
    for sample in samples:
        history_ts = np.asarray(sample["history_ts"], dtype=np.float64)
        future_gt = np.asarray(sample["future_gt"], dtype=np.float64)
        base_forecast = np.asarray(sample["base_forecast"], dtype=np.float64)
        revision_target = np.asarray(sample["revision_target"], dtype=np.float64)
        gt_mask = np.asarray(sample["edit_mask_gt"], dtype=np.float64)
        gt_region = _gt_region(gt_mask)
        edit_spec_gt = extract_gt_edit_spec(sample, history_ts=history_ts, base_forecast=base_forecast)

        if mode == "base_only":
            edited = base_forecast.copy()
            plan = _none_plan()
            edit_spec = _zero_edit_spec(strategy="base_only")
            params: Dict[str, Any] = {}
            pred_region = [0, 0]
            execution_metadata: Dict[str, Any] = {"executor": revision_executor, "tool_name": "none"}
        elif mode == "oracle_region":
            plan = heuristic_revision_plan(sample["context_text"], sample["forecast_horizon"])
            pred_region = gt_region
            if plan["revision_needed"]:
                edit_spec, params = _predict_params(
                    intent=plan["intent"],
                    region=pred_region,
                    history_ts=history_ts,
                    base_forecast=base_forecast,
                    context_text=sample["context_text"],
                    sample=sample,
                    strategy=calibration_strategy,
                    calibration_model_path=calibration_model_path,
                    plan_confidence=plan.get("confidence"),
                    tool_name=plan.get("tool_name"),
                )
                edited, execution_metadata = _apply_with_executor(
                    revision_executor=revision_executor,
                    history_ts=history_ts,
                    base_forecast=base_forecast,
                    intent=plan["intent"],
                    region=pred_region,
                    params=params,
                    tool_name=plan.get("tool_name"),
                    tedit_model_path=tedit_model_path,
                    tedit_config_path=tedit_config_path,
                    tedit_device=tedit_device,
                    sample_metadata=sample,
                )
            else:
                edited = base_forecast.copy()
                edit_spec = _zero_edit_spec(strategy="oracle_region")
                params = {}
                execution_metadata = {"executor": revision_executor, "tool_name": "none"}
        elif mode == "oracle_intent":
            pred_region = gt_region
            plan = _oracle_plan(sample, pred_region)
            if plan["revision_needed"]:
                edit_spec, params = _predict_params(
                    intent=plan["intent"],
                    region=pred_region,
                    history_ts=history_ts,
                    base_forecast=base_forecast,
                    context_text=sample["context_text"],
                    sample=sample,
                    strategy=calibration_strategy,
                    calibration_model_path=calibration_model_path,
                    plan_confidence=plan.get("confidence"),
                    tool_name=plan.get("tool_name"),
                )
                edited, execution_metadata = _apply_with_executor(
                    revision_executor=revision_executor,
                    history_ts=history_ts,
                    base_forecast=base_forecast,
                    intent=plan["intent"],
                    region=pred_region,
                    params=params,
                    tool_name=plan.get("tool_name"),
                    tedit_model_path=tedit_model_path,
                    tedit_config_path=tedit_config_path,
                    tedit_device=tedit_device,
                    sample_metadata=sample,
                )
            else:
                edited = base_forecast.copy()
                edit_spec = _zero_edit_spec(strategy="oracle_intent")
                params = {}
                execution_metadata = {"executor": revision_executor, "tool_name": "none"}
        elif mode == "oracle_tool":
            pred_region = gt_region
            pred_plan = heuristic_revision_plan(sample["context_text"], sample["forecast_horizon"])
            plan = _oracle_plan(sample, pred_region)
            plan["tool_name"] = pred_plan.get("tool_name", "none")
            if plan["revision_needed"]:
                execution_intent = _apply_tool_family_override(plan["intent"], plan["tool_name"])
                edit_spec, params = _predict_params(
                    intent=execution_intent,
                    region=pred_region,
                    history_ts=history_ts,
                    base_forecast=base_forecast,
                    context_text=sample["context_text"],
                    sample=sample,
                    strategy=calibration_strategy,
                    calibration_model_path=calibration_model_path,
                    plan_confidence=plan.get("confidence"),
                    tool_name=plan.get("tool_name"),
                )
                edited, execution_metadata = _apply_with_executor(
                    revision_executor=revision_executor,
                    history_ts=history_ts,
                    base_forecast=base_forecast,
                    intent=execution_intent,
                    region=pred_region,
                    params=params,
                    tool_name=plan.get("tool_name"),
                    tedit_model_path=tedit_model_path,
                    tedit_config_path=tedit_config_path,
                    tedit_device=tedit_device,
                    sample_metadata=sample,
                )
                plan["execution_intent"] = execution_intent
            else:
                edited = base_forecast.copy()
                edit_spec = _zero_edit_spec(strategy="oracle_tool")
                params = {}
                execution_metadata = {"executor": revision_executor, "tool_name": "none"}
        elif mode == "oracle_calibration":
            pred_region = gt_region
            plan = _oracle_plan(sample, pred_region)
            if plan["revision_needed"]:
                edit_spec = dict(edit_spec_gt)
                params = dict(sample["revision_operator_params"].get("params", {}))
                if not params:
                    params = project_edit_spec_to_params(
                        edit_spec=edit_spec,
                        intent=plan["intent"],
                        region=pred_region,
                        history_ts=history_ts,
                        base_forecast=base_forecast,
                    )
                edited, execution_metadata = _apply_with_executor(
                    revision_executor=revision_executor,
                    history_ts=history_ts,
                    base_forecast=base_forecast,
                    intent=plan["intent"],
                    region=pred_region,
                    params=params,
                    tool_name=plan.get("tool_name"),
                    tedit_model_path=tedit_model_path,
                    tedit_config_path=tedit_config_path,
                    tedit_device=tedit_device,
                    sample_metadata=sample,
                )
            else:
                edited = base_forecast.copy()
                edit_spec = _zero_edit_spec(strategy="oracle_calibration")
                params = {}
                execution_metadata = {"executor": revision_executor, "tool_name": "none"}
        elif mode == "global_revision_only":
            plan = heuristic_revision_plan(sample["context_text"], sample["forecast_horizon"])
            if plan["revision_needed"]:
                edited, edit_spec, params, pred_region, execution_metadata = _global_revision(
                    base_forecast=base_forecast,
                    history_ts=history_ts,
                    context_text=sample["context_text"],
                    intent=plan["intent"],
                    sample=sample,
                    calibration_strategy=calibration_strategy,
                    calibration_model_path=calibration_model_path,
                    plan_confidence=plan.get("confidence"),
                    tool_name=plan.get("tool_name"),
                    revision_executor=revision_executor,
                    tedit_model_path=tedit_model_path,
                    tedit_config_path=tedit_config_path,
                    tedit_device=tedit_device,
                )
                plan["localization"]["region"] = pred_region
            else:
                edited = base_forecast.copy()
                edit_spec = _zero_edit_spec(strategy="global_revision_only")
                params = {}
                pred_region = [0, 0]
                execution_metadata = {"executor": revision_executor, "tool_name": "none"}
        else:
            plan = heuristic_revision_plan(sample["context_text"], sample["forecast_horizon"])
            pred_region = plan["localization"]["region"]
            if plan["revision_needed"]:
                edit_spec, params = _predict_params(
                    intent=plan["intent"],
                    region=pred_region,
                    history_ts=history_ts,
                    base_forecast=base_forecast,
                    context_text=sample["context_text"],
                    sample=sample,
                    strategy=calibration_strategy,
                    calibration_model_path=calibration_model_path,
                    plan_confidence=plan.get("confidence"),
                    tool_name=plan.get("tool_name"),
                )
                edited, execution_metadata = _apply_with_executor(
                    revision_executor=revision_executor,
                    history_ts=history_ts,
                    base_forecast=base_forecast,
                    intent=plan["intent"],
                    region=pred_region,
                    params=params,
                    tool_name=plan.get("tool_name"),
                    tedit_model_path=tedit_model_path,
                    tedit_config_path=tedit_config_path,
                    tedit_device=tedit_device,
                    sample_metadata=sample,
                )
            else:
                edited = base_forecast.copy()
                edit_spec = _zero_edit_spec(strategy="localized_full_revision")
                params = {}
                execution_metadata = {"executor": revision_executor, "tool_name": "none"}

        metrics = evaluate_revision_sample(
            base_forecast=base_forecast,
            edited_forecast=edited,
            future_gt=future_gt,
            revision_target=revision_target,
            pred_region=pred_region,
            gt_mask=gt_mask,
        )
        calibration_metrics = evaluate_calibration(
            edit_spec_gt=edit_spec_gt,
            edit_spec_pred=edit_spec,
            base_forecast=base_forecast,
            revision_target=revision_target,
            edited_forecast=edited,
            region=gt_region,
        )
        intent_alignment = compute_intent_alignment(plan, sample)
        result = {
            "sample_id": sample["sample_id"],
            "dataset_name": sample["dataset_name"],
            "context_text": sample["context_text"],
            "mode": mode,
            "plan": plan,
            "pred_region": pred_region,
            "gt_region": gt_region,
            "edit_spec": edit_spec,
            "edit_spec_gt": edit_spec_gt,
            "calibration": params,
            "calibration_strategy": calibration_strategy,
            "calibration_model_path": calibration_model_path,
            "revision_executor": revision_executor,
            "execution_metadata": execution_metadata,
            "metrics": metrics,
            "calibration_metrics": calibration_metrics,
            "intent_alignment": intent_alignment,
            "revision_applicable_gt": sample["revision_applicable_gt"],
            "edit_intent_gt": sample.get("edit_intent_gt"),
            "history_ts": sample["history_ts"],
            "future_gt": sample["future_gt"],
            "base_forecast": sample["base_forecast"],
            "revision_target": sample["revision_target"],
            "edited_forecast": edited.astype(float).tolist(),
        }
        if vis_dir_obj is not None:
            save_path = build_visualization_path(
                vis_dir_obj,
                sample_id=sample["sample_id"],
                target_feature=payload.get("feature", "forecast"),
                task_type=sample.get("shape_gt", "revision"),
                timestamp=run_timestamp,
            )
            save_forecast_revision_visualization(
                sample_id=sample["sample_id"],
                history_ts=history_ts,
                future_gt=future_gt,
                base_forecast=base_forecast,
                revision_target=revision_target,
                edited_forecast=edited,
                gt_region=(gt_region[0], gt_region[1]),
                pred_region=(pred_region[0], pred_region[1]),
                context_text=sample["context_text"],
                metrics=metrics,
                save_path=save_path,
            )
            result["visualization_path"] = str(save_path)
        results.append(result)

    summary = _summarize_results(results)
    output_payload = {
        "summary": summary,
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "visualization_dir": str(vis_dir_obj) if vis_dir_obj is not None else None,
        "calibration_strategy": calibration_strategy,
        "revision_executor": revision_executor,
        "results": results,
    }
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, ensure_ascii=False, indent=2)
    return output_payload


def _summarize_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not results:
        return {"total": 0, "successful": 0, "failed": 0}

    metric_keys = [
        "base_mae_vs_revision_target",
        "edited_mae_vs_revision_target",
        "base_mae_vs_future_gt",
        "edited_mae_vs_future_gt",
        "edited_mse_vs_revision_target",
        "edited_smape_vs_revision_target",
        "future_t_iou",
        "revision_gain",
        "magnitude_calibration_error",
        "outside_region_preservation",
        "over_edit_rate",
    ]
    calibration_keys = [
        "normalized_parameter_error",
        "peak_delta_error",
        "signed_area_error",
        "duration_error",
        "recovery_slope_error",
    ]
    summary: Dict[str, Any] = {"total": len(results), "successful": len(results), "failed": 0}
    for key in metric_keys:
        values = [r["metrics"][key] for r in results if key in r["metrics"]]
        summary[f"avg_{key}"] = float(np.mean(values)) if values else None
    for key in calibration_keys:
        values = [r["calibration_metrics"][key] for r in results if key in r.get("calibration_metrics", {})]
        summary[f"avg_{key}"] = float(np.mean(values)) if values else None
    intent_keys = [
        "revision_needed_match",
        "effect_family_match",
        "direction_match",
        "shape_match",
        "duration_match",
        "strength_match",
        "intent_match_score",
    ]
    for key in intent_keys:
        values = [r["intent_alignment"][key] for r in results if key in r.get("intent_alignment", {})]
        summary[f"avg_{key}"] = float(np.mean(values)) if values else None
    for subset_name, subset_flag in (("applicable", True), ("non_applicable", False)):
        subset = [r for r in results if r.get("revision_applicable_gt") is subset_flag]
        summary[f"{subset_name}_count"] = len(subset)
        if not subset:
            continue
        for key in (
            "base_mae_vs_revision_target",
            "edited_mae_vs_revision_target",
            "base_mae_vs_future_gt",
            "edited_mae_vs_future_gt",
            "future_t_iou",
            "revision_gain",
            "magnitude_calibration_error",
            "outside_region_preservation",
            "over_edit_rate",
        ):
            values = [r["metrics"][key] for r in subset if key in r["metrics"]]
            summary[f"{subset_name}_avg_{key}"] = float(np.mean(values)) if values else None
        for key in calibration_keys:
            values = [r["calibration_metrics"][key] for r in subset if key in r.get("calibration_metrics", {})]
            summary[f"{subset_name}_avg_{key}"] = float(np.mean(values)) if values else None
        values = [r["intent_alignment"]["revision_needed_match"] for r in subset if "intent_alignment" in r]
        summary[f"{subset_name}_avg_revision_needed_match"] = float(np.mean(values)) if values else None
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CPU-safe forecast revision experiments.")
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--mode",
        default="localized_full_revision",
        choices=[
            "base_only",
            "global_revision_only",
            "localized_full_revision",
            "oracle_region",
            "oracle_intent",
            "oracle_tool",
            "oracle_calibration",
        ],
    )
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--vis-dir", default=None)
    parser.add_argument("--no-save-vis", action="store_true")
    parser.add_argument(
        "--calibration-strategy",
        default="rule_local_stats",
        choices=["text_direct_numeric", "discrete_strength_table", "rule_local_stats", "learned_linear", "learned_rule_guarded", "learned_rule_shrunk", "learned_confidence_gated", "learned_reliability_gated"],
    )
    parser.add_argument("--calibration-model", default=None)
    parser.add_argument("--revision-executor", default="profile", choices=["profile", "tedit_hybrid"])
    parser.add_argument("--tedit-model", default=None)
    parser.add_argument("--tedit-config", default=None)
    parser.add_argument("--tedit-device", default="cuda:0")
    args = parser.parse_args()

    run_revision(
        benchmark_path=args.benchmark,
        output_path=args.output,
        mode=args.mode,
        max_samples=args.max_samples,
        vis_dir=args.vis_dir,
        save_visualizations=not args.no_save_vis,
        calibration_strategy=args.calibration_strategy,
        calibration_model_path=args.calibration_model,
        revision_executor=args.revision_executor,
        tedit_model_path=args.tedit_model,
        tedit_config_path=args.tedit_config,
        tedit_device=args.tedit_device,
    )


if __name__ == "__main__":
    main()
