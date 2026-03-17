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
    apply_revision_profile,
    calibrate_revision,
    compute_intent_alignment,
    evaluate_revision_sample,
    heuristic_revision_plan,
)


def _gt_region(mask: np.ndarray) -> List[int]:
    idx = np.where(mask > 0.5)[0]
    if idx.size == 0:
        return [0, 0]
    return [int(idx[0]), int(idx[-1] + 1)]


def _global_revision(base_forecast: np.ndarray, intent: Dict[str, Any]) -> tuple[np.ndarray, Dict[str, Any], List[int]]:
    horizon = len(base_forecast)
    region = [0, horizon]
    params = calibrate_revision(intent, region, base_forecast, base_forecast)
    edited, _ = apply_revision_profile(base_forecast, intent, region, params)
    return edited, params, region


def _strip_big_fields(result: Dict[str, Any]) -> Dict[str, Any]:
    kept = dict(result)
    return kept


def run_revision(
    benchmark_path: str,
    output_path: str,
    mode: str = "localized_full_revision",
    max_samples: int | None = None,
    vis_dir: str | None = None,
    save_visualizations: bool = True,
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

        if mode == "base_only":
            edited = base_forecast.copy()
            plan = {
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
            params: Dict[str, Any] = {}
            pred_region = [0, 0]
        elif mode == "oracle_region":
            plan = heuristic_revision_plan(sample["context_text"], sample["forecast_horizon"])
            pred_region = gt_region
            params = calibrate_revision(plan["intent"], pred_region, history_ts, base_forecast)
            edited, _ = apply_revision_profile(base_forecast, plan["intent"], pred_region, params)
        elif mode == "oracle_intent":
            pred_region = gt_region
            plan = {
                "revision_needed": True,
                "confidence": 1.0,
                "intent": {
                    "effect_family": sample["effect_family_gt"],
                    "direction": sample["direction_gt"],
                    "shape": sample["shape_gt"],
                    "duration": sample["duration_bucket_gt"],
                    "strength": sample["strength_bucket_gt"],
                },
                "localization": {"position_bucket": sample["revision_operator_params"].get("bucket", "mid_horizon"), "region": pred_region},
                "tool_name": "oracle",
            }
            params = calibrate_revision(plan["intent"], pred_region, history_ts, base_forecast)
            edited, _ = apply_revision_profile(base_forecast, plan["intent"], pred_region, params)
        elif mode == "oracle_calibration":
            pred_region = gt_region
            plan = {
                "revision_needed": True,
                "confidence": 1.0,
                "intent": {
                    "effect_family": sample["effect_family_gt"],
                    "direction": sample["direction_gt"],
                    "shape": sample["shape_gt"],
                    "duration": sample["duration_bucket_gt"],
                    "strength": sample["strength_bucket_gt"],
                },
                "localization": {"position_bucket": sample["revision_operator_params"].get("bucket", "mid_horizon"), "region": pred_region},
                "tool_name": "oracle",
            }
            params = dict(sample["revision_operator_params"]["params"])
            edited, _ = apply_revision_profile(base_forecast, plan["intent"], pred_region, params)
        elif mode == "global_revision_only":
            plan = heuristic_revision_plan(sample["context_text"], sample["forecast_horizon"])
            edited, params, pred_region = _global_revision(base_forecast, plan["intent"])
            plan["localization"]["region"] = pred_region
        else:
            plan = heuristic_revision_plan(sample["context_text"], sample["forecast_horizon"])
            pred_region = plan["localization"]["region"]
            params = calibrate_revision(plan["intent"], pred_region, history_ts, base_forecast)
            edited, _ = apply_revision_profile(base_forecast, plan["intent"], pred_region, params)

        metrics = evaluate_revision_sample(
            base_forecast=base_forecast,
            edited_forecast=edited,
            future_gt=future_gt,
            revision_target=revision_target,
            pred_region=pred_region,
            gt_mask=gt_mask,
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
            "calibration": params,
            "metrics": metrics,
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
    summary: Dict[str, Any] = {"total": len(results), "successful": len(results), "failed": 0}
    for key in metric_keys:
        values = [r["metrics"][key] for r in results if key in r["metrics"]]
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
            "oracle_calibration",
        ],
    )
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--vis-dir", default=None)
    parser.add_argument("--no-save-vis", action="store_true")
    args = parser.parse_args()

    run_revision(
        benchmark_path=args.benchmark,
        output_path=args.output,
        mode=args.mode,
        max_samples=args.max_samples,
        vis_dir=args.vis_dir,
        save_visualizations=not args.no_save_vis,
    )


if __name__ == "__main__":
    main()
