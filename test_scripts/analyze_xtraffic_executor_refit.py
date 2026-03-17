from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from modules.forecast_revision import (  # noqa: E402
    apply_revision_profile,
    compute_intent_alignment,
    evaluate_calibration,
    evaluate_revision_sample,
    extract_gt_edit_spec,
    predict_edit_spec,
    project_edit_spec_to_params,
)


@dataclass
class RefitResult:
    sample_id: str
    shape: str
    gt_region: list[int]
    pseudo_params: dict[str, Any]
    heuristic_params: dict[str, Any]
    best_fit_params: dict[str, Any]
    pseudo_metrics: dict[str, float]
    heuristic_metrics: dict[str, float]
    best_fit_metrics: dict[str, float]
    pseudo_calibration: dict[str, float]
    heuristic_calibration: dict[str, float]
    best_fit_calibration: dict[str, float]
    intent_alignment: dict[str, Any]


def _gt_region(mask: np.ndarray) -> list[int]:
    idx = np.where(mask > 0.5)[0]
    if idx.size == 0:
        return [0, 0]
    return [int(idx[0]), int(idx[-1] + 1)]


def _area_error(base: np.ndarray, target: np.ndarray, edited: np.ndarray, region: list[int]) -> float:
    start, end = region
    gt_delta = target[start:end] - base[start:end]
    pred_delta = edited[start:end] - base[start:end]
    return float(abs(np.sum(pred_delta) - np.sum(gt_delta)))


def _fit_step_executor(
    *,
    base_forecast: np.ndarray,
    revision_target: np.ndarray,
    intent: dict[str, Any],
    region: list[int],
    anchor_params: dict[str, Any],
) -> tuple[dict[str, Any], np.ndarray]:
    start, end = region
    region_len = max(1, end - start)
    local_target_delta = revision_target[start:end] - base_forecast[start:end]
    local_scale = max(float(anchor_params.get("local_scale", 1.0)), 1e-3)
    max_abs_target = float(np.max(np.abs(local_target_delta))) if local_target_delta.size else local_scale
    amp_candidates = np.unique(
        np.clip(
            np.concatenate(
                [
                    np.linspace(0.1 * local_scale, max(1.5 * local_scale, 1.5 * max_abs_target), 21),
                    np.array([float(anchor_params.get("amplitude", local_scale)), max_abs_target]),
                ]
            ),
            1e-3,
            None,
        )
    )
    duration_candidates = sorted(set([1, region_len, max(1, region_len // 2), int(anchor_params.get("duration", region_len))]))
    duration_candidates = [d for d in duration_candidates if 1 <= d <= region_len]
    recovery_candidates = np.unique(
        np.clip(
            np.concatenate(
                [np.linspace(0.0, 1.0, 21), np.array([float(anchor_params.get("recovery_rate", 0.0))])],
            ),
            0.0,
            1.0,
        )
    )

    best_score = None
    best_params = None
    best_edited = None
    for amplitude in amp_candidates:
        for duration in duration_candidates:
            for recovery_rate in recovery_candidates:
                params = {
                    "amplitude": float(amplitude),
                    "duration": int(duration),
                    "onset_lag": 0,
                    "recovery_rate": float(recovery_rate),
                    "volatility_scale": 1.0,
                    "executor_family": "math_refit",
                    "local_scale": local_scale,
                    "local_mean": float(anchor_params.get("local_mean", 0.0)),
                    "forecast_min": float(anchor_params.get("forecast_min", float(np.min(base_forecast)))),
                }
                edited, _ = apply_revision_profile(base_forecast, intent, region, params)
                mse = float(np.mean((edited - revision_target) ** 2))
                area_error = _area_error(base_forecast, revision_target, edited, region)
                score = (mse, area_error)
                if best_score is None or score < best_score:
                    best_score = score
                    best_params = params
                    best_edited = edited
    if best_params is None or best_edited is None:
        raise RuntimeError("failed to fit step executor")
    return best_params, best_edited


def analyze_benchmark(benchmark_path: str, output_path: str) -> dict[str, Any]:
    payload = json.loads(Path(benchmark_path).read_text(encoding="utf-8"))
    rows: list[RefitResult] = []

    for sample in payload.get("samples", []):
        if not sample.get("revision_applicable_gt", False):
            continue
        if sample.get("shape_gt") != "step":
            continue

        history_ts = np.asarray(sample["history_ts"], dtype=np.float64)
        future_gt = np.asarray(sample["future_gt"], dtype=np.float64)
        base_forecast = np.asarray(sample["base_forecast"], dtype=np.float64)
        revision_target = np.asarray(sample["revision_target"], dtype=np.float64)
        gt_mask = np.asarray(sample["edit_mask_gt"], dtype=np.float64)
        region = _gt_region(gt_mask)
        intent = dict(sample["edit_intent_gt"])
        context_text = sample["context_text"]

        pseudo_spec = extract_gt_edit_spec(sample, history_ts=history_ts, base_forecast=base_forecast)
        pseudo_params = project_edit_spec_to_params(
            edit_spec=pseudo_spec,
            intent=intent,
            region=region,
            history_ts=history_ts,
            base_forecast=base_forecast,
        )
        pseudo_edited, _ = apply_revision_profile(base_forecast, intent, region, pseudo_params)

        heuristic_spec = predict_edit_spec(
            intent=intent,
            region=region,
            history_ts=history_ts,
            base_forecast=base_forecast,
            context_text=context_text,
            strategy="rule_local_stats",
            sample=sample,
        )
        heuristic_params = project_edit_spec_to_params(
            edit_spec=heuristic_spec,
            intent=intent,
            region=region,
            history_ts=history_ts,
            base_forecast=base_forecast,
        )
        heuristic_edited, _ = apply_revision_profile(base_forecast, intent, region, heuristic_params)

        best_fit_params, best_fit_edited = _fit_step_executor(
            base_forecast=base_forecast,
            revision_target=revision_target,
            intent=intent,
            region=region,
            anchor_params=heuristic_params,
        )

        rows.append(
            RefitResult(
                sample_id=sample["sample_id"],
                shape=sample["shape_gt"],
                gt_region=region,
                pseudo_params=pseudo_params,
                heuristic_params=heuristic_params,
                best_fit_params=best_fit_params,
                pseudo_metrics=evaluate_revision_sample(
                    base_forecast=base_forecast,
                    edited_forecast=pseudo_edited,
                    future_gt=future_gt,
                    revision_target=revision_target,
                    pred_region=region,
                    gt_mask=gt_mask,
                ),
                heuristic_metrics=evaluate_revision_sample(
                    base_forecast=base_forecast,
                    edited_forecast=heuristic_edited,
                    future_gt=future_gt,
                    revision_target=revision_target,
                    pred_region=region,
                    gt_mask=gt_mask,
                ),
                best_fit_metrics=evaluate_revision_sample(
                    base_forecast=base_forecast,
                    edited_forecast=best_fit_edited,
                    future_gt=future_gt,
                    revision_target=revision_target,
                    pred_region=region,
                    gt_mask=gt_mask,
                ),
                pseudo_calibration=evaluate_calibration(
                    edit_spec_gt=pseudo_spec,
                    edit_spec_pred=pseudo_spec,
                    base_forecast=base_forecast,
                    revision_target=revision_target,
                    edited_forecast=pseudo_edited,
                    region=region,
                ),
                heuristic_calibration=evaluate_calibration(
                    edit_spec_gt=pseudo_spec,
                    edit_spec_pred=heuristic_spec,
                    base_forecast=base_forecast,
                    revision_target=revision_target,
                    edited_forecast=heuristic_edited,
                    region=region,
                ),
                best_fit_calibration=evaluate_calibration(
                    edit_spec_gt=pseudo_spec,
                    edit_spec_pred=pseudo_spec,
                    base_forecast=base_forecast,
                    revision_target=revision_target,
                    edited_forecast=best_fit_edited,
                    region=region,
                ),
                intent_alignment=compute_intent_alignment({"revision_needed": True, "intent": intent}, sample),
            )
        )

    if not rows:
        raise ValueError("No applicable step samples found for refit analysis")

    def avg(method: str, key: str, calibration: bool = False) -> float:
        values = []
        for row in rows:
            bucket = getattr(row, f"{method}_{'calibration' if calibration else 'metrics'}")
            values.append(bucket[key])
        return float(mean(values))

    summary = {
        "benchmark_path": benchmark_path,
        "num_samples": len(rows),
        "pseudo": {
            "avg_revision_gain": avg("pseudo", "revision_gain"),
            "avg_edited_mae_vs_revision_target": avg("pseudo", "edited_mae_vs_revision_target"),
            "avg_signed_area_error": avg("pseudo", "signed_area_error", calibration=True),
            "avg_peak_delta_error": avg("pseudo", "peak_delta_error", calibration=True),
        },
        "heuristic": {
            "avg_revision_gain": avg("heuristic", "revision_gain"),
            "avg_edited_mae_vs_revision_target": avg("heuristic", "edited_mae_vs_revision_target"),
            "avg_signed_area_error": avg("heuristic", "signed_area_error", calibration=True),
            "avg_peak_delta_error": avg("heuristic", "peak_delta_error", calibration=True),
        },
        "best_fit": {
            "avg_revision_gain": avg("best_fit", "revision_gain"),
            "avg_edited_mae_vs_revision_target": avg("best_fit", "edited_mae_vs_revision_target"),
            "avg_signed_area_error": avg("best_fit", "signed_area_error", calibration=True),
            "avg_peak_delta_error": avg("best_fit", "peak_delta_error", calibration=True),
        },
        "results": [asdict(row) for row in rows],
    }

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze executor-manifold alignment on XTraffic step samples.")
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    summary = analyze_benchmark(args.benchmark, args.output)
    print(json.dumps({k: v for k, v in summary.items() if k != 'results'}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
