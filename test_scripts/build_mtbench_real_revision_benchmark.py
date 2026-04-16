from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from forecasting import create_baseline
from forecasting.registry import load_baseline
from modules.forecast_revision_benchmark import (
    anchor_forecast_to_history,
    apply_physical_revision_injection,
)
from modules.forecast_revision import ForecastRevisionSample, calibrate_revision
from modules.mtbench_data import load_mtbench_aligned_pairs


def _safe_context_text(text_payload: Dict[str, Any], trend_payload: Dict[str, Any], revision_applicable: bool) -> str:
    content = str(text_payload.get("content") or "").strip()
    headline = content.splitlines()[0].strip() if content else ""
    sentiment = str(text_payload.get("label_sentiment") or "")
    label_type = str(text_payload.get("label_type") or "")
    label_time = str(text_payload.get("label_time") or "")
    direction = _trend_direction(trend_payload)

    if not revision_applicable:
        return "当前没有新的外部修正信号，整体可维持原预测。"

    if _finance_family(trend_payload) == "repricing":
        if direction == "up":
            summary = "市场可能对这条消息进行向上重新定价，带动整个预测窗口的价格路径上修。"
        else:
            summary = "市场可能对这条消息进行向下重新定价，带动整个预测窗口的价格路径下修。"
    else:
        if direction == "up":
            summary = "这条消息更像中短期偏多驱动，未来价格路径可能整体缓慢上修。"
        else:
            summary = "这条消息更像中短期偏空驱动，未来价格路径可能整体缓慢下修。"

    prefix = headline or "近期有新的公司相关消息。"
    suffix = f"文本标签参考：sentiment={sentiment}; type={label_type}; time={label_time}."
    return f"{prefix} {summary} {suffix}"


def _trend_direction(trend_payload: Dict[str, Any]) -> str:
    pct = trend_payload.get("output_percentage_change", trend_payload.get("overall_percentage_change"))
    try:
        pct_value = float(pct)
    except (TypeError, ValueError):
        pct_value = None

    if pct_value is not None:
        if pct_value > 0.25:
            return "up"
        if pct_value < -0.25:
            return "down"
    return "neutral"


def _strength_bucket(trend_payload: Dict[str, Any]) -> str:
    pct = trend_payload.get("output_percentage_change", trend_payload.get("overall_percentage_change"))
    try:
        pct_value = abs(float(pct))
    except (TypeError, ValueError):
        return "medium"
    if pct_value < 1.0:
        return "weak"
    if pct_value < 3.0:
        return "medium"
    return "strong"


def _finance_family(trend_payload: Dict[str, Any]) -> str:
    strength = _strength_bucket(trend_payload)
    return "repricing" if strength == "strong" else "drift_adjust"


def _duration_bucket(horizon: int, family: str) -> str:
    if family == "repricing":
        return "medium" if horizon <= 64 else "long"
    if horizon <= 32:
        return "short"
    return "long"


def _effect_family(direction: str) -> str:
    if direction == "neutral":
        return "none"
    return "level"


def build_mtbench_real_benchmark(
    dataset_path: str,
    output_dir: str,
    baseline_name: str,
    baseline_model_dir: str | None = None,
    max_samples: int | None = None,
    include_non_applicable: bool = False,
) -> Dict[str, Any]:
    records = load_mtbench_aligned_pairs(dataset_path, limit=max_samples)
    if not records:
        raise ValueError("No MTBench records loaded.")

    seq_len = len(records[0].input_window)
    pred_len = len(records[0].output_window)
    if baseline_name == "patchtst" and not baseline_model_dir:
        raise ValueError(
            "baseline_name='patchtst' requires --baseline-model-dir with a trained PatchTST checkpoint."
        )
    baseline = (
        load_baseline(baseline_name, baseline_model_dir)
        if baseline_model_dir
        else create_baseline(
            baseline_name,
            context_length=seq_len,
            prediction_length=pred_len,
            seq_len=seq_len,
            pred_len=pred_len,
        )
    )
    expected_horizon = getattr(baseline, "prediction_length", None)
    expected_context = getattr(baseline, "context_length", None)

    samples: List[Dict[str, Any]] = []
    for idx, record in enumerate(records, start=1):
        history = np.asarray(record.input_window, dtype=np.float64)
        future_gt = np.asarray(record.output_window, dtype=np.float64)
        if expected_context is not None and len(history) < int(expected_context):
            continue
        if expected_horizon is not None and len(future_gt) != int(expected_horizon):
            continue
        raw_base_forecast = np.asarray(baseline.predict(history, len(future_gt)), dtype=np.float64)
        base_forecast, anchor_metadata = anchor_forecast_to_history(history, raw_base_forecast)

        direction = _trend_direction(record.trend)
        family = _finance_family(record.trend)
        revision_applicable_gt = direction != "neutral"
        if include_non_applicable and idx % 5 == 0:
            revision_applicable_gt = False

        effect_family = _effect_family(direction) if revision_applicable_gt else "none"
        shape = ("step" if family == "repricing" else "plateau") if revision_applicable_gt else "none"
        duration_bucket = _duration_bucket(len(future_gt), family) if revision_applicable_gt else "none"
        strength_bucket = _strength_bucket(record.trend) if revision_applicable_gt else "none"

        region = [0, len(future_gt)] if revision_applicable_gt else [0, 0]
        intent_payload = {
            "effect_family": effect_family,
            "direction": direction if revision_applicable_gt else "neutral",
            "shape": shape,
            "duration": duration_bucket,
            "strength": strength_bucket,
        }

        if revision_applicable_gt:
            mask = np.ones(len(future_gt), dtype=int)
            context_text = _safe_context_text(record.text, record.trend, revision_applicable_gt)
            oracle_params = calibrate_revision(intent_payload, region, history, base_forecast)
            revision_target, delta, injection_metadata = apply_physical_revision_injection(
                base_forecast,
                intent_payload,
                region,
                seed=idx + 17,
                params=oracle_params,
            )
        else:
            mask = np.zeros(len(future_gt), dtype=int)
            delta = np.zeros(len(future_gt), dtype=np.float64)
            revision_target = base_forecast.copy()
            context_text = _safe_context_text(record.text, record.trend, revision_applicable_gt)
            oracle_params = {}
            injection_metadata = {"injection_type": "none", "region": region}

        sample = ForecastRevisionSample(
            sample_id=f"{idx:03d}",
            dataset_name="MTBench_finance_aligned_pairs_short",
            history_ts=history.astype(float).tolist(),
            future_gt=future_gt.astype(float).tolist(),
            base_forecast=base_forecast.astype(float).tolist(),
            revision_target=revision_target.astype(float).tolist(),
            context_text=context_text,
            forecast_horizon=len(future_gt),
            edit_mask_gt=mask.astype(int).tolist(),
            delta_gt=delta.astype(float).tolist(),
            revision_applicable_gt=revision_applicable_gt,
            edit_intent_gt={**intent_payload, "label_source": "mtbench_trend_heuristic"},
            effect_family_gt=effect_family,
            direction_gt=direction if revision_applicable_gt else "neutral",
            shape_gt=shape,
            strength_bucket_gt=strength_bucket,
            duration_bucket_gt=duration_bucket,
            revision_operator_family=shape,
            revision_operator_params={
                "region": region,
                "bucket": "full" if revision_applicable_gt else "none",
                "label_source": "mtbench_trend_heuristic_v2",
                "finance_family": family if revision_applicable_gt else "none",
                "params": oracle_params,
                "anchor_metadata": anchor_metadata,
                "injection_metadata": injection_metadata,
                "alignment": record.alignment,
                "trend": record.trend,
                "technical": record.technical,
                "text_metadata": {
                    "published_utc": record.text.get("published_utc"),
                    "label_sentiment": record.text.get("label_sentiment"),
                    "label_time": record.text.get("label_time"),
                    "label_type": record.text.get("label_type"),
                    "article_url": record.text.get("article_url"),
                },
            },
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        samples.append(sample.to_dict())

    payload = {
        "dataset_name": "MTBench_finance_aligned_pairs_short",
        "schema_version": "forecast_revision_real_v2",
        "benchmark_type": "real_context_mtbench",
        "label_source": "mtbench_trend_heuristic_v2",
        "baseline_name": baseline_name,
        "include_non_applicable": include_non_applicable,
        "samples": samples,
    }

    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    json_path = out_root / f"forecast_revision_MTBench_{baseline_name}_{len(samples)}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return {
        "output_path": str(json_path),
        "num_samples": len(samples),
        "baseline_name": baseline_name,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a real MTBench forecast revision benchmark from aligned pairs.")
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--baseline-name", default="patchtst")
    parser.add_argument("--baseline-model-dir", default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--include-non-applicable", action="store_true")
    args = parser.parse_args()

    result = build_mtbench_real_benchmark(
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        baseline_name=args.baseline_name,
        baseline_model_dir=args.baseline_model_dir,
        max_samples=args.max_samples,
        include_non_applicable=args.include_non_applicable,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
