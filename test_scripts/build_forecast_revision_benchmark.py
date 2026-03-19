from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from forecasting import create_baseline
from forecasting.data_utils import load_univariate_series
from forecasting.registry import load_baseline
from modules.forecast_revision_benchmark import (
    anchor_forecast_to_history,
    apply_physical_revision_injection,
)
from modules.forecast_revision import (
    ForecastRevisionSample,
    extract_gt_edit_spec,
)


OPERATORS = ("hump", "step", "flatline", "irregular_noise", "plateau")
BUCKETS = ("early_horizon", "mid_horizon", "late_horizon")

def _operator_spec(index: int, include_no_revision_every: int = 0) -> Dict[str, Any]:
    if include_no_revision_every > 0 and (index + 1) % include_no_revision_every == 0:
        return {
            "operator": "none",
            "bucket": "none",
            "direction": "neutral",
            "strength": "none",
            "duration_bucket": "none",
        }
    positive_index = index
    if include_no_revision_every > 0:
        positive_index = index - ((index + 1) // include_no_revision_every)
    op = OPERATORS[positive_index % len(OPERATORS)]
    bucket = BUCKETS[positive_index % len(BUCKETS)]
    direction = "up" if op in {"hump", "plateau"} else "down"
    if op == "irregular_noise":
        direction = "neutral"
    return {
        "operator": op,
        "bucket": bucket,
        "direction": direction,
        "strength": "medium" if index % 2 == 0 else "strong",
        "duration_bucket": "short" if op == "hump" else "medium",
    }


def _bucket_region(bucket: str, horizon: int, duration_bucket: str) -> List[int]:
    if duration_bucket == "short":
        width = max(4, horizon // 6)
    else:
        width = max(6, horizon // 4)
    if bucket == "early_horizon":
        start = 0
    elif bucket == "late_horizon":
        start = max(0, horizon - width)
    else:
        start = max(0, (horizon - width) // 2)
    return [start, min(horizon, start + width)]


def _generic_context_text(spec: Dict[str, Any]) -> str:
    if spec["operator"] == "none":
        return "当前没有新的外部信号会改变这段未来预测，整体可维持原预测。"
    bucket_phrase = {
        "early_horizon": "在预测窗口前段",
        "mid_horizon": "在预测窗口中段",
        "late_horizon": "在预测窗口后段",
    }[spec["bucket"]]
    if spec["operator"] == "step":
        return f"{bucket_phrase}，系统状态预计会突然切换到更低位并维持一段时间。"
    if spec["operator"] == "flatline":
        return f"{bucket_phrase}，相关指标可能降到极低水平并接近停滞。"
    if spec["operator"] == "irregular_noise":
        return f"{bucket_phrase}，监测输出可能出现持续的杂乱波动和失真。"
    if spec["operator"] == "plateau":
        return f"{bucket_phrase}，相关水平预计会抬升到偏高状态并维持一段时间。"
    return f"{bucket_phrase}，相关指标预计会短时冲高后逐步回落。"


def _traffic_incident_context_text(spec: Dict[str, Any]) -> str:
    if spec["operator"] == "none":
        return "目前没有新的事故、封闭或设备异常信号，未来路况可基本维持原预测。"

    bucket_phrase = {
        "early_horizon": "在预测窗口前段",
        "mid_horizon": "在预测窗口中段",
        "late_horizon": "在预测窗口后段",
    }[spec["bucket"]]

    if spec["operator"] == "step":
        return f"{bucket_phrase}，相关路段可能出现车道封闭或通行受限，通行水平会切换到更低位并维持一段时间。"
    if spec["operator"] == "flatline":
        return f"{bucket_phrase}，受持续性封控或采集异常影响，观测值可能降到极低水平并接近停滞。"
    if spec["operator"] == "irregular_noise":
        return f"{bucket_phrase}，检测设备可能受到干扰，监测输出会出现持续的杂乱波动和失真。"
    if spec["operator"] == "plateau":
        return f"{bucket_phrase}，受持续拥堵或绕行汇入影响，相关路况指标可能抬升到偏高状态并维持一段时间。"
    return f"{bucket_phrase}，受短时事故或突发汇流影响，相关路况指标可能先快速上冲，随后逐步恢复。"


def _context_text(spec: Dict[str, Any], context_style: str = "generic") -> str:
    if context_style == "traffic_incident":
        return _traffic_incident_context_text(spec)
    return _generic_context_text(spec)


def build_benchmark(
    csv_path: str,
    dataset_name: str,
    output_dir: str,
    baseline_name: str,
    baseline_model_dir: str | None,
    seq_len: int,
    pred_len: int,
    num_samples: int,
    target_col: str | None = None,
    include_no_revision_every: int = 0,
    context_style: str = "generic",
) -> Dict[str, Any]:
    series, feature = load_univariate_series(csv_path, target_col)
    if baseline_name == "patchtst" and not baseline_model_dir:
        raise ValueError(
            "baseline_name='patchtst' requires --baseline-model-dir with a trained PatchTST checkpoint."
        )
    baseline = (
        load_baseline(baseline_name, baseline_model_dir)
        if baseline_model_dir
        else create_baseline(baseline_name, context_length=seq_len, prediction_length=pred_len, seq_len=seq_len, pred_len=pred_len)
    )
    samples: List[Dict[str, Any]] = []
    step = max(1, pred_len // 2)

    max_start = max(0, len(series) - seq_len - pred_len - 1)
    starts = list(range(0, max_start + 1, step))
    if not starts:
        raise ValueError("Not enough points to build forecast revision samples.")

    for idx, start in enumerate(starts[:num_samples]):
        history = series[start:start + seq_len]
        future_gt = series[start + seq_len:start + seq_len + pred_len]
        if len(future_gt) < pred_len:
            continue
        raw_base_forecast = np.asarray(baseline.predict(history, pred_len), dtype=np.float64)
        base_forecast, anchor_metadata = anchor_forecast_to_history(history, raw_base_forecast)
        spec = _operator_spec(idx, include_no_revision_every=include_no_revision_every)
        if spec["operator"] == "none":
            region = [0, 0]
            intent = {
                "effect_family": "none",
                "direction": "neutral",
                "shape": "none",
                "duration": "none",
                "strength": "none",
            }
            params = {}
            revision_target = np.asarray(base_forecast, dtype=np.float64).copy()
            delta = np.zeros(pred_len, dtype=np.float64)
            mask = np.zeros(pred_len, dtype=int)
            revision_applicable_gt = False
            injection_metadata = {"injection_type": "none", "region": region}
        else:
            region = _bucket_region(spec["bucket"], pred_len, spec["duration_bucket"])
            intent = {
                "effect_family": "volatility" if spec["operator"] == "irregular_noise" else ("shutdown" if spec["operator"] == "flatline" else "level" if spec["operator"] in {"step", "plateau"} else "impulse"),
                "direction": spec["direction"],
                "shape": spec["operator"],
                "duration": spec["duration_bucket"],
                "strength": spec["strength"],
            }
            params = {
                "amplitude": float(max(np.nanstd(history), np.nanstd(base_forecast), 1e-3) * (1.0 if spec["strength"] == "medium" else 1.6)),
                "duration": int(region[1] - region[0]),
                "onset_lag": 0,
                "recovery_rate": 0.35,
                "volatility_scale": 1.8,
                "floor_value": float(np.nanmin(base_forecast) - max(np.nanstd(base_forecast), 1e-3) * 1.2),
            }
            revision_target, delta, injection_metadata = apply_physical_revision_injection(
                base_forecast,
                intent,
                region,
                seed=idx + 7,
                params=params,
            )
            mask = np.zeros(pred_len, dtype=int)
            mask[region[0]:region[1]] = 1
            revision_applicable_gt = True

        sample = ForecastRevisionSample(
            sample_id=f"{idx + 1:03d}",
            dataset_name=dataset_name,
            history_ts=history.astype(float).tolist(),
            future_gt=future_gt.astype(float).tolist(),
            base_forecast=np.asarray(base_forecast, dtype=np.float64).astype(float).tolist(),
            revision_target=np.asarray(revision_target, dtype=np.float64).astype(float).tolist(),
            context_text=_context_text(spec, context_style=context_style),
            forecast_horizon=pred_len,
            edit_mask_gt=mask.astype(int).tolist(),
            delta_gt=np.asarray(delta, dtype=np.float64).astype(float).tolist(),
            revision_applicable_gt=revision_applicable_gt,
            edit_intent_gt=intent,
            effect_family_gt=intent["effect_family"],
            direction_gt=intent["direction"],
            shape_gt=intent["shape"],
            strength_bucket_gt=intent["strength"],
            duration_bucket_gt=intent["duration"],
            revision_operator_family=spec["operator"],
            revision_operator_params={
                "region": region,
                "bucket": spec["bucket"],
                "params": params,
                "anchor_metadata": anchor_metadata,
                "injection_metadata": injection_metadata,
            },
            edit_spec_gt=None,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        sample_dict = sample.to_dict()
        sample_dict["edit_spec_gt"] = extract_gt_edit_spec(sample_dict)
        samples.append(sample_dict)

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    target_construction_method = "synthetic_physical_injection"
    payload = {
        "dataset_name": dataset_name,
        "feature": feature,
        "task_family": "forecast_revision",
        "application_of": "bettertse_editing",
        "target_construction_method": target_construction_method,
        "baseline_name": baseline_name,
        "context_style": context_style,
        "schema_version": "forecast_revision_v1",
        "samples": samples,
    }
    json_path = output_root / f"forecast_revision_{dataset_name}_{baseline_name}_{len(samples)}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return {"output_path": str(json_path), "num_samples": len(samples), "feature": feature}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build synthetic forecast revision benchmark.")
    parser.add_argument("--csv-path", required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--baseline-name", default="patchtst")
    parser.add_argument("--baseline-model-dir", default=None)
    parser.add_argument("--seq-len", type=int, default=96)
    parser.add_argument("--pred-len", type=int, default=24)
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--target-col", default=None)
    parser.add_argument("--include-no-revision-every", type=int, default=0)
    parser.add_argument("--context-style", choices=["generic", "traffic_incident"], default="generic")
    args = parser.parse_args()

    summary = build_benchmark(
        csv_path=args.csv_path,
        dataset_name=args.dataset_name,
        output_dir=args.output_dir,
        baseline_name=args.baseline_name,
        baseline_model_dir=args.baseline_model_dir,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        num_samples=args.num_samples,
        target_col=args.target_col,
        include_no_revision_every=args.include_no_revision_every,
        context_style=args.context_style,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
