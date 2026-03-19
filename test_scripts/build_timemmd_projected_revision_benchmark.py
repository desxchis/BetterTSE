from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from forecasting import create_baseline
from forecasting.registry import load_baseline
from modules.forecast_revision import ForecastRevisionSample
from modules.forecast_revision_benchmark import (
    anchor_forecast_to_history,
    project_revision_target_from_future,
)
from modules.timemmd_data import load_timemmd_domain


def _normalize_text(text: str) -> str:
    return " ".join(str(text or "").replace("\n", " ").split())


def _select_pred_text(text: str) -> str:
    chunks = [chunk.strip() for chunk in str(text or "").split(";") if chunk.strip()]
    if not chunks:
        return ""
    for chunk in chunks:
        lowered = chunk.lower()
        if "short term" in lowered or "short-term" in lowered or "next 1-3 months" in lowered:
            return chunk
    return chunks[-1]


def _contains_any(text: str, needles: List[str]) -> bool:
    lowered = text.lower()
    return any(token in lowered for token in needles)


def _infer_intent_from_text(text: str) -> Dict[str, Any]:
    normalized = _normalize_text(text)
    lowered = normalized.lower()

    down_tokens = ["decrease", "decline", "drop", "fall", "lower", "downward", "decelerate", "slow down"]
    up_tokens = ["increase", "rise", "grow", "higher", "upward", "surge", "strengthen"]
    stable_tokens = ["remain stable", "stable", "unchanged", "flat", "plateau"]
    strong_tokens = ["sharp", "sharply", "significant", "significantly", "surge", "spike", "collapse"]
    weak_tokens = ["slight", "slightly", "modest", "modestly", "mild", "may"]

    if _contains_any(lowered, stable_tokens) and not _contains_any(lowered, down_tokens + up_tokens):
        return {
            "revision_applicable": False,
            "effect_family": "none",
            "direction": "neutral",
            "shape": "none",
            "duration": "none",
            "strength": "none",
            "bucket": "none",
            "tool_name": "none",
        }

    direction = "down" if _contains_any(lowered, down_tokens) else "up"
    if "volatile" in lowered or "volatility" in lowered:
        shape = "irregular_noise"
        effect_family = "volatility"
        direction = "neutral"
        tool_name = "volatility_increase"
    elif "surge" in lowered or "spike" in lowered:
        shape = "hump"
        effect_family = "impulse"
        tool_name = "spike_inject"
    elif "collapse" in lowered or "zero" in lowered or "shutdown" in lowered:
        shape = "flatline"
        effect_family = "shutdown"
        direction = "down"
        tool_name = "hybrid_down"
    elif "jump" in lowered or "step" in lowered or "reprice" in lowered:
        shape = "step"
        effect_family = "level"
        tool_name = "step_shift"
    else:
        shape = "plateau"
        effect_family = "level"
        tool_name = "hybrid_down" if direction == "down" else "hybrid_up"

    if _contains_any(lowered, strong_tokens):
        strength = "strong"
    elif _contains_any(lowered, weak_tokens):
        strength = "weak"
    else:
        strength = "medium"

    duration = "long" if ("4-18 months" in lowered or "long term" in lowered or "long-term" in lowered) else "medium"
    if "1-3 months" in lowered or "short term" in lowered or "short-term" in lowered:
        duration = "medium"

    return {
        "revision_applicable": True,
        "effect_family": effect_family,
        "direction": direction,
        "shape": shape,
        "duration": duration,
        "strength": strength,
        "bucket": "full_horizon",
        "tool_name": tool_name,
    }


def _build_context_text(raw_fact: str, raw_pred: str, intent: Dict[str, Any]) -> str:
    if not bool(intent.get("revision_applicable", False)):
        return "[RevisionHint] bucket=none; direction=neutral; shape=none; duration=none; strength=none. 当前没有新的外部修正信号，整体可维持原预测。"
    direction_phrase = "整体偏高" if intent["direction"] == "up" else "整体偏低"
    if intent["shape"] == "step":
        effect_phrase = "需要做一次阶跃式重估"
    elif intent["shape"] == "hump":
        effect_phrase = "需要做一次短期冲高后回落的修正"
    elif intent["shape"] == "flatline":
        effect_phrase = "需要压到极低水平"
    elif intent["shape"] == "irregular_noise":
        effect_phrase = "需要提高局部波动"
    else:
        effect_phrase = "需要做一次趋势性修正"
    pred_summary = _normalize_text(raw_pred)
    fact_summary = _normalize_text(raw_fact)
    return (
        f"[RevisionHint] bucket={intent['bucket']}; direction={intent['direction']}; "
        f"shape={intent['shape']}; duration={intent['duration']}; strength={intent['strength']}. "
        f"真实文本上下文：{fact_summary} "
        f"短期指引：{pred_summary} "
        f"请在整个预测窗口内执行修正，整体路径应{direction_phrase}，并且{effect_phrase}。"
    )


def _resolve_future_start(numerical: pd.DataFrame, text_end: pd.Timestamp) -> int | None:
    candidates = numerical.index[numerical["start_date"] > text_end]
    if len(candidates) == 0:
        return None
    return int(candidates[0])


def _direction_matches(base_forecast: np.ndarray, future_gt: np.ndarray, direction: str) -> bool:
    delta_mean = float(np.mean(np.asarray(future_gt, dtype=np.float64) - np.asarray(base_forecast, dtype=np.float64)))
    if abs(delta_mean) < 1e-3:
        return False
    if direction == "up":
        return delta_mean > 0.0
    if direction == "down":
        return delta_mean < 0.0
    return True


def build_timemmd_projected_benchmark(
    *,
    timemmd_root: str,
    domain: str,
    output_dir: str,
    baseline_name: str,
    baseline_model_dir: str | None = None,
    seq_len: int = 96,
    pred_len: int = 24,
    max_samples: int | None = None,
    text_source: str = "report",
    min_projection_gain_ratio: float = 0.08,
    min_explained_delta_ratio: float = 0.08,
) -> Dict[str, Any]:
    bundle = load_timemmd_domain(timemmd_root, domain)
    numerical = bundle["numerical"].copy()
    if text_source == "report":
        texts = bundle["report"].copy()
    elif text_source == "search":
        texts = bundle["search"].copy()
    else:
        texts = bundle["texts"].copy()
    texts = texts.sort_values("end_date").reset_index(drop=True)

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

    samples: List[Dict[str, Any]] = []
    for _, row in texts.iterrows():
        future_start_idx = _resolve_future_start(numerical, row["end_date"])
        if future_start_idx is None:
            continue
        hist_start = future_start_idx - seq_len
        future_end = future_start_idx + pred_len
        if hist_start < 0 or future_end > len(numerical):
            continue

        history = numerical.iloc[hist_start:future_start_idx]["OT"].astype(float).to_numpy()
        future_gt = numerical.iloc[future_start_idx:future_end]["OT"].astype(float).to_numpy()
        if history.size != seq_len or future_gt.size != pred_len:
            continue

        raw_pred_text = _select_pred_text(row["pred"])
        raw_fact_text = _normalize_text(row["fact"])
        intent_struct = _infer_intent_from_text(raw_pred_text or row["pred"])
        context_text = _build_context_text(raw_fact_text, raw_pred_text or row["pred"], intent_struct)

        raw_base = np.asarray(baseline.predict(history, pred_len), dtype=np.float64)
        base_forecast, anchor_metadata = anchor_forecast_to_history(history, raw_base)

        if bool(intent_struct.get("revision_applicable", False)):
            if not _direction_matches(base_forecast, future_gt, str(intent_struct.get("direction", "neutral"))):
                continue
            region = [0, pred_len]
            mask = np.ones(pred_len, dtype=int)
            target, residual, projection_metrics, projection_metadata = project_revision_target_from_future(
                history_ts=history,
                base_forecast=base_forecast,
                future_gt=future_gt,
                intent=intent_struct,
                region=region,
                context_text=context_text,
                seed=int(future_start_idx),
            )
            base_mae = float(projection_metrics.get("base_mae_vs_future_gt", 0.0))
            target_mae = float(projection_metrics.get("mae_target_vs_future_gt", 0.0))
            gain_ratio = ((base_mae - target_mae) / base_mae) if base_mae > 1e-8 else 0.0
            if gain_ratio < float(min_projection_gain_ratio):
                continue
            if float(projection_metrics.get("explained_delta_ratio", 0.0)) < float(min_explained_delta_ratio):
                continue
            delta = target - base_forecast
        else:
            region = [0, 0]
            mask = np.zeros(pred_len, dtype=int)
            target = base_forecast.copy()
            residual = future_gt - target
            projection_metrics = {
                "mae_target_vs_future_gt": float(np.mean(np.abs(target - future_gt))),
                "rmse_target_vs_future_gt": float(np.sqrt(np.mean((target - future_gt) ** 2))),
                "explained_delta_ratio": 0.0,
                "base_mae_vs_future_gt": float(np.mean(np.abs(base_forecast - future_gt))),
                "residual_mae": float(np.mean(np.abs(residual))),
                "delta_mae_scale": float(np.mean(np.abs(future_gt - base_forecast))),
            }
            projection_metadata = {"projection_family": "none", "projection_grid_size": 0, "best_loss": projection_metrics["rmse_target_vs_future_gt"]}
            delta = np.zeros_like(target)

        sample = ForecastRevisionSample(
            sample_id=f"{len(samples) + 1:03d}",
            dataset_name=f"Time-MMD_{domain}_{text_source}",
            history_ts=history.astype(float).tolist(),
            future_gt=future_gt.astype(float).tolist(),
            base_forecast=base_forecast.astype(float).tolist(),
            revision_target=target.astype(float).tolist(),
            context_text=context_text,
            forecast_horizon=pred_len,
            edit_mask_gt=mask.astype(int).tolist(),
            delta_gt=delta.astype(float).tolist(),
            revision_applicable_gt=bool(intent_struct.get("revision_applicable", False)),
            edit_intent_gt=dict(intent_struct),
            effect_family_gt=str(intent_struct.get("effect_family", "none")),
            direction_gt=str(intent_struct.get("direction", "neutral")),
            shape_gt=str(intent_struct.get("shape", "none")),
            strength_bucket_gt=str(intent_struct.get("strength", "none")),
            duration_bucket_gt=str(intent_struct.get("duration", "none")),
            revision_operator_family=str(intent_struct.get("shape", "none")),
            revision_operator_params={
                "region": region,
                "bucket": str(intent_struct.get("bucket", "full_horizon")),
                "tool_name": str(intent_struct.get("tool_name", "none")),
                "anchor_metadata": anchor_metadata,
                "projection_metadata": projection_metadata,
                "text_source": text_source,
                "raw_text": {
                    "fact": raw_fact_text,
                    "pred": _normalize_text(row["pred"]),
                },
                "timemmd_alignment": {
                    "text_start_date": str(row["start_date"].date()),
                    "text_end_date": str(row["end_date"].date()),
                    "history_end_date": str(numerical.iloc[future_start_idx - 1]["end_date"].date()),
                    "future_start_date": str(numerical.iloc[future_start_idx]["start_date"].date()),
                    "future_end_date": str(numerical.iloc[future_end - 1]["end_date"].date()),
                },
                "params": dict(projection_metadata.get("best_params", {})),
            },
            projection_residual=residual.astype(float).tolist(),
            revision_target_source="projected_from_future_and_text",
            target_projection_metrics=projection_metrics,
            intent_struct=dict(intent_struct),
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        samples.append(sample.to_dict())
        if max_samples is not None and len(samples) >= max_samples:
            break

    payload = {
        "dataset_name": f"Time-MMD_{domain}_{text_source}",
        "schema_version": "forecast_revision_projected_v1",
        "benchmark_type": "real_text_projected_revision",
        "baseline_name": baseline_name,
        "domain": domain,
        "text_source": text_source,
        "feature": "OT",
        "samples": samples,
    }

    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    json_path = out_root / f"forecast_revision_TimeMMD_{domain}_{text_source}_{baseline_name}_{len(samples)}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return {"output_path": str(json_path), "num_samples": len(samples)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build projected forecast revision benchmark from Time-MMD.")
    parser.add_argument("--timemmd-root", default="data/Time-MMD")
    parser.add_argument("--domain", default="Energy")
    parser.add_argument("--text-source", choices=["report", "search", "all"], default="report")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--baseline-name", default="dlinear_official")
    parser.add_argument("--baseline-model-dir", default=None)
    parser.add_argument("--seq-len", type=int, default=96)
    parser.add_argument("--pred-len", type=int, default=24)
    parser.add_argument("--max-samples", type=int, default=30)
    parser.add_argument("--min-projection-gain-ratio", type=float, default=0.08)
    parser.add_argument("--min-explained-delta-ratio", type=float, default=0.08)
    args = parser.parse_args()

    text_source = "all" if args.text_source == "all" else args.text_source
    result = build_timemmd_projected_benchmark(
        timemmd_root=args.timemmd_root,
        domain=args.domain,
        output_dir=args.output_dir,
        baseline_name=args.baseline_name,
        baseline_model_dir=args.baseline_model_dir,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        max_samples=args.max_samples,
        text_source=text_source,
        min_projection_gain_ratio=args.min_projection_gain_ratio,
        min_explained_delta_ratio=args.min_explained_delta_ratio,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
