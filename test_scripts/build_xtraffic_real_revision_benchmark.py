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
from forecasting.registry import load_baseline
from modules.forecast_revision import ForecastRevisionSample
from modules.xtraffic_data import XTRAFFIC_CHANNELS, extract_node_series, load_xtraffic_minimal


def _impute_series(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).copy()
    finite = np.isfinite(arr)
    if finite.all():
        return arr
    if not finite.any():
        return np.zeros_like(arr)
    idx = np.arange(len(arr))
    arr[~finite] = np.interp(idx[~finite], idx[finite], arr[finite])
    return arr


def _duration_bucket(duration_minutes: float | None, pred_len: int, freq_minutes: int = 5) -> tuple[str, int]:
    if duration_minutes is None or not np.isfinite(duration_minutes):
        steps = max(12, pred_len // 4)
    else:
        steps = max(6, int(round(float(duration_minutes) / freq_minutes)))
    steps = min(pred_len, steps)
    if steps <= max(12, pred_len // 6):
        bucket = "short"
    elif steps >= max(36, pred_len // 2):
        bucket = "long"
    else:
        bucket = "medium"
    return bucket, int(steps)


def _strength_bucket(drop_z: float) -> str:
    if drop_z < 0.75:
        return "weak"
    if drop_z < 1.25:
        return "medium"
    return "strong"


def _infer_operator(description: str, incident_type: str) -> Dict[str, str]:
    text = f"{incident_type} {description}".lower()
    if any(token in text for token in ("chain", "closure", "closed", "control")):
        return {
            "shape": "flatline",
            "effect_family": "shutdown",
            "direction": "down",
            "tool_name": "hybrid_down",
        }
    if any(token in text for token in ("fog", "wind", "animal", "debris", "hazard")):
        return {
            "shape": "hump",
            "effect_family": "impulse",
            "direction": "down",
            "tool_name": "spike_inject",
        }
    if any(token in text for token in ("fire", "collision", "injur", "hit and run", "car fire")):
        return {
            "shape": "step",
            "effect_family": "level",
            "direction": "down",
            "tool_name": "step_shift",
        }
    return {
        "shape": "step",
        "effect_family": "level",
        "direction": "down",
        "tool_name": "step_shift",
    }


def _response_stats(history: np.ndarray, future: np.ndarray) -> Dict[str, float]:
    h = np.asarray(history, dtype=np.float64)
    f = np.asarray(future, dtype=np.float64)
    h = h[np.isfinite(h)]
    f = f[np.isfinite(f)]
    if h.size == 0 or f.size == 0:
        return {"history_mean": 0.0, "future_mean": 0.0, "history_std": 1e-6, "drop_z": 0.0}
    h_mean = float(np.mean(h))
    f_mean = float(np.mean(f))
    h_std = max(float(np.std(h)), 1e-6)
    return {
        "history_mean": h_mean,
        "future_mean": f_mean,
        "history_std": h_std,
        "drop_z": (h_mean - f_mean) / h_std,
    }


def _refine_duration(
    shape: str,
    duration_minutes: float | None,
    pred_len: int,
    drop_z: float,
    freq_minutes: int = 5,
) -> tuple[str, int]:
    bucket, steps = _duration_bucket(duration_minutes, pred_len, freq_minutes=freq_minutes)
    if shape == "hump":
        if duration_minutes is None or not np.isfinite(duration_minutes):
            steps = min(24, max(6, pred_len // 8))
        else:
            steps = min(max(6, int(round(float(duration_minutes) / freq_minutes))), 24)
        bucket = "short" if steps <= 24 else "medium"
    elif shape == "step":
        if drop_z < 0.8:
            steps = min(steps, 24)
            bucket = "short" if steps <= 24 else bucket
        elif drop_z >= 1.2:
            steps = min(max(steps, 24), pred_len // 2 if pred_len >= 2 else pred_len)
            bucket = "medium" if steps <= max(36, pred_len // 2) else "long"
    elif shape == "flatline":
        steps = min(max(steps, 36), pred_len)
        bucket = "long" if steps >= max(36, pred_len // 2) else "medium"
    return bucket, int(steps)


def _context_text(row: Dict[str, Any], shape: str, duration_bucket: str) -> str:
    desc = str(row.get("incident_description") or "").strip()
    area = str(row.get("incident_area") or "").strip()
    location = str(row.get("incident_location") or "").strip()
    type_text = str(row.get("incident_type") or "").strip()
    duration_text = {
        "short": "短时间内",
        "medium": "一段时间内",
        "long": "较长一段时间内",
    }[duration_bucket]
    prefix = "在预测窗口前段"

    if any(token in desc.lower() for token in ("chain", "closure", "control")):
        return f"{prefix}，{area}附近的{type_text}事件（{desc}，位置：{location}）预计会使相关流量降到极低水平，并在{duration_text}接近停滞。"
    if shape == "hump":
        return f"{prefix}，{area}附近出现{type_text}事件（{desc}，位置：{location}），相关流量可能短时下探，随后逐步恢复。"
    return f"{prefix}，{area}附近发生{type_text}事件（{desc}，位置：{location}），相关流量预计会切换到更低位，并在{duration_text}维持较低水平。"


def _no_revision_context_text(row: Dict[str, Any]) -> str:
    area = str(row.get("incident_area") or "").strip()
    sensor_name = str(row.get("sensor_name") or "").strip()
    location = area or sensor_name or "相关路段"
    return f"在预测窗口前段，{location}附近无新增影响，暂无额外冲击，维持原预测，没有新的修正信号。"


def _build_non_applicable_samples(
    *,
    positive_rows: List[Dict[str, Any]],
    bundle: Any,
    baseline: Any,
    seq_len: int,
    pred_len: int,
    max_non_applicable_abs_z: float,
    target_count: int,
) -> List[Dict[str, Any]]:
    if target_count <= 0 or not positive_rows:
        return []

    incident_steps_by_node: Dict[int, List[int]] = {}
    for row in positive_rows:
        incident_steps_by_node.setdefault(int(row["node_index"]), []).append(int(row["step_index"]))
    for node_index in incident_steps_by_node:
        incident_steps_by_node[node_index] = sorted(set(incident_steps_by_node[node_index]))

    exclusion = pred_len
    samples: List[Dict[str, Any]] = []
    seen_keys: set[Tuple[int, int]] = set()
    shard_limit = int(bundle.num_steps)

    for row in positive_rows:
        if len(samples) >= target_count:
            break
        node_index = int(row["node_index"])
        node_series = extract_node_series(bundle, node_index=node_index, channel=int(row["channel_index"]))
        incident_steps = incident_steps_by_node.get(node_index, [])
        station_id = int(row["station_id"])
        sensor_name = row.get("sensor_name")
        incident_area = row.get("incident_area")

        candidate_starts = list(range(pred_len, max(pred_len + 1, shard_limit - pred_len), pred_len))
        candidate_starts.sort(key=lambda start: abs(start - int(row["future_start_idx"])))

        for future_start_idx in candidate_starts:
            if len(samples) >= target_count:
                break
            future_end_idx = future_start_idx + pred_len
            history_end_idx = future_start_idx
            history_start_idx = history_end_idx - seq_len
            if history_start_idx < 0 or future_end_idx > shard_limit:
                continue
            if any(abs(future_start_idx - step) <= exclusion for step in incident_steps):
                continue
            sample_key = (node_index, future_start_idx)
            if sample_key in seen_keys:
                continue

            history = _impute_series(node_series[history_start_idx:history_end_idx])
            future_gt = _impute_series(node_series[future_start_idx:future_end_idx])
            if history.size != seq_len or future_gt.size != pred_len:
                continue

            response = _response_stats(history, future_gt)
            if abs(response["drop_z"]) > max_non_applicable_abs_z:
                continue

            base_forecast = np.asarray(baseline.predict(history, pred_len), dtype=np.float64)
            mask = np.zeros(pred_len, dtype=int)
            sample = ForecastRevisionSample(
                sample_id=f"NA_{len(samples) + 1:03d}",
                dataset_name="XTraffic",
                history_ts=history.astype(float).tolist(),
                future_gt=future_gt.astype(float).tolist(),
                base_forecast=base_forecast.astype(float).tolist(),
                revision_target=base_forecast.astype(float).tolist(),
                context_text=_no_revision_context_text(
                    {
                        "incident_area": incident_area,
                        "sensor_name": sensor_name,
                    }
                ),
                forecast_horizon=pred_len,
                edit_mask_gt=mask.astype(int).tolist(),
                delta_gt=np.zeros(pred_len, dtype=np.float64).tolist(),
                revision_applicable_gt=False,
                edit_intent_gt={
                    "effect_family": "none",
                    "direction": "neutral",
                    "shape": "none",
                    "duration": "none",
                    "strength": "none",
                    "label_source": "real_no_incident_window",
                },
                effect_family_gt="none",
                direction_gt="neutral",
                shape_gt="none",
                strength_bucket_gt="none",
                duration_bucket_gt="none",
                revision_operator_family="none",
                revision_operator_params={
                    "region": [0, 0],
                    "bucket": "none",
                    "label_source": "real_no_incident_window",
                    "station_id": station_id,
                    "node_index": node_index,
                    "channel_name": row["channel_name"],
                    "response_drop_z": response["drop_z"],
                    "history_mean": response["history_mean"],
                    "future_mean": response["future_mean"],
                },
                timestamp=datetime.now(timezone.utc).isoformat(),
            )
            sample_dict = sample.to_dict()
            sample_dict["real_event_metadata"] = {
                "incident_id": None,
                "incident_time": None,
                "incident_type": "none",
                "incident_description": "no_new_incident_window",
                "incident_location": None,
                "incident_area": incident_area,
                "station_id": station_id,
                "node_index": node_index,
                "sensor_name": sensor_name,
                "history_start_idx": history_start_idx,
                "future_start_idx": future_start_idx,
                "future_end_idx": future_end_idx,
                "response_drop_z": response["drop_z"],
                "label_source": "real_no_incident_window",
            }
            samples.append(sample_dict)
            seen_keys.add(sample_key)

    return samples


def build_real_benchmark(
    candidate_json: str,
    data_dir: str,
    output_dir: str,
    baseline_name: str,
    baseline_model_dir: str | None = None,
    max_samples: int | None = None,
    min_response_z: float = 0.2,
    allowed_types: List[str] | None = None,
    num_non_applicable: int = 0,
    max_non_applicable_abs_z: float = 0.15,
) -> Dict[str, Any]:
    with open(candidate_json, "r", encoding="utf-8") as f:
        candidate_payload = json.load(f)
    candidates = candidate_payload.get("candidates", [])
    if max_samples is not None:
        candidates = candidates[:max_samples]

    bundle = load_xtraffic_minimal(
        data_dir=data_dir,
        shard_name=candidate_payload["shard_name"],
    )
    pred_len = int(candidates[0]["future_end_idx"] - candidates[0]["future_start_idx"]) if candidates else 0
    seq_len = int(candidates[0]["history_end_idx"] - candidates[0]["history_start_idx"]) if candidates else 0
    baseline = (
        load_baseline(baseline_name, baseline_model_dir)
        if baseline_model_dir
        else create_baseline(baseline_name, context_length=seq_len, prediction_length=pred_len, seq_len=seq_len, pred_len=pred_len)
    )

    samples: List[Dict[str, Any]] = []
    positive_rows: List[Dict[str, Any]] = []
    for idx, row in enumerate(candidates, start=1):
        if allowed_types and str(row.get("incident_type")) not in allowed_types:
            continue
        node_series = extract_node_series(bundle, int(row["node_index"]), channel=int(row["channel_index"]))
        history = _impute_series(node_series[int(row["history_start_idx"]):int(row["history_end_idx"])])
        future_gt = _impute_series(node_series[int(row["future_start_idx"]):int(row["future_end_idx"])])
        if history.size != seq_len or future_gt.size != pred_len:
            continue
        response = _response_stats(history, future_gt)
        if response["drop_z"] < min_response_z:
            continue
        base_forecast = baseline.predict(history, pred_len)

        op = _infer_operator(
            description=str(row.get("incident_description") or ""),
            incident_type=str(row.get("incident_type") or ""),
        )
        strength_bucket = _strength_bucket(response["drop_z"])
        duration_bucket, duration_steps = _refine_duration(
            shape=op["shape"],
            duration_minutes=row.get("incident_duration_raw"),
            pred_len=pred_len,
            drop_z=response["drop_z"],
        )
        mask = np.zeros(pred_len, dtype=int)
        mask[:duration_steps] = 1
        context_text = _context_text(row, op["shape"], duration_bucket)

        sample = ForecastRevisionSample(
            sample_id=f"{idx:03d}",
            dataset_name="XTraffic",
            history_ts=history.astype(float).tolist(),
            future_gt=future_gt.astype(float).tolist(),
            base_forecast=np.asarray(base_forecast, dtype=np.float64).astype(float).tolist(),
            revision_target=future_gt.astype(float).tolist(),
            context_text=context_text,
            forecast_horizon=pred_len,
            edit_mask_gt=mask.astype(int).tolist(),
            delta_gt=(future_gt - np.asarray(base_forecast, dtype=np.float64)).astype(float).tolist(),
            revision_applicable_gt=True,
            edit_intent_gt={
                "effect_family": op["effect_family"],
                "direction": op["direction"],
                "shape": op["shape"],
                "duration": duration_bucket,
                "strength": strength_bucket,
                "label_source": "weak_real_incident",
            },
            effect_family_gt=op["effect_family"],
            direction_gt=op["direction"],
            shape_gt=op["shape"],
            strength_bucket_gt=strength_bucket,
            duration_bucket_gt=duration_bucket,
            revision_operator_family=op["shape"],
            revision_operator_params={
                "region": [0, duration_steps],
                "bucket": "early_horizon",
                "label_source": "weak_real_incident",
                "incident_id": row["incident_id"],
                "station_id": row["station_id"],
                "node_index": row["node_index"],
                "channel_name": row["channel_name"],
                "distance_km": row["distance_km"],
                "response_drop_z": response["drop_z"],
                "history_mean": response["history_mean"],
                "future_mean": response["future_mean"],
            },
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        sample_dict = sample.to_dict()
        sample_dict["real_event_metadata"] = {
            "incident_id": row["incident_id"],
            "incident_time": row["incident_time"],
            "incident_type": row["incident_type"],
            "incident_description": row["incident_description"],
            "incident_location": row["incident_location"],
            "incident_area": row["incident_area"],
            "station_id": row["station_id"],
            "node_index": row["node_index"],
            "sensor_name": row["sensor_name"],
            "distance_km": row["distance_km"],
            "response_drop_z": response["drop_z"],
            "history_mean": response["history_mean"],
            "future_mean": response["future_mean"],
        }
        samples.append(sample_dict)
        positive_rows.append(row)

    non_applicable_samples = _build_non_applicable_samples(
        positive_rows=positive_rows,
        bundle=bundle,
        baseline=baseline,
        seq_len=seq_len,
        pred_len=pred_len,
        max_non_applicable_abs_z=max_non_applicable_abs_z,
        target_count=num_non_applicable,
    )
    samples.extend(non_applicable_samples)

    payload = {
        "dataset_name": "XTraffic",
        "schema_version": "forecast_revision_real_v1",
        "benchmark_type": "real_context_weak_labels",
        "label_source": "incident_to_text_heuristic",
        "channel_name": candidate_payload["channel_name"],
        "shard_name": candidate_payload["shard_name"],
        "baseline_name": baseline_name,
        "min_response_z": min_response_z,
        "allowed_types": allowed_types,
        "num_non_applicable": num_non_applicable,
        "max_non_applicable_abs_z": max_non_applicable_abs_z,
        "samples": samples,
    }

    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    json_path = out_root / f"forecast_revision_XTraffic_{baseline_name}_{len(samples)}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return {
        "output_path": str(json_path),
        "num_samples": len(samples),
        "baseline_name": baseline_name,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a real-context XTraffic forecast revision benchmark from aligned candidates.")
    parser.add_argument("--candidate-json", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--baseline-name", default="naive_last")
    parser.add_argument("--baseline-model-dir", default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--min-response-z", type=float, default=0.2)
    parser.add_argument("--allowed-types", nargs="*", default=None)
    parser.add_argument("--num-non-applicable", type=int, default=0)
    parser.add_argument("--max-non-applicable-abs-z", type=float, default=0.15)
    args = parser.parse_args()

    result = build_real_benchmark(
        candidate_json=args.candidate_json,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        baseline_name=args.baseline_name,
        baseline_model_dir=args.baseline_model_dir,
        max_samples=args.max_samples,
        min_response_z=args.min_response_z,
        allowed_types=args.allowed_types,
        num_non_applicable=args.num_non_applicable,
        max_non_applicable_abs_z=args.max_non_applicable_abs_z,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
