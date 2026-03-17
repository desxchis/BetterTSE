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

from modules.xtraffic_data import (
    XTRAFFIC_CHANNELS,
    align_incident_to_sensor,
    extract_node_series,
    load_xtraffic_minimal,
    time_to_step_index,
)


def _default_context_text(row: pd.Series) -> str:
    area = str(row.get("AREA") or "").strip()
    desc = str(row.get("DESCRIPTION") or "").strip()
    location = str(row.get("LOCATION") or "").strip()
    dt = row.get("dt")
    time_text = dt.strftime("%Y-%m-%d %H:%M") if pd.notna(dt) else "unknown time"
    parts = [f"{time_text}"]
    if area:
        parts.append(area)
    if desc:
        parts.append(desc)
    if location:
        parts.append(location)
    return "；".join(parts)


def build_candidates(
    data_dir: str,
    output_dir: str,
    shard_name: str = "p01_done.npy",
    seq_len: int = 288,
    pred_len: int = 144,
    channel: int = 0,
    max_incidents: int | None = None,
    max_distance_km: float = 5.0,
) -> Dict[str, Any]:
    bundle = load_xtraffic_minimal(data_dir=data_dir, shard_name=shard_name)
    incidents = bundle.incidents.copy()

    # Keep only incidents that can possibly define a history/future window inside the shard.
    incidents = incidents[incidents["dt"].notna()].copy()
    incidents = incidents.sort_values("dt").reset_index(drop=True)
    if max_incidents is not None:
        incidents = incidents.head(max_incidents)

    candidates: List[Dict[str, Any]] = []
    for _, row in incidents.iterrows():
        step_idx = time_to_step_index(bundle, row["dt"])
        if step_idx is None:
            continue
        history_start = step_idx - seq_len
        future_end = step_idx + pred_len
        if history_start < 0 or future_end > bundle.num_steps:
            continue

        alignment = align_incident_to_sensor(bundle, row)
        if alignment is None:
            continue
        if alignment["distance_km"] > max_distance_km:
            continue

        node_series = extract_node_series(bundle, alignment["node_index"], channel=channel)
        history = node_series[history_start:step_idx]
        future = node_series[step_idx:future_end]
        if history.size != seq_len or future.size != pred_len:
            continue
        if not np.isfinite(history).any() or not np.isfinite(future).any():
            continue

        candidates.append(
            {
                "incident_id": int(row["incident_id"]) if pd.notna(row["incident_id"]) else None,
                "incident_time": row["dt"].isoformat(),
                "incident_duration_raw": None if pd.isna(row["duration"]) else float(row["duration"]),
                "incident_type": row.get("Type"),
                "incident_description": row.get("DESCRIPTION"),
                "incident_location": row.get("LOCATION"),
                "incident_area": row.get("AREA"),
                "incident_fwy": None if pd.isna(row.get("Fwy")) else int(row["Fwy"]),
                "incident_direction": row.get("Freeway_direction"),
                "incident_lat": None if pd.isna(row.get("Latitude")) else float(row["Latitude"]),
                "incident_lng": None if pd.isna(row.get("Longitude")) else float(row["Longitude"]),
                "context_text": _default_context_text(row),
                "station_id": alignment["station_id"],
                "node_index": alignment["node_index"],
                "sensor_name": alignment["sensor_name"],
                "sensor_direction": alignment["sensor_direction"],
                "sensor_fwy": alignment["sensor_fwy"],
                "sensor_lat": alignment["sensor_lat"],
                "sensor_lng": alignment["sensor_lng"],
                "distance_km": alignment["distance_km"],
                "channel_index": channel,
                "channel_name": XTRAFFIC_CHANNELS[channel],
                "shard_name": shard_name,
                "step_index": step_idx,
                "history_start_idx": history_start,
                "history_end_idx": step_idx,
                "future_start_idx": step_idx,
                "future_end_idx": future_end,
                "history_nan_ratio": float(np.isnan(history).mean()),
                "future_nan_ratio": float(np.isnan(future).mean()),
            }
        )

    summary = {
        "dataset_name": "XTraffic",
        "schema_version": "xtraffic_real_revision_candidates_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_dir": str(data_dir),
        "shard_name": shard_name,
        "seq_len": seq_len,
        "pred_len": pred_len,
        "channel_index": channel,
        "channel_name": XTRAFFIC_CHANNELS[channel],
        "num_candidates": len(candidates),
        "max_distance_km": max_distance_km,
        "candidates": candidates,
    }

    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    json_path = out_root / f"xtraffic_candidates_{Path(shard_name).stem}_{XTRAFFIC_CHANNELS[channel]}.json"
    csv_path = out_root / f"xtraffic_candidates_{Path(shard_name).stem}_{XTRAFFIC_CHANNELS[channel]}.csv"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    pd.DataFrame(candidates).to_csv(csv_path, index=False)
    return {
        "output_json": str(json_path),
        "output_csv": str(csv_path),
        "num_candidates": len(candidates),
        "channel_name": XTRAFFIC_CHANNELS[channel],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build aligned XTraffic incident candidates for real forecast revision.")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--shard-name", default="p01_done.npy")
    parser.add_argument("--seq-len", type=int, default=288)
    parser.add_argument("--pred-len", type=int, default=144)
    parser.add_argument("--channel", type=int, default=0, choices=[0, 1, 2])
    parser.add_argument("--max-incidents", type=int, default=None)
    parser.add_argument("--max-distance-km", type=float, default=5.0)
    args = parser.parse_args()

    result = build_candidates(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        shard_name=args.shard_name,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        channel=args.channel,
        max_incidents=args.max_incidents,
        max_distance_km=args.max_distance_km,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
