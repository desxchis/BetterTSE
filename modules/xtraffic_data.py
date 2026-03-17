from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd


# XTraffic paper/project description states the shard contains
# traffic flow, lane occupancy, and average speed in 5-minute steps.
XTRAFFIC_CHANNELS = ("flow", "occupancy", "speed")


@dataclass
class XTrafficMinimalBundle:
    shard_name: str
    shard_start: pd.Timestamp
    freq_minutes: int
    series: np.memmap
    node_order: np.ndarray
    sensor_meta: pd.DataFrame
    incidents: pd.DataFrame

    @property
    def num_steps(self) -> int:
        return int(self.series.shape[0])

    @property
    def num_nodes(self) -> int:
        return int(self.series.shape[1])

    @property
    def num_channels(self) -> int:
        return int(self.series.shape[2])

    @property
    def shard_end(self) -> pd.Timestamp:
        return self.shard_start + pd.Timedelta(minutes=self.freq_minutes * (self.num_steps - 1))


def _normalize_direction(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    text = str(value).strip().upper()
    return text[:1] if text else ""


def _haversine_km(lat1: np.ndarray, lon1: np.ndarray, lat2: float, lon2: float) -> np.ndarray:
    r = 6371.0
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0) ** 2
    return 2.0 * r * np.arcsin(np.sqrt(a))


def load_xtraffic_minimal(
    data_dir: str | Path,
    shard_name: str = "p01_done.npy",
    incidents_file: str = "incidents_y2023.csv",
    sensor_meta_file: str = "sensor_meta_feature.csv",
    node_order_file: str = "node_order.npy",
    shard_start: str = "2023-01-01 00:00:00",
    freq_minutes: int = 5,
) -> XTrafficMinimalBundle:
    root = Path(data_dir)
    series = np.load(root / shard_name, mmap_mode="r")
    node_order = np.load(root / node_order_file)

    sensor_meta = pd.read_csv(root / sensor_meta_file, sep="\t").copy()
    sensor_meta["station_id"] = pd.to_numeric(sensor_meta["station_id"], errors="coerce").astype("Int64")
    sensor_meta["Direction"] = sensor_meta["Direction"].map(_normalize_direction)
    sensor_meta["Fwy"] = pd.to_numeric(sensor_meta["Fwy"], errors="coerce").astype("Int64")
    sensor_meta["Lat"] = pd.to_numeric(sensor_meta["Lat"], errors="coerce")
    sensor_meta["Lng"] = pd.to_numeric(sensor_meta["Lng"], errors="coerce")

    incidents = pd.read_csv(root / incidents_file, sep="\t", low_memory=False).copy()
    incidents["dt"] = pd.to_datetime(incidents["dt"], errors="coerce")
    incidents["duration"] = pd.to_numeric(incidents["duration"], errors="coerce")
    incidents["Fwy"] = pd.to_numeric(incidents["Fwy"], errors="coerce").astype("Int64")
    incidents["Freeway_direction"] = incidents["Freeway_direction"].map(_normalize_direction)
    incidents["Latitude"] = pd.to_numeric(incidents["Latitude"], errors="coerce")
    incidents["Longitude"] = pd.to_numeric(incidents["Longitude"], errors="coerce")
    incidents["incident_id"] = pd.to_numeric(incidents["incident_id"], errors="coerce").astype("Int64")

    return XTrafficMinimalBundle(
        shard_name=shard_name,
        shard_start=pd.Timestamp(shard_start),
        freq_minutes=freq_minutes,
        series=series,
        node_order=node_order.astype(np.int64),
        sensor_meta=sensor_meta,
        incidents=incidents,
    )


def build_xtraffic_time_index(bundle: XTrafficMinimalBundle) -> pd.DatetimeIndex:
    return pd.date_range(bundle.shard_start, periods=bundle.num_steps, freq=f"{bundle.freq_minutes}min")


def time_to_step_index(bundle: XTrafficMinimalBundle, timestamp: pd.Timestamp) -> int | None:
    if pd.isna(timestamp):
        return None
    if timestamp < bundle.shard_start or timestamp > bundle.shard_end:
        return None
    delta = timestamp - bundle.shard_start
    return int(delta.total_seconds() // (bundle.freq_minutes * 60))


def align_incident_to_sensor(
    bundle: XTrafficMinimalBundle,
    incident_row: pd.Series,
) -> Dict[str, Any] | None:
    lat = incident_row.get("Latitude")
    lon = incident_row.get("Longitude")
    if pd.isna(lat) or pd.isna(lon):
        return None

    meta = bundle.sensor_meta
    fwy = incident_row.get("Fwy")
    direction = incident_row.get("Freeway_direction", "")

    candidates = meta[meta["Lat"].notna() & meta["Lng"].notna()].copy()
    if pd.notna(fwy):
        same_fwy = candidates[candidates["Fwy"] == int(fwy)]
        if not same_fwy.empty:
            candidates = same_fwy
    if direction:
        same_dir = candidates[candidates["Direction"] == direction]
        if not same_dir.empty:
            candidates = same_dir

    distances = _haversine_km(
        candidates["Lat"].to_numpy(dtype=np.float64),
        candidates["Lng"].to_numpy(dtype=np.float64),
        float(lat),
        float(lon),
    )
    best_idx = int(np.argmin(distances))
    best_row = candidates.iloc[best_idx]
    station_id = int(best_row["station_id"])

    node_matches = np.where(bundle.node_order == station_id)[0]
    if node_matches.size == 0:
        return None

    return {
        "station_id": station_id,
        "node_index": int(node_matches[0]),
        "sensor_name": best_row.get("Name"),
        "sensor_direction": best_row.get("Direction"),
        "sensor_fwy": int(best_row["Fwy"]) if pd.notna(best_row["Fwy"]) else None,
        "sensor_lat": float(best_row["Lat"]),
        "sensor_lng": float(best_row["Lng"]),
        "distance_km": float(distances[best_idx]),
    }


def extract_node_series(
    bundle: XTrafficMinimalBundle,
    node_index: int,
    channel: int = 0,
) -> np.ndarray:
    if channel < 0 or channel >= bundle.num_channels:
        raise ValueError(f"channel out of range: {channel}")
    return np.asarray(bundle.series[:, node_index, channel], dtype=np.float64)
