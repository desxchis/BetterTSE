from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


@dataclass
class TimeMMDTextRecord:
    source: str
    start_date: str
    end_date: str
    fact: str
    pred: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TimeMMDAlignedRecord:
    domain: str
    text_source: str
    text_start_date: str
    text_end_date: str
    history_end_date: str
    future_start_date: str
    future_end_date: str
    context_text_raw: str
    context_text_compact: str
    text_fact: str
    text_pred: str
    history_ts: List[float]
    future_gt: List[float]
    timestamps: Dict[str, List[str]]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _normalize_text_frame(frame: pd.DataFrame, *, source: str) -> pd.DataFrame:
    df = frame.copy()
    rename_map = {"preds": "pred"}
    df = df.rename(columns=rename_map)
    required = ["start_date", "end_date", "fact", "pred"]
    missing = [name for name in required if name not in df.columns]
    if missing:
        raise ValueError(f"Time-MMD textual file missing columns: {missing}")
    df["source"] = source
    df["start_date"] = pd.to_datetime(df["start_date"])
    df["end_date"] = pd.to_datetime(df["end_date"])
    df["fact"] = df["fact"].fillna("").astype(str)
    df["pred"] = df["pred"].fillna("").astype(str)
    return df[["source", "start_date", "end_date", "fact", "pred"]]


def load_timemmd_domain(root_dir: str | Path, domain: str) -> Dict[str, pd.DataFrame]:
    root = Path(root_dir)
    numerical_path = root / "numerical" / domain / f"{domain}.csv"
    report_path = root / "textual" / domain / f"{domain}_report.csv"
    search_path = root / "textual" / domain / f"{domain}_search.csv"
    if not numerical_path.exists():
        raise FileNotFoundError(f"Missing Time-MMD numerical file: {numerical_path}")
    if not report_path.exists():
        raise FileNotFoundError(f"Missing Time-MMD report file: {report_path}")
    if not search_path.exists():
        raise FileNotFoundError(f"Missing Time-MMD search file: {search_path}")

    numerical = pd.read_csv(numerical_path)
    if "OT" not in numerical.columns:
        raise ValueError(f"Time-MMD numerical file missing OT column: {numerical_path}")
    if "start_date" not in numerical.columns or "end_date" not in numerical.columns:
        raise ValueError(f"Time-MMD numerical file must contain start_date/end_date: {numerical_path}")
    numerical["start_date"] = pd.to_datetime(numerical["start_date"])
    numerical["end_date"] = pd.to_datetime(numerical["end_date"])
    numerical = numerical.sort_values("start_date").reset_index(drop=True)

    report = _normalize_text_frame(pd.read_csv(report_path), source="report")
    search = _normalize_text_frame(pd.read_csv(search_path), source="search")
    texts = pd.concat([report, search], ignore_index=True).sort_values("end_date").reset_index(drop=True)
    return {"numerical": numerical, "report": report, "search": search, "texts": texts}


def summarize_timemmd_domain(root_dir: str | Path, domain: str) -> Dict[str, Any]:
    bundle = load_timemmd_domain(root_dir, domain)
    numerical = bundle["numerical"]
    texts = bundle["texts"]
    return {
        "domain": domain,
        "num_points": int(len(numerical)),
        "text_points": int(len(texts)),
        "start_date_min": str(numerical["start_date"].min().date()),
        "end_date_max": str(numerical["end_date"].max().date()),
        "text_sources": sorted(texts["source"].unique().tolist()),
    }
