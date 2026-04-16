from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd


def _looks_headerless(columns: pd.Index) -> bool:
    parsed = pd.to_numeric(pd.Series(columns, dtype="object"), errors="coerce")
    return bool(parsed.notna().all())


def _read_csv_frame(csv_path: str) -> pd.DataFrame:
    path = Path(csv_path)
    df = pd.read_csv(path)

    if _looks_headerless(df.columns):
        df = pd.read_csv(path, header=None)
        df.columns = [f"feature_{idx}" for idx in range(df.shape[1])]
    return df


def load_csv_frame(csv_path: str) -> pd.DataFrame:
    df = _read_csv_frame(csv_path)

    if "date" in df.columns:
        df = df.drop(columns=["date"])
    return df


def inspect_csv_frame(csv_path: str) -> Dict[str, Any]:
    df = load_csv_frame(csv_path)
    numeric_df = df.apply(pd.to_numeric, errors="coerce")
    feature_columns = [str(col) for col in numeric_df.columns]
    return {
        "num_rows": int(len(numeric_df)),
        "num_features": int(len(feature_columns)),
        "feature_columns": feature_columns,
        "default_target_col": feature_columns[0] if feature_columns else None,
        "has_date": bool("date" in _read_csv_frame(csv_path).columns),
    }


def materialize_ltsf_compatible_csv(
    csv_path: str,
    output_path: str | Path,
    *,
    pandas_freq: str,
) -> Dict[str, Any]:
    raw_df = _read_csv_frame(csv_path).copy()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    if "date" not in raw_df.columns:
        raw_df.insert(
            0,
            "date",
            pd.date_range(start="2000-01-01", periods=len(raw_df), freq=pandas_freq),
        )
    else:
        raw_df["date"] = pd.to_datetime(raw_df["date"])

    renamed = []
    for idx, col in enumerate(raw_df.columns):
        if col == "date":
            renamed.append("date")
        else:
            renamed.append(f"feature_{idx - 1}")
    raw_df.columns = renamed
    raw_df.to_csv(output, index=False)

    return {
        "csv_path": str(output.resolve()),
        "target_col": "feature_0",
        "num_rows": int(len(raw_df)),
        "num_features": int(len(raw_df.columns) - 1),
        "has_synthetic_date": bool("date" not in _read_csv_frame(csv_path).columns),
    }


def load_univariate_series(csv_path: str, target_col: str | None = None) -> tuple[np.ndarray, str]:
    df = load_csv_frame(csv_path)

    if target_col and target_col in df.columns:
        feature = target_col
    elif target_col is not None and str(target_col).isdigit():
        col_idx = int(str(target_col))
        if 0 <= col_idx < len(df.columns):
            feature = str(df.columns[col_idx])
        else:
            feature = str(df.columns[0])
    else:
        feature = str(df.columns[0])

    series = pd.to_numeric(df[feature], errors="coerce").astype(np.float64).values
    return series, feature
