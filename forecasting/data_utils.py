from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _looks_headerless(columns: pd.Index) -> bool:
    parsed = pd.to_numeric(pd.Series(columns, dtype="object"), errors="coerce")
    return bool(parsed.notna().all())


def load_univariate_series(csv_path: str, target_col: str | None = None) -> tuple[np.ndarray, str]:
    path = Path(csv_path)
    df = pd.read_csv(path)

    if _looks_headerless(df.columns):
        df = pd.read_csv(path, header=None)
        df.columns = [f"feature_{idx}" for idx in range(df.shape[1])]

    if "date" in df.columns:
        df = df.drop(columns=["date"])

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
