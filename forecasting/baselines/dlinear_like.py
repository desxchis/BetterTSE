from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from forecasting.base import ForecastBaseline


class DLinearLikeBaseline(ForecastBaseline):
    """A tiny linear-trend baseline used as a CPU-safe DLinear-style proxy."""

    name = "dlinear_like"

    def predict(self, history: np.ndarray, horizon: int) -> np.ndarray:
        history = np.asarray(history, dtype=np.float64)
        idx = np.arange(len(history), dtype=np.float64)
        finite_mask = np.isfinite(history)
        if finite_mask.sum() < 2:
            fill_value = float(np.nanmean(history)) if np.isfinite(np.nanmean(history)) else 0.0
            return np.full(horizon, fill_value, dtype=np.float64)

        x = idx[finite_mask]
        y = history[finite_mask]
        A = np.column_stack([x, np.ones_like(x)])
        slope, bias = np.linalg.lstsq(A, y, rcond=None)[0]
        future_x = np.arange(len(history), len(history) + horizon, dtype=np.float64)
        return slope * future_x + bias

    def save(self, output_dir: str | Path) -> None:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        with open(output / "baseline_state.json", "w", encoding="utf-8") as f:
            json.dump({"name": self.name, "config": self.config}, f, ensure_ascii=False, indent=2)
