from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from forecasting.base import ForecastBaseline


class NaiveLastBaseline(ForecastBaseline):
    name = "naive_last"

    def predict(self, history: np.ndarray, horizon: int) -> np.ndarray:
        history = np.asarray(history, dtype=np.float64)
        finite = history[np.isfinite(history)]
        last_value = float(finite[-1]) if finite.size else 0.0
        return np.full(horizon, last_value, dtype=np.float64)

    def save(self, output_dir: str | Path) -> None:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        with open(output / "baseline_state.json", "w", encoding="utf-8") as f:
            json.dump({"name": self.name, "config": self.config}, f, ensure_ascii=False, indent=2)
