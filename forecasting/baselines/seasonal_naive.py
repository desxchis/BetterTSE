from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from forecasting.base import ForecastBaseline


class SeasonalNaiveBaseline(ForecastBaseline):
    """Repeat the last observed seasonal cycle."""

    name = "seasonal_naive"

    def __init__(self, **config):
        super().__init__(**config)
        self.season_length = int(self.config.get("season_length", 24))

    def predict(self, history: np.ndarray, horizon: int) -> np.ndarray:
        history = np.asarray(history, dtype=np.float64)
        finite = history[np.isfinite(history)]
        if finite.size == 0:
            return np.zeros(horizon, dtype=np.float64)
        season = min(max(1, self.season_length), len(finite))
        template = finite[-season:]
        repeats = int(np.ceil(horizon / season))
        return np.tile(template, repeats)[:horizon].astype(np.float64)

    def save(self, output_dir: str | Path) -> None:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        with open(output / "baseline_state.json", "w", encoding="utf-8") as f:
            json.dump({"name": self.name, "config": self.config}, f, ensure_ascii=False, indent=2)
