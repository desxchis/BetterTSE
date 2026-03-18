from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from forecasting.base import ForecastBaseline


class HoltLinearBaseline(ForecastBaseline):
    """Lightweight Holt linear trend baseline without external stats packages."""

    name = "holt_linear"

    def __init__(self, **config):
        super().__init__(**config)
        self.alpha = float(self.config.get("alpha", 0.6))
        self.beta = float(self.config.get("beta", 0.2))

    def predict(self, history: np.ndarray, horizon: int) -> np.ndarray:
        history = np.asarray(history, dtype=np.float64)
        finite = history[np.isfinite(history)]
        if finite.size == 0:
            return np.zeros(horizon, dtype=np.float64)
        if finite.size == 1:
            return np.full(horizon, float(finite[-1]), dtype=np.float64)

        level = float(finite[0])
        trend = float(finite[1] - finite[0])
        alpha = min(max(self.alpha, 1e-3), 0.999)
        beta = min(max(self.beta, 1e-3), 0.999)
        for value in finite[1:]:
            prev_level = level
            level = alpha * float(value) + (1.0 - alpha) * (level + trend)
            trend = beta * (level - prev_level) + (1.0 - beta) * trend
        steps = np.arange(1, horizon + 1, dtype=np.float64)
        return level + steps * trend

    def save(self, output_dir: str | Path) -> None:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        with open(output / "baseline_state.json", "w", encoding="utf-8") as f:
            json.dump({"name": self.name, "config": self.config}, f, ensure_ascii=False, indent=2)
