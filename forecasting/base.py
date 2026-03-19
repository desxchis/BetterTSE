from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


class ForecastBaseline(ABC):
    """Common interface for forecasting baselines used in revision benchmarks."""

    name: str = "baseline"

    def __init__(self, **config: Any) -> None:
        self.config = dict(config)

    def fit(self, train_split: np.ndarray, val_split: Optional[np.ndarray] = None) -> "ForecastBaseline":
        return self

    def fit_windows(
        self,
        history_windows: np.ndarray,
        future_windows: np.ndarray,
        val_history_windows: Optional[np.ndarray] = None,
        val_future_windows: Optional[np.ndarray] = None,
    ) -> "ForecastBaseline":
        del history_windows, future_windows, val_history_windows, val_future_windows
        raise NotImplementedError(f"{self.__class__.__name__} does not support window-based training.")

    @abstractmethod
    def predict(self, history: np.ndarray, horizon: int) -> np.ndarray:
        raise NotImplementedError

    def save(self, output_dir: str | Path) -> None:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    @classmethod
    def load(cls, model_dir: str | Path, **config: Any) -> "ForecastBaseline":
        return cls(**config)

    def describe(self) -> Dict[str, Any]:
        return {"name": self.name, "config": dict(self.config)}
