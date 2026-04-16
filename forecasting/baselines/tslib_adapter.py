from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from forecasting.base import ForecastBaseline


class TSLibExternalBaseline(ForecastBaseline):
    """Registry-stable adapter for paper baselines trained through TSLib."""

    name = "tslib_external"
    tslib_model_name = "unknown"
    paper_role = "mainline"

    def __init__(self, **config: Any) -> None:
        super().__init__(**config)
        self.context_length = int(self.config.get("context_length", self.config.get("seq_len", 96)))
        self.prediction_length = int(self.config.get("prediction_length", self.config.get("pred_len", 24)))
        self.model_dir = str(self.config.get("model_dir", ""))
        self._artifact_ready = False
        self._artifact_metadata: Dict[str, Any] = {}
        artifact = self.config.get("tslib_artifact")
        if isinstance(artifact, dict):
            self._artifact_metadata.update(artifact)
            self._artifact_ready = bool(artifact.get("export_ready", False))

    def fit(self, train_split: np.ndarray, val_split: Optional[np.ndarray] = None) -> "TSLibExternalBaseline":
        del train_split, val_split
        raise RuntimeError(
            f"{self.name} is a TSLib-backed paper baseline. "
            "This repository currently exposes the adapter and metadata contract only; "
            "formal training should be launched through the TSLib integration path."
        )

    def fit_windows(
        self,
        history_windows: np.ndarray,
        future_windows: np.ndarray,
        val_history_windows: Optional[np.ndarray] = None,
        val_future_windows: Optional[np.ndarray] = None,
    ) -> "TSLibExternalBaseline":
        del history_windows, future_windows, val_history_windows, val_future_windows
        raise RuntimeError(
            f"{self.name} is a TSLib-backed paper baseline. "
            "Windowed training is defined by the exported TSLib training stack, not by the local adapter."
        )

    def predict(self, history: np.ndarray, horizon: int) -> np.ndarray:
        del history
        if horizon != self.prediction_length:
            raise ValueError(
                f"{self.name} was configured for prediction_length={self.prediction_length}, got horizon={horizon}."
            )
        raise RuntimeError(
            f"{self.name} adapter is configured, but no callable exported inference artifact is available in "
            f"'{self.model_dir or '<unset>'}'. Train/export the model first before using it in benchmark builders."
        )

    def save(self, output_dir: str | Path) -> None:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        payload = {
            "name": self.name,
            "config": self.config,
            "context_length": self.context_length,
            "prediction_length": self.prediction_length,
            "baseline_source": "tslib",
            "baseline_family": self.tslib_model_name,
            "paper_role": self.paper_role,
            "tslib_artifact": {
                "export_ready": self._artifact_ready,
                **self._artifact_metadata,
            },
        }
        with open(output / "baseline_state.json", "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, model_dir: str | Path, **config: Any) -> "TSLibExternalBaseline":
        model_dir = Path(model_dir)
        merged = dict(config)
        state_path = model_dir / "baseline_state.json"
        if state_path.exists():
            with open(state_path, "r", encoding="utf-8") as f:
                state = json.load(f)
            merged.update(state.get("config", {}))
            merged["context_length"] = state.get("context_length", merged.get("context_length", 96))
            merged["prediction_length"] = state.get("prediction_length", merged.get("prediction_length", 24))
            if "tslib_artifact" in state:
                merged["tslib_artifact"] = state["tslib_artifact"]
        merged["model_dir"] = str(model_dir)
        return cls(**merged)

    def describe(self) -> Dict[str, Any]:
        payload = super().describe()
        payload.update(
            {
                "baseline_source": "tslib",
                "baseline_family": self.tslib_model_name,
                "paper_role": self.paper_role,
                "artifact_ready": self._artifact_ready,
                "tslib_artifact": dict(self._artifact_metadata),
            }
        )
        return payload


class DLinearTSLibBaseline(TSLibExternalBaseline):
    name = "dlinear_tslib"
    tslib_model_name = "DLinear"
    paper_role = "mainline"


class PatchTSTTSLibBaseline(TSLibExternalBaseline):
    name = "patchtst_tslib"
    tslib_model_name = "PatchTST"
    paper_role = "mainline"


class ITransformerTSLibBaseline(TSLibExternalBaseline):
    name = "itransformer_tslib"
    tslib_model_name = "iTransformer"
    paper_role = "mainline"


class TimeMixerTSLibBaseline(TSLibExternalBaseline):
    name = "timemixer_tslib"
    tslib_model_name = "TimeMixer"
    paper_role = "mainline"


class AutoformerTSLibBaseline(TSLibExternalBaseline):
    name = "autoformer_tslib"
    tslib_model_name = "Autoformer"
    paper_role = "historical_control"
