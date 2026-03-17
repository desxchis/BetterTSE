from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any, Optional

import numpy as np

from forecasting.base import ForecastBaseline


class PatchTSTHFBaseline(ForecastBaseline):
    """PatchTST adapter placeholder.

    The adapter is intentionally lightweight for CPU-only v1. If the installed
    transformers build exposes PatchTST, this class can be extended later into
    a full train/infer baseline without changing the registry contract.
    """

    name = "patchtst"

    def __init__(self, **config: Any) -> None:
        super().__init__(**config)
        self._transformers_error: Optional[Exception] = None
        self._torch_error: Optional[Exception] = None
        self._available = importlib.util.find_spec("transformers") is not None
        self.model = None
        self.device = str(self.config.get("device", "cpu"))
        self.context_length = int(self.config.get("context_length", self.config.get("seq_len", 96)))
        self.prediction_length = int(self.config.get("prediction_length", self.config.get("pred_len", 24)))

    def _ensure_runtime(self) -> None:
        if not self._available:
            raise RuntimeError("transformers is not installed; PatchTST baseline unavailable.")
        if self.model is not None:
            return
        try:
            import torch  # type: ignore
            from transformers import PatchTSTConfig, PatchTSTForPrediction  # type: ignore

            self._torch = torch
            self._PatchTSTConfig = PatchTSTConfig
            self._PatchTSTForPrediction = PatchTSTForPrediction
        except Exception as exc:  # pragma: no cover - environment dependent
            self._transformers_error = exc
            raise RuntimeError(f"PatchTST runtime import failed: {exc}") from exc

    def _build_model(self, num_input_channels: int = 1) -> None:
        self._ensure_runtime()
        cfg = self._PatchTSTConfig(
            num_input_channels=num_input_channels,
            context_length=self.context_length,
            prediction_length=self.prediction_length,
            distribution_output=None,
            loss="mse",
            patch_length=int(self.config.get("patch_length", 8)),
            patch_stride=int(self.config.get("patch_stride", 8)),
            d_model=int(self.config.get("d_model", 32)),
            ffn_dim=int(self.config.get("ffn_dim", 64)),
            num_hidden_layers=int(self.config.get("num_hidden_layers", 2)),
            num_attention_heads=int(self.config.get("num_attention_heads", 4)),
            dropout=float(self.config.get("dropout", 0.0)) if "dropout" in self.config else 0.0,
            head_dropout=float(self.config.get("head_dropout", 0.0)),
        )
        self.model = self._PatchTSTForPrediction(cfg)
        self.model.to(self.device)
        self.model.train()

    def fit(self, train_split: np.ndarray, val_split: Optional[np.ndarray] = None) -> "PatchTSTHFBaseline":
        train_split = np.asarray(train_split, dtype=np.float32)
        if train_split.ndim != 1:
            raise ValueError("PatchTSTHFBaseline currently expects a univariate 1D training series.")
        if len(train_split) < self.context_length + self.prediction_length:
            raise ValueError("Training series is shorter than context_length + prediction_length.")

        self._build_model(num_input_channels=1)
        torch = self._torch
        optimizer = torch.optim.Adam(self.model.parameters(), lr=float(self.config.get("lr", 1e-3)))
        epochs = int(self.config.get("epochs", 3))
        batch_size = int(self.config.get("batch_size", 16))

        windows_x = []
        windows_y = []
        for start in range(0, len(train_split) - self.context_length - self.prediction_length + 1):
            hist = train_split[start:start + self.context_length]
            fut = train_split[start + self.context_length:start + self.context_length + self.prediction_length]
            if np.isnan(hist).any() or np.isnan(fut).any():
                continue
            windows_x.append(hist)
            windows_y.append(fut)
        if not windows_x:
            raise ValueError("No valid sliding windows available for PatchTST training.")

        x = torch.tensor(np.asarray(windows_x), dtype=torch.float32, device=self.device).unsqueeze(-1)
        y = torch.tensor(np.asarray(windows_y), dtype=torch.float32, device=self.device).unsqueeze(-1)

        for _ in range(epochs):
            perm = torch.randperm(x.shape[0], device=self.device)
            for i in range(0, x.shape[0], batch_size):
                idx = perm[i:i + batch_size]
                batch_x = x[idx]
                batch_y = y[idx]
                optimizer.zero_grad()
                outputs = self.model(past_values=batch_x, future_values=batch_y)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
        self.model.eval()
        return self

    def predict(self, history: np.ndarray, horizon: int) -> np.ndarray:
        history = np.asarray(history, dtype=np.float32)
        if horizon != self.prediction_length:
            raise ValueError(
                f"PatchTST baseline was configured for prediction_length={self.prediction_length}, got horizon={horizon}."
            )
        if self.model is None:
            raise RuntimeError("PatchTST model is not initialized. Call fit() or load() before predict().")
        if history.ndim != 1:
            raise ValueError("PatchTSTHFBaseline currently expects a univariate 1D history series.")
        if len(history) < self.context_length:
            raise ValueError("History length is shorter than configured context_length.")

        torch = self._torch
        context = history[-self.context_length:]
        context = np.nan_to_num(context, nan=float(np.nanmean(history)) if np.isfinite(np.nanmean(history)) else 0.0)
        x = torch.tensor(context[None, :, None], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            outputs = self.model(past_values=x)
        pred = outputs.prediction_outputs.detach().cpu().numpy()[0, :, 0]
        return pred.astype(np.float64)

    def save(self, output_dir: str | Path) -> None:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        state_path = output / "baseline_state.json"
        with open(state_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "name": self.name,
                    "config": self.config,
                    "context_length": self.context_length,
                    "prediction_length": self.prediction_length,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        if self.model is not None:
            self._torch.save(self.model.state_dict(), output / "model.pt")

    @classmethod
    def load(cls, model_dir: str | Path, **config: Any) -> "PatchTSTHFBaseline":
        model_dir = Path(model_dir)
        state_path = model_dir / "baseline_state.json"
        loaded_config = {}
        if state_path.exists():
            with open(state_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            loaded_config.update(loaded.get("config", {}))
            loaded_config["context_length"] = loaded.get("context_length", loaded_config.get("context_length", 96))
            loaded_config["prediction_length"] = loaded.get("prediction_length", loaded_config.get("prediction_length", 24))
        loaded_config.update(config)
        inst = cls(**loaded_config)
        model_path = model_dir / "model.pt"
        if model_path.exists():
            inst._build_model(num_input_channels=1)
            state_dict = inst._torch.load(model_path, map_location=inst.device)
            inst.model.load_state_dict(state_dict)
            inst.model.eval()
        return inst

    def describe(self) -> dict[str, Any]:
        info = super().describe()
        info["available"] = self._available
        info["runtime_ready"] = self.model is not None
        if self._transformers_error is not None:
            info["runtime_error"] = repr(self._transformers_error)
        return info
