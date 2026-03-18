from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any, Optional

import numpy as np

from forecasting.base import ForecastBaseline


class _OfficialSequenceModel:
    """
    PyTorch official-style sequence model:
    two LSTMCell layers + linear head, autoregressive over time.
    """

    def __init__(self, torch: Any, hidden_size: int) -> None:
        self._torch = torch
        nn = torch.nn
        self.lstm1 = nn.LSTMCell(1, hidden_size)
        self.lstm2 = nn.LSTMCell(hidden_size, hidden_size)
        self.linear = nn.Linear(hidden_size, 1)

    def parameters(self) -> Any:
        return list(self.lstm1.parameters()) + list(self.lstm2.parameters()) + list(self.linear.parameters())

    def to(self, device: str) -> "_OfficialSequenceModel":
        self.lstm1.to(device)
        self.lstm2.to(device)
        self.linear.to(device)
        return self

    def train(self) -> None:
        self.lstm1.train()
        self.lstm2.train()
        self.linear.train()

    def eval(self) -> None:
        self.lstm1.eval()
        self.lstm2.eval()
        self.linear.eval()

    def state_dict(self) -> dict[str, Any]:
        return {
            "lstm1": self.lstm1.state_dict(),
            "lstm2": self.lstm2.state_dict(),
            "linear": self.linear.state_dict(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.lstm1.load_state_dict(state["lstm1"])
        self.lstm2.load_state_dict(state["lstm2"])
        self.linear.load_state_dict(state["linear"])

    def forward(self, input_seq: Any, future: int = 0) -> Any:
        """
        input_seq: [B, T] tensor.
        output: [B, T + future] autoregressive predictions.
        """
        torch = self._torch
        batch_size = input_seq.size(0)
        h_t = torch.zeros(batch_size, self.lstm1.hidden_size, dtype=input_seq.dtype, device=input_seq.device)
        c_t = torch.zeros(batch_size, self.lstm1.hidden_size, dtype=input_seq.dtype, device=input_seq.device)
        h_t2 = torch.zeros(batch_size, self.lstm2.hidden_size, dtype=input_seq.dtype, device=input_seq.device)
        c_t2 = torch.zeros(batch_size, self.lstm2.hidden_size, dtype=input_seq.dtype, device=input_seq.device)
        outputs = []

        for x_t in input_seq.chunk(input_seq.size(1), dim=1):
            h_t, c_t = self.lstm1(x_t, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            out = self.linear(h_t2)
            outputs.append(out)

        for _ in range(int(future)):
            h_t, c_t = self.lstm1(out, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            out = self.linear(h_t2)
            outputs.append(out)

        return torch.cat(outputs, dim=1)


class LSTMOfficialBaseline(ForecastBaseline):
    """
    LSTM baseline aligned to PyTorch official example architecture:
    https://github.com/pytorch/examples/tree/main/time_sequence_prediction
    """

    name = "lstm_official"

    def __init__(self, **config: Any) -> None:
        super().__init__(**config)
        self._available = importlib.util.find_spec("torch") is not None
        self._runtime_error: Optional[Exception] = None
        self.device = str(self.config.get("device", "cpu"))
        self.context_length = int(self.config.get("context_length", self.config.get("seq_len", 96)))
        self.prediction_length = int(self.config.get("prediction_length", self.config.get("pred_len", 24)))
        self.hidden_size = int(self.config.get("hidden_size", 51))
        self.epochs = int(self.config.get("epochs", 10))
        self.batch_size = int(self.config.get("batch_size", 32))
        self.lr = float(self.config.get("lr", 1e-2))
        self.optimizer_name = str(self.config.get("optimizer", "lbfgs")).lower()
        self.model: _OfficialSequenceModel | None = None
        self.norm_mean = float(self.config.get("norm_mean", 0.0))
        self.norm_std = float(self.config.get("norm_std", 1.0))
        self.norm_std = max(self.norm_std, 1e-6)

    def _ensure_runtime(self) -> None:
        if not self._available:
            raise RuntimeError("torch is not installed; lstm_official baseline unavailable.")
        try:
            import torch  # type: ignore
        except Exception as exc:  # pragma: no cover
            self._runtime_error = exc
            raise RuntimeError(f"LSTM runtime import failed: {exc}") from exc
        self._torch = torch

    def _build_model(self) -> None:
        self._ensure_runtime()
        self.model = _OfficialSequenceModel(self._torch, hidden_size=self.hidden_size).to(self.device)
        self.model.train()

    def _build_windows(self, series: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        arr = np.asarray(series, dtype=np.float32)
        if arr.ndim != 1:
            raise ValueError("LSTMOfficialBaseline expects a univariate 1D training series.")
        total = self.context_length + self.prediction_length
        if len(arr) < total:
            raise ValueError("Training series is shorter than context_length + prediction_length.")
        xs = []
        ys = []
        for start in range(0, len(arr) - total + 1):
            hist = arr[start : start + self.context_length]
            fut = arr[start + self.context_length : start + total]
            if np.isnan(hist).any() or np.isnan(fut).any():
                continue
            xs.append(hist)
            ys.append(fut)
        if not xs:
            raise ValueError("No valid sliding windows available for lstm_official training.")
        return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)

    def _compute_norm_stats(self, series: np.ndarray) -> None:
        finite = np.asarray(series, dtype=np.float64)
        finite = finite[np.isfinite(finite)]
        if finite.size == 0:
            self.norm_mean = 0.0
            self.norm_std = 1.0
            return
        self.norm_mean = float(np.mean(finite))
        self.norm_std = max(float(np.std(finite)), 1e-6)

    def _normalize(self, values: np.ndarray) -> np.ndarray:
        arr = np.asarray(values, dtype=np.float64)
        return ((arr - self.norm_mean) / self.norm_std).astype(np.float32)

    def _denormalize(self, values: np.ndarray) -> np.ndarray:
        arr = np.asarray(values, dtype=np.float64)
        return arr * self.norm_std + self.norm_mean

    def fit(self, train_split: np.ndarray, val_split: Optional[np.ndarray] = None) -> "LSTMOfficialBaseline":
        del val_split
        train_split = np.asarray(train_split, dtype=np.float64)
        self._compute_norm_stats(train_split)
        self._build_model()
        assert self.model is not None
        torch = self._torch

        normalized = self._normalize(train_split)
        x_np, y_np = self._build_windows(np.asarray(normalized, dtype=np.float64))
        x = torch.tensor(x_np, dtype=torch.float32, device=self.device)
        y = torch.tensor(y_np, dtype=torch.float32, device=self.device)

        if self.optimizer_name == "lbfgs":
            optimizer = torch.optim.LBFGS(self.model.parameters(), lr=self.lr)
        else:
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = torch.nn.MSELoss()

        self.model.train()
        batch = max(1, min(self.batch_size, x.shape[0]))
        for _ in range(self.epochs):
            perm = torch.randperm(x.shape[0], device=self.device)
            for i in range(0, x.shape[0], batch):
                idx = perm[i : i + batch]
                bx = x[idx]
                by = y[idx]

                def closure() -> Any:
                    optimizer.zero_grad()
                    pred = self.model.forward(bx, future=self.prediction_length)
                    pred_future = pred[:, self.context_length : self.context_length + self.prediction_length]
                    loss = criterion(pred_future, by)
                    loss.backward()
                    return loss

                if self.optimizer_name == "lbfgs":
                    optimizer.step(closure)
                else:
                    closure()
                    optimizer.step()

        self.model.eval()
        return self

    def predict(self, history: np.ndarray, horizon: int) -> np.ndarray:
        if horizon != self.prediction_length:
            raise ValueError(
                f"lstm_official baseline was configured for prediction_length={self.prediction_length}, got horizon={horizon}."
            )
        if self.model is None:
            raise RuntimeError("lstm_official model is not initialized. Call fit() or load() before predict().")
        hist = np.asarray(history, dtype=np.float32)
        if hist.ndim != 1:
            raise ValueError("LSTMOfficialBaseline expects a univariate 1D history series.")
        if len(hist) < self.context_length:
            raise ValueError("History length is shorter than configured context_length.")

        torch = self._torch
        ctx = hist[-self.context_length :]
        if np.isnan(ctx).any():
            fill = float(np.nanmean(ctx)) if np.isfinite(np.nanmean(ctx)) else 0.0
            ctx = np.nan_to_num(ctx, nan=fill)
        ctx_norm = self._normalize(ctx)
        x = torch.tensor(ctx_norm[None, :], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            out = self.model.forward(x, future=self.prediction_length)
        pred = out[:, self.context_length : self.context_length + self.prediction_length]
        pred_denorm = self._denormalize(pred.detach().cpu().numpy()[0].astype(np.float64))
        return pred_denorm.astype(np.float64)

    def save(self, output_dir: str | Path) -> None:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        with open(output / "baseline_state.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "name": self.name,
                    "config": self.config,
                    "context_length": self.context_length,
                    "prediction_length": self.prediction_length,
                    "hidden_size": self.hidden_size,
                    "norm_mean": self.norm_mean,
                    "norm_std": self.norm_std,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        if self.model is not None:
            self._torch.save(self.model.state_dict(), output / "model.pt")

    @classmethod
    def load(cls, model_dir: str | Path, **config: Any) -> "LSTMOfficialBaseline":
        model_dir = Path(model_dir)
        loaded_config: dict[str, Any] = {}
        state_path = model_dir / "baseline_state.json"
        if state_path.exists():
            with open(state_path, "r", encoding="utf-8") as f:
                saved = json.load(f)
            loaded_config.update(saved.get("config", {}))
            loaded_config["context_length"] = saved.get("context_length", loaded_config.get("context_length", 96))
            loaded_config["prediction_length"] = saved.get("prediction_length", loaded_config.get("prediction_length", 24))
            loaded_config["hidden_size"] = saved.get("hidden_size", loaded_config.get("hidden_size", 51))
            loaded_config["norm_mean"] = saved.get("norm_mean", loaded_config.get("norm_mean", 0.0))
            loaded_config["norm_std"] = saved.get("norm_std", loaded_config.get("norm_std", 1.0))
        loaded_config.update(config)

        inst = cls(**loaded_config)
        model_path = model_dir / "model.pt"
        if model_path.exists():
            inst._build_model()
            assert inst.model is not None
            state_dict = inst._torch.load(model_path, map_location=inst.device)
            inst.model.load_state_dict(state_dict)
            inst.model.eval()
        return inst

    def describe(self) -> dict[str, Any]:
        info = super().describe()
        info["available"] = self._available
        info["runtime_ready"] = self.model is not None
        info["source"] = "pytorch_official_time_sequence_prediction_aligned"
        if self._runtime_error is not None:
            info["runtime_error"] = repr(self._runtime_error)
        return info
