from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

import numpy as np

from forecasting.base import ForecastBaseline


class _MovingAvg:
    def __init__(self, torch: Any, kernel_size: int, stride: int) -> None:
        nn = torch.nn
        self.kernel_size = int(kernel_size)
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)
        self._torch = torch

    def __call__(self, x: Any) -> Any:
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = self._torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        return x.permute(0, 2, 1)


class _SeriesDecomp:
    def __init__(self, torch: Any, kernel_size: int) -> None:
        self.moving_avg = _MovingAvg(torch, kernel_size=kernel_size, stride=1)

    def __call__(self, x: Any) -> tuple[Any, Any]:
        moving_mean = self.moving_avg(x)
        return x - moving_mean, moving_mean


class _OfficialDLinearModel:
    """Official LTSF-Linear DLinear architecture, adapted as a local wrapper."""

    def __init__(self, torch: Any, *, seq_len: int, pred_len: int, enc_in: int = 1, individual: bool = False, moving_avg: int = 25) -> None:
        self._torch = torch
        self.seq_len = int(seq_len)
        self.pred_len = int(pred_len)
        self.channels = int(enc_in)
        self.individual = bool(individual)
        self.decompsition = _SeriesDecomp(torch, kernel_size=int(moving_avg))
        nn = torch.nn
        if self.individual:
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()
            for _ in range(self.channels):
                self.Linear_Seasonal.append(nn.Linear(self.seq_len, self.pred_len))
                self.Linear_Trend.append(nn.Linear(self.seq_len, self.pred_len))
        else:
            self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len)
            self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)

    def parameters(self) -> Any:
        if self.individual:
            params = []
            for module in self.Linear_Seasonal:
                params.extend(list(module.parameters()))
            for module in self.Linear_Trend:
                params.extend(list(module.parameters()))
            return params
        return list(self.Linear_Seasonal.parameters()) + list(self.Linear_Trend.parameters())

    def to(self, device: str) -> "_OfficialDLinearModel":
        if self.individual:
            self.Linear_Seasonal.to(device)
            self.Linear_Trend.to(device)
        else:
            self.Linear_Seasonal.to(device)
            self.Linear_Trend.to(device)
        return self

    def train(self) -> None:
        self.Linear_Seasonal.train()
        self.Linear_Trend.train()

    def eval(self) -> None:
        self.Linear_Seasonal.eval()
        self.Linear_Trend.eval()

    def state_dict(self) -> dict[str, Any]:
        return {
            "Linear_Seasonal": self.Linear_Seasonal.state_dict(),
            "Linear_Trend": self.Linear_Trend.state_dict(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.Linear_Seasonal.load_state_dict(state["Linear_Seasonal"])
        self.Linear_Trend.load_state_dict(state["Linear_Trend"])

    def forward(self, x: Any) -> Any:
        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init = seasonal_init.permute(0, 2, 1)
        trend_init = trend_init.permute(0, 2, 1)
        if self.individual:
            seasonal_output = self._torch.zeros(
                [seasonal_init.size(0), seasonal_init.size(1), self.pred_len],
                dtype=seasonal_init.dtype,
                device=seasonal_init.device,
            )
            trend_output = self._torch.zeros(
                [trend_init.size(0), trend_init.size(1), self.pred_len],
                dtype=trend_init.dtype,
                device=trend_init.device,
            )
            for i in range(self.channels):
                seasonal_output[:, i, :] = self.Linear_Seasonal[i](seasonal_init[:, i, :])
                trend_output[:, i, :] = self.Linear_Trend[i](trend_init[:, i, :])
        else:
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            trend_output = self.Linear_Trend(trend_init)
        return (seasonal_output + trend_output).permute(0, 2, 1)


class DLinearOfficialBaseline(ForecastBaseline):
    name = "dlinear_official"

    def __init__(self, **config: Any) -> None:
        super().__init__(**config)
        self._available = importlib.util.find_spec("torch") is not None
        self._runtime_error: Optional[Exception] = None
        self.device = str(self.config.get("device", "cpu"))
        self.context_length = int(self.config.get("context_length", self.config.get("seq_len", 96)))
        self.prediction_length = int(self.config.get("prediction_length", self.config.get("pred_len", 24)))
        self.batch_size = int(self.config.get("batch_size", 32))
        self.epochs = int(self.config.get("epochs", 20))
        self.lr = float(self.config.get("lr", 1e-3))
        self.individual = bool(self.config.get("individual", False))
        self.moving_avg = int(self.config.get("moving_avg", 25))
        self.model: _OfficialDLinearModel | None = None
        self.norm_mean = float(self.config.get("norm_mean", 0.0))
        self.norm_std = max(float(self.config.get("norm_std", 1.0)), 1e-6)

    def _ensure_runtime(self) -> None:
        if not self._available:
            raise RuntimeError("torch is not installed; dlinear_official baseline unavailable.")
        try:
            import torch  # type: ignore
        except Exception as exc:  # pragma: no cover
            self._runtime_error = exc
            raise RuntimeError(f"DLinear runtime import failed: {exc}") from exc
        self._torch = torch

    def _build_model(self) -> None:
        self._ensure_runtime()
        self.model = _OfficialDLinearModel(
            self._torch,
            seq_len=self.context_length,
            pred_len=self.prediction_length,
            enc_in=1,
            individual=self.individual,
            moving_avg=self.moving_avg,
        ).to(self.device)
        self.model.train()

    def _compute_norm_stats(self, values: np.ndarray) -> None:
        arr = np.asarray(values, dtype=np.float64)
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            self.norm_mean = 0.0
            self.norm_std = 1.0
            return
        self.norm_mean = float(np.mean(finite))
        self.norm_std = max(float(np.std(finite)), 1e-6)

    def _normalize(self, values: np.ndarray) -> np.ndarray:
        return ((np.asarray(values, dtype=np.float64) - self.norm_mean) / self.norm_std).astype(np.float32)

    def _denormalize(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(values, dtype=np.float64) * self.norm_std + self.norm_mean

    def _build_windows(self, series: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        arr = np.asarray(series, dtype=np.float32).flatten()
        total = self.context_length + self.prediction_length
        if arr.size < total:
            raise ValueError("Training series is shorter than context_length + prediction_length.")
        xs = []
        ys = []
        for start in range(0, arr.size - total + 1):
            hist = arr[start : start + self.context_length]
            fut = arr[start + self.context_length : start + total]
            if np.isnan(hist).any() or np.isnan(fut).any():
                continue
            xs.append(hist)
            ys.append(fut)
        if not xs:
            raise ValueError("No valid windows available for dlinear_official training.")
        return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)

    def _fit_from_windows(self, x_np: np.ndarray, y_np: np.ndarray) -> "DLinearOfficialBaseline":
        self._build_model()
        assert self.model is not None
        torch = self._torch
        x = torch.tensor(np.asarray(x_np, dtype=np.float32), dtype=torch.float32, device=self.device).unsqueeze(-1)
        y = torch.tensor(np.asarray(y_np, dtype=np.float32), dtype=torch.float32, device=self.device).unsqueeze(-1)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = torch.nn.MSELoss()
        batch = max(1, min(self.batch_size, x.shape[0]))

        self.model.train()
        for _ in range(self.epochs):
            perm = torch.randperm(x.shape[0], device=self.device)
            for i in range(0, x.shape[0], batch):
                idx = perm[i : i + batch]
                optimizer.zero_grad()
                pred = self.model.forward(x[idx])
                loss = criterion(pred, y[idx])
                loss.backward()
                optimizer.step()
        self.model.eval()
        return self

    def fit(self, train_split: np.ndarray, val_split: Optional[np.ndarray] = None) -> "DLinearOfficialBaseline":
        del val_split
        train_split = np.asarray(train_split, dtype=np.float64)
        self._compute_norm_stats(train_split)
        x_np, y_np = self._build_windows(self._normalize(train_split))
        return self._fit_from_windows(x_np, y_np)

    def fit_windows(
        self,
        history_windows: np.ndarray,
        future_windows: np.ndarray,
        val_history_windows: Optional[np.ndarray] = None,
        val_future_windows: Optional[np.ndarray] = None,
    ) -> "DLinearOfficialBaseline":
        del val_history_windows, val_future_windows
        x_np = np.asarray(history_windows, dtype=np.float64)
        y_np = np.asarray(future_windows, dtype=np.float64)
        if x_np.ndim != 2 or y_np.ndim != 2:
            raise ValueError("fit_windows expects 2D arrays [N, T].")
        if x_np.shape[1] != self.context_length or y_np.shape[1] != self.prediction_length:
            raise ValueError("fit_windows received mismatched window lengths.")
        train_values = np.concatenate([x_np.reshape(-1), y_np.reshape(-1)], axis=0)
        self._compute_norm_stats(train_values)
        return self._fit_from_windows(self._normalize(x_np), self._normalize(y_np))

    def predict(self, history: np.ndarray, horizon: int) -> np.ndarray:
        if horizon != self.prediction_length:
            raise ValueError(
                f"dlinear_official baseline was configured for prediction_length={self.prediction_length}, got horizon={horizon}."
            )
        if self.model is None:
            raise RuntimeError("dlinear_official model is not initialized. Call fit() or load() before predict().")
        hist = np.asarray(history, dtype=np.float64).flatten()
        if hist.size < self.context_length:
            raise ValueError("History length is shorter than configured context_length.")
        ctx = hist[-self.context_length :]
        if np.isnan(ctx).any():
            fill = float(np.nanmean(ctx)) if np.isfinite(np.nanmean(ctx)) else 0.0
            ctx = np.nan_to_num(ctx, nan=fill)
        ctx_norm = self._normalize(ctx)
        x = self._torch.tensor(ctx_norm[None, :, None], dtype=self._torch.float32, device=self.device)
        with self._torch.no_grad():
            pred = self.model.forward(x)
        pred_np = pred.detach().cpu().numpy()[0, :, 0]
        return self._denormalize(pred_np).astype(np.float64)

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
                    "individual": self.individual,
                    "moving_avg": self.moving_avg,
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
    def load(cls, model_dir: str | Path, **config: Any) -> "DLinearOfficialBaseline":
        model_dir = Path(model_dir)
        loaded_config: dict[str, Any] = {}
        state_path = model_dir / "baseline_state.json"
        if state_path.exists():
            with open(state_path, "r", encoding="utf-8") as f:
                saved = json.load(f)
            loaded_config.update(saved.get("config", {}))
            loaded_config["context_length"] = saved.get("context_length", loaded_config.get("context_length", 96))
            loaded_config["prediction_length"] = saved.get("prediction_length", loaded_config.get("prediction_length", 24))
            loaded_config["individual"] = saved.get("individual", loaded_config.get("individual", False))
            loaded_config["moving_avg"] = saved.get("moving_avg", loaded_config.get("moving_avg", 25))
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
        info["source"] = "LTSF-Linear_official_DLinear_aligned"
        if self._runtime_error is not None:
            info["runtime_error"] = repr(self._runtime_error)
        return info
