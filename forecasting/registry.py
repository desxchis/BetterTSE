from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from forecasting.base import ForecastBaseline
from forecasting.baselines import (
    AutoformerTSLibBaseline,
    DLinearOfficialBaseline,
    DLinearLikeBaseline,
    DLinearTSLibBaseline,
    HoltLinearBaseline,
    ITransformerTSLibBaseline,
    LSTMOfficialBaseline,
    NaiveLastBaseline,
    PatchTSTHFBaseline,
    PatchTSTTSLibBaseline,
    SeasonalNaiveBaseline,
    TimeMixerTSLibBaseline,
)


_REGISTRY = {
    "naive_last": NaiveLastBaseline,
    "dlinear_official": DLinearOfficialBaseline,
    "dlinear_like": DLinearLikeBaseline,
    "holt_linear": HoltLinearBaseline,
    "lstm_official": LSTMOfficialBaseline,
    "seasonal_naive": SeasonalNaiveBaseline,
    "patchtst": PatchTSTHFBaseline,
    "dlinear_tslib": DLinearTSLibBaseline,
    "patchtst_tslib": PatchTSTTSLibBaseline,
    "itransformer_tslib": ITransformerTSLibBaseline,
    "timemixer_tslib": TimeMixerTSLibBaseline,
    "autoformer_tslib": AutoformerTSLibBaseline,
}


def create_baseline(name: str, **config: Any) -> ForecastBaseline:
    if name not in _REGISTRY:
        raise ValueError(f"Unknown baseline '{name}'. Available: {sorted(_REGISTRY)}")
    return _REGISTRY[name](**config)


def load_baseline(name: str, model_dir: str | Path, **config: Any) -> ForecastBaseline:
    if name not in _REGISTRY:
        raise ValueError(f"Unknown baseline '{name}'. Available: {sorted(_REGISTRY)}")
    cls = _REGISTRY[name]
    model_dir = Path(model_dir).expanduser().resolve()
    merged_config = {}
    state_path = model_dir / "baseline_state.json"
    if state_path.exists():
        with open(state_path, "r", encoding="utf-8") as f:
            state = json.load(f)
        merged_config.update(state.get("config", {}))
    merged_config.update(config)
    return cls.load(model_dir, **merged_config)


def get_available_baselines() -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for name, cls in _REGISTRY.items():
        items.append(
            {
                "name": name,
                "class": cls.__name__,
                "paper_role": getattr(cls, "paper_role", "engineering"),
                "baseline_source": "tslib" if name.endswith("_tslib") else "local",
            }
        )
    return items
