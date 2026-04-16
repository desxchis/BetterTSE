"""Forecast baseline registry for forecast-revision experiments."""

from forecasting.dataset_catalog import (
    ENGINEERING_BASELINES,
    PAPER_HISTORICAL_CONTROL_BASELINES,
    PAPER_MAINLINE_BASELINES,
    get_standard_dataset,
    list_standard_datasets,
    resolve_standard_dataset,
)
from forecasting.registry import create_baseline, get_available_baselines, load_baseline

__all__ = [
    "create_baseline",
    "get_available_baselines",
    "load_baseline",
    "get_standard_dataset",
    "list_standard_datasets",
    "resolve_standard_dataset",
    "PAPER_MAINLINE_BASELINES",
    "PAPER_HISTORICAL_CONTROL_BASELINES",
    "ENGINEERING_BASELINES",
]
