from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List


@dataclass(frozen=True)
class StandardForecastDataset:
    dataset_id: str
    display_name: str
    dataset_family: str
    default_csv_path: str
    default_target_col: str | None
    default_context_lengths: tuple[int, ...]
    default_prediction_lengths: tuple[int, ...]
    split_policy: str

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["default_context_lengths"] = list(self.default_context_lengths)
        payload["default_prediction_lengths"] = list(self.default_prediction_lengths)
        payload["available"] = Path(self.default_csv_path).exists()
        return payload


_STANDARD_FORECAST_DATASETS: dict[str, StandardForecastDataset] = {
    "traffic": StandardForecastDataset(
        dataset_id="traffic",
        display_name="Traffic",
        dataset_family="ltsf_numeric",
        default_csv_path="data/traffic.csv",
        default_target_col="0",
        default_context_lengths=(96, 192, 336, 720),
        default_prediction_lengths=(96, 192, 336, 720),
        split_policy="ltsf_official_like",
    ),
    "weather": StandardForecastDataset(
        dataset_id="weather",
        display_name="Weather",
        dataset_family="ltsf_numeric",
        default_csv_path="data/Weather.csv",
        default_target_col="0",
        default_context_lengths=(96, 192, 336, 720),
        default_prediction_lengths=(96, 192, 336, 720),
        split_policy="ltsf_official_like",
    ),
    "etth1": StandardForecastDataset(
        dataset_id="etth1",
        display_name="ETTh1",
        dataset_family="ltsf_numeric",
        default_csv_path="data/ETTh1.csv",
        default_target_col="0",
        default_context_lengths=(96, 192, 336, 720),
        default_prediction_lengths=(96, 192, 336, 720),
        split_policy="ltsf_official_like",
    ),
    "etth2": StandardForecastDataset(
        dataset_id="etth2",
        display_name="ETTh2",
        dataset_family="ltsf_numeric",
        default_csv_path="data/ETTh2.csv",
        default_target_col="0",
        default_context_lengths=(96, 192, 336, 720),
        default_prediction_lengths=(96, 192, 336, 720),
        split_policy="ltsf_official_like",
    ),
    "ettm1": StandardForecastDataset(
        dataset_id="ettm1",
        display_name="ETTm1",
        dataset_family="ltsf_numeric",
        default_csv_path="data/ETTm1.csv",
        default_target_col="0",
        default_context_lengths=(96, 192, 336, 720),
        default_prediction_lengths=(96, 192, 336, 720),
        split_policy="ltsf_official_like",
    ),
    "ettm2": StandardForecastDataset(
        dataset_id="ettm2",
        display_name="ETTm2",
        dataset_family="ltsf_numeric",
        default_csv_path="data/ETTm2.csv",
        default_target_col="0",
        default_context_lengths=(96, 192, 336, 720),
        default_prediction_lengths=(96, 192, 336, 720),
        split_policy="ltsf_official_like",
    ),
    "electricity": StandardForecastDataset(
        dataset_id="electricity",
        display_name="Electricity",
        dataset_family="ltsf_numeric",
        default_csv_path="data/electricity.csv",
        default_target_col="0",
        default_context_lengths=(96, 192, 336, 720),
        default_prediction_lengths=(96, 192, 336, 720),
        split_policy="ltsf_official_like",
    ),
}

PAPER_MAINLINE_BASELINES = (
    "dlinear_tslib",
    "patchtst_tslib",
    "itransformer_tslib",
    "timemixer_tslib",
)
PAPER_HISTORICAL_CONTROL_BASELINES = ("autoformer_tslib",)
ENGINEERING_BASELINES = (
    "naive_last",
    "seasonal_naive",
    "holt_linear",
    "dlinear_like",
    "dlinear_official",
    "patchtst",
    "lstm_official",
)


def list_standard_datasets() -> List[Dict[str, Any]]:
    return [dataset.to_dict() for dataset in _STANDARD_FORECAST_DATASETS.values()]


def get_standard_dataset(dataset_id: str) -> StandardForecastDataset:
    key = str(dataset_id).strip().lower()
    if key not in _STANDARD_FORECAST_DATASETS:
        raise ValueError(
            f"Unknown standard forecasting dataset '{dataset_id}'. Available: {sorted(_STANDARD_FORECAST_DATASETS)}"
        )
    return _STANDARD_FORECAST_DATASETS[key]


def resolve_standard_dataset(dataset_id: str) -> Dict[str, Any]:
    dataset = get_standard_dataset(dataset_id)
    payload = dataset.to_dict()
    payload["csv_path"] = dataset.default_csv_path
    payload["target_col"] = dataset.default_target_col
    return payload
