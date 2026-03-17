from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from forecasting import create_baseline
from forecasting.data_utils import load_univariate_series


def train_baseline(
    csv_path: str,
    dataset_name: str,
    model_name: str,
    output_dir: str,
    target_col: str | None = None,
    seq_len: int = 96,
    pred_len: int = 24,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    **model_config,
) -> dict:
    series, feature = load_univariate_series(csv_path, target_col)
    n = len(series)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    train_split = series[:train_end]
    val_split = series[train_end:val_end] if val_end > train_end else None

    baseline = create_baseline(
        model_name,
        context_length=seq_len,
        prediction_length=pred_len,
        seq_len=seq_len,
        pred_len=pred_len,
        **model_config,
    )
    baseline.fit(train_split, val_split)

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    baseline.save(output_root)

    summary = {
        "dataset_name": dataset_name,
        "feature": feature,
        "model_name": model_name,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "train_points": int(len(train_split)),
        "val_points": int(len(val_split) if val_split is not None else 0),
        "config": baseline.describe(),
    }
    with open(output_root / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a forecasting baseline for revision experiments.")
    parser.add_argument("--csv-path", required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--model-name", required=True, choices=["naive_last", "dlinear_like", "patchtst"])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--target-col", default=None)
    parser.add_argument("--seq-len", type=int, default=96)
    parser.add_argument("--pred-len", type=int, default=24)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    summary = train_baseline(
        csv_path=args.csv_path,
        dataset_name=args.dataset_name,
        model_name=args.model_name,
        output_dir=args.output_dir,
        target_col=args.target_col,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        device=args.device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
