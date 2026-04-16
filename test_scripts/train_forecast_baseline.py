from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from forecasting.data_utils import load_univariate_series
from forecasting.dataset_catalog import list_standard_datasets, resolve_standard_dataset
from forecasting.registry import create_baseline
from forecasting.tslib_bridge import build_tslib_training_manifest, write_tslib_training_manifest
from modules.mtbench_data import load_mtbench_aligned_pairs
from modules.timemmd_data import load_timemmd_domain
from modules.xtraffic_data import XTRAFFIC_CHANNELS, extract_node_series, load_xtraffic_minimal


def _resolve_xtraffic_channel(channel: str) -> int:
    text = str(channel).strip().lower()
    if text.isdigit():
        idx = int(text)
        if idx < 0 or idx >= len(XTRAFFIC_CHANNELS):
            raise ValueError(f"xtraffic channel index out of range: {idx}")
        return idx
    if text in XTRAFFIC_CHANNELS:
        return XTRAFFIC_CHANNELS.index(text)
    raise ValueError(f"unknown xtraffic channel '{channel}', expected one of {XTRAFFIC_CHANNELS} or an index.")


def _impute_series(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).copy()
    finite = np.isfinite(arr)
    if finite.all():
        return arr
    if not finite.any():
        return np.zeros_like(arr)
    idx = np.arange(len(arr))
    arr[~finite] = np.interp(idx[~finite], idx[finite], arr[finite])
    return arr


def _resolve_standard_dataset_args(args: argparse.Namespace) -> dict[str, object] | None:
    if not args.dataset_id:
        return None
    if args.dataset_kind != "csv":
        raise ValueError("--dataset-id currently maps to standard csv forecasting datasets, so --dataset-kind must remain 'csv'.")
    dataset = resolve_standard_dataset(args.dataset_id)
    csv_path = args.csv_path or dataset["csv_path"]
    target_col = args.target_col if args.target_col is not None else dataset["target_col"]
    if not Path(str(csv_path)).exists():
        raise FileNotFoundError(
            f"standard forecasting dataset '{args.dataset_id}' expects csv_path='{csv_path}', but the file is missing. "
            "Add the dataset file first or override --csv-path explicitly."
        )
    return {
        "dataset_id": dataset["dataset_id"],
        "dataset_family": dataset["dataset_family"],
        "split_policy": dataset["split_policy"],
        "csv_path": str(csv_path),
        "target_col": None if target_col is None else str(target_col),
    }


def _load_training_series(
    *,
    dataset_kind: str,
    csv_path: str | None,
    target_col: str | None,
    xtraffic_data_dir: str | None,
    xtraffic_shard_name: str,
    xtraffic_node_index: int,
    xtraffic_node_indices: str | None,
    xtraffic_channel: str,
    mtbench_path: str | None,
    mtbench_limit: int | None,
    timemmd_root: str | None,
    timemmd_domain: str,
) -> tuple[np.ndarray, str]:
    if dataset_kind == "csv":
        if not csv_path:
            raise ValueError("--csv-path is required when --dataset-kind=csv")
        series, feature = load_univariate_series(csv_path, target_col)
        return np.asarray(series, dtype=np.float64), feature

    if dataset_kind == "xtraffic":
        if not xtraffic_data_dir:
            raise ValueError("--xtraffic-data-dir is required when --dataset-kind=xtraffic")
        bundle = load_xtraffic_minimal(data_dir=xtraffic_data_dir, shard_name=xtraffic_shard_name)
        channel_index = _resolve_xtraffic_channel(xtraffic_channel)
        if xtraffic_node_indices:
            node_ids = [int(token.strip()) for token in str(xtraffic_node_indices).split(",") if token.strip()]
        else:
            node_ids = [int(xtraffic_node_index)]
        if not node_ids:
            raise ValueError("xtraffic node list is empty.")
        parts = []
        for node_id in node_ids:
            node_series = extract_node_series(bundle, node_index=node_id, channel=channel_index)
            parts.append(_impute_series(node_series))
        series = np.concatenate(parts, axis=0)
        if len(node_ids) == 1:
            feature = f"xtraffic_node_{node_ids[0]}_{XTRAFFIC_CHANNELS[channel_index]}"
        else:
            feature = f"xtraffic_nodes_{len(node_ids)}_{XTRAFFIC_CHANNELS[channel_index]}"
        return series.astype(np.float64), feature

    if dataset_kind == "mtbench":
        if not mtbench_path:
            raise ValueError("--mtbench-path is required when --dataset-kind=mtbench")
        records = load_mtbench_aligned_pairs(mtbench_path, limit=mtbench_limit)
        if not records:
            raise ValueError("No MTBench aligned pairs loaded.")
        parts = []
        for rec in records:
            parts.append(np.asarray(rec.input_window, dtype=np.float64))
            parts.append(np.asarray(rec.output_window, dtype=np.float64))
        series = np.concatenate(parts, axis=0)
        series = _impute_series(series)
        return series.astype(np.float64), "mtbench_finance_concat"

    if dataset_kind == "timemmd":
        if not timemmd_root:
            raise ValueError("--timemmd-root is required when --dataset-kind=timemmd")
        bundle = load_timemmd_domain(timemmd_root, timemmd_domain)
        series = bundle["numerical"]["OT"].astype(float).to_numpy()
        return _impute_series(series).astype(np.float64), f"timemmd_{timemmd_domain}_OT"

    raise ValueError(f"unsupported dataset-kind: {dataset_kind}")


def _build_windows_from_series_list(
    series_list: list[np.ndarray],
    context_length: int,
    prediction_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    xs = []
    ys = []
    total = context_length + prediction_length
    for series in series_list:
        arr = np.asarray(series, dtype=np.float64).flatten()
        if len(arr) < total:
            continue
        for start in range(0, len(arr) - total + 1):
            hist = arr[start : start + context_length]
            fut = arr[start + context_length : start + total]
            if np.isnan(hist).any() or np.isnan(fut).any():
                continue
            xs.append(hist.astype(np.float32))
            ys.append(fut.astype(np.float32))
    if not xs:
        raise ValueError("No valid windows could be built from the provided series list.")
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)


def _load_structured_training_windows(
    *,
    dataset_kind: str,
    csv_path: str | None,
    target_col: str | None,
    xtraffic_data_dir: str | None,
    xtraffic_shard_name: str,
    xtraffic_node_index: int,
    xtraffic_node_indices: str | None,
    xtraffic_channel: str,
    mtbench_path: str | None,
    mtbench_limit: int | None,
    timemmd_root: str | None,
    timemmd_domain: str,
    context_length: int,
    prediction_length: int,
) -> tuple[np.ndarray, np.ndarray, str] | None:
    if dataset_kind == "mtbench":
        if not mtbench_path:
            raise ValueError("--mtbench-path is required when --dataset-kind=mtbench")
        records = load_mtbench_aligned_pairs(mtbench_path, limit=mtbench_limit)
        xs = []
        ys = []
        for rec in records:
            hist = np.asarray(rec.input_window, dtype=np.float64)
            fut = np.asarray(rec.output_window, dtype=np.float64)
            if hist.size != context_length or fut.size != prediction_length:
                continue
            if np.isnan(hist).any() or np.isnan(fut).any():
                continue
            xs.append(hist.astype(np.float32))
            ys.append(fut.astype(np.float32))
        if not xs:
            raise ValueError("No valid MTBench aligned windows matched the requested context/prediction lengths.")
        return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32), "mtbench_finance_aligned_windows"

    if dataset_kind == "xtraffic" and xtraffic_node_indices:
        if not xtraffic_data_dir:
            raise ValueError("--xtraffic-data-dir is required when --dataset-kind=xtraffic")
        bundle = load_xtraffic_minimal(data_dir=xtraffic_data_dir, shard_name=xtraffic_shard_name)
        channel_index = _resolve_xtraffic_channel(xtraffic_channel)
        node_ids = [int(token.strip()) for token in str(xtraffic_node_indices).split(",") if token.strip()]
        series_list = [
            _impute_series(extract_node_series(bundle, node_index=node_id, channel=channel_index))
            for node_id in node_ids
        ]
        xs, ys = _build_windows_from_series_list(series_list, context_length, prediction_length)
        feature = f"xtraffic_nodes_{len(node_ids)}_{XTRAFFIC_CHANNELS[channel_index]}_windowed"
        return xs, ys, feature

    if dataset_kind == "timemmd":
        if not timemmd_root:
            raise ValueError("--timemmd-root is required when --dataset-kind=timemmd")
        bundle = load_timemmd_domain(timemmd_root, timemmd_domain)
        series = _impute_series(bundle["numerical"]["OT"].astype(float).to_numpy())
        xs, ys = _build_windows_from_series_list([series], context_length, prediction_length)
        return xs, ys, f"timemmd_{timemmd_domain}_OT_windowed"

    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Train or materialize a forecast baseline for revision benchmarks.")
    parser.add_argument("--dataset-kind", choices=["csv", "xtraffic", "mtbench", "timemmd"], default="csv")
    parser.add_argument("--dataset-id", default=None)
    parser.add_argument("--list-standard-datasets", action="store_true")
    parser.add_argument("--csv-path", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--baseline-name", default=None)
    parser.add_argument("--target-col", default=None)
    parser.add_argument("--split-policy", default=None)
    parser.add_argument("--training-split-id", default="default")
    parser.add_argument("--xtraffic-data-dir", default=None)
    parser.add_argument("--xtraffic-shard-name", default="p01_done.npy")
    parser.add_argument("--xtraffic-node-index", type=int, default=0)
    parser.add_argument("--xtraffic-node-indices", default=None)
    parser.add_argument("--xtraffic-channel", default="flow")
    parser.add_argument("--mtbench-path", default=None)
    parser.add_argument("--mtbench-limit", type=int, default=None)
    parser.add_argument("--timemmd-root", default="data/Time-MMD")
    parser.add_argument("--timemmd-domain", default="Energy")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--context-length", type=int, default=96)
    parser.add_argument("--prediction-length", type=int, default=24)
    parser.add_argument("--season-length", type=int, default=24)
    parser.add_argument("--alpha", type=float, default=0.6)
    parser.add_argument("--beta", type=float, default=0.2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--optimizer", default=None)
    parser.add_argument("--hidden-size", type=int, default=None)
    parser.add_argument("--tslib-root", default=None)
    args = parser.parse_args()

    if args.list_standard_datasets:
        print(json.dumps(list_standard_datasets(), ensure_ascii=False, indent=2))
        return

    if not args.output_dir or not args.baseline_name:
        raise ValueError("--output-dir and --baseline-name are required unless --list-standard-datasets is used.")

    dataset_info = _resolve_standard_dataset_args(args)
    if dataset_info is not None:
        args.csv_path = str(dataset_info["csv_path"])
        args.target_col = dataset_info["target_col"]

    dataset_id = str(dataset_info["dataset_id"]) if dataset_info is not None else None
    dataset_family = str(dataset_info["dataset_family"]) if dataset_info is not None else args.dataset_kind
    split_policy = str(dataset_info["split_policy"]) if dataset_info is not None else (args.split_policy or "series_train_tail_holdout")

    series, feature = _load_training_series(
        dataset_kind=args.dataset_kind,
        csv_path=args.csv_path,
        target_col=args.target_col,
        xtraffic_data_dir=args.xtraffic_data_dir,
        xtraffic_shard_name=args.xtraffic_shard_name,
        xtraffic_node_index=args.xtraffic_node_index,
        xtraffic_node_indices=args.xtraffic_node_indices,
        xtraffic_channel=args.xtraffic_channel,
        mtbench_path=args.mtbench_path,
        mtbench_limit=args.mtbench_limit,
        timemmd_root=args.timemmd_root,
        timemmd_domain=args.timemmd_domain,
    )
    train_end = max(args.context_length + args.prediction_length, int(len(series) * args.train_ratio))
    train_end = min(train_end, len(series))
    train_split = np.asarray(series[:train_end], dtype=np.float64)
    val_split = np.asarray(series[train_end:], dtype=np.float64) if train_end < len(series) else None

    baseline_kwargs = {
        "context_length": args.context_length,
        "prediction_length": args.prediction_length,
        "seq_len": args.context_length,
        "pred_len": args.prediction_length,
        "season_length": args.season_length,
        "alpha": args.alpha,
        "beta": args.beta,
        "device": args.device,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "dataset_id": dataset_id,
        "dataset_family": dataset_family,
        "split_policy": split_policy,
        "training_split_id": args.training_split_id,
        "feature": feature,
    }
    if args.optimizer:
        baseline_kwargs["optimizer"] = args.optimizer
    if args.hidden_size is not None:
        baseline_kwargs["hidden_size"] = int(args.hidden_size)

    materialized_without_training = args.baseline_name.endswith("_tslib")
    manifest_paths = None
    if materialized_without_training:
        if not args.csv_path:
            raise ValueError("TSLib paper baselines currently require a csv-backed standard dataset input.")
        training_manifest = build_tslib_training_manifest(
            baseline_name=args.baseline_name,
            dataset_id=dataset_id,
            csv_path=args.csv_path,
            context_length=args.context_length,
            prediction_length=args.prediction_length,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            output_dir=args.output_dir,
            training_split_id=args.training_split_id,
            split_policy=split_policy,
            feature=feature,
            tslib_root=args.tslib_root,
        )
        baseline_kwargs["tslib_artifact"] = training_manifest

    baseline = create_baseline(args.baseline_name, **baseline_kwargs)
    structured = _load_structured_training_windows(
        dataset_kind=args.dataset_kind,
        csv_path=args.csv_path,
        target_col=args.target_col,
        xtraffic_data_dir=args.xtraffic_data_dir,
        xtraffic_shard_name=args.xtraffic_shard_name,
        xtraffic_node_index=args.xtraffic_node_index,
        xtraffic_node_indices=args.xtraffic_node_indices,
        xtraffic_channel=args.xtraffic_channel,
        mtbench_path=args.mtbench_path,
        mtbench_limit=args.mtbench_limit,
        timemmd_root=args.timemmd_root,
        timemmd_domain=args.timemmd_domain,
        context_length=args.context_length,
        prediction_length=args.prediction_length,
    )

    if materialized_without_training:
        train_points = int(len(train_split))
        val_points = int(len(val_split)) if val_split is not None else 0
        training_status = "materialized_interface_only"
    elif structured is not None and args.baseline_name in {"lstm_official", "dlinear_official"}:
        train_x, train_y, feature = structured
        baseline.fit_windows(train_x, train_y)
        train_points = int(train_x.shape[0])
        val_points = 0
        training_status = "trained"
    else:
        baseline.fit(train_split, val_split)
        train_points = int(len(train_split))
        val_points = int(len(val_split)) if val_split is not None else 0
        training_status = "trained"
    baseline.save(args.output_dir)
    if materialized_without_training:
        manifest_paths = write_tslib_training_manifest(args.output_dir, training_manifest)

    payload = {
        "baseline": baseline.describe(),
        "dataset_kind": args.dataset_kind,
        "dataset_id": dataset_id,
        "dataset_family": dataset_family,
        "split_policy": split_policy,
        "training_split_id": args.training_split_id,
        "feature": feature,
        "train_points": train_points,
        "val_points": val_points,
        "training_status": training_status,
        "materialized_without_training": materialized_without_training,
        "tslib_training_manifest": manifest_paths,
        "output_dir": str(Path(args.output_dir).resolve()),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
