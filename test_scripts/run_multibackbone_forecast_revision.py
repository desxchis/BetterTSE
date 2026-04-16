from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from run_forecast_revision import run_revision
from test_scripts.build_forecast_revision_benchmark import build_benchmark
from test_scripts.build_timemmd_projected_revision_benchmark import build_timemmd_projected_benchmark


EXAMPLE_BACKBONES = ["dlinear_official", "patchtst"]
DEFAULT_MODES = [
    "base_only",
    "heuristic_revision",
    "direct_delta_regression",
    "wo_parameter_calibration",
    "localized_full_revision",
]


def _parse_key_value_pairs(items: Iterable[str]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"expected NAME=VALUE format, got '{item}'")
        key, value = item.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key or not value:
            raise ValueError(f"invalid NAME=VALUE pair: '{item}'")
        mapping[key] = value
    return mapping


def _run_subprocess(cmd: List[str]) -> Dict[str, Any]:
    started = datetime.now(timezone.utc).isoformat()
    cp = subprocess.run(
        cmd,
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "cmd": cmd,
        "returncode": cp.returncode,
        "stdout": cp.stdout[-20000:],
        "stderr": cp.stderr[-20000:],
        "started_utc": started,
        "ended_utc": datetime.now(timezone.utc).isoformat(),
    }


def _prepare_baseline(
    *,
    baseline_name: str,
    output_dir: Path,
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
    timemmd_root: str,
    timemmd_domain: str,
    context_length: int,
    prediction_length: int,
    season_length: int,
    alpha: float,
    beta: float,
    device: str,
    epochs: int,
    batch_size: int,
    lr: float,
    optimizer: str | None,
    hidden_size: int | None,
) -> Dict[str, Any]:
    cmd = [
        sys.executable,
        "test_scripts/train_forecast_baseline.py",
        "--dataset-kind",
        dataset_kind,
        "--output-dir",
        str(output_dir),
        "--baseline-name",
        baseline_name,
        "--context-length",
        str(context_length),
        "--prediction-length",
        str(prediction_length),
        "--season-length",
        str(season_length),
        "--alpha",
        str(alpha),
        "--beta",
        str(beta),
        "--device",
        device,
        "--epochs",
        str(epochs),
        "--batch-size",
        str(batch_size),
        "--lr",
        str(lr),
        "--timemmd-root",
        timemmd_root,
        "--timemmd-domain",
        timemmd_domain,
        "--xtraffic-shard-name",
        xtraffic_shard_name,
        "--xtraffic-node-index",
        str(xtraffic_node_index),
        "--xtraffic-channel",
        xtraffic_channel,
    ]
    if csv_path:
        cmd.extend(["--csv-path", csv_path])
    if target_col is not None:
        cmd.extend(["--target-col", target_col])
    if xtraffic_data_dir:
        cmd.extend(["--xtraffic-data-dir", xtraffic_data_dir])
    if xtraffic_node_indices:
        cmd.extend(["--xtraffic-node-indices", xtraffic_node_indices])
    if mtbench_path:
        cmd.extend(["--mtbench-path", mtbench_path])
    if mtbench_limit is not None:
        cmd.extend(["--mtbench-limit", str(mtbench_limit)])
    if optimizer:
        cmd.extend(["--optimizer", optimizer])
    if hidden_size is not None:
        cmd.extend(["--hidden-size", str(hidden_size)])

    rec = _run_subprocess(cmd)
    if rec["returncode"] != 0:
        raise RuntimeError(
            f"baseline training failed for {baseline_name}: {rec['stderr'][-1000:]}"
        )
    return rec


def _summary_row(
    *,
    baseline_name: str,
    model_dir: str | None,
    benchmark_path: str,
    localized_summary: Dict[str, Any] | None,
    base_summary: Dict[str, Any] | None,
) -> Dict[str, Any]:
    localized_summary = localized_summary or {}
    base_summary = base_summary or {}
    return {
        "baseline_name": baseline_name,
        "model_dir": model_dir,
        "benchmark_path": benchmark_path,
        "avg_revision_gain": localized_summary.get("avg_revision_gain"),
        "applicable_avg_revision_gain": localized_summary.get("applicable_avg_revision_gain"),
        "avg_future_t_iou": localized_summary.get("avg_future_t_iou"),
        "avg_edited_mae_vs_revision_target": localized_summary.get("avg_edited_mae_vs_revision_target"),
        "avg_base_mae_vs_revision_target": base_summary.get("avg_base_mae_vs_revision_target"),
        "avg_revision_needed_match": localized_summary.get("avg_revision_needed_match"),
        "applicable_count": localized_summary.get("applicable_count"),
        "total": localized_summary.get("total"),
    }


def _benchmark_metadata_for_kind(dataset_kind: str) -> tuple[str, str]:
    if dataset_kind == "timemmd":
        return "future_guided_projection", "future_guided_projected_revision"
    return "synthetic_physical_injection", "controlled_synthetic_revision"


def _format_value(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _write_markdown_report(
    *,
    output_path: Path,
    payload: Dict[str, Any],
) -> None:
    lines = [
        "# Multi-Backbone Forecast Revision Report",
        "",
        "This report belongs to the forecast-revision application line of BetterTSE.",
        "",
        f"- generated_utc: {payload['generated_utc']}",
        f"- dataset_name: {payload['dataset_name']}",
        f"- task_family: {payload['task_family']}",
        f"- application_of: {payload['application_of']}",
        f"- target_construction_method: {payload['target_construction_method']}",
        f"- modes: {', '.join(payload['modes'])}",
        "",
        "## Backbone Comparison",
        "",
        "| Backbone | base target MAE | localized target MAE | localized gain | applicable gain | future tIoU | revision_needed_match |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["comparison_rows"]:
        lines.append(
            f"| {row['baseline_name']} | {_format_value(row['avg_base_mae_vs_revision_target'])} | "
            f"{_format_value(row['avg_edited_mae_vs_revision_target'])} | {_format_value(row['avg_revision_gain'])} | "
            f"{_format_value(row['applicable_avg_revision_gain'])} | {_format_value(row['avg_future_t_iou'])} | "
            f"{_format_value(row['avg_revision_needed_match'])} |"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Pure editing remains the method core; this report only covers the downstream forecast-revision line.",
            "- The listed backbones are only the ones selected for this run; they are not a locked global backbone set.",
            "- Newer forecasting models can be integrated as long as they produce a compatible `base_forecast`.",
            "- This report currently covers the controlled synthetic revision regime built from physical injection on top of `base_forecast`.",
            "- The future-guided projected revision regime should be reported separately when using projected benchmark builders.",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def run_multibackbone_experiment(
    *,
    dataset_name: str,
    output_dir: str,
    baseline_names: List[str],
    baseline_model_dirs: Dict[str, str],
    prepare_missing_baselines: bool,
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
    timemmd_root: str,
    timemmd_domain: str,
    timemmd_text_source: str,
    seq_len: int,
    pred_len: int,
    num_samples: int,
    include_no_revision_every: int,
    context_style: str,
    modes: List[str],
    calibration_strategy: str,
    revision_executor: str,
    tedit_model_path: str | None,
    tedit_config_path: str | None,
    tedit_device: str,
    max_samples: int | None,
    device: str,
    season_length: int,
    epochs: int,
    batch_size: int,
    lr: float,
    optimizer: str | None,
    hidden_size: int | None,
) -> Dict[str, Any]:
    out_root = Path(output_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    benchmark_root = out_root / "benchmarks"
    run_root = out_root / "runs"
    baseline_root = out_root / "baselines"
    training_logs: Dict[str, Any] = {}
    baselines_payload: Dict[str, Any] = {}
    comparison_rows: List[Dict[str, Any]] = []
    task_family = "forecast_revision"
    application_of = "bettertse_editing"
    target_construction_method, target_regime = _benchmark_metadata_for_kind(dataset_kind)

    for baseline_name in baseline_names:
        model_dir = baseline_model_dirs.get(baseline_name)
        if not model_dir and prepare_missing_baselines:
            prepared_dir = baseline_root / baseline_name
            training_logs[baseline_name] = _prepare_baseline(
                baseline_name=baseline_name,
                output_dir=prepared_dir,
                dataset_kind=dataset_kind,
                csv_path=csv_path,
                target_col=target_col,
                xtraffic_data_dir=xtraffic_data_dir,
                xtraffic_shard_name=xtraffic_shard_name,
                xtraffic_node_index=xtraffic_node_index,
                xtraffic_node_indices=xtraffic_node_indices,
                xtraffic_channel=xtraffic_channel,
                mtbench_path=mtbench_path,
                mtbench_limit=mtbench_limit,
                timemmd_root=timemmd_root,
                timemmd_domain=timemmd_domain,
                context_length=seq_len,
                prediction_length=pred_len,
                season_length=season_length,
                alpha=0.6,
                beta=0.2,
                device=device,
                epochs=epochs,
                batch_size=batch_size,
                lr=lr,
                optimizer=optimizer,
                hidden_size=hidden_size,
            )
            model_dir = str(prepared_dir)

        if dataset_kind == "timemmd":
            benchmark_summary = build_timemmd_projected_benchmark(
                timemmd_root=timemmd_root,
                domain=timemmd_domain,
                output_dir=str(benchmark_root / baseline_name),
                baseline_name=baseline_name,
                baseline_model_dir=model_dir,
                seq_len=seq_len,
                pred_len=pred_len,
                max_samples=num_samples,
                text_source=timemmd_text_source,
            )
        else:
            benchmark_summary = build_benchmark(
                csv_path=csv_path or "",
                dataset_name=dataset_name,
                output_dir=str(benchmark_root / baseline_name),
                baseline_name=baseline_name,
                baseline_model_dir=model_dir,
                seq_len=seq_len,
                pred_len=pred_len,
                num_samples=num_samples,
                target_col=target_col,
                include_no_revision_every=include_no_revision_every,
                context_style=context_style,
            )
        benchmark_path = benchmark_summary["output_path"]

        mode_payloads: Dict[str, Any] = {}
        for mode in modes:
            mode_output = run_root / baseline_name / f"{mode}.json"
            vis_dir = run_root / baseline_name / "visualizations" / mode
            result = run_revision(
                benchmark_path=benchmark_path,
                output_path=str(mode_output),
                mode=mode,
                max_samples=max_samples,
                vis_dir=str(vis_dir),
                save_visualizations=(mode == "localized_full_revision"),
                calibration_strategy=calibration_strategy,
                revision_executor=revision_executor,
                tedit_model_path=tedit_model_path,
                tedit_config_path=tedit_config_path,
                tedit_device=tedit_device,
            )
            mode_payloads[mode] = {
                "output_path": str(mode_output),
                "summary": result.get("summary", {}),
                "normalized_summary": result.get("normalized_summary", {}),
                "visualization_dir": result.get("visualization_dir"),
            }

        comparison_row = _summary_row(
            baseline_name=baseline_name,
            model_dir=model_dir,
            benchmark_path=benchmark_path,
            localized_summary=mode_payloads.get("localized_full_revision", {}).get("summary"),
            base_summary=mode_payloads.get("base_only", {}).get("summary"),
        )
        comparison_rows.append(comparison_row)
        baselines_payload[baseline_name] = {
            "model_dir": model_dir,
            "benchmark_path": benchmark_path,
            "benchmark_summary": benchmark_summary,
            "task_family": task_family,
            "application_of": application_of,
            "target_construction_method": target_construction_method,
            "target_regime": target_regime,
            "modes": mode_payloads,
        }

    comparison_rows.sort(
        key=lambda row: float(row["avg_revision_gain"]) if row["avg_revision_gain"] is not None else float("-inf"),
        reverse=True,
    )
    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_name": dataset_name,
        "task_family": task_family,
        "application_of": application_of,
        "target_construction_method": target_construction_method,
        "target_regime": target_regime,
        "modes": modes,
        "baseline_names": baseline_names,
        "comparison_rows": comparison_rows,
        "training_logs": training_logs,
        "baselines": baselines_payload,
    }
    json_path = out_root / "multibackbone_summary.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path = out_root / "multibackbone_report.md"
    _write_markdown_report(output_path=md_path, payload=payload)
    payload["summary_path"] = str(json_path)
    payload["report_path"] = str(md_path)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Run forecast-revision experiments for any selected set of base forecasting backbones.")
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset-kind", choices=["csv", "xtraffic", "mtbench", "timemmd"], default="csv")
    parser.add_argument("--csv-path", default=None)
    parser.add_argument("--target-col", default=None)
    parser.add_argument("--xtraffic-data-dir", default=None)
    parser.add_argument("--xtraffic-shard-name", default="p01_done.npy")
    parser.add_argument("--xtraffic-node-index", type=int, default=0)
    parser.add_argument("--xtraffic-node-indices", default=None)
    parser.add_argument("--xtraffic-channel", default="flow")
    parser.add_argument("--mtbench-path", default=None)
    parser.add_argument("--mtbench-limit", type=int, default=None)
    parser.add_argument("--timemmd-root", default="data/Time-MMD")
    parser.add_argument("--timemmd-domain", default="Energy")
    parser.add_argument("--timemmd-text-source", choices=["report", "search", "all"], default="report")
    parser.add_argument("--baseline-names", nargs="+", default=list(EXAMPLE_BACKBONES), help="Backbones to include in this run. Defaults to example repo-supported backbones, not a locked project-wide set.")
    parser.add_argument("--baseline-model-dir", action="append", default=[], help="Repeat NAME=PATH for pre-trained baselines.")
    parser.add_argument("--prepare-missing-baselines", action="store_true")
    parser.add_argument("--seq-len", type=int, default=96)
    parser.add_argument("--pred-len", type=int, default=24)
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--include-no-revision-every", type=int, default=0)
    parser.add_argument("--context-style", choices=["generic", "traffic_incident"], default="generic")
    parser.add_argument("--modes", nargs="+", default=list(DEFAULT_MODES))
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--calibration-strategy", default="rule_local_stats")
    parser.add_argument("--revision-executor", default="profile", choices=["profile", "tedit_hybrid"])
    parser.add_argument("--tedit-model", default=None)
    parser.add_argument("--tedit-config", default=None)
    parser.add_argument("--tedit-device", default="cuda:0")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--season-length", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--optimizer", default=None)
    parser.add_argument("--hidden-size", type=int, default=None)
    args = parser.parse_args()

    if args.dataset_kind == "csv" and not args.csv_path:
        raise ValueError("--csv-path is required when --dataset-kind=csv")
    if args.dataset_kind == "xtraffic" and not args.xtraffic_data_dir:
        raise ValueError("--xtraffic-data-dir is required when --dataset-kind=xtraffic")
    if args.dataset_kind == "mtbench" and not args.mtbench_path:
        raise ValueError("--mtbench-path is required when --dataset-kind=mtbench")

    result = run_multibackbone_experiment(
        dataset_name=args.dataset_name,
        output_dir=args.output_dir,
        baseline_names=list(args.baseline_names),
        baseline_model_dirs=_parse_key_value_pairs(args.baseline_model_dir),
        prepare_missing_baselines=bool(args.prepare_missing_baselines),
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
        timemmd_text_source=args.timemmd_text_source,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        num_samples=args.num_samples,
        include_no_revision_every=args.include_no_revision_every,
        context_style=args.context_style,
        modes=list(args.modes),
        calibration_strategy=args.calibration_strategy,
        revision_executor=args.revision_executor,
        tedit_model_path=args.tedit_model,
        tedit_config_path=args.tedit_config,
        tedit_device=args.tedit_device,
        max_samples=args.max_samples,
        device=args.device,
        season_length=args.season_length,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        optimizer=args.optimizer,
        hidden_size=args.hidden_size,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
