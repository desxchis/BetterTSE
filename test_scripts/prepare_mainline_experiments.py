from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from forecasting import (
    PAPER_HISTORICAL_CONTROL_BASELINES,
    PAPER_MAINLINE_BASELINES,
    get_standard_dataset,
    resolve_standard_dataset,
)
from forecasting.registry import create_baseline
from forecasting.tslib_bridge import build_tslib_training_manifest, write_tslib_training_manifest

DEFAULT_PURE_DATASET = "etth1"
DEFAULT_REVISION_DATASETS = ("traffic", "weather", "etth1", "ettm1", "electricity")
DEFAULT_REVISION_BACKBONES = PAPER_MAINLINE_BASELINES + PAPER_HISTORICAL_CONTROL_BASELINES

PURE_EDITING_OPERATORS = ("hump", "step", "plateau", "flatline", "irregular_noise", "none")
PURE_EDITING_INSTRUCTION_BUCKETS = ("direct", "indirect", "ambiguous", "multi_view")
PURE_EDITING_METRICS = ("mae_vs_target", "mse_vs_target", "t_iou", "preservation_mae")
REVISION_METRICS = (
    "edited_mae_vs_revision_target",
    "revision_gain",
    "edited_mae_vs_future_gt",
    "future_t_iou",
)


def _parse_csv_arg(raw: str | None, *, default: Iterable[str] = ()) -> List[str]:
    if raw is None:
        return [str(item).strip() for item in default if str(item).strip()]
    return [token.strip() for token in str(raw).split(",") if token.strip()]


def _now_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _quote_command(parts: List[str]) -> str:
    rendered: List[str] = []
    for part in parts:
        token = str(part)
        if token == "":
            rendered.append("''")
        elif any(ch.isspace() for ch in token):
            rendered.append(json.dumps(token))
        else:
            rendered.append(token)
    return " ".join(rendered)


def _dataset_inventory(dataset_ids: Iterable[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for dataset_id in dataset_ids:
        spec = get_standard_dataset(dataset_id)
        payload = spec.to_dict()
        payload["csv_path"] = spec.default_csv_path
        rows.append(payload)
    return rows


def _pure_editing_schema() -> Dict[str, Any]:
    return {
        "task_family": "pure_time_series_editing",
        "target_regime": "controlled_physical_injection",
        "required_fields": [
            "task_type",
            "series_id",
            "base_ts",
            "context_text",
            "intent_label",
            "localization_label",
            "canonical_tool_label",
            "hybrid_tool_label",
            "parameter_label",
            "target_ts",
            "mask_gt",
        ],
        "planned_operator_set": list(PURE_EDITING_OPERATORS),
        "instruction_buckets": list(PURE_EDITING_INSTRUCTION_BUCKETS),
        "primary_metrics": list(PURE_EDITING_METRICS),
        "notes": [
            "The current repository already exposes controlled physical injection through build_event_driven_testset.py.",
            "The planned operator set includes plateau and none; the current builder is close but not fully parity-complete yet.",
        ],
    }


def _revision_schema() -> Dict[str, Any]:
    return {
        "task_family": "forecast_revision",
        "application_of": "bettertse_editing",
        "required_fields": [
            "task_type",
            "series_id",
            "history_ts",
            "base_forecast",
            "context_text",
            "intent_label",
            "localization_label",
            "canonical_tool_label",
            "hybrid_tool_label",
            "parameter_label",
            "revision_target",
            "future_gt",
        ],
        "primary_metrics": list(REVISION_METRICS),
        "notes": [
            "The repository already uses ForecastRevisionSample as the runnable source-of-truth payload.",
            "revision_target should remain distinct from future_gt and be reported separately in evaluation.",
        ],
    }


def _logger_schema() -> Dict[str, Any]:
    return {
        "shared_fields": [
            "sample_id",
            "task_family",
            "dataset_name",
            "target_regime",
            "model_name",
            "mode",
            "context_text",
            "intent_alignment",
            "localization_prediction",
            "tool_prediction",
            "metrics",
            "visualization_path",
        ],
        "pure_editing_metrics": list(PURE_EDITING_METRICS),
        "revision_metrics": list(REVISION_METRICS),
    }


def _write_schema_artifacts(schema_root: Path) -> Dict[str, str]:
    schema_root.mkdir(parents=True, exist_ok=True)
    paths = {
        "pure_editing_schema": schema_root / "pure_editing_sample_schema.json",
        "forecast_revision_schema": schema_root / "forecast_revision_sample_schema.json",
        "result_logger_schema": schema_root / "result_logger_schema.json",
    }
    paths["pure_editing_schema"].write_text(json.dumps(_pure_editing_schema(), ensure_ascii=False, indent=2), encoding="utf-8")
    paths["forecast_revision_schema"].write_text(json.dumps(_revision_schema(), ensure_ascii=False, indent=2), encoding="utf-8")
    paths["result_logger_schema"].write_text(json.dumps(_logger_schema(), ensure_ascii=False, indent=2), encoding="utf-8")
    return {key: str(value) for key, value in paths.items()}


def _materialize_paper_baseline_contract(
    *,
    dataset_id: str,
    baseline_name: str,
    output_dir: Path,
    context_length: int,
    prediction_length: int,
    epochs: int,
    batch_size: int,
    lr: float,
    training_split_id: str,
) -> Dict[str, Any]:
    dataset = resolve_standard_dataset(dataset_id)
    manifest = build_tslib_training_manifest(
        baseline_name=baseline_name,
        dataset_id=dataset_id,
        csv_path=str(dataset["csv_path"]),
        context_length=context_length,
        prediction_length=prediction_length,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        output_dir=output_dir,
        training_split_id=training_split_id,
        split_policy=str(dataset["split_policy"]),
        feature=str(dataset["target_col"] or "0"),
        tslib_root=None,
    )
    baseline = create_baseline(
        baseline_name,
        context_length=context_length,
        prediction_length=prediction_length,
        seq_len=context_length,
        pred_len=prediction_length,
        dataset_id=dataset_id,
        dataset_family=str(dataset["dataset_family"]),
        split_policy=str(dataset["split_policy"]),
        training_split_id=training_split_id,
        feature=str(dataset["target_col"] or "0"),
        tslib_artifact=manifest,
    )
    baseline.save(output_dir)
    manifest_paths = write_tslib_training_manifest(output_dir, manifest)
    return {
        "dataset_id": dataset_id,
        "baseline_name": baseline_name,
        "output_dir": str(output_dir),
        "baseline_state_path": str(output_dir / "baseline_state.json"),
        "tslib_training_manifest": manifest_paths,
        "status": "contract_materialized",
        "export_ready": False,
    }


def _build_stage_commands(
    *,
    output_root: Path,
    pure_dataset_id: str,
    revision_dataset_ids: List[str],
    revision_backbones: List[str],
    pure_num_samples: int,
    revision_num_samples: int,
    seq_len: int,
    pred_len: int,
    max_samples: int,
) -> List[Dict[str, Any]]:
    pure_dataset = get_standard_dataset(pure_dataset_id)
    pure_benchmark_dir = output_root / "benchmarks" / "pure_editing"
    pure_results_dir = output_root / "runs" / "pure_editing"
    pure_testset = pure_benchmark_dir / f"{pure_dataset.display_name}_event_driven_{pure_num_samples}.json"

    stages: List[Dict[str, Any]] = [
        {
            "name": "pure_editing_benchmark_build",
            "type": "pure_editing",
            "status": "ready_now",
            "output_dir": str(pure_benchmark_dir),
            "expected_artifact": str(pure_testset),
            "command": [
                sys.executable,
                "test_scripts/build_event_driven_testset.py",
                "--csv-path",
                pure_dataset.default_csv_path,
                "--dataset-name",
                pure_dataset.display_name,
                "--output-dir",
                str(pure_benchmark_dir),
                "--num-samples",
                str(pure_num_samples),
                "--seq-len",
                str(seq_len),
            ],
            "notes": [
                "Builds the current controlled physical pure-editing benchmark entrypoint.",
                "This is the closest existing builder to the planned P1 benchmark.",
            ],
        },
        {
            "name": "pure_editing_full_pipeline_smoke",
            "type": "pure_editing",
            "status": "ready_now",
            "output_dir": str(pure_results_dir),
            "expected_artifact": str(pure_results_dir / "pipeline_results.json"),
            "command": [
                sys.executable,
                "run_pipeline.py",
                "--testset",
                str(pure_testset),
                "--output",
                str(pure_results_dir / "pipeline_results.json"),
                "--max-samples",
                str(max_samples),
            ],
            "notes": [
                "Runs the existing pure-editing pipeline without requiring TEdit checkpoints.",
                "Needs API credentials at execution time because the planner still calls the configured LLM backend.",
            ],
        },
        {
            "name": "pure_editing_baseline_pack",
            "type": "pure_editing",
            "status": "ready_now",
            "output_dir": str(pure_results_dir / "baselines"),
            "expected_artifact": str(pure_results_dir / "baselines" / "full_bettertse.json"),
            "command": [
                sys.executable,
                "run_pipeline.py",
                "--testset",
                str(pure_testset),
                "--output",
                str(pure_results_dir / "baselines" / "full_bettertse.json"),
                "--mode",
                "full_bettertse",
                "--max-samples",
                str(max_samples),
            ],
            "notes": [
                "Direct-Edit, w-o Localization, and w-o Canonical Layer are now exposed as separate CLI modes in run_pipeline.py.",
                "This stage records the shared benchmark path and one representative full-mode launcher.",
            ],
        },
    ]

    for dataset_id in revision_dataset_ids:
        dataset = get_standard_dataset(dataset_id)
        benchmark_dir = output_root / "benchmarks" / "forecast_revision" / dataset_id
        benchmark_template = benchmark_dir / f"forecast_revision_{dataset.display_name}_{{baseline_name}}_{revision_num_samples}.json"
        for baseline_name in revision_backbones:
            stages.append(
                {
                    "name": f"forecast_revision_controlled_builder__{dataset_id}__{baseline_name}",
                    "type": "forecast_revision",
                    "status": "ready_after_backbone_training",
                    "output_dir": str(benchmark_dir),
                    "expected_artifact": str(benchmark_template).replace("{baseline_name}", baseline_name),
                    "command": [
                        sys.executable,
                        "test_scripts/build_forecast_revision_benchmark.py",
                        "--csv-path",
                        dataset.default_csv_path,
                        "--dataset-name",
                        dataset.display_name,
                        "--output-dir",
                        str(benchmark_dir),
                        "--baseline-name",
                        baseline_name,
                        "--baseline-model-dir",
                        str(output_root / "prepared_backbones" / dataset_id / baseline_name),
                        "--num-samples",
                        str(revision_num_samples),
                        "--seq-len",
                        str(seq_len),
                        "--pred-len",
                        str(pred_len),
                    ],
                    "notes": [
                        "Controlled synthetic revision benchmark builder is already available.",
                        "Execution becomes runnable after the staged backbone directory contains a trained or exported inference artifact.",
                    ],
                }
            )
            stages.append(
                {
                    "name": f"forecast_revision_eval__{dataset_id}__{baseline_name}",
                    "type": "forecast_revision",
                    "status": "ready_after_backbone_training",
                    "output_dir": str(output_root / "runs" / "forecast_revision" / dataset_id / baseline_name),
                    "expected_artifact": str(output_root / "runs" / "forecast_revision" / dataset_id / baseline_name / "localized_full_revision.json"),
                    "command": [
                        sys.executable,
                        "run_forecast_revision.py",
                        "--benchmark",
                        str(benchmark_template).replace("{baseline_name}", baseline_name),
                        "--output",
                        str(output_root / "runs" / "forecast_revision" / dataset_id / baseline_name / "localized_full_revision.json"),
                        "--mode",
                        "localized_full_revision",
                        "--max-samples",
                        str(revision_num_samples),
                    ],
                    "notes": [
                        "Uses the existing structured what-where-how-much revision pipeline.",
                        "Current runnable comparison modes include base_only, heuristic_revision, direct_delta_regression, wo_parameter_calibration, and localized_full_revision.",
                    ],
                }
            )
    stages.append(
        {
            "name": "forecast_revision_projected_target_mainline",
            "type": "forecast_revision",
            "status": "partial_repo_support",
            "output_dir": str(output_root / "benchmarks" / "forecast_revision_projected"),
            "expected_artifact": str(output_root / "benchmarks" / "forecast_revision_projected"),
            "command": [],
            "notes": [
                "The repo already contains projected or real-target builders for TimeMMD and XTraffic variants.",
                "A general future-guided projected builder for the full standard LTSF dataset pool is not yet exposed as one shared CLI entrypoint.",
            ],
        }
    )
    return stages


def _render_readme(plan: Dict[str, Any]) -> str:
    lines = [
        "# Mainline Experiment Preparation",
        "",
        "Prepared by test_scripts/prepare_mainline_experiments.py.",
        "",
        "## Purpose",
        "",
        "Prepare BetterTSE experiment assets without starting large experiment runs.",
        "",
        "## Prepared Now",
        "",
        "- unified sample-schema specs for pure editing, forecast revision, and result logging",
        "- staged paper-backbone contracts for the requested forecasting models",
        "- dataset inventory and readiness manifest",
        "- runnable command scripts for stages that already exist in the repository",
        "",
        "## Current Gaps",
        "",
        "- a single generic projected-target builder across all standard LTSF datasets is still not exposed",
        "",
        "## Ready Commands",
        "",
    ]
    for stage in plan["stages"]:
        if stage["status"] not in {"ready_now", "ready_after_backbone_training"}:
            continue
        lines.append(f"- {stage['name']}")
        lines.append(f"  command: {_quote_command(stage['command'])}")
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- ready_now means the command matches current repository support immediately.",
            "- ready_after_backbone_training means the directory and launcher are prepared, but a trained or exported backbone artifact must exist first.",
            "",
        ]
    )
    return "\n".join(lines)


def _render_shell(stages: List[Dict[str, Any]], allowed_statuses: set[str]) -> str:
    lines = ["#!/usr/bin/env bash", "set -euo pipefail", ""]
    for stage in stages:
        if stage["status"] not in allowed_statuses or not stage["command"]:
            continue
        lines.append(f"# {stage['name']}")
        lines.append(_quote_command(stage["command"]))
        lines.append("")
    return "\n".join(lines)


def _write_plan_artifacts(plan: Dict[str, Any], output_root: Path) -> Dict[str, str]:
    manifest_path = output_root / "experiment_plan.json"
    readme_path = output_root / "README.md"
    ready_shell = output_root / "run_ready_now.sh"
    after_training_shell = output_root / "run_after_backbone_training.sh"
    manifest_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    readme_path.write_text(_render_readme(plan), encoding="utf-8")
    ready_shell.write_text(_render_shell(plan["stages"], {"ready_now"}), encoding="utf-8")
    after_training_shell.write_text(_render_shell(plan["stages"], {"ready_after_backbone_training"}), encoding="utf-8")
    try:
        ready_shell.chmod(0o755)
        after_training_shell.chmod(0o755)
    except OSError:
        pass
    return {
        "manifest_path": str(manifest_path),
        "readme_path": str(readme_path),
        "run_ready_now": str(ready_shell),
        "run_after_backbone_training": str(after_training_shell),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare BetterTSE mainline experiment assets without launching the experiments.")
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--pure-dataset-id", default=DEFAULT_PURE_DATASET)
    parser.add_argument("--revision-dataset-ids", default=",".join(DEFAULT_REVISION_DATASETS))
    parser.add_argument("--revision-backbones", default=",".join(DEFAULT_REVISION_BACKBONES))
    parser.add_argument("--pure-num-samples", type=int, default=64)
    parser.add_argument("--revision-num-samples", type=int, default=64)
    parser.add_argument("--seq-len", type=int, default=192)
    parser.add_argument("--pred-len", type=int, default=96)
    parser.add_argument("--context-length", type=int, default=192)
    parser.add_argument("--prediction-length", type=int, default=96)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--training-split-id", default="paper_mainline_v1")
    parser.add_argument("--max-samples", type=int, default=3)
    args = parser.parse_args()

    output_root = Path(args.output_root) if args.output_root else ROOT / "tmp" / "experiment_preparation" / f"mainline_{_now_slug()}"
    output_root.mkdir(parents=True, exist_ok=True)

    pure_dataset_id = str(args.pure_dataset_id).strip().lower()
    revision_dataset_ids = _parse_csv_arg(args.revision_dataset_ids, default=DEFAULT_REVISION_DATASETS)
    revision_backbones = _parse_csv_arg(args.revision_backbones, default=DEFAULT_REVISION_BACKBONES)

    schema_paths = _write_schema_artifacts(output_root / "schemas")

    prepared_backbones: List[Dict[str, Any]] = []
    for dataset_id in revision_dataset_ids:
        for baseline_name in revision_backbones:
            baseline_dir = output_root / "prepared_backbones" / dataset_id / baseline_name
            prepared_backbones.append(
                _materialize_paper_baseline_contract(
                    dataset_id=dataset_id,
                    baseline_name=baseline_name,
                    output_dir=baseline_dir,
                    context_length=args.context_length,
                    prediction_length=args.prediction_length,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    lr=args.lr,
                    training_split_id=args.training_split_id,
                )
            )

    plan = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "output_root": str(output_root),
        "purpose": "Prepare datasets, model contracts, schemas, and runnable command manifests for the BetterTSE pure-editing and forecast-revision experiment mainlines without launching the experiments.",
        "datasets": {
            "pure_editing": _dataset_inventory([pure_dataset_id]),
            "forecast_revision": _dataset_inventory(revision_dataset_ids),
        },
        "requested_revision_backbones": revision_backbones,
        "prepared_backbones": prepared_backbones,
        "schema_artifacts": schema_paths,
        "stages": _build_stage_commands(
            output_root=output_root,
            pure_dataset_id=pure_dataset_id,
            revision_dataset_ids=revision_dataset_ids,
            revision_backbones=revision_backbones,
            pure_num_samples=args.pure_num_samples,
            revision_num_samples=args.revision_num_samples,
            seq_len=args.seq_len,
            pred_len=args.pred_len,
            max_samples=args.max_samples,
        ),
        "notes": [
            "This preparation step intentionally avoids launching baseline training, benchmark construction, or evaluation runs.",
            "Paper backbones are staged through TSLib-compatible contracts so the training and export step can be attached later without changing the revision interface.",
            "Existing repository support is reused as-is; missing experiment baselines are reported as gaps instead of being approximated through ad hoc code paths.",
        ],
    }
    artifacts = _write_plan_artifacts(plan, output_root)

    print(json.dumps({"plan": plan, "artifacts": artifacts}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
