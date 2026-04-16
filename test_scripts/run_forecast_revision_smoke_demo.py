from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from run_forecast_revision import run_revision
from test_scripts.build_forecast_revision_benchmark import build_benchmark


def _repo_relative(path: str | Path) -> str:
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(_ROOT))
    except ValueError:
        return str(resolved)


def _build_demo_readme(
    *,
    output_dir: Path,
    benchmark_path: str,
    result_path: str,
    vis_dir: Path,
    run_summary: Dict[str, Any],
) -> str:
    results: List[Dict[str, Any]] = run_summary.get("results", [])
    lines: List[str] = [
        "# Forecast Revision Smoke Demo",
        "",
        "## Purpose",
        "",
        "This is a CPU-safe smoke pipeline for the current forecast-revision task mode.",
        "It only checks that the main path can run end to end:",
        "",
        "- build a small synthetic revision benchmark from `data/Weather.csv`",
        "- run `localized_full_revision` with `rule_local_stats`",
        "- save result JSON and PNG visualizations",
        "",
        "## Run Commands",
        "",
        "```bash",
        "python test_scripts/run_forecast_revision_smoke_demo.py \\",
        "  --output-dir tmp/forecast_revision_smoke_demo",
        "```",
        "",
        "## Key Outputs",
        "",
        f"- benchmark: `{_repo_relative(benchmark_path)}`",
        f"- results: `{_repo_relative(result_path)}`",
        f"- visualizations: `{_repo_relative(vis_dir)}`",
        "",
        "## Aggregate Summary",
        "",
        "```json",
        json.dumps(run_summary.get("summary", {}), ensure_ascii=False, indent=2),
        "```",
        "",
        "## Task Samples",
        "",
    ]

    for result in results[:3]:
        gt_intent = result.get("edit_intent_gt", {})
        lines.extend(
            [
                f"### Sample {result.get('sample_id', 'unknown')}",
                "",
                f"- context: `{result.get('context_text', '')}`",
                f"- gt intent: `{json.dumps(gt_intent, ensure_ascii=False)}`",
                f"- predicted region: `{result.get('pred_region')}`",
                f"- gt region: `{result.get('gt_region')}`",
                f"- revision gain: `{result.get('metrics', {}).get('revision_gain')}`",
                f"- visualization: `{_repo_relative(result.get('visualization_path', ''))}`",
                "",
            ]
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a CPU-safe forecast revision smoke demo with sample tasks and visualizations.")
    parser.add_argument("--output-dir", default="tmp/forecast_revision_smoke_demo")
    parser.add_argument("--csv-path", default="data/Weather.csv")
    parser.add_argument("--dataset-name", default="WeatherSmoke")
    parser.add_argument("--baseline-name", default="patchtst")
    parser.add_argument("--baseline-model-dir", default=None)
    parser.add_argument("--seq-len", type=int, default=96)
    parser.add_argument("--pred-len", type=int, default=24)
    parser.add_argument("--num-samples", type=int, default=3)
    parser.add_argument("--target-col", default="0")
    parser.add_argument("--context-style", choices=["generic", "traffic_incident"], default="generic")
    parser.add_argument("--include-no-revision-every", type=int, default=3)
    parser.add_argument("--mode", default="localized_full_revision", choices=["base_only", "global_revision_only", "localized_full_revision"])
    parser.add_argument("--calibration-strategy", default="rule_local_stats", choices=["text_direct_numeric", "discrete_strength_table", "rule_local_stats"])
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.baseline_name == "patchtst" and not args.baseline_model_dir:
        raise ValueError(
            "baseline_name='patchtst' requires --baseline-model-dir pointing to a trained PatchTST checkpoint."
        )

    benchmark_summary = build_benchmark(
        csv_path=args.csv_path,
        dataset_name=args.dataset_name,
        output_dir=str(output_dir / "benchmark"),
        baseline_name=args.baseline_name,
        baseline_model_dir=args.baseline_model_dir,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        num_samples=args.num_samples,
        target_col=args.target_col,
        include_no_revision_every=args.include_no_revision_every,
        context_style=args.context_style,
    )
    benchmark_path = benchmark_summary["output_path"]
    result_path = output_dir / "revision_results.json"
    vis_dir = output_dir / "visualizations"

    run_summary = run_revision(
        benchmark_path=benchmark_path,
        output_path=str(result_path),
        mode=args.mode,
        max_samples=args.num_samples,
        vis_dir=str(vis_dir),
        save_visualizations=True,
        calibration_strategy=args.calibration_strategy,
        calibration_model_path=None,
    )

    readme_path = output_dir / "README.md"
    readme_path.write_text(
        _build_demo_readme(
            output_dir=output_dir,
            benchmark_path=benchmark_path,
            result_path=str(result_path),
            vis_dir=vis_dir,
            run_summary=run_summary,
        ),
        encoding="utf-8",
    )

    payload = {
        "output_dir": str(output_dir.resolve()),
        "benchmark_path": str(Path(benchmark_path).resolve()),
        "result_path": str(result_path.resolve()),
        "visualization_dir": str(vis_dir.resolve()),
        "readme_path": str(readme_path.resolve()),
        "mode": args.mode,
        "calibration_strategy": args.calibration_strategy,
        "num_samples": args.num_samples,
        "summary": run_summary.get("summary", {}),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
