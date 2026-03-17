from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from run_forecast_revision import run_revision


DEFAULT_METHODS = [
    {"name": "text_direct_numeric", "mode": "oracle_intent", "calibration_strategy": "text_direct_numeric"},
    {"name": "discrete_strength_table", "mode": "oracle_intent", "calibration_strategy": "discrete_strength_table"},
    {"name": "rule_local_stats", "mode": "oracle_intent", "calibration_strategy": "rule_local_stats"},
    {"name": "learned_linear", "mode": "oracle_intent", "calibration_strategy": "learned_linear"},
    {"name": "oracle_calibration", "mode": "oracle_calibration", "calibration_strategy": "rule_local_stats"},
]


def _method_row(method_name: str, payload: dict) -> dict:
    summary = payload.get("summary", {})
    return {
        "method": method_name,
        "avg_normalized_parameter_error": summary.get("avg_normalized_parameter_error"),
        "avg_peak_delta_error": summary.get("avg_peak_delta_error"),
        "avg_signed_area_error": summary.get("avg_signed_area_error"),
        "avg_duration_error": summary.get("avg_duration_error"),
        "avg_recovery_slope_error": summary.get("avg_recovery_slope_error"),
        "avg_revision_gain": summary.get("avg_revision_gain"),
        "avg_edited_mae_vs_revision_target": summary.get("avg_edited_mae_vs_revision_target"),
        "avg_outside_region_preservation": summary.get("avg_outside_region_preservation"),
    }


def _render_markdown(rows: list[dict]) -> str:
    lines = [
        "# Forecast Revision Calibration Benchmark",
        "",
        "| Method | NPE | Peak Delta | Signed Area | Duration | Recovery Slope | Revision Gain | Edited MAE | Outside Preservation |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {method} | {avg_normalized_parameter_error:.4f} | {avg_peak_delta_error:.4f} | {avg_signed_area_error:.4f} | {avg_duration_error:.4f} | {avg_recovery_slope_error:.4f} | {avg_revision_gain:.4f} | {avg_edited_mae_vs_revision_target:.4f} | {avg_outside_region_preservation:.4f} |".format(
                method=row["method"],
                avg_normalized_parameter_error=row["avg_normalized_parameter_error"] or 0.0,
                avg_peak_delta_error=row["avg_peak_delta_error"] or 0.0,
                avg_signed_area_error=row["avg_signed_area_error"] or 0.0,
                avg_duration_error=row["avg_duration_error"] or 0.0,
                avg_recovery_slope_error=row["avg_recovery_slope_error"] or 0.0,
                avg_revision_gain=row["avg_revision_gain"] or 0.0,
                avg_edited_mae_vs_revision_target=row["avg_edited_mae_vs_revision_target"] or 0.0,
                avg_outside_region_preservation=row["avg_outside_region_preservation"] or 0.0,
            )
        )
    return "\n".join(lines) + "\n"


def run_calibration_benchmark(benchmark_path: str, output_dir: str, max_samples: int | None = None, methods: list[str] | None = None, calibration_model: str | None = None) -> dict:
    selected = []
    requested = set(methods or [])
    for spec in DEFAULT_METHODS:
        if not requested or spec["name"] in requested:
            selected.append(spec)
    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    rows = []
    method_outputs = {}
    for spec in selected:
        output_path = out_root / f"calibration_{spec['name']}.json"
        payload = run_revision(
            benchmark_path=benchmark_path,
            output_path=str(output_path),
            mode=spec["mode"],
            max_samples=max_samples,
            save_visualizations=False,
            calibration_strategy=spec["calibration_strategy"],
            calibration_model_path=calibration_model,
        )
        rows.append(_method_row(spec["name"], payload))
        method_outputs[spec["name"]] = {
            "mode": spec["mode"],
            "calibration_strategy": spec["calibration_strategy"],
            "output_path": str(output_path),
            "summary": payload.get("summary", {}),
        }

    summary = {
        "benchmark_path": benchmark_path,
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "methods": method_outputs,
        "table_rows": rows,
    }
    summary_path = out_root / "calibration_benchmark_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    markdown_path = out_root / "calibration_benchmark_summary.md"
    markdown_path.write_text(_render_markdown(rows), encoding='utf-8')
    summary["summary_path"] = str(summary_path)
    summary["markdown_path"] = str(markdown_path)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run oracle-region/oracle-intent calibration benchmark for forecast revision.")
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--methods",
        nargs="*",
        choices=[spec["name"] for spec in DEFAULT_METHODS],
        default=None,
    )
    parser.add_argument("--calibration-model", default=None)
    args = parser.parse_args()

    result = run_calibration_benchmark(
        benchmark_path=args.benchmark,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        methods=args.methods,
        calibration_model=args.calibration_model,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
