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


DEFAULT_MODES = [
    "localized_full_revision",
    "oracle_region",
    "oracle_tool",
    "oracle_intent",
    "oracle_calibration",
]


def _row(mode: str, payload: dict) -> dict:
    s = payload.get("summary", {})
    return {
        "mode": mode,
        "avg_revision_gain": s.get("avg_revision_gain"),
        "avg_edited_mae_vs_revision_target": s.get("avg_edited_mae_vs_revision_target"),
        "avg_normalized_parameter_error": s.get("avg_normalized_parameter_error"),
        "avg_signed_area_error": s.get("avg_signed_area_error"),
        "avg_duration_error": s.get("avg_duration_error"),
        "avg_recovery_slope_error": s.get("avg_recovery_slope_error"),
        "avg_future_t_iou": s.get("avg_future_t_iou"),
    }


def _render_markdown(rows: list[dict]) -> str:
    lines = [
        "# Forecast Revision Semi-Oracle Suite",
        "",
        "| Mode | Revision Gain | Edited MAE | NPE | Signed Area | Duration | Recovery Slope | t-IoU |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {mode} | {avg_revision_gain:.4f} | {avg_edited_mae_vs_revision_target:.4f} | {avg_normalized_parameter_error:.4f} | {avg_signed_area_error:.4f} | {avg_duration_error:.4f} | {avg_recovery_slope_error:.4f} | {avg_future_t_iou:.4f} |".format(
                mode=row["mode"],
                avg_revision_gain=row["avg_revision_gain"] or 0.0,
                avg_edited_mae_vs_revision_target=row["avg_edited_mae_vs_revision_target"] or 0.0,
                avg_normalized_parameter_error=row["avg_normalized_parameter_error"] or 0.0,
                avg_signed_area_error=row["avg_signed_area_error"] or 0.0,
                avg_duration_error=row["avg_duration_error"] or 0.0,
                avg_recovery_slope_error=row["avg_recovery_slope_error"] or 0.0,
                avg_future_t_iou=row["avg_future_t_iou"] or 0.0,
            )
        )
    return "\n".join(lines) + "\n"


def run_suite(benchmark_path: str, output_dir: str, calibration_strategy: str = "rule_local_stats", max_samples: int | None = None, calibration_model: str | None = None) -> dict:
    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    rows = []
    modes = {}
    for mode in DEFAULT_MODES:
        output_path = out_root / f"{mode}.json"
        payload = run_revision(
            benchmark_path=benchmark_path,
            output_path=str(output_path),
            mode=mode,
            max_samples=max_samples,
            save_visualizations=False,
            calibration_strategy=calibration_strategy,
            calibration_model_path=calibration_model,
        )
        modes[mode] = {
            "output_path": str(output_path),
            "summary": payload.get("summary", {}),
        }
        rows.append(_row(mode, payload))
    summary = {
        "benchmark_path": benchmark_path,
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "calibration_strategy": calibration_strategy,
        "modes": modes,
        "table_rows": rows,
    }
    summary_path = out_root / "semi_oracle_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_path = out_root / "semi_oracle_summary.md"
    markdown_path.write_text(_render_markdown(rows), encoding="utf-8")
    summary["summary_path"] = str(summary_path)
    summary["markdown_path"] = str(markdown_path)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run semi-oracle degradation suite for forecast revision.")
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--calibration-strategy",
        default="rule_local_stats",
        choices=["text_direct_numeric", "discrete_strength_table", "rule_local_stats", "learned_linear"],
    )
    parser.add_argument("--calibration-model", default=None)
    args = parser.parse_args()
    result = run_suite(
        benchmark_path=args.benchmark,
        output_dir=args.output_dir,
        calibration_strategy=args.calibration_strategy,
        max_samples=args.max_samples,
        calibration_model=args.calibration_model,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
