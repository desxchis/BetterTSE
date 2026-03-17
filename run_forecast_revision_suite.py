from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from run_forecast_revision import run_revision


DEFAULT_MODES = [
    "base_only",
    "global_revision_only",
    "localized_full_revision",
    "oracle_region",
    "oracle_intent",
    "oracle_calibration",
]


def run_suite(
    benchmark_path: str,
    output_dir: str,
    modes: list[str] | None = None,
    max_samples: int | None = None,
    save_visualizations: bool = True,
) -> dict:
    selected_modes = modes or list(DEFAULT_MODES)
    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    suite = {
        "benchmark_path": benchmark_path,
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "modes": {},
    }

    for mode in selected_modes:
        mode_output = out_root / f"pipeline_results_{mode}.json"
        mode_vis_dir = out_root / "visualizations" / mode
        payload = run_revision(
            benchmark_path=benchmark_path,
            output_path=str(mode_output),
            mode=mode,
            max_samples=max_samples,
            vis_dir=str(mode_vis_dir),
            save_visualizations=save_visualizations and mode == "localized_full_revision",
        )
        suite["modes"][mode] = {
            "output_path": str(mode_output),
            "summary": payload.get("summary", {}),
            "visualization_dir": payload.get("visualization_dir"),
        }

    summary_path = out_root / "suite_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(suite, f, ensure_ascii=False, indent=2)
    suite["summary_path"] = str(summary_path)
    return suite


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a standard forecast revision experiment suite.")
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--modes", nargs="*", default=None)
    parser.add_argument("--no-save-vis", action="store_true")
    args = parser.parse_args()

    result = run_suite(
        benchmark_path=args.benchmark,
        output_dir=args.output_dir,
        modes=args.modes,
        max_samples=args.max_samples,
        save_visualizations=not args.no_save_vis,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
