from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from run_forecast_revision_suite import run_suite


@dataclass(frozen=True)
class CheckpointSpec:
    key: str
    label: str
    reference_summary_path: Path


DEFAULT_CHECKPOINTS: List[CheckpointSpec] = [
    CheckpointSpec(
        key="weather_dlinear_v4_cal_fast",
        label="Weather v4 controlled checkpoint (`dlinear_like`)",
        reference_summary_path=_ROOT / "results/forecast_revision/runs/weather_dlinear_v4_cal_fast/suite_summary.json",
    ),
    CheckpointSpec(
        key="xtraffic_dlinear_v2_real",
        label="XTraffic v2 narrowed real checkpoint (`dlinear_like`)",
        reference_summary_path=_ROOT / "results/forecast_revision/runs/xtraffic_p01_speed_dlinear_v2_real_suite/suite_summary.json",
    ),
    CheckpointSpec(
        key="xtraffic_dlinear_v2_nonapp",
        label="XTraffic v2 non-app gate checkpoint (`dlinear_like`)",
        reference_summary_path=_ROOT / "results/forecast_revision/runs/xtraffic_p01_speed_dlinear_v2_nonapp_suite/suite_summary.json",
    ),
    CheckpointSpec(
        key="mtbench_dlinear_v2_100",
        label="MTBench finance v2 100-sample checkpoint (`dlinear_like`)",
        reference_summary_path=_ROOT / "results/forecast_revision/runs/mtbench_finance_dlinear_v2_100_suite/suite_summary.json",
    ),
]


def _load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _compare_summary_dicts(
    reference: Dict[str, Any],
    rerun: Dict[str, Any],
    tolerance: float,
) -> Dict[str, Any]:
    all_keys = sorted(set(reference.keys()) | set(rerun.keys()))
    mismatches: List[Dict[str, Any]] = []
    max_abs_diff = 0.0

    for key in all_keys:
        if key not in reference or key not in rerun:
            mismatches.append(
                {
                    "key": key,
                    "reference_present": key in reference,
                    "rerun_present": key in rerun,
                }
            )
            continue

        ref_value = reference[key]
        rerun_value = rerun[key]

        if isinstance(ref_value, (int, float)) and isinstance(rerun_value, (int, float)) and not isinstance(ref_value, bool) and not isinstance(rerun_value, bool):
            diff = abs(float(ref_value) - float(rerun_value))
            max_abs_diff = max(max_abs_diff, diff)
            if diff > tolerance:
                mismatches.append(
                    {
                        "key": key,
                        "reference": ref_value,
                        "rerun": rerun_value,
                        "abs_diff": diff,
                    }
                )
            continue

        if ref_value != rerun_value:
            mismatches.append(
                {
                    "key": key,
                    "reference": ref_value,
                    "rerun": rerun_value,
                }
            )

    return {
        "exact_within_tolerance": len(mismatches) == 0,
        "mismatch_count": len(mismatches),
        "max_abs_diff": max_abs_diff,
        "mismatches": mismatches,
    }


def run_repro_check(
    output_dir: str,
    tolerance: float = 1e-12,
    checkpoint_keys: List[str] | None = None,
) -> Dict[str, Any]:
    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    selected_specs = [
        spec for spec in DEFAULT_CHECKPOINTS if checkpoint_keys is None or spec.key in checkpoint_keys
    ]
    if not selected_specs:
        raise ValueError("No checkpoints selected for reproducibility check.")

    report: Dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "tolerance": tolerance,
        "checkpoints": [],
    }

    for spec in selected_specs:
        reference_payload = _load_json(spec.reference_summary_path)
        benchmark_path = reference_payload["benchmark_path"]
        reference_modes = sorted(reference_payload.get("modes", {}).keys())

        rerun_root = out_root / spec.key
        rerun_result = run_suite(
            benchmark_path=benchmark_path,
            output_dir=str(rerun_root),
            modes=reference_modes,
            save_visualizations=False,
        )
        rerun_payload = _load_json(Path(rerun_result["summary_path"]))

        mode_reports: Dict[str, Any] = {}
        exact_match = True
        overall_max_abs_diff = 0.0

        for mode in reference_modes:
            reference_summary = reference_payload["modes"][mode]["summary"]
            rerun_summary = rerun_payload["modes"][mode]["summary"]
            comparison = _compare_summary_dicts(reference_summary, rerun_summary, tolerance=tolerance)
            mode_reports[mode] = comparison
            exact_match = exact_match and bool(comparison["exact_within_tolerance"])
            overall_max_abs_diff = max(overall_max_abs_diff, float(comparison["max_abs_diff"]))

        report["checkpoints"].append(
            {
                "key": spec.key,
                "label": spec.label,
                "reference_summary_path": str(spec.reference_summary_path),
                "benchmark_path": benchmark_path,
                "rerun_summary_path": str(rerun_result["summary_path"]),
                "exact_match": exact_match,
                "max_abs_diff": overall_max_abs_diff,
                "mode_reports": mode_reports,
            }
        )

    report["all_exact_match"] = all(item["exact_match"] for item in report["checkpoints"])

    json_path = out_root / "repro_check_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    md_lines = [
        "# Forecast Revision Repro Check",
        "",
        f"- created_at_utc: `{report['created_at_utc']}`",
        f"- tolerance: `{tolerance}`",
        f"- all_exact_match: `{report['all_exact_match']}`",
        "",
    ]
    for item in report["checkpoints"]:
        md_lines.extend(
            [
                f"## {item['label']}",
                "",
                f"- key: `{item['key']}`",
                f"- exact_match: `{item['exact_match']}`",
                f"- max_abs_diff: `{item['max_abs_diff']}`",
                f"- benchmark: `{item['benchmark_path']}`",
                f"- reference: `{item['reference_summary_path']}`",
                f"- rerun: `{item['rerun_summary_path']}`",
                "",
            ]
        )
        for mode, mode_report in item["mode_reports"].items():
            md_lines.append(
                f"- mode `{mode}`: exact=`{mode_report['exact_within_tolerance']}`, mismatches=`{mode_report['mismatch_count']}`, max_abs_diff=`{mode_report['max_abs_diff']}`"
            )
        md_lines.append("")

    md_path = out_root / "repro_check_report.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines).strip() + "\n")

    report["json_path"] = str(json_path)
    report["markdown_path"] = str(md_path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run small reproducibility reruns on current forecast-revision checkpoints.")
    parser.add_argument(
        "--output-dir",
        default="results/forecast_revision/repro_checks/latest",
    )
    parser.add_argument("--tolerance", type=float, default=1e-12)
    parser.add_argument("--checkpoints", nargs="*", default=None)
    args = parser.parse_args()

    report = run_repro_check(
        output_dir=args.output_dir,
        tolerance=args.tolerance,
        checkpoint_keys=args.checkpoints,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
