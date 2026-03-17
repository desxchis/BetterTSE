from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from modules.experiment_visualization import save_forecast_revision_visualization


@dataclass
class CaseSpec:
    key: str
    title: str
    source_label: str
    run_path: Path
    selector: Callable[[List[Dict[str, Any]]], Dict[str, Any]]


def _load_results(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload.get("results", [])


def _select_max_by(results: List[Dict[str, Any]], predicate: Callable[[Dict[str, Any]], bool], metric: str) -> Dict[str, Any]:
    filtered = [r for r in results if predicate(r)]
    if not filtered:
        raise ValueError(f"No samples matched selector for metric '{metric}'.")
    return max(filtered, key=lambda r: float(r["metrics"].get(metric, float("-inf"))))


def _weather_selector(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    return _select_max_by(
        results,
        lambda r: bool(r.get("revision_applicable_gt")),
        "revision_gain",
    )


def _xtraffic_positive_selector(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    return _select_max_by(
        results,
        lambda r: bool(r.get("revision_applicable_gt")),
        "revision_gain",
    )


def _xtraffic_noop_selector(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    return _select_max_by(
        results,
        lambda r: (
            not bool(r.get("revision_applicable_gt"))
            and float(r["intent_alignment"].get("revision_needed_match", 0.0)) >= 1.0
            and float(r["metrics"].get("over_edit_rate", 1.0)) <= 0.0
        ),
        "outside_region_preservation",
    )


def _mtbench_repricing_selector(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    return _select_max_by(
        results,
        lambda r: bool(r.get("revision_applicable_gt")) and r.get("edit_intent_gt", {}).get("shape") == "step",
        "revision_gain",
    )


def _mtbench_drift_selector(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    return _select_max_by(
        results,
        lambda r: bool(r.get("revision_applicable_gt")) and r.get("edit_intent_gt", {}).get("shape") == "plateau",
        "revision_gain",
    )


def _build_case_specs() -> List[CaseSpec]:
    return [
        CaseSpec(
            key="weather_controlled",
            title="Weather Controlled Positive Case",
            source_label="Weather v4 (`dlinear_like`)",
            run_path=_ROOT / "results/forecast_revision/runs/weather_dlinear_v4_cal/pipeline_results_localized_full_revision.json",
            selector=_weather_selector,
        ),
        CaseSpec(
            key="xtraffic_positive",
            title="XTraffic Real Positive Case",
            source_label="XTraffic v2 nonapp (`dlinear_like`, applicable)",
            run_path=_ROOT / "results/forecast_revision/runs/xtraffic_p01_speed_dlinear_v2_nonapp_suite/pipeline_results_localized_full_revision.json",
            selector=_xtraffic_positive_selector,
        ),
        CaseSpec(
            key="xtraffic_noop",
            title="XTraffic Real No-Op Case",
            source_label="XTraffic v2 nonapp (`dlinear_like`, non-applicable)",
            run_path=_ROOT / "results/forecast_revision/runs/xtraffic_p01_speed_dlinear_v2_nonapp_suite/pipeline_results_localized_full_revision.json",
            selector=_xtraffic_noop_selector,
        ),
        CaseSpec(
            key="mtbench_repricing",
            title="MTBench Native-Text Repricing Case",
            source_label="MTBench finance v2 100 (`dlinear_like`, repricing)",
            run_path=_ROOT / "results/forecast_revision/runs/mtbench_finance_dlinear_v2_100_suite/pipeline_results_localized_full_revision.json",
            selector=_mtbench_repricing_selector,
        ),
        CaseSpec(
            key="mtbench_drift",
            title="MTBench Native-Text Drift-Adjust Case",
            source_label="MTBench finance v2 100 (`dlinear_like`, drift_adjust)",
            run_path=_ROOT / "results/forecast_revision/runs/mtbench_finance_dlinear_v2_100_suite/pipeline_results_localized_full_revision.json",
            selector=_mtbench_drift_selector,
        ),
    ]


def _save_case_figure(case_key: str, result: Dict[str, Any], output_dir: Path) -> Path:
    save_path = output_dir / f"{case_key}_{result['sample_id']}.png"
    save_forecast_revision_visualization(
        sample_id=result["sample_id"],
        history_ts=np.asarray(result["history_ts"], dtype=np.float64),
        future_gt=np.asarray(result["future_gt"], dtype=np.float64),
        base_forecast=np.asarray(result["base_forecast"], dtype=np.float64),
        revision_target=np.asarray(result["revision_target"], dtype=np.float64),
        edited_forecast=np.asarray(result["edited_forecast"], dtype=np.float64),
        gt_region=tuple(result["gt_region"]),
        pred_region=tuple(result["pred_region"]),
        context_text=result["context_text"],
        metrics=result["metrics"],
        save_path=save_path,
    )
    return save_path


def _format_case_md(case: CaseSpec, result: Dict[str, Any], fig_path: Path) -> str:
    metrics = result["metrics"]
    alignment = result.get("intent_alignment", {})
    gt_intent = result.get("edit_intent_gt", {})
    plan = result.get("plan", {})
    incident_meta = result.get("revision_operator_params") or gt_intent
    lines = [
        f"## {case.title}",
        "",
        f"- source: {case.source_label}",
        f"- sample_id: `{result['sample_id']}`",
        f"- figure: [{fig_path.name}]({fig_path.as_posix()})",
        f"- applicable: `{result.get('revision_applicable_gt')}`",
        f"- GT intent: `{gt_intent.get('shape', 'none')}` / `{gt_intent.get('direction', 'neutral')}` / `{gt_intent.get('strength', 'none')}`",
        f"- predicted intent: `{plan.get('intent', {}).get('shape', 'none')}` / `{plan.get('intent', {}).get('direction', 'neutral')}` / `{plan.get('intent', {}).get('strength', 'none')}`",
        f"- region: pred `{result['pred_region']}` vs gt `{result['gt_region']}`",
        f"- revision_gain: `{metrics.get('revision_gain', 0.0):.4f}`",
        f"- future_t_iou: `{metrics.get('future_t_iou', 0.0):.4f}`",
        f"- magnitude_error: `{metrics.get('magnitude_calibration_error', 0.0):.4f}`",
        f"- over_edit_rate: `{metrics.get('over_edit_rate', 0.0):.4f}`",
        f"- revision_needed_match: `{alignment.get('revision_needed_match', 0.0):.4f}`",
        "",
        "Context:",
        "",
        f"> {result['context_text'][:400]}",
        "",
    ]
    if incident_meta:
        lines.extend(
            [
                "Notes:",
                "",
                f"- source metadata: `{str(incident_meta)[:500]}`",
                "",
            ]
        )
    return "\n".join(lines)


def build_unified_case_studies(output_dir: str) -> Dict[str, Any]:
    out_root = Path(output_dir)
    vis_dir = out_root / "visualizations"
    out_root.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

    cases_output: List[Dict[str, Any]] = []
    sections: List[str] = [
        "# Unified Forecast Revision Case Studies",
        "",
        "This file consolidates one controlled case and four real-data cases across the current stable checkpoint line.",
        "",
    ]

    for case in _build_case_specs():
        results = _load_results(case.run_path)
        selected = case.selector(results)
        fig_path = _save_case_figure(case.key, selected, vis_dir)
        sections.append(_format_case_md(case, selected, fig_path))
        cases_output.append(
            {
                "key": case.key,
                "title": case.title,
                "source_label": case.source_label,
                "run_path": str(case.run_path),
                "sample_id": selected["sample_id"],
                "figure_path": str(fig_path),
                "revision_gain": selected["metrics"]["revision_gain"],
                "future_t_iou": selected["metrics"]["future_t_iou"],
                "applicable": selected["revision_applicable_gt"],
            }
        )

    md_path = out_root / "CASE_STUDIES_20260317.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(sections).strip() + "\n")

    summary_path = out_root / "case_studies_index.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({"cases": cases_output, "markdown_path": str(md_path)}, f, ensure_ascii=False, indent=2)

    return {
        "markdown_path": str(md_path),
        "summary_path": str(summary_path),
        "num_cases": len(cases_output),
        "visualization_dir": str(vis_dir),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build unified forecast revision case studies across Weather, XTraffic, and MTBench.")
    parser.add_argument(
        "--output-dir",
        default="results/forecast_revision/case_studies/20260317_unified",
    )
    args = parser.parse_args()

    result = build_unified_case_studies(args.output_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
