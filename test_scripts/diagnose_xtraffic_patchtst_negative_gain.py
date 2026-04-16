from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from run_forecast_revision import run_revision


def _gt_region_from_mask(mask: np.ndarray) -> List[int]:
    idx = np.where(mask > 0.5)[0]
    if idx.size == 0:
        return [0, 0]
    return [int(idx[0]), int(idx[-1] + 1)]


def _mean(values: List[float]) -> float | None:
    if not values:
        return None
    return float(np.mean(values))


def _direction_from_delta(values: np.ndarray, eps: float = 1e-6) -> str:
    v = float(np.mean(values)) if values.size else 0.0
    if v > eps:
        return "up"
    if v < -eps:
        return "down"
    return "neutral"


def _safe_overlap_iou(a: Tuple[int, int], b: Tuple[int, int]) -> float | None:
    a0, a1 = a
    b0, b1 = b
    if a1 <= a0 or b1 <= b0:
        return None
    inter = max(0, min(a1, b1) - max(a0, b0))
    union = max(a1, b1) - min(a0, b0)
    if union <= 0:
        return None
    return float(inter / union)


def _json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2)


def summarize_benchmark_samples(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(samples)
    applicable = [s for s in samples if bool(s.get("revision_applicable_gt", False))]
    non_applicable = [s for s in samples if not bool(s.get("revision_applicable_gt", False))]

    effect_counter = Counter()
    shape_counter = Counter()
    duration_counter = Counter()
    region_lengths: List[int] = []
    direction_checks: List[float] = []

    for s in applicable:
        intent = s.get("edit_intent_gt", {}) or {}
        effect_counter[str(intent.get("effect_family", s.get("effect_family_gt", "unknown")))] += 1
        shape_counter[str(intent.get("shape", s.get("shape_gt", "unknown")))] += 1
        duration_counter[str(intent.get("duration", s.get("duration_bucket_gt", "unknown")))] += 1

        mask = np.asarray(s.get("edit_mask_gt", []), dtype=np.float64)
        region = _gt_region_from_mask(mask)
        region_len = int(max(0, region[1] - region[0]))
        region_lengths.append(region_len)
        if region_len > 0:
            base = np.asarray(s.get("base_forecast", []), dtype=np.float64)
            target = np.asarray(s.get("revision_target", []), dtype=np.float64)
            delta = target[region[0] : region[1]] - base[region[0] : region[1]]
            inferred = _direction_from_delta(delta)
            gt_direction = str(intent.get("direction", s.get("direction_gt", "neutral")))
            direction_checks.append(1.0 if inferred == gt_direction else 0.0)

    return {
        "total_samples": total,
        "applicable_count": len(applicable),
        "non_applicable_count": len(non_applicable),
        "applicable_ratio": float(len(applicable) / total) if total else 0.0,
        "effect_family_distribution": dict(effect_counter),
        "shape_distribution": dict(shape_counter),
        "duration_distribution": dict(duration_counter),
        "region_length": {
            "mean": _mean(region_lengths),
            "min": int(min(region_lengths)) if region_lengths else None,
            "max": int(max(region_lengths)) if region_lengths else None,
        },
        "direction_consistency_rate": _mean(direction_checks),
    }


def summarize_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_bucket: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    duration_bucket: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    worst_rows: List[Dict[str, Any]] = []
    executor_checks = {
        "editor_region_within_bounds_rate": [],
        "editor_pred_iou": [],
    }

    for r in results:
        metrics = r.get("metrics", {})
        intent = r.get("edit_intent_gt", {}) or {}
        shape = str(intent.get("shape", r.get("shape_gt", "unknown")))
        duration = str(intent.get("duration", r.get("duration_bucket_gt", "unknown")))
        for key in (
            "revision_gain",
            "base_mae_vs_revision_target",
            "edited_mae_vs_revision_target",
            "base_mae_vs_future_gt",
            "edited_mae_vs_future_gt",
            "outside_region_preservation",
            "over_edit_rate",
        ):
            if key in metrics:
                by_bucket[shape][key].append(float(metrics[key]))
                duration_bucket[duration][key].append(float(metrics[key]))

        exec_meta = r.get("execution_metadata", {}) or {}
        editor_region = exec_meta.get("editor_region")
        pred_region = r.get("pred_region")
        window_len = exec_meta.get("editor_window_len_resampled")
        future_offset = exec_meta.get("future_offset_resampled")
        if isinstance(editor_region, list) and len(editor_region) == 2 and isinstance(window_len, int):
            within = 1.0 if (0 <= int(editor_region[0]) <= int(editor_region[1]) <= int(window_len)) else 0.0
            executor_checks["editor_region_within_bounds_rate"].append(within)
            if isinstance(pred_region, list) and len(pred_region) == 2 and isinstance(future_offset, int):
                local_editor = (
                    int(editor_region[0]) - int(future_offset),
                    int(editor_region[1]) - int(future_offset),
                )
                local_pred = (int(pred_region[0]), int(pred_region[1]))
                iou = _safe_overlap_iou(local_editor, local_pred)
                if iou is not None:
                    executor_checks["editor_pred_iou"].append(iou)

        worst_rows.append(
            {
                "sample_id": r.get("sample_id"),
                "revision_gain": float(metrics.get("revision_gain", 0.0)),
                "over_edit_rate": float(metrics.get("over_edit_rate", 0.0)),
                "outside_region_preservation": float(metrics.get("outside_region_preservation", 0.0)),
                "tool_name": exec_meta.get("tool_name"),
                "pred_region": r.get("pred_region"),
                "gt_region": r.get("gt_region"),
                "editor_region": exec_meta.get("editor_region"),
                "future_offset_resampled": exec_meta.get("future_offset_resampled"),
                "intent_match_score": float(r.get("intent_alignment", {}).get("intent_match_score", 0.0)),
            }
        )

    def _collapse(buckets: Dict[str, Dict[str, List[float]]]) -> Dict[str, Dict[str, float | None]]:
        out: Dict[str, Dict[str, float | None]] = {}
        for k, vals in buckets.items():
            out[k] = {metric: _mean(v) for metric, v in vals.items()}
        return out

    worst_rows.sort(key=lambda x: x["revision_gain"])
    return {
        "by_shape": _collapse(by_bucket),
        "by_duration": _collapse(duration_bucket),
        "executor_checks": {
            "editor_region_within_bounds_rate": _mean(executor_checks["editor_region_within_bounds_rate"]),
            "avg_editor_pred_iou": _mean(executor_checks["editor_pred_iou"]),
        },
        "worst_samples_topk": worst_rows[:10],
    }


def compare_ab(tedit_payload: Dict[str, Any], profile_payload: Dict[str, Any]) -> Dict[str, Any]:
    ts = tedit_payload.get("summary", {})
    ps = profile_payload.get("summary", {})
    keys = [
        "avg_revision_gain",
        "avg_base_mae_vs_revision_target",
        "avg_edited_mae_vs_revision_target",
        "avg_base_mae_vs_future_gt",
        "avg_edited_mae_vs_future_gt",
        "avg_outside_region_preservation",
        "avg_over_edit_rate",
    ]
    diff = {}
    for k in keys:
        tv = ts.get(k)
        pv = ps.get(k)
        if tv is not None and pv is not None:
            diff[k] = float(tv - pv)
        else:
            diff[k] = None
    return {"tedit_summary": ts, "profile_summary": ps, "tedit_minus_profile": diff}


def render_markdown(report: Dict[str, Any]) -> str:
    lines = [
        "# XTraffic PatchTST Negative-Gain Diagnosis",
        "",
        "## Dataset Health",
        "",
        "```json",
        _json_dumps(report["sample_health"]),
        "```",
        "",
        "## A/B Summary (`tedit_hybrid` vs `profile`)",
        "",
        "```json",
        _json_dumps(report["ab_compare"]),
        "```",
        "",
        "## TEdit Result Breakdown",
        "",
        "```json",
        _json_dumps(report["tedit_breakdown"]),
        "```",
        "",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose XTraffic negative revision gain under patchtst baseline.")
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--calibration-strategy", default="rule_local_stats")
    parser.add_argument("--tedit-model", required=True)
    parser.add_argument("--tedit-config", required=True)
    parser.add_argument("--tedit-device", default="cuda:0")
    args = parser.parse_args()

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    with open(args.benchmark, "r", encoding="utf-8") as f:
        benchmark = json.load(f)
    samples = benchmark.get("samples", [])
    if args.max_samples is not None:
        samples = samples[: args.max_samples]
        benchmark = dict(benchmark)
        benchmark["samples"] = samples
        bench_path = out_root / "benchmark_slice.json"
        with open(bench_path, "w", encoding="utf-8") as f:
            json.dump(benchmark, f, ensure_ascii=False, indent=2)
        benchmark_path = str(bench_path)
    else:
        benchmark_path = args.benchmark

    sample_health = summarize_benchmark_samples(samples)

    tedit_output = out_root / "localized_tedit_hybrid.json"
    tedit_payload = run_revision(
        benchmark_path=benchmark_path,
        output_path=str(tedit_output),
        mode="localized_full_revision",
        max_samples=args.max_samples,
        save_visualizations=False,
        calibration_strategy=args.calibration_strategy,
        revision_executor="tedit_hybrid",
        tedit_model_path=args.tedit_model,
        tedit_config_path=args.tedit_config,
        tedit_device=args.tedit_device,
    )

    profile_output = out_root / "localized_profile.json"
    profile_payload = run_revision(
        benchmark_path=benchmark_path,
        output_path=str(profile_output),
        mode="localized_full_revision",
        max_samples=args.max_samples,
        save_visualizations=False,
        calibration_strategy=args.calibration_strategy,
        revision_executor="profile",
    )

    report = {
        "benchmark_path": benchmark_path,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "sample_health": sample_health,
        "ab_compare": compare_ab(tedit_payload, profile_payload),
        "tedit_breakdown": summarize_results(tedit_payload.get("results", [])),
    }

    report_json = out_root / "diagnosis_report.json"
    report_md = out_root / "diagnosis_report.md"
    report_json.write_text(_json_dumps(report), encoding="utf-8")
    report_md.write_text(render_markdown(report), encoding="utf-8")
    print(
        _json_dumps(
            {
                "diagnosis_report_json": str(report_json),
                "diagnosis_report_md": str(report_md),
                "tedit_output": str(tedit_output),
                "profile_output": str(profile_output),
            }
        )
    )


if __name__ == "__main__":
    main()
