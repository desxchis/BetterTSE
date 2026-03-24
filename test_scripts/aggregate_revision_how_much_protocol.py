from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

METHOD_FILES = {
    "teacher_search_oracle": "teacher_search_oracle.json",
    "teacher_distilled_shrunk": "teacher_distilled_shrunk.json",
    "heuristic_revision": "heuristic_revision.json",
    "rule_local_stats": "rule_local_stats.json",
    "direct_delta_regression": "direct_delta_regression.json",
}

LOWER_IS_BETTER_METRICS = (
    "avg_edited_mae_vs_revision_target",
    "avg_edited_mae_vs_future_gt",
    "avg_magnitude_calibration_error",
    "avg_signed_area_error",
    "avg_duration_error",
    "avg_recovery_slope_error",
)

HIGHER_IS_BETTER_METRICS = (
    "avg_revision_gain",
    "avg_future_t_iou",
)

PER_SAMPLE_METRICS = (
    "edited_mae_vs_revision_target",
    "edited_mae_vs_future_gt",
    "revision_gain",
    "future_t_iou",
    "magnitude_calibration_error",
    "signed_area_error",
    "duration_error",
    "recovery_slope_error",
    "normalized_parameter_error",
    "peak_delta_error",
)

BUCKET_FIELDS = (
    ("effect_family", "effect_family"),
    ("shape", "shape"),
    ("duration_bucket", "duration_bucket"),
    ("strength_bucket", "strength_bucket"),
)


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(result) or math.isinf(result):
        return None
    return result


def _mean_std(values: Iterable[float | None]) -> Dict[str, float | None]:
    cleaned = [float(v) for v in values if v is not None]
    if not cleaned:
        return {"mean": None, "std": None, "count": 0}
    mean = sum(cleaned) / len(cleaned)
    if len(cleaned) == 1:
        std = 0.0
    else:
        var = sum((v - mean) ** 2 for v in cleaned) / (len(cleaned) - 1)
        std = math.sqrt(max(var, 0.0))
    return {"mean": mean, "std": std, "count": len(cleaned)}


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _gt_labels(result: Dict[str, Any]) -> Dict[str, str]:
    record = result.get("experiment_record") or {}
    ground_truth = record.get("ground_truth") or {}
    intent = ground_truth.get("intent_label") or {}
    return {
        "effect_family": str(intent.get("effect_family", "none")),
        "shape": str(intent.get("shape", "none")),
        "duration": str(intent.get("duration", "none")),
        "strength": str(intent.get("strength", "none")),
    }


def _result_rows(backbone: str, seed: int, method: str, payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for result in payload.get("results", []):
        metrics = result.get("metrics") or {}
        calibration = result.get("calibration_metrics") or {}
        labels = _gt_labels(result)
        row = {
            "backbone": backbone,
            "seed": seed,
            "method": method,
            "sample_id": result.get("sample_id"),
            "effect_family": labels["effect_family"],
            "shape": labels["shape"],
            "duration_bucket": labels["duration"],
            "strength_bucket": labels["strength"],
        }
        for key in PER_SAMPLE_METRICS:
            if key in metrics:
                row[key] = _safe_float(metrics.get(key))
            elif key in calibration:
                row[key] = _safe_float(calibration.get(key))
            else:
                row[key] = None
        rows.append(row)
    return rows


def _summary_row(backbone: str, seed: int, method: str, payload: Dict[str, Any], path: Path) -> Dict[str, Any]:
    summary = payload.get("summary") or {}
    row: Dict[str, Any] = {
        "backbone": backbone,
        "seed": seed,
        "method": method,
        "path": str(path),
        "count": int(summary.get("total", 0) or 0),
    }
    for key in LOWER_IS_BETTER_METRICS + HIGHER_IS_BETTER_METRICS:
        row[key] = _safe_float(summary.get(key))
    return row


def _aggregate_summary(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["backbone"], row["method"])].append(row)

    table: List[Dict[str, Any]] = []
    for (backbone, method), items in sorted(grouped.items()):
        entry: Dict[str, Any] = {
            "backbone": backbone,
            "method": method,
            "seed_count": len(items),
        }
        for key in LOWER_IS_BETTER_METRICS + HIGHER_IS_BETTER_METRICS:
            stats = _mean_std(item.get(key) for item in items)
            entry[f"{key}_mean"] = stats["mean"]
            entry[f"{key}_std"] = stats["std"]
        table.append(entry)
    return table


def _aggregate_buckets(rows: List[Dict[str, Any]], bucket_key: str) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        bucket_value = str(row.get(bucket_key, "none"))
        grouped[(row["backbone"], row["method"], bucket_value)].append(row)

    table: List[Dict[str, Any]] = []
    for (backbone, method, bucket_value), items in sorted(grouped.items()):
        entry: Dict[str, Any] = {
            "backbone": backbone,
            "method": method,
            bucket_key: bucket_value,
            "count": len(items),
        }
        for key in PER_SAMPLE_METRICS:
            stats = _mean_std(item.get(key) for item in items)
            entry[f"{key}_mean"] = stats["mean"]
        table.append(entry)
    return table


def _oracle_gap_table(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, int], Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        grouped[(row["backbone"], row["seed"])][row["method"]] = row

    table: List[Dict[str, Any]] = []
    for (backbone, seed), methods in sorted(grouped.items()):
        oracle = methods.get("teacher_search_oracle")
        distilled = methods.get("teacher_distilled_shrunk")
        heuristic = methods.get("heuristic_revision")
        rule = methods.get("rule_local_stats")
        if not oracle or not distilled:
            continue
        entry: Dict[str, Any] = {
            "backbone": backbone,
            "seed": seed,
        }
        for metric in LOWER_IS_BETTER_METRICS:
            oracle_val = _safe_float(oracle.get(metric))
            student_val = _safe_float(distilled.get(metric))
            heuristic_val = _safe_float(heuristic.get(metric) if heuristic else None)
            rule_val = _safe_float(rule.get(metric) if rule else None)
            entry[f"{metric}_oracle"] = oracle_val
            entry[f"{metric}_distilled"] = student_val
            entry[f"{metric}_heuristic"] = heuristic_val
            entry[f"{metric}_rule"] = rule_val
            entry[f"{metric}_distilled_minus_oracle"] = None if oracle_val is None or student_val is None else student_val - oracle_val
            if heuristic_val is None or oracle_val is None or student_val is None:
                entry[f"{metric}_gap_closed_vs_heuristic"] = None
                entry[f"{metric}_gap_ratio_vs_heuristic"] = None
            else:
                denom = heuristic_val - oracle_val
                if abs(denom) < 1e-12:
                    entry[f"{metric}_gap_closed_vs_heuristic"] = None
                    entry[f"{metric}_gap_ratio_vs_heuristic"] = None
                else:
                    gap_ratio = (student_val - oracle_val) / denom
                    entry[f"{metric}_gap_ratio_vs_heuristic"] = gap_ratio
                    entry[f"{metric}_gap_closed_vs_heuristic"] = 1.0 - gap_ratio
        for metric in HIGHER_IS_BETTER_METRICS:
            oracle_val = _safe_float(oracle.get(metric))
            student_val = _safe_float(distilled.get(metric))
            heuristic_val = _safe_float(heuristic.get(metric) if heuristic else None)
            entry[f"{metric}_oracle"] = oracle_val
            entry[f"{metric}_distilled"] = student_val
            entry[f"{metric}_heuristic"] = heuristic_val
            entry[f"{metric}_distilled_minus_oracle"] = None if oracle_val is None or student_val is None else student_val - oracle_val
            if heuristic_val is None or oracle_val is None or student_val is None:
                entry[f"{metric}_gap_closed_vs_heuristic"] = None
                entry[f"{metric}_gap_ratio_vs_heuristic"] = None
            else:
                denom = oracle_val - heuristic_val
                if abs(denom) < 1e-12:
                    entry[f"{metric}_gap_closed_vs_heuristic"] = None
                    entry[f"{metric}_gap_ratio_vs_heuristic"] = None
                else:
                    gap_ratio = (oracle_val - student_val) / denom
                    entry[f"{metric}_gap_ratio_vs_heuristic"] = gap_ratio
                    entry[f"{metric}_gap_closed_vs_heuristic"] = 1.0 - gap_ratio
        table.append(entry)
    return table


def _scan_runs(root_dir: Path) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    summary_rows: List[Dict[str, Any]] = []
    sample_rows: List[Dict[str, Any]] = []
    candidate_dirs: List[Path]
    if any(path.is_dir() and path.name.startswith("seed_") for path in root_dir.iterdir()):
        candidate_dirs = [root_dir]
    else:
        candidate_dirs = sorted(path for path in root_dir.iterdir() if path.is_dir())
    for backbone_dir in candidate_dirs:
        backbone = backbone_dir.name
        for seed_dir in sorted(path for path in backbone_dir.iterdir() if path.is_dir() and path.name.startswith("seed_")):
            try:
                seed = int(seed_dir.name.split("_", 1)[1])
            except ValueError:
                continue
            for method, filename in METHOD_FILES.items():
                path = seed_dir / filename
                if not path.exists():
                    continue
                payload = _load_json(path)
                summary_rows.append(_summary_row(backbone, seed, method, payload, path))
                sample_rows.extend(_result_rows(backbone, seed, method, payload))
    return summary_rows, sample_rows


def _format_float(value: Any) -> str:
    if value is None:
        return "NA"
    return f"{float(value):.4f}"


def _write_markdown(output_path: Path, payload: Dict[str, Any]) -> None:
    lines = [
        "# Revision How-Much Protocol Aggregate",
        "",
        f"- root_dir: {payload['root_dir']}",
        f"- total_run_files: {payload['total_run_files']}",
        f"- generated_utc: {payload['generated_utc']}",
        "",
        "## Backbone x Method",
        "",
        "| Backbone | Method | Seeds | Target MAE | Revision Gain | Future MAE | Magnitude Err | Signed Area Err | Duration Err | Recovery Err |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in payload["summary_by_backbone_method"]:
        lines.append(
            f"| {row['backbone']} | {row['method']} | {row['seed_count']} | "
            f"{_format_float(row['avg_edited_mae_vs_revision_target_mean'])} | "
            f"{_format_float(row['avg_revision_gain_mean'])} | "
            f"{_format_float(row['avg_edited_mae_vs_future_gt_mean'])} | "
            f"{_format_float(row['avg_magnitude_calibration_error_mean'])} | "
            f"{_format_float(row['avg_signed_area_error_mean'])} | "
            f"{_format_float(row['avg_duration_error_mean'])} | "
            f"{_format_float(row['avg_recovery_slope_error_mean'])} |"
        )
    lines.extend(
        [
            "",
            "## Distilled vs Oracle Gap",
            "",
            "| Backbone | Seed | Target MAE gap closed vs heuristic | Revision gain gap closed vs heuristic | Future MAE gap closed vs heuristic |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in payload["oracle_gap_rows"]:
        lines.append(
            f"| {row['backbone']} | {row['seed']} | "
            f"{_format_float(row.get('avg_edited_mae_vs_revision_target_gap_closed_vs_heuristic'))} | "
            f"{_format_float(row.get('avg_revision_gain_gap_closed_vs_heuristic'))} | "
            f"{_format_float(row.get('avg_edited_mae_vs_future_gt_gap_closed_vs_heuristic'))} |"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Full per-sample bucket summaries are written to `aggregate_results.json`.",
            "- `gap_closed_vs_heuristic = 1` means the distilled student reached oracle on that metric relative to heuristic.",
            "- Lower-is-better metrics use target/future/calibration errors; higher-is-better metrics use revision gain and future t-IoU.",
            "",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def aggregate(root_dir: str, output_dir: str) -> Dict[str, Any]:
    root = Path(root_dir)
    summary_rows, sample_rows = _scan_runs(root)
    bucket_payload = {
        name: _aggregate_buckets(sample_rows, field)
        for name, field in BUCKET_FIELDS
    }
    payload = {
        "root_dir": str(root.resolve()),
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "total_run_files": len(summary_rows),
        "summary_rows": summary_rows,
        "summary_by_backbone_method": _aggregate_summary(summary_rows),
        "oracle_gap_rows": _oracle_gap_table(summary_rows),
        "bucket_summaries": bucket_payload,
    }
    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "aggregate_results.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_markdown(out_root / "README.md", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate multi-seed revision how-much protocol results.")
    parser.add_argument("--root-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    payload = aggregate(args.root_dir, args.output_dir)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
