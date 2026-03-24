from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from modules.pure_editing_volatility import (
    classify_volatility_pattern,
    classify_volatility_subpattern,
    compute_volatility_audit_metrics,
    heuristic_volatility_operator,
    search_best_volatility_operator,
)


OPERATORS = ("global_subwindow", "burst_local", "envelope_noise", "piecewise_envelope_noise")


def _mean(rows: List[Dict[str, Any]], key: str) -> float:
    return float(np.mean([float(row[key]) for row in rows])) if rows else 0.0


def _summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {"total": len(rows)}
    heuristic_rows = rows
    summary["heuristic_better_rate_anchor"] = float(np.mean([1.0 for _ in heuristic_rows])) if heuristic_rows else 0.0
    for operator_name in OPERATORS:
        op_rows = [row for row in rows if row["operator_name"] == operator_name]
        summary[operator_name] = {
            "count": len(op_rows),
            "teacher_better_rate": float(np.mean([1.0 if row["teacher_better"] else 0.0 for row in op_rows])) if op_rows else 0.0,
            "avg_teacher_mae_vs_target": _mean(op_rows, "teacher_mae_vs_target"),
            "avg_heuristic_mae_vs_target": _mean(op_rows, "heuristic_mae_vs_target"),
            "avg_teacher_local_std_error": _mean(op_rows, "teacher_local_std_error"),
            "avg_heuristic_local_std_error": _mean(op_rows, "heuristic_local_std_error"),
            "avg_teacher_roughness_error": _mean(op_rows, "teacher_roughness_error"),
            "avg_heuristic_roughness_error": _mean(op_rows, "heuristic_roughness_error"),
            "avg_teacher_windowed_energy_profile_error": _mean(op_rows, "teacher_windowed_energy_profile_error"),
            "avg_heuristic_windowed_energy_profile_error": _mean(op_rows, "heuristic_windowed_energy_profile_error"),
            "avg_teacher_preservation_mae": _mean(op_rows, "teacher_preservation_mae"),
            "avg_heuristic_preservation_mae": _mean(op_rows, "heuristic_preservation_mae"),
        }
    return summary


def _bucket_summary(rows: List[Dict[str, Any]], field: str) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = {}
    for operator_name in OPERATORS:
        operator_rows = [row for row in rows if row["operator_name"] == operator_name]
        values = sorted({str(row.get(field, "none")) for row in operator_rows})
        out[operator_name] = []
        for value in values:
            bucket_rows = [row for row in operator_rows if str(row.get(field, "none")) == value]
            out[operator_name].append(
                {
                    field: value,
                    "count": len(bucket_rows),
                    "teacher_better_rate": float(np.mean([1.0 if row["teacher_better"] else 0.0 for row in bucket_rows])) if bucket_rows else 0.0,
                    "teacher_mae_vs_target": _mean(bucket_rows, "teacher_mae_vs_target"),
                    "heuristic_mae_vs_target": _mean(bucket_rows, "heuristic_mae_vs_target"),
                    "teacher_local_std_error": _mean(bucket_rows, "teacher_local_std_error"),
                    "heuristic_local_std_error": _mean(bucket_rows, "heuristic_local_std_error"),
                    "teacher_windowed_energy_profile_error": _mean(bucket_rows, "teacher_windowed_energy_profile_error"),
                    "heuristic_windowed_energy_profile_error": _mean(bucket_rows, "heuristic_windowed_energy_profile_error"),
                }
            )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit pure-editing volatility operators on a volatility-only subset.")
    parser.add_argument("--testset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-samples", type=int, default=24)
    args = parser.parse_args()

    payload = json.loads(Path(args.testset).read_text(encoding="utf-8"))
    samples = [sample for sample in payload.get("samples", []) if sample.get("injection_operator") == "noise_injection"][: args.max_samples]

    rows: List[Dict[str, Any]] = []
    for sample in samples:
        base_ts = np.asarray(sample["base_ts"], dtype=np.float32)
        target_ts = np.asarray(sample["target_ts"], dtype=np.float32)
        region = [int(sample["gt_start"]), int(sample["gt_end"]) + 1]
        target_region = target_ts[region[0] : region[1]]
        base_region = base_ts[region[0] : region[1]]
        volatility_pattern = classify_volatility_pattern(target_region, base_region)
        volatility_subpattern = classify_volatility_subpattern(target_region, base_region)
        heuristic_ts = heuristic_volatility_operator(base_ts, region)
        heuristic_metrics = compute_volatility_audit_metrics(
            base_ts=base_ts,
            target_ts=target_ts,
            edited_ts=heuristic_ts,
            region=region,
        )
        for operator_name in OPERATORS:
            teacher = search_best_volatility_operator(
                operator_name=operator_name,
                base_ts=base_ts,
                target_ts=target_ts,
                region=region,
            )
            rows.append(
                {
                    "sample_id": sample["sample_id"],
                    "operator_name": operator_name,
                    "duration_bucket": (sample.get("stress_metadata") or {}).get("duration_bucket", "unknown"),
                    "strength_bucket": (sample.get("stress_metadata") or {}).get("strength_bucket", "unknown"),
                    "target_energy_type": (sample.get("stress_metadata") or {}).get("target_energy_type", "unknown"),
                    "volatility_pattern": volatility_pattern,
                    "volatility_subpattern": volatility_subpattern,
                    "teacher_params": teacher.params,
                    "teacher_search_space_size": teacher.search_space_size,
                    "teacher_mae_vs_target": teacher.metrics["mae_vs_target"],
                    "teacher_mse_vs_target": teacher.metrics["mse_vs_target"],
                    "teacher_preservation_mae": teacher.metrics["preservation_mae"],
                    "teacher_local_std_error": teacher.metrics["local_std_error"],
                    "teacher_roughness_error": teacher.metrics["roughness_error"],
                    "teacher_windowed_energy_profile_error": teacher.metrics["windowed_energy_profile_error"],
                    "heuristic_mae_vs_target": heuristic_metrics["mae_vs_target"],
                    "heuristic_mse_vs_target": heuristic_metrics["mse_vs_target"],
                    "heuristic_preservation_mae": heuristic_metrics["preservation_mae"],
                    "heuristic_local_std_error": heuristic_metrics["local_std_error"],
                    "heuristic_roughness_error": heuristic_metrics["roughness_error"],
                    "heuristic_windowed_energy_profile_error": heuristic_metrics["windowed_energy_profile_error"],
                    "teacher_better": teacher.metrics["mae_vs_target"] < heuristic_metrics["mae_vs_target"],
                }
            )

    output = {
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "testset": args.testset,
        "max_samples": args.max_samples,
        "summary": _summarize(rows),
        "bucket_summaries": {
            "duration_bucket": _bucket_summary(rows, "duration_bucket"),
            "target_energy_type": _bucket_summary(rows, "target_energy_type"),
            "volatility_pattern": _bucket_summary(rows, "volatility_pattern"),
            "volatility_subpattern": _bucket_summary(rows, "volatility_subpattern"),
        },
        "results": rows,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
