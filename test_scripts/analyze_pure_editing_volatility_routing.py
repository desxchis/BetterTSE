from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from modules.pure_editing_volatility import classify_volatility_subpattern


EXPECTED_TOOL_BY_SUBPATTERN = {
    "uniform_variance": "volatility_global_scale",
    "local_burst": "volatility_local_burst",
    "monotonic_envelope": "volatility_envelope_monotonic",
}


def _mean(values: List[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def analyze(testset_path: str, result_path: str) -> Dict[str, Any]:
    testset = json.loads(Path(testset_path).read_text(encoding="utf-8"))
    results = json.loads(Path(result_path).read_text(encoding="utf-8"))
    sample_map = {sample["sample_id"]: sample for sample in testset["samples"]}

    volatility_rows: List[Dict[str, Any]] = []
    for row in results.get("results", []):
        sample = sample_map.get(row["sample_id"])
        if not sample or sample.get("injection_operator") != "noise_injection":
            continue
        base_ts = np.asarray(sample["base_ts"], dtype=np.float32)
        target_ts = np.asarray(sample["target_ts"], dtype=np.float32)
        start = int(sample["gt_start"])
        end = int(sample["gt_end"]) + 1
        subpattern = classify_volatility_subpattern(target_ts[start:end], base_ts[start:end])
        predicted_tool = (
            (row.get("llm_plan") or {}).get("tool_name")
            or ((row.get("llm_plan") or {}).get("execution") or {}).get("tool_name")
            or (row.get("experiment_record") or {}).get("prediction", {}).get("hybrid_tool_label")
            or "none"
        )
        expected_tool = EXPECTED_TOOL_BY_SUBPATTERN.get(subpattern)
        volatility_rows.append(
            {
                "sample_id": row["sample_id"],
                "subpattern": subpattern,
                "expected_tool": expected_tool,
                "predicted_tool": predicted_tool,
                "route_correct": bool(expected_tool is not None and predicted_tool == expected_tool),
                "is_preview_case": subpattern == "non_monotonic_envelope",
                "target_mae": float((row.get("metrics") or {}).get("mae_vs_target", 0.0)),
                "preservation_mae": float((row.get("metrics") or {}).get("preservation_mae", 0.0)),
            }
        )

    by_tool: Dict[str, Dict[str, Any]] = {}
    for tool_name in sorted({row["predicted_tool"] for row in volatility_rows}):
        subset = [row for row in volatility_rows if row["predicted_tool"] == tool_name]
        by_tool[tool_name] = {
            "count": len(subset),
            "avg_target_mae": _mean([row["target_mae"] for row in subset]),
            "avg_preservation_mae": _mean([row["preservation_mae"] for row in subset]),
            "route_correct_rate": _mean([1.0 if row["route_correct"] else 0.0 for row in subset if not row["is_preview_case"]]),
        }

    by_subpattern: Dict[str, Dict[str, Any]] = {}
    for subpattern in sorted({row["subpattern"] for row in volatility_rows}):
        subset = [row for row in volatility_rows if row["subpattern"] == subpattern]
        by_subpattern[subpattern] = {
            "count": len(subset),
            "expected_tool": EXPECTED_TOOL_BY_SUBPATTERN.get(subpattern),
            "routed_tools": {tool: sum(1 for row in subset if row["predicted_tool"] == tool) for tool in sorted({row["predicted_tool"] for row in subset})},
            "route_correct_rate": _mean([1.0 if row["route_correct"] else 0.0 for row in subset if not row["is_preview_case"]]),
            "avg_target_mae": _mean([row["target_mae"] for row in subset]),
        }

    summary = {
        "total_volatility_cases": len(volatility_rows),
        "preview_case_count": sum(1 for row in volatility_rows if row["is_preview_case"]),
        "fallback_or_unsupported_count": sum(
            1 for row in volatility_rows
            if (not row["is_preview_case"]) and row["predicted_tool"] not in set(EXPECTED_TOOL_BY_SUBPATTERN.values())
        ),
        "overall_route_correct_rate": _mean([1.0 if row["route_correct"] else 0.0 for row in volatility_rows if not row["is_preview_case"]]),
    }

    return {
        "testset": testset_path,
        "result_path": result_path,
        "summary": summary,
        "by_predicted_tool": by_tool,
        "by_subpattern": by_subpattern,
        "rows": volatility_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze volatility split routing on pure-editing pipeline outputs.")
    parser.add_argument("--testset", required=True)
    parser.add_argument("--result", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    report = analyze(args.testset, args.result)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
