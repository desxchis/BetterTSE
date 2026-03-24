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

from config import get_api_config
from modules.llm import CustomLLMClient, get_event_driven_plan
from modules.pure_editing_volatility import classify_volatility_subpattern, resolve_volatility_subtype_route


EXPECTED_SUBTYPE_BY_SUBPATTERN = {
    "uniform_variance": "global_scale",
    "local_burst": "local_burst",
    "monotonic_envelope": "envelope_monotonic",
    "non_monotonic_envelope": "preview_non_monotonic",
}


def _mean(values: List[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _prompt_text(sample: Dict[str, Any]) -> str:
    if sample.get("vague_prompt"):
        return str(sample["vague_prompt"])
    prompts = sample.get("event_prompts", [])
    if prompts:
        sorted_ep = sorted(prompts, key=lambda x: abs(x.get("level", 0) - 2))
        return str(sorted_ep[0].get("prompt", ""))
    return str(sample.get("causal_scenario", "") or sample.get("technical_ground_truth", "") or "")


def run_route_closure(
    testset_path: str,
    max_samples: int | None = None,
    routing_source: str = "planner_llm",
) -> Dict[str, Any]:
    testset = json.loads(Path(testset_path).read_text(encoding="utf-8"))
    samples = [sample for sample in testset.get("samples", []) if sample.get("injection_operator") == "noise_injection"]
    if max_samples is not None:
        samples = samples[:max_samples]

    llm_client = None
    if routing_source == "planner_llm":
        api_cfg = get_api_config()
        llm_client = CustomLLMClient(
            model_name=api_cfg["model_name"],
            base_url=api_cfg["base_url"],
            api_key=api_cfg["api_key"],
            temperature=0.3,
        )

    rows: List[Dict[str, Any]] = []
    for sample in samples:
        prompt_text = _prompt_text(sample)
        base_ts = np.asarray(sample["base_ts"], dtype=np.float32)
        target_ts = np.asarray(sample["target_ts"], dtype=np.float32)
        start = int(sample["gt_start"])
        end = int(sample["gt_end"]) + 1
        subpattern = classify_volatility_subpattern(target_ts[start:end], base_ts[start:end])
        expected_subtype = EXPECTED_SUBTYPE_BY_SUBPATTERN.get(subpattern, "none")

        if routing_source == "planner_llm":
            plan = get_event_driven_plan(
                news_text="",
                instruction_text=prompt_text,
                ts_length=len(base_ts),
                llm=llm_client,
            )
            routing = plan.get("volatility_routing") or {}
            proposed_subtype = routing.get("proposed_subtype") or (plan.get("intent") or {}).get("volatility_subtype") or "none"
            guarded_subtype = routing.get("guarded_subtype") or proposed_subtype
            final_subtype = routing.get("final_subtype") or plan.get("volatility_subtype") or "none"
            tool_name = plan.get("tool_name") or (plan.get("execution") or {}).get("tool_name")
            guard_reason = routing.get("guard_reason", "")
        else:
            routed = resolve_volatility_subtype_route(
                text=prompt_text,
                region=[start, end],
                ts_length=len(base_ts),
            )
            plan = {
                "tool_name": routed["tool_name"],
                "volatility_subtype": routed["final_subtype"],
                "volatility_routing": {
                    "proposed_subtype": routed["proposed_subtype"],
                    "guarded_subtype": routed["guarded_subtype"],
                    "final_subtype": routed["final_subtype"],
                    "guard_reason": routed["guard_reason"],
                    "is_preview": routed["is_preview"],
                },
            }
            proposed_subtype = routed["proposed_subtype"]
            guarded_subtype = routed["guarded_subtype"]
            final_subtype = routed["final_subtype"]
            tool_name = routed["tool_name"]
            guard_reason = routed["guard_reason"]
        rows.append(
            {
                "sample_id": sample.get("sample_id"),
                "prompt_text": prompt_text,
                "subpattern": subpattern,
                "expected_subtype": expected_subtype,
                "proposed_subtype": proposed_subtype,
                "guarded_subtype": guarded_subtype,
                "final_subtype": final_subtype,
                "guard_reason": guard_reason,
                "tool_name": tool_name,
                "subtype_correct": bool(final_subtype == expected_subtype),
                "is_preview_case": expected_subtype == "preview_non_monotonic",
            }
        )

    by_subpattern: Dict[str, Dict[str, Any]] = {}
    for subpattern in sorted({row["subpattern"] for row in rows}):
        subset = [row for row in rows if row["subpattern"] == subpattern]
        by_subpattern[subpattern] = {
            "count": len(subset),
            "expected_subtype": EXPECTED_SUBTYPE_BY_SUBPATTERN.get(subpattern),
            "proposed_subtypes": {value: sum(1 for row in subset if row["proposed_subtype"] == value) for value in sorted({row["proposed_subtype"] for row in subset})},
            "guarded_subtypes": {value: sum(1 for row in subset if row["guarded_subtype"] == value) for value in sorted({row["guarded_subtype"] for row in subset})},
            "final_subtypes": {value: sum(1 for row in subset if row["final_subtype"] == value) for value in sorted({row["final_subtype"] for row in subset})},
            "subtype_correct_rate": _mean([1.0 if row["subtype_correct"] else 0.0 for row in subset]),
        }

    supported_rows = [row for row in rows if not row["is_preview_case"]]
    summary = {
        "total_volatility_cases": len(rows),
        "supported_case_count": len(supported_rows),
        "preview_case_count": sum(1 for row in rows if row["is_preview_case"]),
        "supported_route_accuracy": _mean([1.0 if row["subtype_correct"] else 0.0 for row in supported_rows]),
        "preview_not_misrouted_rate": _mean([1.0 if row["final_subtype"] == "preview_non_monotonic" else 0.0 for row in rows if row["is_preview_case"]]),
    }

    return {
        "testset": testset_path,
        "routing_source": routing_source,
        "summary": summary,
        "by_subpattern": by_subpattern,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run route-only closure for volatility subtype routing.")
    parser.add_argument("--testset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--routing-source", choices=["planner_llm", "text_guard_only"], default="planner_llm")
    args = parser.parse_args()

    report = run_route_closure(args.testset, max_samples=args.max_samples, routing_source=args.routing_source)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
