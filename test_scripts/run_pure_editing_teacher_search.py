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

from modules.experiment_logging import extract_pure_editing_gt_labels
from modules.pure_editing_how_much import (
    compute_pure_editing_parameter_metrics,
    teacher_search_pure_editing_params,
)
from tool.tedit_wrapper import TEditWrapper, get_tedit_instance
from tool.ts_editors import spike_inject, step_shift, volatility_increase, hybrid_down_soft, hybrid_up_soft


def _heuristic_execute(
    *,
    tool_name: str,
    base_ts: np.ndarray,
    region: List[int],
    direction: str,
    tedit: TEditWrapper | None,
) -> np.ndarray:
    start, end = int(region[0]), int(region[1])
    scale = max(float(np.std(base_ts[start:end])), float(np.std(base_ts)) * 0.5, 1e-3)
    sign = -1.0 if str(direction).lower() in {"down", "downward", "negative"} else 1.0
    if tool_name == "spike_inject":
        return spike_inject(
            ts=base_ts,
            start_idx=start,
            end_idx=end,
            center=(start + end) // 2,
            amplitude=sign * scale * 2.0,
            width=max(2.0, (end - start) / 6.0),
        )
    if tool_name == "step_shift":
        return step_shift(
            ts=base_ts,
            start_idx=start,
            end_idx=end,
            level_shift=sign * scale * 2.0,
        )
    if tool_name == "volatility_increase":
        return volatility_increase(ts=base_ts, start_idx=start, end_idx=end, amplify_factor=2.0)
    if tool_name == "hybrid_up":
        if tedit is None:
            raise ValueError("tedit is required for hybrid_up")
        return hybrid_up_soft(ts=base_ts, start_idx=start, end_idx=end, math_shift=abs(scale * 2.0), tedit=tedit)
    if tool_name == "hybrid_down":
        if tedit is None:
            raise ValueError("tedit is required for hybrid_down")
        return hybrid_down_soft(ts=base_ts, start_idx=start, end_idx=end, math_shift=-abs(scale * 2.0), tedit=tedit)
    raise ValueError(f"unsupported heuristic tool: {tool_name}")


def _summarize(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not results:
        return {"total": 0}
    keys = [
        "teacher_mae_vs_target",
        "teacher_mse_vs_target",
        "teacher_preservation_mae",
        "teacher_peak_delta_error",
        "teacher_signed_area_error",
        "heuristic_mae_vs_target",
        "heuristic_mse_vs_target",
        "heuristic_preservation_mae",
        "heuristic_peak_delta_error",
        "heuristic_signed_area_error",
    ]
    summary: Dict[str, Any] = {"total": len(results)}
    for key in keys:
        values = [float(item[key]) for item in results]
        summary[f"avg_{key}"] = float(np.mean(values))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run pure-editing tool-conditioned teacher-search prototype.")
    parser.add_argument("--testset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-samples", type=int, default=10)
    parser.add_argument("--tedit-model", default="")
    parser.add_argument("--tedit-config", default="")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    payload = json.loads(Path(args.testset).read_text(encoding="utf-8"))
    samples = list(payload.get("samples", []))[: args.max_samples]

    tedit = None
    if args.tedit_model and args.tedit_config:
        tedit = get_tedit_instance(
            model_path=args.tedit_model,
            config_path=args.tedit_config,
            device=args.device,
            force_reload=True,
        )

    results: List[Dict[str, Any]] = []
    for sample in samples:
        gt_config = {
            "start_step": sample.get("gt_start", 0),
            "end_step": sample.get("gt_end", 0),
            "injection_operator": sample.get("injection_operator", "none"),
            "edit_intent_gt": sample.get("edit_intent_gt", {}),
            "task_type": sample.get("task_type", "unknown"),
        }
        labels = extract_pure_editing_gt_labels(sample, gt_config)
        tool_name = labels["hybrid_tool_label"]
        if tool_name in {"none", "season_enhance", "season_reduce", "ensemble_smooth"}:
            continue
        region = list((labels["localization_label"] or {}).get("region", [0, len(sample.get("base_ts", []))]))
        direction = str((labels["intent_label"] or {}).get("direction", "up"))

        base_ts = np.asarray(sample["base_ts"], dtype=np.float32)
        target_ts = np.asarray(sample["target_ts"], dtype=np.float32)

        heuristic_ts = _heuristic_execute(
            tool_name=tool_name,
            base_ts=base_ts,
            region=region,
            direction=direction,
            tedit=tedit,
        )
        heuristic_metrics = compute_pure_editing_parameter_metrics(
            base_ts=base_ts,
            target_ts=target_ts,
            edited_ts=heuristic_ts,
            region=region,
        )
        teacher = teacher_search_pure_editing_params(
            tool_name=tool_name,
            base_ts=base_ts,
            target_ts=target_ts,
            region=region,
            direction=direction,
            tedit=tedit,
        )
        results.append(
            {
                "sample_id": sample.get("sample_id"),
                "tool_name": tool_name,
                "direction": direction,
                "region": region,
                "teacher_params": teacher.params,
                "teacher_search_space_size": teacher.search_space_size,
                "teacher_objective": teacher.objective,
                "teacher_mae_vs_target": teacher.metrics["mae_vs_target"],
                "teacher_mse_vs_target": teacher.metrics["mse_vs_target"],
                "teacher_preservation_mae": teacher.metrics["preservation_mae"],
                "teacher_peak_delta_error": teacher.metrics["peak_delta_error"],
                "teacher_signed_area_error": teacher.metrics["signed_area_error"],
                "heuristic_mae_vs_target": heuristic_metrics["mae_vs_target"],
                "heuristic_mse_vs_target": heuristic_metrics["mse_vs_target"],
                "heuristic_preservation_mae": heuristic_metrics["preservation_mae"],
                "heuristic_peak_delta_error": heuristic_metrics["peak_delta_error"],
                "heuristic_signed_area_error": heuristic_metrics["signed_area_error"],
                "teacher_improves_target_mae": teacher.metrics["mae_vs_target"] < heuristic_metrics["mae_vs_target"],
                "teacher_improves_preservation": teacher.metrics["preservation_mae"] <= heuristic_metrics["preservation_mae"],
            }
        )

    output = {
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "testset": args.testset,
        "max_samples": args.max_samples,
        "summary": _summarize(results),
        "results": results,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
