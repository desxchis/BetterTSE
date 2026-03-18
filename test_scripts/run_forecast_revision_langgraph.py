from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys
from typing import Any, Dict, List

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from agent.agent import A1
from modules.forecast_revision import (
    calibrate_revision,
    evaluate_calibration,
    evaluate_revision_sample,
    extract_gt_edit_spec,
    heuristic_revision_plan,
)
from modules.forecast_revision_executor import apply_tedit_hybrid_revision


def _gt_region(mask: np.ndarray) -> List[int]:
    idx = np.where(mask > 0.5)[0]
    if idx.size == 0:
        return [0, 0]
    return [int(idx[0]), int(idx[-1] + 1)]


def _iso_range(start: datetime, length: int, step_minutes: int = 60) -> list[str]:
    return [(start + timedelta(minutes=step_minutes * i)).strftime("%Y-%m-%dT%H:%M:%SZ") for i in range(length)]


def _build_agent_input(sample: Dict[str, Any]) -> str:
    history = np.asarray(sample["history_ts"], dtype=np.float64).flatten().tolist()
    base_forecast = np.asarray(sample["base_forecast"], dtype=np.float64).flatten().tolist()

    now = datetime.now(timezone.utc).replace(microsecond=0)
    history_ts = _iso_range(now - timedelta(hours=len(history)), len(history), step_minutes=60)
    forecast_ts = _iso_range(now, len(base_forecast), step_minutes=60)
    payload = {
        "history": {"timestamps": history_ts, "values": history},
        "forecast": {"timestamps": forecast_ts, "values": base_forecast},
        "context": sample.get("context_text", ""),
    }
    return json.dumps(payload, ensure_ascii=False)


def _extract_last_event(events: list[dict[str, Any]], event_type: str) -> dict[str, Any] | None:
    for e in reversed(events):
        if isinstance(e, dict) and e.get("type") == event_type:
            return e
    return None


def _infer_intent_from_agent(
    *,
    sample: Dict[str, Any],
    planner_step: Dict[str, Any] | None,
    planner_edit: Dict[str, Any] | None,
    pred_region: List[int],
) -> tuple[Dict[str, Any], str | None]:
    fallback = heuristic_revision_plan(sample.get("context_text", ""), int(sample.get("forecast_horizon", 0)))
    fallback_intent = fallback.get("intent", {}) if isinstance(fallback, dict) else {}
    fallback_tool = fallback.get("tool_name") if isinstance(fallback, dict) else None

    if isinstance(planner_edit, dict):
        parsed = planner_edit.get("parsed", {}) or {}
        name = str(parsed.get("name", ""))
        params = parsed.get("parameters", {}) or {}
        if name == "apply_trend_in_region":
            slope = float(params.get("slope", 0.0))
            direction = "up" if slope > 0 else "down" if slope < 0 else "neutral"
            strength = "weak" if abs(slope) < 0.1 else "medium" if abs(slope) < 0.3 else "strong"
            region_len = max(0, int(pred_region[1]) - int(pred_region[0]))
            horizon = int(sample.get("forecast_horizon", region_len))
            if region_len <= max(4, horizon // 4):
                duration = "short"
            elif region_len <= max(8, horizon // 2):
                duration = "medium"
            else:
                duration = "long"
            intent = {
                "effect_family": "level",
                "direction": direction,
                "shape": "step" if abs(slope) >= 0.25 else "plateau",
                "duration": duration,
                "strength": strength,
            }
            tool_name = "hybrid_up" if direction == "up" else "hybrid_down" if direction == "down" else "none"
            return intent, tool_name
        return fallback_intent, fallback_tool

    if isinstance(planner_step, dict):
        parsed = planner_step.get("parsed", {}) or {}
        name = str(parsed.get("name", ""))
        if name in {"hybrid_up", "hybrid_down", "step_shift", "spike_inject", "volatility_increase"}:
            intent = dict(fallback_intent)
            if name == "hybrid_up":
                intent.update({"effect_family": "level", "direction": "up", "shape": "plateau"})
            elif name == "hybrid_down":
                intent.update({"effect_family": "level", "direction": "down", "shape": "plateau"})
            return intent, name
    return fallback_intent, fallback_tool


def run_langgraph_revision(
    *,
    benchmark_path: str,
    output_path: str,
    llm_name: str,
    source: str | None,
    base_url: str | None,
    api_key: str,
    max_samples: int | None,
    tedit_model_path: str,
    tedit_config_path: str,
    tedit_device: str,
) -> Dict[str, Any]:
    with open(benchmark_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    samples = payload.get("samples", [])
    if max_samples is not None:
        samples = samples[:max_samples]

    agent = A1(
        llm_name=llm_name,
        source=source,
        base_url=base_url,
        api_key=api_key,
        enable_tedit=True,
        enable_instruction_decomposition=False,
        tedit_model_path=tedit_model_path,
        tedit_config_path=tedit_config_path,
        tedit_device=tedit_device,
    )
    agent.set_editing_mode(True)

    results: list[dict[str, Any]] = []
    for sample in samples:
        history_ts = np.asarray(sample["history_ts"], dtype=np.float64)
        future_gt = np.asarray(sample["future_gt"], dtype=np.float64)
        base_forecast = np.asarray(sample["base_forecast"], dtype=np.float64)
        revision_target = np.asarray(sample["revision_target"], dtype=np.float64)
        gt_mask = np.asarray(sample["edit_mask_gt"], dtype=np.float64)
        gt_region = _gt_region(gt_mask)
        edit_spec_gt = extract_gt_edit_spec(sample, history_ts=history_ts, base_forecast=base_forecast)

        user_input = _build_agent_input(sample)
        pipeline_outputs: list[dict[str, Any]] = []
        agent_error: str | None = None
        try:
            for _, _, events in agent.go(user_input):
                pipeline_outputs = events
        except Exception as exc:
            pipeline_outputs = [{"type": "agent.error", "error": str(exc)}]
            agent_error = str(exc)

        planner_step = _extract_last_event(pipeline_outputs, "planner.step_json")
        planner_edit = _extract_last_event(pipeline_outputs, "planner.edit_json")
        composer_out = _extract_last_event(pipeline_outputs, "composer.output")
        editor_out = _extract_last_event(pipeline_outputs, "editor.edit_summary")

        pred_region = [0, len(base_forecast)]
        if isinstance(planner_step, dict):
            parsed = planner_step.get("parsed", {}) or {}
            if "update_start_idx" in parsed and "update_end_idx" in parsed:
                pred_region = [int(parsed["update_start_idx"]), int(parsed["update_end_idx"])]
        if isinstance(planner_edit, dict):
            parsed = planner_edit.get("parsed", {}) or {}
            if "start_idx" in parsed and "end_idx" in parsed:
                pred_region = [int(parsed["start_idx"]), int(parsed["end_idx"])]

        intent, tool_name = _infer_intent_from_agent(
            sample=sample,
            planner_step=planner_step,
            planner_edit=planner_edit,
            pred_region=pred_region,
        )
        if bool(sample.get("revision_applicable_gt", True)) and intent.get("effect_family", "none") != "none":
            params = calibrate_revision(
                intent=intent,
                region=pred_region,
                history_ts=history_ts,
                base_forecast=base_forecast,
            )
            edited, _, execution_metadata = apply_tedit_hybrid_revision(
                history_ts=history_ts,
                base_forecast=base_forecast,
                intent=intent,
                region=pred_region,
                params=params,
                preferred_tool_name=tool_name,
                tedit_model_path=tedit_model_path,
                tedit_config_path=tedit_config_path,
                tedit_device=tedit_device,
                sample_metadata=sample,
            )
        else:
            edited = base_forecast.copy()
            params = {}
            execution_metadata = {"executor": "tedit_hybrid", "tool_name": "none"}

        metrics = evaluate_revision_sample(
            base_forecast=base_forecast,
            edited_forecast=edited,
            future_gt=future_gt,
            revision_target=revision_target,
            pred_region=pred_region,
            gt_mask=gt_mask,
        )
        calibration_metrics = evaluate_calibration(
            edit_spec_gt=edit_spec_gt,
            edit_spec_pred=edit_spec_gt,
            base_forecast=base_forecast,
            revision_target=revision_target,
            edited_forecast=edited,
            region=gt_region,
        )
        results.append(
            {
                "sample_id": sample.get("sample_id"),
                "dataset_name": sample.get("dataset_name"),
                "mode": "langgraph_revision",
                "pred_region": pred_region,
                "gt_region": gt_region,
                "metrics": metrics,
                "calibration_metrics": calibration_metrics,
                "execution_metadata": {
                    "planner_step": planner_step,
                    "planner_edit": planner_edit,
                    "composer_output": composer_out,
                    "editor_output": editor_out,
                    "agent_error": agent_error,
                    "intent": intent,
                    "tool_name": tool_name,
                    "executor": execution_metadata,
                    "calibration_params": params,
                },
                "revision_applicable_gt": sample.get("revision_applicable_gt"),
                "history_ts": sample.get("history_ts"),
                "future_gt": sample.get("future_gt"),
                "base_forecast": sample.get("base_forecast"),
                "revision_target": sample.get("revision_target"),
                "edited_forecast": edited.astype(float).tolist(),
            }
        )

    def _avg(key: str) -> float | None:
        vals = [float(r["metrics"][key]) for r in results if key in r.get("metrics", {})]
        return float(np.mean(vals)) if vals else None

    summary = {
        "total": len(results),
        "successful": len(results),
        "failed": 0,
        "avg_base_mae_vs_revision_target": _avg("base_mae_vs_revision_target"),
        "avg_edited_mae_vs_revision_target": _avg("edited_mae_vs_revision_target"),
        "avg_revision_gain": _avg("revision_gain"),
        "avg_over_edit_rate": _avg("over_edit_rate"),
        "avg_outside_region_preservation": _avg("outside_region_preservation"),
    }
    out = {
        "summary": summary,
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "planner_backend": "langgraph_agent",
        "results": results,
    }
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Run forecast revision via LangGraph agent (editing path).")
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-samples", type=int, default=5)
    parser.add_argument("--llm-name", default="claude-sonnet-4-20250514")
    parser.add_argument("--source", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--tedit-model", required=True)
    parser.add_argument("--tedit-config", required=True)
    parser.add_argument("--tedit-device", default="cuda:0")
    args = parser.parse_args()

    payload = run_langgraph_revision(
        benchmark_path=args.benchmark,
        output_path=args.output,
        llm_name=args.llm_name,
        source=args.source,
        base_url=args.base_url,
        api_key=args.api_key,
        max_samples=args.max_samples,
        tedit_model_path=args.tedit_model,
        tedit_config_path=args.tedit_config,
        tedit_device=args.tedit_device,
    )
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
