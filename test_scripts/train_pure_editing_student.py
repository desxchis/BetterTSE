from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from modules.pure_editing_how_much import (
    compute_pure_editing_parameter_metrics,
    teacher_search_pure_editing_params,
)
from modules.pure_editing_student import (
    derive_student_tool_and_region,
    fit_tool_conditioned_student,
    load_student_model,
    predict_tool_conditioned_params,
    save_student_model,
)
from tool.tedit_wrapper import get_tedit_instance
from tool.ts_editors import (
    hybrid_down_soft,
    hybrid_up_soft,
    spike_inject,
    step_shift,
    volatility_burst_local,
    volatility_envelope_monotonic,
    volatility_global_scale,
)


def _execute_tool(
    *,
    tool_name: str,
    params: Dict[str, Any],
    base_ts: np.ndarray,
    region: List[int],
    tedit,
) -> np.ndarray:
    start, end = int(region[0]), int(region[1])
    base = np.asarray(base_ts, dtype=np.float32)
    if tool_name == "spike_inject":
        return spike_inject(
            ts=base,
            start_idx=start,
            end_idx=end,
            center=int(params["center"]),
            amplitude=float(params["amplitude"]),
            width=float(params["width"]),
        )
    if tool_name == "step_shift":
        return step_shift(
            ts=base,
            start_idx=start,
            end_idx=end,
            level_shift=float(params["level_shift"]),
            left_ramp_steps=int(params.get("left_ramp_steps", 1)),
            right_ramp_steps=int(params.get("right_ramp_steps", 1)),
        )
    if tool_name == "hybrid_up":
        return hybrid_up_soft(
            ts=base,
            start_idx=start,
            end_idx=end,
            math_shift=float(params["math_shift"]),
            tedit=tedit,
        )
    if tool_name == "hybrid_down":
        return hybrid_down_soft(
            ts=base,
            start_idx=start,
            end_idx=end,
            math_shift=float(params["math_shift"]),
            tedit=tedit,
        )
    if tool_name == "volatility_global_scale":
        return volatility_global_scale(base_ts=base, region=region, **params)
    if tool_name == "volatility_local_burst":
        return volatility_burst_local(base_ts=base, region=region, **params)
    if tool_name == "volatility_envelope_monotonic":
        return volatility_envelope_monotonic(base_ts=base, region=region, **params)
    raise ValueError(f"unsupported tool_name: {tool_name}")


def _heuristic_params(tool_name: str, base_ts: np.ndarray, region: List[int]) -> Dict[str, Any]:
    start, end = int(region[0]), int(region[1])
    local = np.asarray(base_ts[start:end], dtype=np.float64)
    local_std = max(float(np.std(local)), float(np.std(base_ts)) * 0.25, 1e-3)
    region_len = max(1, end - start)
    if tool_name == "spike_inject":
        return {
            "amplitude": local_std * 2.0,
            "width": max(2.0, region_len / 6.0),
            "center": int((start + end) // 2),
        }
    if tool_name == "step_shift":
        return {
            "level_shift": local_std * 2.0,
            "left_ramp_steps": max(1, region_len // 10),
            "right_ramp_steps": max(1, region_len // 5),
        }
    if tool_name == "hybrid_up":
        return {"math_shift": local_std * 2.0}
    if tool_name == "hybrid_down":
        return {"math_shift": -local_std * 2.0}
    if tool_name == "volatility_global_scale":
        return {
            "base_noise_scale": 1.0,
            "local_std_target_ratio": 2.0,
            "baseline_offset_ratio": 0.05,
            "trend_preserve": 0.0,
        }
    if tool_name == "volatility_local_burst":
        return {
            "background_scale": 0.5,
            "burst_center": 0.5,
            "burst_width": 0.25,
            "burst_amplitude": 2.4,
            "burst_envelope_sharpness": 0.8,
            "baseline_offset_ratio": 0.05,
        }
    if tool_name == "volatility_envelope_monotonic":
        return {
            "base_noise_scale": 1.0,
            "start_scale": 0.3,
            "end_scale": 2.0,
            "baseline_offset_ratio": 0.05,
            "trend_preserve": 0.0,
        }
    raise ValueError(f"unsupported heuristic tool_name: {tool_name}")


def _load_samples(testsets: List[str]) -> List[Dict[str, Any]]:
    samples: List[Dict[str, Any]] = []
    for path in testsets:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        samples.extend(list(payload.get("samples", [])))
    return samples


def _prepare_rows(samples: List[Dict[str, Any]], *, tedit) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for sample in samples:
        derived = derive_student_tool_and_region(sample)
        if derived is None:
            continue
        tool_name = str(derived["tool_name"])
        region = list(derived["region"])
        prompt_text = str(derived["prompt_text"])
        intent = dict(derived["intent"])
        direction = str(intent.get("direction", "up"))
        base_ts = np.asarray(sample["base_ts"], dtype=np.float32)
        target_ts = np.asarray(sample["target_ts"], dtype=np.float32)
        teacher = teacher_search_pure_editing_params(
            tool_name=tool_name,
            base_ts=base_ts,
            target_ts=target_ts,
            region=region,
            direction=direction,
            tedit=tedit,
        )
        rows.append(
            {
                "sample_id": sample.get("sample_id"),
                "dataset_name": sample.get("dataset_name"),
                "tool_name": tool_name,
                "region": region,
                "prompt_text": prompt_text,
                "intent": intent,
                "base_ts": base_ts.tolist(),
                "target_ts": target_ts.tolist(),
                "teacher_params": dict(teacher.params),
                "teacher_metrics": dict(teacher.metrics),
                "teacher_objective": float(teacher.objective),
                "teacher_search_space_size": int(teacher.search_space_size),
            }
        )
    return rows


def _split_rows(rows: List[Dict[str, Any]], *, train_ratio: float, seed: int) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    by_tool: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        by_tool.setdefault(str(row["tool_name"]), []).append(row)
    rng = random.Random(seed)
    train_rows: List[Dict[str, Any]] = []
    heldout_rows: List[Dict[str, Any]] = []
    for tool_name, tool_rows in by_tool.items():
        shuffled = list(tool_rows)
        rng.shuffle(shuffled)
        if len(shuffled) == 1:
            train_rows.extend(shuffled)
            continue
        split = max(1, min(len(shuffled) - 1, int(round(len(shuffled) * train_ratio))))
        train_rows.extend(shuffled[:split])
        heldout_rows.extend(shuffled[split:])
    return train_rows, heldout_rows


def _evaluate_rows(rows: List[Dict[str, Any]], *, model: Dict[str, Any], tedit) -> Dict[str, Any]:
    detailed = []
    for row in rows:
        base_ts = np.asarray(row["base_ts"], dtype=np.float32)
        target_ts = np.asarray(row["target_ts"], dtype=np.float32)
        region = list(row["region"])
        tool_name = str(row["tool_name"])
        intent = dict(row["intent"])
        prompt_text = str(row["prompt_text"])

        teacher_params = dict(row["teacher_params"])
        heuristic_params = _heuristic_params(tool_name, base_ts, region)
        student_params = predict_tool_conditioned_params(
            model=model,
            tool_name=tool_name,
            base_ts=base_ts,
            region=region,
            prompt_text=prompt_text,
            intent=intent,
        )
        if student_params is None:
            continue

        teacher_ts = _execute_tool(tool_name=tool_name, params=teacher_params, base_ts=base_ts, region=region, tedit=tedit)
        heuristic_ts = _execute_tool(tool_name=tool_name, params=heuristic_params, base_ts=base_ts, region=region, tedit=tedit)
        student_ts = _execute_tool(tool_name=tool_name, params=student_params, base_ts=base_ts, region=region, tedit=tedit)

        teacher_metrics = compute_pure_editing_parameter_metrics(
            base_ts=base_ts, target_ts=target_ts, edited_ts=teacher_ts, region=region
        )
        heuristic_metrics = compute_pure_editing_parameter_metrics(
            base_ts=base_ts, target_ts=target_ts, edited_ts=heuristic_ts, region=region
        )
        student_metrics = compute_pure_editing_parameter_metrics(
            base_ts=base_ts, target_ts=target_ts, edited_ts=student_ts, region=region
        )
        detailed.append(
            {
                "sample_id": row["sample_id"],
                "dataset_name": row["dataset_name"],
                "tool_name": tool_name,
                "student_params": student_params,
                "teacher_params": teacher_params,
                "heuristic_params": heuristic_params,
                "student_mae_vs_target": student_metrics["mae_vs_target"],
                "teacher_mae_vs_target": teacher_metrics["mae_vs_target"],
                "heuristic_mae_vs_target": heuristic_metrics["mae_vs_target"],
                "student_preservation_mae": student_metrics["preservation_mae"],
                "teacher_preservation_mae": teacher_metrics["preservation_mae"],
                "heuristic_preservation_mae": heuristic_metrics["preservation_mae"],
                "student_beats_heuristic": student_metrics["mae_vs_target"] < heuristic_metrics["mae_vs_target"],
                "student_teacher_gap": student_metrics["mae_vs_target"] - teacher_metrics["mae_vs_target"],
            }
        )

    def _avg(key: str) -> float:
        if not detailed:
            return 0.0
        return float(np.mean([float(row[key]) for row in detailed]))

    tool_summary = {}
    for tool_name in sorted({str(row["tool_name"]) for row in detailed}):
        subset = [row for row in detailed if row["tool_name"] == tool_name]
        tool_summary[tool_name] = {
            "count": len(subset),
            "student_mae_vs_target": float(np.mean([row["student_mae_vs_target"] for row in subset])),
            "teacher_mae_vs_target": float(np.mean([row["teacher_mae_vs_target"] for row in subset])),
            "heuristic_mae_vs_target": float(np.mean([row["heuristic_mae_vs_target"] for row in subset])),
            "student_better_rate_vs_heuristic": float(np.mean([1.0 if row["student_beats_heuristic"] else 0.0 for row in subset])),
        }

    return {
        "summary": {
            "total": len(detailed),
            "student_mae_vs_target": _avg("student_mae_vs_target"),
            "teacher_mae_vs_target": _avg("teacher_mae_vs_target"),
            "heuristic_mae_vs_target": _avg("heuristic_mae_vs_target"),
            "student_preservation_mae": _avg("student_preservation_mae"),
            "teacher_preservation_mae": _avg("teacher_preservation_mae"),
            "heuristic_preservation_mae": _avg("heuristic_preservation_mae"),
            "student_better_rate_vs_heuristic": float(np.mean([1.0 if row["student_beats_heuristic"] else 0.0 for row in detailed])) if detailed else 0.0,
            "student_teacher_gap": _avg("student_teacher_gap"),
        },
        "by_tool": tool_summary,
        "rows": detailed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a tool-conditioned pure-editing how-much student.")
    parser.add_argument("--testset", action="append", required=True, help="Repeatable pure-editing benchmark JSON path.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--tedit-model", default="")
    parser.add_argument("--tedit-config", default="")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    samples = _load_samples(args.testset)
    if args.max_samples is not None:
        samples = samples[: args.max_samples]

    tedit = None
    if args.tedit_model and args.tedit_config:
        tedit = get_tedit_instance(
            model_path=args.tedit_model,
            config_path=args.tedit_config,
            device=args.device,
            force_reload=True,
        )

    rows = _prepare_rows(samples, tedit=tedit)
    train_rows, heldout_rows = _split_rows(rows, train_ratio=args.train_ratio, seed=args.seed)
    model = fit_tool_conditioned_student(train_rows, alpha=args.alpha)

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    model_path = out_root / "tool_conditioned_pure_editing_student.json"
    save_student_model(model, str(model_path))

    teacher_dump_path = out_root / "teacher_dump.json"
    teacher_dump_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    train_path = out_root / "train_split.json"
    train_path.write_text(json.dumps(train_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    heldout_path = out_root / "heldout_split.json"
    heldout_path.write_text(json.dumps(heldout_rows, ensure_ascii=False, indent=2), encoding="utf-8")

    eval_payload = _evaluate_rows(heldout_rows, model=model, tedit=tedit)
    eval_path = out_root / "heldout_eval.json"
    eval_path.write_text(json.dumps(eval_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({
        "teacher_dump": str(teacher_dump_path),
        "model_path": str(model_path),
        "train_count": len(train_rows),
        "heldout_count": len(heldout_rows),
        "heldout_eval": str(eval_path),
        "summary": eval_payload["summary"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
