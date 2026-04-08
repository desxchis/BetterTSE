from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

import sys

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tool.tedit_wrapper import TEditWrapper


STRENGTHS = [
    ("weak", 0),
    ("medium", 1),
    ("strong", 2),
]


def _compute_metrics(source_ts: np.ndarray, target_ts: np.ndarray, edited_ts: np.ndarray, mask_gt: np.ndarray) -> Dict[str, float | None]:
    edit_mask = mask_gt.astype(bool)
    bg_mask = ~edit_mask
    metrics: Dict[str, float | None] = {
        "edit_gain": None,
        "bg_mae": None,
        "target_mae_edit_region": None,
    }
    if np.any(edit_mask):
        metrics["edit_gain"] = float(np.mean(np.abs(edited_ts[edit_mask] - source_ts[edit_mask])))
        metrics["target_mae_edit_region"] = float(np.mean(np.abs(edited_ts[edit_mask] - target_ts[edit_mask])))
    if np.any(bg_mask):
        metrics["bg_mae"] = float(np.mean(np.abs(edited_ts[bg_mask] - source_ts[bg_mask])))
    return metrics


def _aggregate(values: List[float | None]) -> float | None:
    filtered = [float(v) for v in values if v is not None]
    if not filtered:
        return None
    return float(np.mean(filtered))


def _trend_attrs(direction: str) -> tuple[np.ndarray, np.ndarray]:
    src_attrs = np.array([0, 0, 0], dtype=np.int64)
    if direction == "down":
        tgt_attrs = np.array([1, 0, 1], dtype=np.int64)
    else:
        tgt_attrs = np.array([1, 1, 1], dtype=np.int64)
    return src_attrs, tgt_attrs


def _family_direction(family: Dict[str, Any]) -> str:
    samples = family.get("samples", [])
    for sample in samples:
        direction = str(sample.get("direction", ""))
        if direction in {"up", "down"}:
            return direction
        if direction in {"upward", "downward"}:
            return "up" if direction == "upward" else "down"
    return "up"


def run_eval(
    benchmark_path: Path,
    model_path: str,
    config_path: str,
    output_path: Path,
    *,
    max_families: int,
    edit_steps: int,
    sampler: str,
    seed: int,
    device: str,
    smooth_radius: float,
    bg_drift_threshold: float,
    probe_json: str | None,
) -> Dict[str, Any]:
    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
    all_families = [fam for fam in benchmark.get("families", []) if str(fam.get("tool_name")) == "trend_injection"]
    families = all_families[:max_families]
    if not families:
        raise ValueError("No trend_injection families found in benchmark.")

    probe_gate = None
    if probe_json:
        probe = json.loads(Path(probe_json).read_text(encoding="utf-8"))
        probe_gate = {
            "probe_path": probe_json,
            "diff_0_2_linf": float(probe.get("diff_0_2_linf", 0.0)),
            "pass": bool(float(probe.get("diff_0_2_linf", 0.0)) > 0.0),
        }

    wrapper = TEditWrapper(model_path=model_path, config_path=config_path, device=device)
    wrapper.set_edit_steps(edit_steps)

    family_rows: List[Dict[str, Any]] = []
    per_strength_rows: Dict[str, List[Dict[str, Any]]] = {name: [] for name, _ in STRENGTHS}

    for family_idx, family in enumerate(families):
        direction = _family_direction(family)
        src_attrs, tgt_attrs = _trend_attrs(direction)
        samples = {str(sample["strength_text"]): sample for sample in family.get("samples", [])}
        row: Dict[str, Any] = {
            "family_id": family.get("family_id"),
            "tool_name": family.get("tool_name"),
            "direction": direction,
            "region": family.get("region"),
            "per_strength": {},
        }
        gains: List[float | None] = []

        for strength_name, strength_label in STRENGTHS:
            sample = samples[strength_name]
            source_ts = np.asarray(sample["source_ts"], dtype=np.float32)
            target_ts = np.asarray(sample["target_ts"], dtype=np.float32)
            mask_gt = np.asarray(sample["mask_gt"], dtype=np.int64)
            start_idx, end_idx = [int(v) for v in sample["region"]]
            instruction_text = str(sample["instruction_text"])

            torch.manual_seed(seed + family_idx)
            np.random.seed(seed + family_idx)
            edited_ts = wrapper.edit_region_soft(
                ts=source_ts,
                start_idx=start_idx,
                end_idx=end_idx,
                src_attrs=src_attrs,
                tgt_attrs=tgt_attrs,
                n_samples=1,
                sampler=sampler,
                smooth_radius=smooth_radius,
                strength_label=strength_label,
                task_id=None,
                instruction_text=instruction_text,
            )
            edited_ts = np.asarray(edited_ts, dtype=np.float32).reshape(-1)
            metrics = _compute_metrics(source_ts, target_ts, edited_ts, mask_gt)
            gains.append(metrics["edit_gain"])
            record = {
                "family_id": family.get("family_id"),
                "strength_text": strength_name,
                "strength_label": strength_label,
                "instruction_text": instruction_text,
                **metrics,
            }
            per_strength_rows[strength_name].append(record)
            row["per_strength"][strength_name] = record

        monotonic_hit = bool(gains[0] is not None and gains[1] is not None and gains[2] is not None and gains[0] < gains[1] < gains[2])
        weak_bg = row["per_strength"]["weak"]["bg_mae"]
        strong_bg = row["per_strength"]["strong"]["bg_mae"]
        bg_delta = None if weak_bg is None or strong_bg is None else float(strong_bg - weak_bg)
        row["weak_edit_gain"] = gains[0]
        row["medium_edit_gain"] = gains[1]
        row["strong_edit_gain"] = gains[2]
        row["strong_minus_weak_edit_gain"] = None if gains[0] is None or gains[2] is None else float(gains[2] - gains[0])
        row["bg_mae_strong_minus_weak"] = bg_delta
        row["monotonic_hit"] = monotonic_hit
        row["preservation_pass"] = bool(bg_delta is not None and bg_delta <= bg_drift_threshold)
        family_rows.append(row)

    summary = {
        "num_families": len(family_rows),
        "family_filter": "trend_injection",
        "probe_gate": probe_gate,
        "edit_gain_mean": {
            strength_name: _aggregate([row["edit_gain"] for row in per_strength_rows[strength_name]])
            for strength_name, _ in STRENGTHS
        },
        "bg_mae_mean": {
            strength_name: _aggregate([row["bg_mae"] for row in per_strength_rows[strength_name]])
            for strength_name, _ in STRENGTHS
        },
        "target_mae_edit_region_mean": {
            strength_name: _aggregate([row["target_mae_edit_region"] for row in per_strength_rows[strength_name]])
            for strength_name, _ in STRENGTHS
        },
        "monotonic_hit_rate": float(np.mean([float(row["monotonic_hit"]) for row in family_rows])),
        "strong_minus_weak_edit_gain_mean": _aggregate([row["strong_minus_weak_edit_gain"] for row in family_rows]),
        "bg_mae_strong_minus_weak": _aggregate([row["bg_mae_strong_minus_weak"] for row in family_rows]),
        "preservation_pass_rate": float(np.mean([float(row["preservation_pass"]) for row in family_rows])),
    }

    payload = {
        "config": {
            "benchmark_path": str(benchmark_path),
            "model_path": model_path,
            "config_path": config_path,
            "max_families": max_families,
            "edit_steps": edit_steps,
            "sampler": sampler,
            "seed": seed,
            "device": device,
            "smooth_radius": smooth_radius,
            "bg_drift_threshold": bg_drift_threshold,
        },
        "summary": summary,
        "families": family_rows,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path = output_path.with_suffix(".md")
    lines = [
        "# Trend Monotonic Eval",
        "",
        f"- benchmark: `{benchmark_path}`",
        f"- num_families: {summary['num_families']}",
        f"- monotonic_hit_rate: {summary['monotonic_hit_rate']:.4f}",
        f"- strong_minus_weak_edit_gain_mean: {summary['strong_minus_weak_edit_gain_mean']}",
        f"- bg_mae_strong_minus_weak: {summary['bg_mae_strong_minus_weak']}",
        f"- preservation_pass_rate: {summary['preservation_pass_rate']:.4f}",
    ]
    if probe_gate is not None:
        lines.append(f"- probe_gate_pass: {probe_gate['pass']}")
        lines.append(f"- probe_diff_0_2_linf: {probe_gate['diff_0_2_linf']}")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Run trend-family monotonic evaluation for strength-conditioned TEdit.")
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-families", type=int, default=18)
    parser.add_argument("--edit-steps", type=int, default=10)
    parser.add_argument("--sampler", default="ddim")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--smooth-radius", type=float, default=3.0)
    parser.add_argument("--bg-drift-threshold", type=float, default=0.05)
    parser.add_argument("--probe-json", default="")
    args = parser.parse_args()

    payload = run_eval(
        benchmark_path=Path(args.benchmark),
        model_path=args.model_path,
        config_path=args.config_path,
        output_path=Path(args.output),
        max_families=args.max_families,
        edit_steps=args.edit_steps,
        sampler=args.sampler,
        seed=args.seed,
        device=args.device,
        smooth_radius=args.smooth_radius,
        bg_drift_threshold=args.bg_drift_threshold,
        probe_json=args.probe_json.strip() or None,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
