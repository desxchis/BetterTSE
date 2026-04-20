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


LEGACY_STRENGTH_TO_SCALAR = {
    "weak": 0.0,
    "medium": 0.5,
    "strong": 1.0,
}
DEFAULT_SWEEP = [0.0, 0.25, 0.5, 0.75, 1.0]
FLOAT_TOL = 1.0e-6


def _compute_metrics(source_ts: np.ndarray, target_ts: np.ndarray, edited_ts: np.ndarray, mask_gt: np.ndarray) -> Dict[str, float | None]:
    edit_mask = mask_gt.astype(bool)
    bg_mask = ~edit_mask
    metrics: Dict[str, float | None] = {
        "edit_gain": None,
        "bg_mae": None,
        "background_leak_max": None,
        "target_mae_edit_region": None,
    }
    if np.any(edit_mask):
        metrics["edit_gain"] = float(np.mean(np.abs(edited_ts[edit_mask] - source_ts[edit_mask])))
        metrics["target_mae_edit_region"] = float(np.mean(np.abs(edited_ts[edit_mask] - target_ts[edit_mask])))
    if np.any(bg_mask):
        bg_abs = np.abs(edited_ts[bg_mask] - source_ts[bg_mask])
        metrics["bg_mae"] = float(np.mean(bg_abs))
        metrics["background_leak_max"] = float(np.max(bg_abs))
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


def _resolve_strength_scalar(sample: Dict[str, Any]) -> float:
    scalar = sample.get("strength_scalar")
    if scalar is not None:
        return float(scalar)
    strength_text = str(sample.get("strength_text", ""))
    if strength_text in LEGACY_STRENGTH_TO_SCALAR:
        return float(LEGACY_STRENGTH_TO_SCALAR[strength_text])
    raise ValueError(f"Sample missing strength_scalar and unknown strength_text={strength_text}")


def _average_tied_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < len(sorted_values):
        end = start + 1
        while end < len(sorted_values) and sorted_values[end] == sorted_values[start]:
            end += 1
        avg_rank = 0.5 * (start + end - 1)
        ranks[order[start:end]] = avg_rank
        start = end
    return ranks


def _spearman_rho(x: List[float], y: List[float]) -> float | None:
    if len(x) < 2 or len(y) < 2 or len(x) != len(y):
        return None
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    if np.allclose(x_arr, x_arr[0]) or np.allclose(y_arr, y_arr[0]):
        return None
    x_rank = _average_tied_ranks(x_arr)
    y_rank = _average_tied_ranks(y_arr)
    x_rank -= x_rank.mean()
    y_rank -= y_rank.mean()
    denom = float(np.sqrt(np.sum(x_rank * x_rank) * np.sum(y_rank * y_rank)))
    if denom <= 1.0e-12:
        return None
    return float(np.sum(x_rank * y_rank) / denom)


def _parse_sweep_values(raw: str) -> List[float]:
    if not raw.strip():
        return list(DEFAULT_SWEEP)
    values = sorted({float(token.strip()) for token in raw.split(",") if token.strip()})
    if not values:
        return list(DEFAULT_SWEEP)
    return values


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
    sweep_values: List[float],
) -> Dict[str, Any]:
    if not benchmark_path.exists() or not benchmark_path.is_file():
        raise ValueError(f"Benchmark file not found: {benchmark_path}")
    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
    all_families = [fam for fam in benchmark.get("families", []) if str(fam.get("tool_name")) == "trend_injection"]
    families = all_families[:max_families]
    if not families:
        raise ValueError("No trend_injection families found in benchmark.")
    if not model_path or not config_path:
        raise ValueError("model_path and config_path are required")

    required_keys = {"source_ts", "target_ts", "mask_gt", "region", "instruction_text"}
    required_anchor_scalars = {0.0, 0.5, 1.0}
    for family in families:
        samples = family.get("samples", [])
        if not isinstance(samples, list) or not samples:
            raise ValueError(f"Family {family.get('family_id')} has no samples")
        anchor_scalars = set()
        for sample in samples:
            missing_keys = sorted(required_keys - set(sample.keys()))
            if missing_keys:
                raise ValueError(f"Family {family.get('family_id')} sample missing keys {missing_keys}")
            anchor_scalars.add(float(_resolve_strength_scalar(sample)))
        if not required_anchor_scalars.issubset(anchor_scalars):
            raise ValueError(f"Family {family.get('family_id')} missing required anchors {sorted(required_anchor_scalars - anchor_scalars)}")

    wrapper = TEditWrapper(model_path=model_path, config_path=config_path, device=device)
    wrapper.set_edit_steps(edit_steps)

    family_rows: List[Dict[str, Any]] = []
    per_sweep_rows: Dict[str, List[Dict[str, Any]]] = {f"{value:.4f}": [] for value in sweep_values}

    for family_idx, family in enumerate(families):
        direction = _family_direction(family)
        src_attrs, tgt_attrs = _trend_attrs(direction)
        samples = [dict(sample) for sample in family.get("samples", [])]
        for sample in samples:
            sample["strength_scalar"] = _resolve_strength_scalar(sample)
        samples.sort(key=lambda sample: float(sample["strength_scalar"]))
        anchor_by_scalar = {float(sample["strength_scalar"]): sample for sample in samples}
        nearest_anchor = [min(anchor_by_scalar.keys(), key=lambda anchor: abs(anchor - value)) for value in sweep_values]

        source_ts = np.asarray(samples[0]["source_ts"], dtype=np.float32)
        mask_gt = np.asarray(samples[0]["mask_gt"], dtype=np.int64)
        region = samples[0]["region"]
        if not isinstance(region, (list, tuple)) or len(region) != 2:
            raise ValueError(f"Family {family.get('family_id')} has invalid region={region}")
        start_idx, end_idx = [int(v) for v in region]
        if not (0 <= start_idx < end_idx <= len(source_ts)):
            raise ValueError(f"Family {family.get('family_id')} has out-of-range region={region}")
        instruction_text = str(samples[len(samples) // 2]["instruction_text"])

        row: Dict[str, Any] = {
            "family_id": family.get("family_id"),
            "tool_name": family.get("tool_name"),
            "direction": direction,
            "duration_bucket": str(samples[0].get("duration_bucket", family.get("duration_bucket", "unknown"))),
            "region": family.get("region"),
            "anchor_strength_scalar": [float(sample["strength_scalar"]) for sample in samples],
            "anchor_target_edit_gain": [
                float(np.mean(np.abs(np.asarray(sample["target_ts"], dtype=np.float32)[mask_gt.astype(bool)] - source_ts[mask_gt.astype(bool)])))
                for sample in samples
            ],
            "sweep": [],
        }

        sweep_gains: List[float] = []
        sweep_scalars: List[float] = []
        sweep_bg: List[float] = []
        target_mae_values: List[float] = []

        for sweep_idx, (strength_scalar, anchor_scalar) in enumerate(zip(sweep_values, nearest_anchor)):
            anchor_sample = anchor_by_scalar[anchor_scalar]
            target_ts = np.asarray(anchor_sample["target_ts"], dtype=np.float32)

            torch.manual_seed(seed + family_idx * 97 + sweep_idx)
            np.random.seed(seed + family_idx * 97 + sweep_idx)
            edited_ts = wrapper.edit_region_soft(
                ts=source_ts,
                start_idx=start_idx,
                end_idx=end_idx,
                src_attrs=src_attrs,
                tgt_attrs=tgt_attrs,
                n_samples=1,
                sampler=sampler,
                smooth_radius=smooth_radius,
                strength_scalar=float(strength_scalar),
                task_id=None,
                instruction_text=instruction_text,
            )
            edited_ts = np.asarray(edited_ts, dtype=np.float32).reshape(-1)
            if edited_ts.shape != source_ts.shape:
                raise RuntimeError(f"Family {family.get('family_id')} output shape {edited_ts.shape} != input shape {source_ts.shape}")
            metrics = _compute_metrics(source_ts, target_ts, edited_ts, mask_gt)
            if metrics["edit_gain"] is not None:
                sweep_scalars.append(float(strength_scalar))
                sweep_gains.append(float(metrics["edit_gain"]))
            if metrics["bg_mae"] is not None:
                sweep_bg.append(float(metrics["bg_mae"]))
            if metrics["target_mae_edit_region"] is not None:
                target_mae_values.append(float(metrics["target_mae_edit_region"]))
            record = {
                "family_id": family.get("family_id"),
                "strength_scalar": float(strength_scalar),
                "nearest_anchor_scalar": float(anchor_scalar),
                "instruction_text": instruction_text,
                **metrics,
            }
            per_sweep_rows[f"{strength_scalar:.4f}"].append(record)
            row["sweep"].append(record)

        adjacent_monotonic = bool(
            len(sweep_gains) >= 2
            and all((sweep_gains[idx + 1] + FLOAT_TOL) >= sweep_gains[idx] for idx in range(len(sweep_gains) - 1))
        )
        rho = _spearman_rho(sweep_scalars, sweep_gains)
        gain_range = None if len(sweep_gains) < 2 else float(sweep_gains[-1] - sweep_gains[0])
        off_anchor_values = [value for value in sweep_values if all(abs(value - anchor) > FLOAT_TOL for anchor in anchor_by_scalar.keys())]
        off_anchor_records = [record for record in row["sweep"] if any(abs(record["strength_scalar"] - value) <= FLOAT_TOL for value in off_anchor_values)]
        off_anchor_monotonic = bool(
            len(off_anchor_records) < 2
            or all(
                (off_anchor_records[idx + 1]["edit_gain"] is not None)
                and (off_anchor_records[idx]["edit_gain"] is not None)
                and (off_anchor_records[idx + 1]["edit_gain"] + FLOAT_TOL) >= off_anchor_records[idx]["edit_gain"]
                for idx in range(len(off_anchor_records) - 1)
            )
        )

        row["adjacent_monotonic_pass"] = adjacent_monotonic
        row["spearman_rho_strength_gain"] = rho
        row["gain_range"] = gain_range
        row["bg_mae_mean"] = _aggregate(sweep_bg)
        row["target_mae_edit_region_mean"] = _aggregate(target_mae_values)
        row["preservation_pass"] = bool((row["bg_mae_mean"] is not None) and (row["bg_mae_mean"] <= bg_drift_threshold))
        row["off_anchor_monotonic_pass"] = off_anchor_monotonic
        family_rows.append(row)

    probe_gate = None
    if probe_json:
        probe = json.loads(Path(probe_json).read_text(encoding="utf-8"))
        probe_gate = {
            "probe_path": probe_json,
            "diff_0_2_linf": float(probe.get("diff_0_2_linf", 0.0)),
            "pass": bool(float(probe.get("diff_0_2_linf", 0.0)) > 0.0),
        }

    duration_bucket_rows: Dict[str, List[Dict[str, Any]]] = {}
    for row in family_rows:
        duration_bucket_rows.setdefault(str(row.get("duration_bucket", "unknown")), []).append(row)

    summary = {
        "num_families": len(family_rows),
        "family_filter": "trend_injection",
        "probe_gate": probe_gate,
        "sweep_values": sweep_values,
        "edit_gain_mean": {
            strength_scalar: _aggregate([row["edit_gain"] for row in per_sweep_rows[strength_scalar]])
            for strength_scalar in sorted(per_sweep_rows.keys())
        },
        "bg_mae_mean": {
            strength_scalar: _aggregate([row["bg_mae"] for row in per_sweep_rows[strength_scalar]])
            for strength_scalar in sorted(per_sweep_rows.keys())
        },
        "target_mae_edit_region_mean": {
            strength_scalar: _aggregate([row["target_mae_edit_region"] for row in per_sweep_rows[strength_scalar]])
            for strength_scalar in sorted(per_sweep_rows.keys())
        },
        "adjacent_monotonic_pass_rate": float(np.mean([float(row["adjacent_monotonic_pass"]) for row in family_rows])),
        "off_anchor_monotonic_pass_rate": float(np.mean([float(row["off_anchor_monotonic_pass"]) for row in family_rows])),
        "gain_range_mean": _aggregate([row["gain_range"] for row in family_rows]),
        "family_spearman_rho_mean": _aggregate([row["spearman_rho_strength_gain"] for row in family_rows]),
        "preservation_pass_rate": float(np.mean([float(row["preservation_pass"]) for row in family_rows])),
        "duration_bucket_summary": {
            bucket: {
                "num_families": len(bucket_rows),
                "adjacent_monotonic_pass_rate": float(np.mean([float(item["adjacent_monotonic_pass"]) for item in bucket_rows])),
                "off_anchor_monotonic_pass_rate": float(np.mean([float(item["off_anchor_monotonic_pass"]) for item in bucket_rows])),
                "gain_range_mean": _aggregate([item["gain_range"] for item in bucket_rows]),
                "family_spearman_rho_mean": _aggregate([item["spearman_rho_strength_gain"] for item in bucket_rows]),
                "bg_mae_mean": _aggregate([item["bg_mae_mean"] for item in bucket_rows]),
                "preservation_pass_rate": float(np.mean([float(item["preservation_pass"]) for item in bucket_rows])),
            }
            for bucket, bucket_rows in sorted(duration_bucket_rows.items())
        },
    }
    summary["minimum_usable_gain_range_mean"] = summary["gain_range_mean"]
    summary["family_spearman_rho_strength_gain_mean"] = summary["family_spearman_rho_mean"]

    if len(family_rows) != int(summary["num_families"]):
        raise RuntimeError(f"Family count mismatch: rows={len(family_rows)} summary={summary['num_families']}")
    if int(summary["num_families"]) <= 0:
        raise RuntimeError("Monotonic eval produced zero families")
    for metric_key in ("adjacent_monotonic_pass_rate", "off_anchor_monotonic_pass_rate", "preservation_pass_rate"):
        if summary.get(metric_key) is None:
            raise RuntimeError(f"Missing summary metric {metric_key}")

    payload = {
        "status": {
            "ok": True,
            "stage": "run_tedit_trend_monotonic_eval",
        },
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
            "sweep_values": sweep_values,
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
        f"- adjacent_monotonic_pass_rate: {summary['adjacent_monotonic_pass_rate']:.4f}",
        f"- off_anchor_monotonic_pass_rate: {summary['off_anchor_monotonic_pass_rate']:.4f}",
        f"- gain_range_mean: {summary['gain_range_mean']}",
        f"- family_spearman_rho_mean: {summary['family_spearman_rho_mean']}",
        f"- preservation_pass_rate: {summary['preservation_pass_rate']:.4f}",
    ]
    if probe_gate is not None:
        lines.append(f"- probe_gate_pass: {probe_gate['pass']}")
        lines.append(f"- probe_diff_0_2_linf: {probe_gate['diff_0_2_linf']}")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Run scalar trend-family monotonic evaluation for strength-conditioned TEdit.")
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
    parser.add_argument("--probe-json", default=None)
    parser.add_argument("--sweep-values", default=",".join(str(value) for value in DEFAULT_SWEEP))
    args = parser.parse_args()

    payload = run_eval(
        benchmark_path=Path(args.benchmark),
        model_path=args.model_path,
        config_path=args.config_path,
        output_path=Path(args.output),
        max_families=int(args.max_families),
        edit_steps=int(args.edit_steps),
        sampler=str(args.sampler),
        seed=int(args.seed),
        device=str(args.device),
        smooth_radius=float(args.smooth_radius),
        bg_drift_threshold=float(args.bg_drift_threshold),
        probe_json=args.probe_json,
        sweep_values=_parse_sweep_values(args.sweep_values),
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
