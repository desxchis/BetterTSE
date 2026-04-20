from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


LEGACY_STRENGTH_TO_SCALAR = {
    "weak": 0.0,
    "medium": 0.5,
    "strong": 1.0,
}
FLOAT_TOL = 1.0e-6


def _edit_gain(source_ts: np.ndarray, target_ts: np.ndarray, mask_gt: np.ndarray) -> float | None:
    edit_mask = mask_gt.astype(bool)
    if not np.any(edit_mask):
        return None
    return float(np.mean(np.abs(target_ts[edit_mask] - source_ts[edit_mask])))


def _background_leakage(source_ts: np.ndarray, target_ts: np.ndarray, mask_gt: np.ndarray) -> float | None:
    bg_mask = ~mask_gt.astype(bool)
    if not np.any(bg_mask):
        return None
    return float(np.max(np.abs(target_ts[bg_mask] - source_ts[bg_mask])))


def _summarize(values: List[float | None]) -> Dict[str, float | None]:
    filtered = [float(v) for v in values if v is not None]
    if not filtered:
        return {"mean": None, "min": None, "max": None}
    return {
        "mean": float(np.mean(filtered)),
        "min": float(np.min(filtered)),
        "max": float(np.max(filtered)),
    }


def _resolve_strength_scalar(sample: Dict[str, Any]) -> float:
    scalar = sample.get("strength_scalar")
    if scalar is not None:
        return float(scalar)
    strength_text = str(sample.get("strength_text", ""))
    if strength_text in LEGACY_STRENGTH_TO_SCALAR:
        return float(LEGACY_STRENGTH_TO_SCALAR[strength_text])
    raise ValueError(f"Sample missing strength_scalar and unknown strength_text={strength_text}")


def _spearman_rho(x: List[float], y: List[float]) -> float | None:
    if len(x) < 2 or len(y) < 2 or len(x) != len(y):
        return None
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    x_rank = np.argsort(np.argsort(x_arr)).astype(np.float64)
    y_rank = np.argsort(np.argsort(y_arr)).astype(np.float64)
    x_rank -= x_rank.mean()
    y_rank -= y_rank.mean()
    denom = float(np.sqrt(np.sum(x_rank * x_rank) * np.sum(y_rank * y_rank)))
    if denom <= 1.0e-12:
        return None
    return float(np.sum(x_rank * y_rank) / denom)


def check_benchmark(path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    families = list(data.get("families", []))

    family_rows: List[Dict[str, Any]] = []
    all_bg_leaks: List[float | None] = []
    tool_counts: Dict[str, int] = {}
    duration_counts: Dict[str, int] = {}
    noise_subtype_counts: Dict[str, int] = {}
    scalar_coverages: List[float] = []
    family_rhos: List[float] = []

    scalar_order_valid_count = 0
    monotonic_target_count = 0
    zero_bg_leak_count = 0
    family_valid_count = 0

    for family in families:
        samples = [dict(sample) for sample in family.get("samples", [])]
        for sample in samples:
            sample["strength_scalar"] = _resolve_strength_scalar(sample)
        samples.sort(key=lambda sample: float(sample["strength_scalar"]))
        strengths = [float(sample["strength_scalar"]) for sample in samples]
        scalar_order_valid = len(strengths) >= 2 and all((right - left) > FLOAT_TOL for left, right in zip(strengths[:-1], strengths[1:]))
        if scalar_order_valid:
            scalar_order_valid_count += 1
        if strengths:
            scalar_coverages.append(float(max(strengths) - min(strengths)))

        gains: List[float | None] = []
        bg_leaks: List[float | None] = []
        edit_region_fractions: List[float] = []
        for sample in samples:
            source_ts = np.asarray(sample["source_ts"], dtype=np.float32)
            target_ts = np.asarray(sample["target_ts"], dtype=np.float32)
            mask_gt = np.asarray(sample["mask_gt"], dtype=np.int64)
            gains.append(_edit_gain(source_ts, target_ts, mask_gt))
            bg_leak = _background_leakage(source_ts, target_ts, mask_gt)
            bg_leaks.append(bg_leak)
            all_bg_leaks.append(bg_leak)
            edit_region_fractions.append(float(np.mean(mask_gt.astype(np.float32))))

        valid_gain_pairs = [(s, g) for s, g in zip(strengths, gains) if g is not None]
        monotonic_target = bool(
            scalar_order_valid
            and len(valid_gain_pairs) >= 2
            and all((valid_gain_pairs[idx + 1][1] + FLOAT_TOL) >= valid_gain_pairs[idx][1] for idx in range(len(valid_gain_pairs) - 1))
        )
        if monotonic_target:
            monotonic_target_count += 1

        family_bg_leak = None if all(v is None for v in bg_leaks) else float(max(v for v in bg_leaks if v is not None))
        if family_bg_leak is not None and family_bg_leak <= 1e-6:
            zero_bg_leak_count += 1

        rho = _spearman_rho(
            [pair[0] for pair in valid_gain_pairs],
            [pair[1] for pair in valid_gain_pairs],
        )
        if rho is not None:
            family_rhos.append(rho)

        family_valid = bool(scalar_order_valid and monotonic_target and family_bg_leak is not None and family_bg_leak <= 1e-6)
        if family_valid:
            family_valid_count += 1

        tool_name = str(family.get("tool_name", "unknown"))
        duration_bucket = str(family.get("duration_bucket", "unknown"))
        noise_subtype = str(family.get("noise_subtype", "") or "")
        tool_counts[tool_name] = tool_counts.get(tool_name, 0) + 1
        duration_counts[duration_bucket] = duration_counts.get(duration_bucket, 0) + 1
        if noise_subtype:
            noise_subtype_counts[noise_subtype] = noise_subtype_counts.get(noise_subtype, 0) + 1

        family_rows.append(
            {
                "family_id": family.get("family_id"),
                "tool_name": tool_name,
                "duration_bucket": duration_bucket,
                "noise_subtype": noise_subtype or None,
                "strength_scalar": strengths,
                "num_samples": len(samples),
                "scalar_order_valid": scalar_order_valid,
                "target_edit_gain_by_scalar": [
                    {"strength_scalar": float(s), "edit_gain": None if g is None else float(g)}
                    for s, g in zip(strengths, gains)
                ],
                "target_monotonic": monotonic_target,
                "edit_region_fraction_mean": None if not edit_region_fractions else float(np.mean(edit_region_fractions)),
                "background_leak_max": family_bg_leak,
                "spearman_rho_strength_gain": rho,
                "family_valid": family_valid,
            }
        )

    summary = {
        "benchmark_type": data.get("benchmark_type"),
        "num_families": len(families),
        "num_samples": int(data.get("num_samples", 0)),
        "family_valid_rate": float(family_valid_count / max(1, len(families))),
        "scalar_order_valid_rate": float(scalar_order_valid_count / max(1, len(families))),
        "target_monotonic_rate": float(monotonic_target_count / max(1, len(families))),
        "zero_background_leak_rate": float(zero_bg_leak_count / max(1, len(families))),
        "scalar_coverage": _summarize(scalar_coverages),
        "background_leak": _summarize(all_bg_leaks),
        "family_spearman_rho": _summarize(family_rhos),
        "tool_counts": tool_counts,
        "duration_counts": duration_counts,
        "noise_subtype_counts": noise_subtype_counts,
        "health_pass": bool(
            len(families) > 0
            and family_valid_count == len(families)
        ),
    }
    return {
        "benchmark_path": str(path),
        "summary": summary,
        "families": family_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run health checks on scalar-ordered TEdit strength families.")
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    report = check_benchmark(Path(args.benchmark))
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    md_path = output_path.with_suffix(".md")
    s = report["summary"]
    lines = [
        "# Benchmark Health",
        "",
        f"- benchmark: `{args.benchmark}`",
        f"- num_families: {s['num_families']}",
        f"- family_valid_rate: {s['family_valid_rate']:.4f}",
        f"- scalar_order_valid_rate: {s['scalar_order_valid_rate']:.4f}",
        f"- target_monotonic_rate: {s['target_monotonic_rate']:.4f}",
        f"- zero_background_leak_rate: {s['zero_background_leak_rate']:.4f}",
        f"- scalar_coverage_max: {s['scalar_coverage']['max']}",
        f"- background_leak_max: {s['background_leak']['max']}",
        f"- family_spearman_rho_mean: {s['family_spearman_rho']['mean']}",
        f"- health_pass: {s['health_pass']}",
        "",
        "## Tool Counts",
        "",
    ]
    for key, value in sorted(s["tool_counts"].items()):
        lines.append(f"- `{key}`: {value}")
    lines.extend(["", "## Duration Counts", ""])
    for key, value in sorted(s["duration_counts"].items()):
        lines.append(f"- `{key}`: {value}")
    if s["noise_subtype_counts"]:
        lines.extend(["", "## Noise Subtype Counts", ""])
        for key, value in sorted(s["noise_subtype_counts"].items()):
            lines.append(f"- `{key}`: {value}")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
