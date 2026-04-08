from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


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


def check_benchmark(path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    families = list(data.get("families", []))
    strengths_expected = ["weak", "medium", "strong"]

    family_rows: List[Dict[str, Any]] = []
    all_bg_leaks: List[float | None] = []
    tool_counts: Dict[str, int] = {}
    duration_counts: Dict[str, int] = {}

    complete_strength_count = 0
    monotonic_target_count = 0
    zero_bg_leak_count = 0

    for family in families:
        samples = list(family.get("samples", []))
        strength_map = {str(sample.get("strength_text")): sample for sample in samples}
        strengths = [s for s in strengths_expected if s in strength_map]
        complete_strength = strengths == strengths_expected
        if complete_strength:
            complete_strength_count += 1

        gains: List[float | None] = []
        bg_leaks: List[float | None] = []
        edit_region_fractions: List[float] = []
        for strength_text in strengths_expected:
            sample = strength_map.get(strength_text)
            if sample is None:
                gains.append(None)
                bg_leaks.append(None)
                continue
            source_ts = np.asarray(sample["source_ts"], dtype=np.float32)
            target_ts = np.asarray(sample["target_ts"], dtype=np.float32)
            mask_gt = np.asarray(sample["mask_gt"], dtype=np.int64)
            gains.append(_edit_gain(source_ts, target_ts, mask_gt))
            bg_leak = _background_leakage(source_ts, target_ts, mask_gt)
            bg_leaks.append(bg_leak)
            all_bg_leaks.append(bg_leak)
            edit_region_fractions.append(float(np.mean(mask_gt.astype(np.float32))))

        monotonic_target = bool(
            complete_strength
            and gains[0] is not None
            and gains[1] is not None
            and gains[2] is not None
            and gains[0] < gains[1] < gains[2]
        )
        if monotonic_target:
            monotonic_target_count += 1

        family_bg_leak = None if all(v is None for v in bg_leaks) else float(max(v for v in bg_leaks if v is not None))
        if family_bg_leak is not None and family_bg_leak <= 1e-6:
            zero_bg_leak_count += 1

        tool_name = str(family.get("tool_name", "unknown"))
        duration_bucket = str(family.get("duration_bucket", "unknown"))
        tool_counts[tool_name] = tool_counts.get(tool_name, 0) + 1
        duration_counts[duration_bucket] = duration_counts.get(duration_bucket, 0) + 1

        family_rows.append(
            {
                "family_id": family.get("family_id"),
                "tool_name": tool_name,
                "duration_bucket": duration_bucket,
                "strengths": strengths,
                "complete_strength": complete_strength,
                "target_edit_gain": {
                    "weak": gains[0],
                    "medium": gains[1],
                    "strong": gains[2],
                },
                "target_monotonic": monotonic_target,
                "edit_region_fraction_mean": None if not edit_region_fractions else float(np.mean(edit_region_fractions)),
                "background_leak_max": family_bg_leak,
            }
        )

    summary = {
        "benchmark_type": data.get("benchmark_type"),
        "num_families": len(families),
        "num_samples": int(data.get("num_samples", 0)),
        "complete_strength_rate": float(complete_strength_count / max(1, len(families))),
        "target_monotonic_rate": float(monotonic_target_count / max(1, len(families))),
        "zero_background_leak_rate": float(zero_bg_leak_count / max(1, len(families))),
        "background_leak": _summarize(all_bg_leaks),
        "tool_counts": tool_counts,
        "duration_counts": duration_counts,
        "health_pass": bool(
            len(families) > 0
            and complete_strength_count == len(families)
            and monotonic_target_count == len(families)
            and all((v is None or v <= 1e-6) for v in all_bg_leaks)
        ),
    }
    return {
        "benchmark_path": str(path),
        "summary": summary,
        "families": family_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run health checks on a discrete weak/medium/strong benchmark.")
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
        f"- complete_strength_rate: {s['complete_strength_rate']:.4f}",
        f"- target_monotonic_rate: {s['target_monotonic_rate']:.4f}",
        f"- zero_background_leak_rate: {s['zero_background_leak_rate']:.4f}",
        f"- background_leak_max: {s['background_leak']['max']}",
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
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
