from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from test_scripts.build_event_driven_testset import (
    CSVDataLoader,
    EventDrivenSample,
    InjectorFactory,
)


DEFAULT_INJECTION_TYPES = [
    "trend_injection",
    "seasonality_injection",
    "step_change",
    "multiplier",
    "hard_zero",
    "noise_injection",
]

TASK_NAME_TO_ID = {
    "trend_up": 0,
    "trend_down": 1,
    "seasonality_neutral": 2,
    "volatility_neutral": 3,
    "mixed_neutral": 4,
}

DURATION_BUCKET_SPECS = {
    "short": (0.08, 0.12),
    "medium": (0.14, 0.20),
    "long": (0.24, 0.34),
}


def _template_prompt(feature_desc: str, effect_family: str, shape: str, duration_bucket: str) -> str:
    return (
        f"请把{feature_desc}在一个{duration_bucket}窗口里调整成更接近{effect_family}/{shape}的局部变化，"
        f"只改那一段，非编辑区尽量保持不变。"
    )


def _duration_from_bucket(seq_len: int, bucket: str, rng: np.random.RandomState) -> int:
    lo, hi = DURATION_BUCKET_SPECS[bucket]
    return max(6, int(round(seq_len * float(rng.uniform(lo, hi)))))


def _start_step_for_duration(seq_len: int, duration: int, rng: np.random.RandomState) -> int:
    margin = max(4, seq_len // 12)
    hi = max(margin + 1, seq_len - duration - margin)
    return int(rng.randint(margin, hi))


def _strength_bucket(
    *,
    injection_type: str,
    injection_config: Dict[str, Any],
    base_ts: np.ndarray,
) -> str:
    base = np.asarray(base_ts, dtype=np.float64)
    scale = max(float(np.std(base)), float(np.max(base) - np.min(base)) * 0.1, 1e-6)
    if injection_type == "trend_injection":
        value = abs(float(injection_config.get("amplitude", 0.0))) / scale
    elif injection_type == "seasonality_injection":
        value = abs(float(injection_config.get("seasonal_amplitude", 0.0))) / scale
    elif injection_type == "step_change":
        value = abs(float(injection_config.get("magnitude", 0.0))) / scale
    elif injection_type == "multiplier":
        value = abs(float(injection_config.get("multiplier", 1.0)) - 1.0)
    elif injection_type == "hard_zero":
        floor = float(injection_config.get("floor_value", np.min(base)))
        value = abs(float(np.mean(base)) - floor) / scale
    elif injection_type == "noise_injection":
        value = abs(float(injection_config.get("noise_std", 0.0))) / scale
    else:
        value = 1.0
    if value < 0.75:
        return "weak"
    if value < 1.5:
        return "medium"
    return "strong"


def _target_energy_type(base_ts: np.ndarray, target_ts: np.ndarray, start_step: int, end_step: int) -> str:
    delta = np.asarray(target_ts, dtype=np.float64) - np.asarray(base_ts, dtype=np.float64)
    local = delta[start_step : end_step + 1]
    if local.size == 0:
        return "none"
    peak = float(np.max(np.abs(local)))
    area = float(np.mean(np.abs(local)))
    if peak <= 1e-8:
        return "none"
    if area <= peak * 0.30:
        return "peak_dominant"
    if area >= peak * 0.60:
        return "area_dominant"
    return "mixed"


def _strength_label_id(bucket: str) -> int:
    return {"weak": 0, "medium": 1, "strong": 2}.get(str(bucket), 1)


def _task_id_from_intent(intent: Dict[str, Any]) -> int:
    effect_family = str(intent.get("effect_family", "unknown"))
    direction = str(intent.get("direction", "neutral"))
    if effect_family == "trend":
        return TASK_NAME_TO_ID["trend_down" if direction == "down" else "trend_up"]
    if effect_family == "seasonality":
        return TASK_NAME_TO_ID["seasonality_neutral"]
    if effect_family == "volatility":
        return TASK_NAME_TO_ID["volatility_neutral"]
    return TASK_NAME_TO_ID["mixed_neutral"]


def build_stress_benchmark(
    *,
    csv_path: str,
    dataset_name: str,
    output_dir: str,
    num_samples: int,
    seq_len: int,
    random_seed: int,
    injection_types: List[str] | None = None,
) -> Dict[str, Any]:
    rng = np.random.RandomState(random_seed)
    data_loader = CSVDataLoader(csv_path, dataset_name)
    injector_factory = InjectorFactory(random_seed)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    chosen_types = list(injection_types or DEFAULT_INJECTION_TYPES)
    per_type_base = num_samples // len(chosen_types)
    remainder = num_samples % len(chosen_types)
    target_counts = {
        injection_type: per_type_base + (1 if idx < remainder else 0)
        for idx, injection_type in enumerate(chosen_types)
    }

    samples: List[Dict[str, Any]] = []
    sample_idx = 1
    for injection_type in chosen_types:
        for local_idx in range(target_counts[injection_type]):
            duration_bucket = ["short", "medium", "long"][local_idx % 3]
            feature = str(rng.choice(data_loader.features))
            start_idx = int(rng.randint(0, max(1, len(data_loader.data) - seq_len - 1)))
            base_ts, _ = data_loader.get_sequence(start_idx, seq_len, feature)
            injector = injector_factory.create_injector(injection_type)
            duration = _duration_from_bucket(seq_len, duration_bucket, rng)
            start_step = _start_step_for_duration(seq_len, duration, rng)
            target_ts, mask_gt, injection_config = injector.inject(base_ts, start_step, duration)
            edit_intent_gt = injector.get_edit_intent(injection_config)
            edit_intent_gt["duration"] = duration_bucket
            edit_intent_gt["strength"] = _strength_bucket(
                injection_type=injection_type,
                injection_config=injection_config,
                base_ts=base_ts,
            )
            feature_desc = data_loader.feature_descriptions.get(feature, injector_factory.get_feature_description(feature))
            prompt_text = _template_prompt(
                feature_desc=feature_desc,
                effect_family=str(edit_intent_gt.get("effect_family", "unknown")),
                shape=str(edit_intent_gt.get("shape", "unknown")),
                duration_bucket=duration_bucket,
            )
            end_step = int(injection_config["end_step"])
            sample = EventDrivenSample(
                sample_id=f"stress_{sample_idx:03d}",
                dataset_name=dataset_name,
                target_feature=feature,
                feature_description=feature_desc,
                task_type=injector.get_task_type(),
                legacy_task_type=injector.get_legacy_task_type(),
                injection_operator=injector.get_injection_operator(),
                edit_intent_gt=edit_intent_gt,
                gt_start=int(injection_config["start_step"]),
                gt_end=end_step,
                event_prompts=[
                    {
                        "level": 0,
                        "level_name": "stress_template",
                        "perspective": "protocol",
                        "prompt": prompt_text,
                    }
                ],
                technical_ground_truth=f"[Stress Benchmark] {injection_type} {duration_bucket}",
                base_ts=np.asarray(base_ts, dtype=float).tolist(),
                target_ts=np.asarray(target_ts, dtype=float).tolist(),
                mask_gt=np.asarray(mask_gt, dtype=int).tolist(),
                injection_config=injection_config,
                causal_scenario=f"stress benchmark::{injection_type}::{duration_bucket}",
                seq_len=seq_len,
                timestamp=datetime.now().isoformat(),
            )
            sample_payload = asdict(sample)
            sample_payload["vague_prompt"] = prompt_text
            sample_payload["source_ts"] = sample_payload["base_ts"]
            sample_payload["instruction_text"] = prompt_text
            sample_payload["strength_label"] = _strength_label_id(edit_intent_gt["strength"])
            sample_payload["task_id"] = _task_id_from_intent(edit_intent_gt)
            sample_payload["region"] = [int(sample.gt_start), int(sample.gt_end)]
            sample_payload["stress_metadata"] = {
                "duration_bucket": duration_bucket,
                "strength_bucket": edit_intent_gt["strength"],
                "strength_label": sample_payload["strength_label"],
                "task_id": sample_payload["task_id"],
                "target_energy_type": _target_energy_type(base_ts, target_ts, sample.gt_start, sample.gt_end),
                "injection_type": injection_type,
            }
            samples.append(sample_payload)
            sample_idx += 1

    output_payload = {
        "benchmark_type": "pure_editing_how_much_stress",
        "dataset_name": dataset_name,
        "seq_len": seq_len,
        "num_samples": len(samples),
        "random_seed": random_seed,
        "injection_types": chosen_types,
        "tool_quota": target_counts,
        "samples": samples,
    }
    json_path = output_root / f"pure_editing_how_much_stress_{dataset_name}_{len(samples)}.json"
    json_path.write_text(json.dumps(output_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    summary_path = output_root / "README.md"
    summary_lines = [
        "# Pure Editing How-Much Stress Benchmark",
        "",
        f"- dataset_name: {dataset_name}",
        f"- num_samples: {len(samples)}",
        f"- seq_len: {seq_len}",
        f"- random_seed: {random_seed}",
        "",
        "## Tool Quota",
        "",
    ]
    for injection_type, count in target_counts.items():
        summary_lines.append(f"- `{injection_type}`: {count}")
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    return {"json_path": str(json_path), "summary_path": str(summary_path), "num_samples": len(samples)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a tool-balanced pure-editing how-much stress benchmark.")
    parser.add_argument("--csv-path", required=True)
    parser.add_argument("--dataset-name", default="ETTh1")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-samples", type=int, default=50)
    parser.add_argument("--seq-len", type=int, default=192)
    parser.add_argument("--random-seed", type=int, default=7)
    parser.add_argument("--injection-types", default="")
    args = parser.parse_args()

    injection_types = None
    if args.injection_types.strip():
        injection_types = [token.strip() for token in args.injection_types.split(",") if token.strip()]

    result = build_stress_benchmark(
        csv_path=args.csv_path,
        dataset_name=args.dataset_name,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        seq_len=args.seq_len,
        random_seed=args.random_seed,
        injection_types=injection_types,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
