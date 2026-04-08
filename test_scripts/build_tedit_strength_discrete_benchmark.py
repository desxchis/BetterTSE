from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from test_scripts.build_event_driven_testset import CSVDataLoader, InjectorFactory


DEFAULT_INJECTION_TYPES = [
    "trend_injection",
    "step_change",
    "multiplier",
    "hard_zero",
    "noise_injection",
]

STRENGTH_TEXT_TO_LABEL = {
    "weak": 0,
    "medium": 1,
    "strong": 2,
}

STRENGTH_ZH = {
    "weak": "轻微",
    "medium": "中等",
    "strong": "明显",
}

DURATION_BUCKET_SPECS = {
    "short": (0.08, 0.12),
    "medium": (0.14, 0.20),
    "long": (0.24, 0.34),
}

STRENGTH_PARAMETER_SPECS = {
    "trend_injection": {
        "weak": {"amplitude_ratio": 0.12},
        "medium": {"amplitude_ratio": 0.24},
        "strong": {"amplitude_ratio": 0.36},
    },
    "step_change": {
        "weak": {"magnitude_ratio": 0.18},
        "medium": {"magnitude_ratio": 0.36},
        "strong": {"magnitude_ratio": 0.60},
    },
    "multiplier": {
        "weak": {"multiplier": 1.25},
        "medium": {"multiplier": 1.75},
        "strong": {"multiplier": 2.40},
    },
    "hard_zero": {
        "weak": {"floor_percentile": 30.0},
        "medium": {"floor_percentile": 12.0},
        "strong": {"floor_percentile": 3.0},
    },
    "noise_injection": {
        "weak": {"noise_std_ratio": 0.9},
        "medium": {"noise_std_ratio": 1.6},
        "strong": {"noise_std_ratio": 2.4},
    },
}

TASK_NAME_TO_ID = {
    "trend_up": 0,
    "trend_down": 1,
    "seasonality_neutral": 2,
    "volatility_neutral": 3,
    "mixed_neutral": 4,
}


def _duration_from_bucket(seq_len: int, bucket: str, rng: np.random.RandomState) -> int:
    lo, hi = DURATION_BUCKET_SPECS[bucket]
    return max(6, int(round(seq_len * float(rng.uniform(lo, hi)))))


def _start_step_for_duration(seq_len: int, duration: int, rng: np.random.RandomState) -> int:
    margin = max(4, seq_len // 12)
    hi = max(margin + 1, seq_len - duration - margin)
    return int(rng.randint(margin, hi))


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


def _target_energy_type(base_ts: np.ndarray, target_ts: np.ndarray, start_step: int, end_step: int) -> str:
    delta = np.asarray(target_ts, dtype=np.float64) - np.asarray(base_ts, dtype=np.float64)
    local = delta[start_step:end_step]
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


def _prompt_text(feature_desc: str, effect_family: str, shape: str, duration_bucket: str, strength_text: str) -> str:
    return (
        f"请把{feature_desc}在一个{duration_bucket}窗口里做{STRENGTH_ZH[strength_text]}的"
        f"{effect_family}/{shape}局部变化，只改那一段，非编辑区尽量保持不变。"
    )


def _build_multiplier_target(base_ts: np.ndarray, start_step: int, duration: int, multiplier: float) -> tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    target_ts = base_ts.copy()
    mask_gt = np.zeros(len(base_ts), dtype=np.int64)
    end_step = min(start_step + duration, len(base_ts))
    region_len = end_step - start_step
    ramp_out = min(max(3, region_len // 5), 10)
    scale = np.ones(region_len, dtype=np.float64) * float(multiplier)
    scale[-ramp_out:] = np.linspace(float(multiplier), 1.0, ramp_out)
    target_ts[start_step:end_step] = base_ts[start_step:end_step] * scale
    mask_gt[start_step:end_step] = 1
    return target_ts, mask_gt, {
        "injection_type": "multiplier",
        "multiplier": float(multiplier),
        "ramp_out": int(ramp_out),
        "start_step": int(start_step),
        "end_step": int(end_step),
        "duration": int(duration),
    }


def _build_hard_zero_target(base_ts: np.ndarray, start_step: int, duration: int, floor_percentile: float) -> tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    target_ts = base_ts.copy()
    mask_gt = np.zeros(len(base_ts), dtype=np.int64)
    end_step = min(start_step + duration, len(base_ts))
    region_len = end_step - start_step
    ramp = min(5, max(1, region_len // 6))
    floor_value = float(np.percentile(base_ts, floor_percentile))
    target_ts[start_step:end_step] = floor_value
    entry_val = float(base_ts[start_step])
    target_ts[start_step:start_step + ramp] = np.linspace(entry_val, floor_value, ramp)
    exit_val = float(base_ts[end_step]) if end_step < len(base_ts) else float(base_ts[-1])
    target_ts[end_step - ramp:end_step] = np.linspace(floor_value, exit_val, ramp)
    mask_gt[start_step:end_step] = 1
    return target_ts, mask_gt, {
        "injection_type": "hard_zero",
        "floor_value": floor_value,
        "floor_percentile": float(floor_percentile),
        "ramp": int(ramp),
        "start_step": int(start_step),
        "end_step": int(end_step),
        "duration": int(duration),
    }


def _build_noise_target(
    base_ts: np.ndarray,
    start_step: int,
    duration: int,
    noise_std_ratio: float,
    baseline_ratio: float,
    family_noise: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    target_ts = base_ts.copy()
    mask_gt = np.zeros(len(base_ts), dtype=np.int64)
    end_step = min(start_step + duration, len(base_ts))
    data_std = max(float(np.std(base_ts)), 1e-6)
    data_min = float(np.min(base_ts))
    data_range = float(np.max(base_ts)) - data_min
    baseline = data_min + data_range * float(baseline_ratio)
    noise_std = data_std * float(noise_std_ratio)
    target_ts[start_step:end_step] = baseline + family_noise[: end_step - start_step] * noise_std
    mask_gt[start_step:end_step] = 1
    return target_ts, mask_gt, {
        "injection_type": "noise_injection",
        "baseline": baseline,
        "baseline_ratio": float(baseline_ratio),
        "noise_std": noise_std,
        "noise_std_ratio": float(noise_std_ratio),
        "start_step": int(start_step),
        "end_step": int(end_step),
        "duration": int(duration),
    }


def _build_trend_target(base_ts: np.ndarray, start_step: int, duration: int, amplitude_ratio: float, direction: str) -> tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    target_ts = base_ts.copy()
    mask_gt = np.zeros(len(base_ts), dtype=np.int64)
    end_step = min(start_step + duration, len(base_ts))
    region_len = end_step - start_step
    data_range = max(float(np.max(base_ts)) - float(np.min(base_ts)), 1e-6)
    amplitude = data_range * float(amplitude_ratio)
    if direction == "downward":
        amplitude = -abs(amplitude)
    else:
        amplitude = abs(amplitude)
    t = np.linspace(0.0, 2.0 * np.pi, region_len)
    hump = (1.0 - np.cos(t)) / 2.0
    target_ts[start_step:end_step] += hump * amplitude
    mask_gt[start_step:end_step] = 1
    return target_ts, mask_gt, {
        "injection_type": "trend_injection",
        "direction": direction,
        "amplitude": float(amplitude),
        "amplitude_ratio": float(amplitude_ratio),
        "start_step": int(start_step),
        "end_step": int(end_step),
        "duration": int(duration),
    }


def _build_step_target(base_ts: np.ndarray, start_step: int, duration: int, magnitude_ratio: float, direction: str) -> tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    target_ts = base_ts.copy()
    mask_gt = np.zeros(len(base_ts), dtype=np.int64)
    end_step = min(start_step + duration, len(base_ts))
    region_len = end_step - start_step
    data_range = max(float(np.max(base_ts)) - float(np.min(base_ts)), 1e-6)
    step_magnitude = data_range * float(magnitude_ratio)
    if direction == "down":
        step_magnitude = -abs(step_magnitude)
    else:
        step_magnitude = abs(step_magnitude)
    ramp_out = min(max(3, region_len // 3), 20)
    magnitude_array = np.ones(region_len, dtype=np.float64) * step_magnitude
    magnitude_array[-ramp_out:] = np.linspace(step_magnitude, 0.0, ramp_out)
    target_ts[start_step:end_step] += magnitude_array
    mask_gt[start_step:end_step] = 1
    return target_ts, mask_gt, {
        "injection_type": "step_change",
        "direction": direction,
        "magnitude": float(step_magnitude),
        "magnitude_ratio": float(magnitude_ratio),
        "ramp_out": int(ramp_out),
        "start_step": int(start_step),
        "end_step": int(end_step),
        "duration": int(duration),
    }


def _build_family_variant(
    injection_type: str,
    strength_text: str,
    base_ts: np.ndarray,
    start_step: int,
    duration: int,
    shared_state: Dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    spec = STRENGTH_PARAMETER_SPECS[injection_type][strength_text]
    if injection_type == "multiplier":
        return _build_multiplier_target(base_ts, start_step, duration, multiplier=spec["multiplier"])
    if injection_type == "hard_zero":
        return _build_hard_zero_target(base_ts, start_step, duration, floor_percentile=spec["floor_percentile"])
    if injection_type == "noise_injection":
        return _build_noise_target(
            base_ts,
            start_step,
            duration,
            noise_std_ratio=spec["noise_std_ratio"],
            baseline_ratio=float(shared_state["baseline_ratio"]),
            family_noise=np.asarray(shared_state["family_noise"], dtype=np.float64),
        )
    if injection_type == "trend_injection":
        return _build_trend_target(
            base_ts,
            start_step,
            duration,
            amplitude_ratio=spec["amplitude_ratio"],
            direction=str(shared_state["direction"]),
        )
    if injection_type == "step_change":
        return _build_step_target(
            base_ts,
            start_step,
            duration,
            magnitude_ratio=spec["magnitude_ratio"],
            direction=str(shared_state["direction"]),
        )
    raise ValueError(f"Unsupported injection_type: {injection_type}")


def build_discrete_benchmark(
    *,
    csv_path: str,
    dataset_name: str,
    output_dir: str,
    num_families: int,
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
    strengths = ["weak", "medium", "strong"]

    families: List[Dict[str, Any]] = []
    flat_samples: List[Dict[str, Any]] = []
    for family_idx in range(num_families):
        injection_type = chosen_types[family_idx % len(chosen_types)]
        duration_bucket = ["short", "medium", "long"][family_idx % 3]
        feature = str(rng.choice(data_loader.features))
        start_idx = int(rng.randint(0, max(1, len(data_loader.data) - seq_len - 1)))
        base_ts, _ = data_loader.get_sequence(start_idx, seq_len, feature)
        duration = _duration_from_bucket(seq_len, duration_bucket, rng)
        start_step = _start_step_for_duration(seq_len, duration, rng)

        shared_state: Dict[str, Any] = {}
        if injection_type == "trend_injection":
            shared_state["direction"] = str(rng.choice(["upward", "downward"]))
        elif injection_type == "step_change":
            shared_state["direction"] = str(rng.choice(["up", "down"]))
        elif injection_type == "noise_injection":
            shared_state["baseline_ratio"] = float(rng.uniform(0.02, 0.08))
            shared_state["family_noise"] = rng.normal(loc=0.0, scale=1.0, size=duration)

        injector = injector_factory.create_injector(injection_type)
        feature_desc = data_loader.feature_descriptions.get(feature, injector_factory.get_feature_description(feature))

        family_samples: List[Dict[str, Any]] = []
        for strength_text in strengths:
            target_ts, mask_gt, injection_config = _build_family_variant(
                injection_type=injection_type,
                strength_text=strength_text,
                base_ts=np.asarray(base_ts, dtype=np.float64),
                start_step=start_step,
                duration=duration,
                shared_state=shared_state,
            )
            intent = injector.get_edit_intent(injection_config)
            intent["duration"] = duration_bucket
            intent["strength"] = strength_text
            prompt_text = _prompt_text(
                feature_desc=feature_desc,
                effect_family=str(intent.get("effect_family", "unknown")),
                shape=str(intent.get("shape", "unknown")),
                duration_bucket=duration_bucket,
                strength_text=strength_text,
            )
            sample_payload = {
                "family_id": f"family_{family_idx + 1:03d}",
                "sample_id": f"family_{family_idx + 1:03d}_{strength_text}",
                "dataset_name": dataset_name,
                "target_feature": feature,
                "feature_description": feature_desc,
                "source_ts": np.asarray(base_ts, dtype=float).tolist(),
                "target_ts": np.asarray(target_ts, dtype=float).tolist(),
                "mask_gt": np.asarray(mask_gt, dtype=int).tolist(),
                "instruction_text": prompt_text,
                "region": [int(start_step), int(injection_config["end_step"])],
                "strength_text": strength_text,
                "strength_label": STRENGTH_TEXT_TO_LABEL[strength_text],
                "task_id": _task_id_from_intent(intent),
                "tool_name": injection_type,
                "effect_family": str(intent.get("effect_family", "unknown")),
                "shape": str(intent.get("shape", "unknown")),
                "direction": str(intent.get("direction", "neutral")),
                "duration_bucket": duration_bucket,
                "target_energy_type": _target_energy_type(base_ts, target_ts, start_step, int(injection_config["end_step"])),
                "injection_config": injection_config,
                "timestamp": datetime.now().isoformat(),
            }
            family_samples.append(sample_payload)
            flat_samples.append(sample_payload)

        families.append(
            {
                "family_id": f"family_{family_idx + 1:03d}",
                "tool_name": injection_type,
                "target_feature": feature,
                "duration_bucket": duration_bucket,
                "region": [int(start_step), int(start_step + duration)],
                "source_ts": np.asarray(base_ts, dtype=float).tolist(),
                "samples": family_samples,
            }
        )

    payload = {
        "benchmark_type": "tedit_discrete_strength_mainline",
        "dataset_name": dataset_name,
        "seq_len": seq_len,
        "num_families": len(families),
        "num_samples": len(flat_samples),
        "random_seed": random_seed,
        "injection_types": chosen_types,
        "strengths": strengths,
        "families": families,
        "samples": flat_samples,
    }
    json_path = output_root / f"tedit_discrete_strength_benchmark_{dataset_name}_{len(families)}families.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    summary_path = output_root / "README.md"
    summary_lines = [
        "# TEdit Discrete Strength Benchmark",
        "",
        f"- dataset_name: {dataset_name}",
        f"- num_families: {len(families)}",
        f"- num_samples: {len(flat_samples)}",
        f"- seq_len: {seq_len}",
        f"- random_seed: {random_seed}",
        "",
        "## Strength Families",
        "",
        "- each family fixes source/tool/region/shape template and varies only weak/medium/strong",
    ]
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    return {
        "json_path": str(json_path),
        "summary_path": str(summary_path),
        "num_families": len(families),
        "num_samples": len(flat_samples),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a discrete weak/medium/strong benchmark for TEdit strength-control mainline.")
    parser.add_argument("--csv-path", required=True)
    parser.add_argument("--dataset-name", default="ETTh1")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-families", type=int, default=20)
    parser.add_argument("--seq-len", type=int, default=192)
    parser.add_argument("--random-seed", type=int, default=7)
    parser.add_argument("--injection-types", default="")
    args = parser.parse_args()

    injection_types = None
    if args.injection_types.strip():
        injection_types = [token.strip() for token in args.injection_types.split(",") if token.strip()]

    result = build_discrete_benchmark(
        csv_path=args.csv_path,
        dataset_name=args.dataset_name,
        output_dir=args.output_dir,
        num_families=args.num_families,
        seq_len=args.seq_len,
        random_seed=args.random_seed,
        injection_types=injection_types,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
