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

try:
    from test_scripts.build_event_driven_testset import CSVDataLoader, InjectorFactory
except Exception:
    CSVDataLoader = None
    InjectorFactory = None


class _FallbackCSVDataLoader:
    def __init__(self, csv_path: str, dataset_name: str):
        self.csv_path = csv_path
        self.dataset_name = dataset_name
        header = np.genfromtxt(csv_path, delimiter=",", names=True, max_rows=1, dtype=None, encoding="utf-8")
        column_names = list(header.dtype.names or [])
        self.features = [col for col in column_names if col != "date"]
        if not self.features:
            raise ValueError(f"No usable feature columns found in {csv_path}")
        self._series = {
            feature: np.genfromtxt(csv_path, delimiter=",", names=True, dtype=np.float64, encoding="utf-8")[feature]
            for feature in self.features
        }
        first_feature = self.features[0]
        self.length = int(np.asarray(self._series[first_feature]).shape[0])
        self.data = np.empty((self.length,), dtype=np.float64)
        self.feature_descriptions = {feature: feature for feature in self.features}

    def get_sequence(self, start_idx: int, seq_len: int, feature: str):
        values = np.asarray(self._series[feature], dtype=np.float64)
        end_idx = min(len(values), start_idx + seq_len)
        seq = values[start_idx:end_idx]
        if seq.shape[0] < seq_len:
            raise ValueError(f"Sequence too short for feature={feature}, start_idx={start_idx}, seq_len={seq_len}")
        return seq, {"feature": feature, "start_idx": int(start_idx), "end_idx": int(end_idx)}


class _FallbackInjector:
    def __init__(self, injection_type: str):
        self.injection_type = injection_type

    def get_edit_intent(self, injection_config: Dict[str, Any]) -> Dict[str, Any]:
        direction = _normalize_direction(injection_config.get("direction", "neutral"))
        if self.injection_type == "trend_injection":
            return {
                "effect_family": "impulse",
                "shape": "hump",
                "direction": direction,
                "duration": "medium",
                "recovery": "gradual",
                "onset": "gradual",
            }
        if self.injection_type == "step_change":
            return {
                "effect_family": "level",
                "shape": "step",
                "direction": direction,
                "duration": "medium",
                "recovery": "gradual",
                "onset": "abrupt",
            }
        if self.injection_type == "multiplier":
            return {
                "effect_family": "level",
                "shape": "scaled_surge",
                "direction": "upward",
                "duration": "medium",
                "recovery": "gradual",
                "onset": "abrupt",
            }
        if self.injection_type == "hard_zero":
            return {
                "effect_family": "shutdown",
                "shape": "flatline",
                "direction": "downward",
                "duration": "medium",
                "recovery": "gradual",
                "onset": "gradual",
            }
        if self.injection_type == "noise_injection":
            return {
                "effect_family": "volatility",
                "shape": "irregular_noise",
                "direction": "neutral",
                "duration": "medium",
                "recovery": "none",
                "onset": "abrupt",
            }
        return {
            "effect_family": "unknown",
            "shape": "unknown",
            "direction": "neutral",
            "duration": "medium",
            "recovery": "none",
            "onset": "unknown",
        }


class _FallbackInjectorFactory:
    def __init__(self, random_seed: int):
        self.random_seed = int(random_seed)

    def create_injector(self, injection_type: str):
        return _FallbackInjector(injection_type)

    def get_feature_description(self, feature: str) -> str:
        return str(feature)


def _create_data_loader(csv_path: str, dataset_name: str):
    if CSVDataLoader is not None:
        return CSVDataLoader(csv_path, dataset_name)
    return _FallbackCSVDataLoader(csv_path, dataset_name)


def _create_injector_factory(random_seed: int):
    if InjectorFactory is not None:
        return InjectorFactory(random_seed)
    return _FallbackInjectorFactory(random_seed)


def _ensure_region_fields(sample_payload: Dict[str, Any], start_step: int, end_step: int, seq_len: int) -> None:
    sample_payload["region_start"] = int(start_step)
    sample_payload["region_end"] = int(end_step)
    sample_payload["series_length"] = int(seq_len)
    sample_payload.setdefault("strength_scalar", float(sample_payload.get("strength_label", 0)))
    sample_payload.setdefault("task_id", None)
    sample_payload.setdefault("instruction_text", "")


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

STRENGTH_TEXT_TO_SCALAR = {
    "weak": 0.0,
    "medium": 0.5,
    "strong": 1.0,
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

HARD_ZERO_MIN_SOURCE_ABS_MEAN = 2.0
HARD_ZERO_MIN_TARGET_GAIN_GAP = 0.05
HARD_ZERO_MAX_RESAMPLE_ATTEMPTS = 256

TASK_NAME_TO_ID = {
    "trend_up": 0,
    "trend_down": 1,
    "step_up": 2,
    "step_down": 3,
    "multiplier": 4,
    "hard_zero": 5,
    "noise_injection": 6,
    "mixed_neutral": 7,
}

FAMILY_SEMANTIC_RULES = {
    "trend_injection": {
        "attr_strategy": "proxy",
        "task_name_by_direction": {
            "upward": "trend_up",
            "downward": "trend_down",
            "neutral": "trend_up",
        },
    },
    "step_change": {
        "attr_strategy": "neutral",
        "task_name_by_direction": {
            "upward": "step_up",
            "downward": "step_down",
            "neutral": "step_up",
        },
    },
    "multiplier": {
        "attr_strategy": "neutral",
        "task_name": "multiplier",
    },
    "hard_zero": {
        "attr_strategy": "neutral",
        "task_name": "hard_zero",
    },
    "noise_injection": {
        "attr_strategy": "neutral",
        "task_name": "noise_injection",
    },
}

STEP_CHANGE_PROXY_MAP = {
    "upward": "neutral",
    "downward": "neutral",
}

FAMILY_TAG_TEMPLATES = {
    "trend_injection": "trend_proxy_{direction}",
    "step_change": "step_{attr_strategy}_{direction}",
    "multiplier": "multiplier_neutral",
    "hard_zero": "hard_zero_neutral",
    "noise_injection": "noise_injection_neutral",
}

INSTRUCTION_TEMPLATES = {
    "trend_injection": "请把{feature_desc}在一个{duration_bucket}窗口里做{strength_zh}的trend/hump局部变化，保持显式趋势方向为{direction_label}，只改那一段，非编辑区尽量保持不变。",
    "step_change": "请把{feature_desc}在一个{duration_bucket}窗口里做{strength_zh}的step/level局部变化，保持显式台阶方向为{direction_label}，只改那一段，非编辑区尽量保持不变。",
    "multiplier": "请把{feature_desc}在一个{duration_bucket}窗口里做{strength_zh}的乘性放大（multiplier）局部变化，明确体现按倍数抬升的语义，只改那一段，非编辑区尽量保持不变。",
    "hard_zero": "请把{feature_desc}在一个{duration_bucket}窗口里做{strength_zh}的归零/贴地（hard zero, flatline）局部变化，明确体现关停贴地语义，只改那一段，非编辑区尽量保持不变。",
    "noise_injection": "请把{feature_desc}在一个{duration_bucket}窗口里做{strength_zh}的volatility/noise局部变化，明确体现不规则噪声扰动语义，只改那一段，非编辑区尽量保持不变。",
}


def _normalize_direction(direction: Any) -> str:
    token = str(direction or "").strip().lower()
    if token in {"down", "downward", "decrease", "decreasing", "negative"}:
        return "downward"
    if token in {"up", "upward", "increase", "increasing", "positive"}:
        return "upward"
    return "neutral"


def _resolve_attr_strategy(injection_type: str, direction: str) -> str:
    rule = FAMILY_SEMANTIC_RULES.get(injection_type, {})
    base_strategy = str(rule.get("attr_strategy", "neutral"))
    if injection_type != "step_change":
        return base_strategy
    proxy_mode = STEP_CHANGE_PROXY_MAP.get(_normalize_direction(direction), "neutral")
    if proxy_mode == "proxy":
        return "proxy"
    return "neutral"


def _task_id_from_family_semantics(tool_name: str, direction: str) -> int:
    canonical_tool = str(tool_name).strip().lower()
    rule = FAMILY_SEMANTIC_RULES.get(canonical_tool)
    if rule is None:
        return TASK_NAME_TO_ID["mixed_neutral"]
    if "task_name" in rule:
        return TASK_NAME_TO_ID[str(rule["task_name"])]
    normalized_direction = _normalize_direction(direction)
    task_name = rule["task_name_by_direction"].get(normalized_direction, rule["task_name_by_direction"].get("neutral", "mixed_neutral"))
    return TASK_NAME_TO_ID[str(task_name)]


def _family_semantic_tag(tool_name: str, direction: str, attr_strategy: str) -> str:
    template = FAMILY_TAG_TEMPLATES.get(tool_name, "mixed_{attr_strategy}_{direction}")
    return template.format(direction=_normalize_direction(direction), attr_strategy=str(attr_strategy))


def _prompt_text(
    *,
    tool_name: str,
    feature_desc: str,
    effect_family: str,
    shape: str,
    duration_bucket: str,
    strength_text: str,
    direction: str,
) -> str:
    template = INSTRUCTION_TEMPLATES.get(
        tool_name,
        "请把{feature_desc}在一个{duration_bucket}窗口里做{strength_zh}的{effect_family}/{shape}局部变化，只改那一段，非编辑区尽量保持不变。",
    )
    return template.format(
        feature_desc=feature_desc,
        duration_bucket=duration_bucket,
        strength_zh=STRENGTH_ZH[strength_text],
        effect_family=effect_family,
        shape=shape,
        direction_label=_normalize_direction(direction),
    )


def _duration_from_bucket(seq_len: int, bucket: str, rng: np.random.RandomState) -> int:
    lo, hi = DURATION_BUCKET_SPECS[bucket]
    return max(6, int(round(seq_len * float(rng.uniform(lo, hi)))))


def _start_step_for_duration(seq_len: int, duration: int, rng: np.random.RandomState) -> int:
    margin = max(4, seq_len // 12)
    hi = max(margin + 1, seq_len - duration - margin)
    return int(rng.randint(margin, hi))


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


def _hard_zero_family_has_strength_signal(family_samples: List[Dict[str, Any]]) -> bool:
    """Reject hard_zero families whose low-percentile floor is not a usable shutdown target."""
    if len(family_samples) != 3:
        return False
    samples = sorted(family_samples, key=lambda sample: float(sample.get("strength_scalar", 0.0)))
    start_step = int(samples[0]["region_start"])
    end_step = int(samples[0]["region_end"])
    if end_step <= start_step:
        return False
    source = np.asarray(samples[0]["source_ts"][start_step:end_step], dtype=np.float64)
    if source.size == 0:
        return False
    source_abs_mean = float(np.mean(np.abs(source)))
    if source_abs_mean < HARD_ZERO_MIN_SOURCE_ABS_MEAN:
        return False

    gains: List[float] = []
    floor_distances: List[float] = []
    for sample in samples:
        target = np.asarray(sample["target_ts"][start_step:end_step], dtype=np.float64)
        gains.append(float(np.mean(np.abs(target - source))))
        floor_distances.append(float(np.mean(np.abs(target))))

    gain_gaps = [gains[index + 1] - gains[index] for index in range(len(gains) - 1)]
    floor_gaps = [floor_distances[index + 1] - floor_distances[index] for index in range(len(floor_distances) - 1)]
    return bool(
        min(gain_gaps) >= HARD_ZERO_MIN_TARGET_GAIN_GAP
        and max(floor_gaps) <= 1.0e-8
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
    base_region = np.asarray(base_ts[start_step:end_step], dtype=np.float64)
    floor_value = float(np.percentile(base_region, floor_percentile))
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
    region_len = end_step - start_step
    base_region = np.asarray(base_ts[start_step:end_step], dtype=np.float64)
    centered_noise = np.asarray(family_noise[:region_len], dtype=np.float64)
    centered_noise = centered_noise - float(np.mean(centered_noise))
    noise_scale = np.abs(centered_noise)
    max_scale = float(np.max(noise_scale))
    if max_scale > 1.0e-8:
        noise_scale = noise_scale / max_scale
    target_ts[start_step:end_step] = base_region + noise_scale * float(noise_std_ratio)
    mask_gt[start_step:end_step] = 1
    return target_ts, mask_gt, {
        "injection_type": "noise_injection",
        "baseline_ratio": float(baseline_ratio),
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
    if _normalize_direction(direction) == "downward":
        amplitude = -abs(amplitude)
    else:
        amplitude = abs(amplitude)
    t = np.linspace(0.0, 2.0 * np.pi, region_len)
    hump = (1.0 - np.cos(t)) / 2.0
    target_ts[start_step:end_step] += hump * amplitude
    mask_gt[start_step:end_step] = 1
    return target_ts, mask_gt, {
        "injection_type": "trend_injection",
        "direction": _normalize_direction(direction),
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
    if _normalize_direction(direction) == "downward":
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
        "direction": _normalize_direction(direction),
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
    data_loader = _create_data_loader(csv_path, dataset_name)
    injector_factory = _create_injector_factory(random_seed)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    chosen_types = list(injection_types or DEFAULT_INJECTION_TYPES)
    strengths = ["weak", "medium", "strong"]

    families: List[Dict[str, Any]] = []
    flat_samples: List[Dict[str, Any]] = []
    for family_idx in range(num_families):
        injection_type = chosen_types[family_idx % len(chosen_types)]
        duration_bucket = ["short", "medium", "long"][family_idx % 3]
        max_attempts = HARD_ZERO_MAX_RESAMPLE_ATTEMPTS if injection_type == "hard_zero" else 1
        family_samples: List[Dict[str, Any]] = []
        for attempt_idx in range(max_attempts):
            feature = str(rng.choice(data_loader.features))
            start_idx = int(rng.randint(0, max(1, len(data_loader.data) - seq_len - 1)))
            base_ts, _ = data_loader.get_sequence(start_idx, seq_len, feature)
            base_ts = np.asarray(base_ts, dtype=np.float64)
            duration = _duration_from_bucket(seq_len, duration_bucket, rng)
            start_step = _start_step_for_duration(seq_len, duration, rng)

            shared_state: Dict[str, Any] = {}
            if injection_type == "trend_injection":
                shared_state["direction"] = str(rng.choice(["upward", "downward"]))
            elif injection_type == "step_change":
                shared_state["direction"] = str(rng.choice(["upward", "downward"]))
            elif injection_type == "noise_injection":
                shared_state["baseline_ratio"] = float(rng.uniform(0.02, 0.08))
                shared_state["family_noise"] = rng.normal(loc=0.0, scale=1.0, size=duration)

            injector = injector_factory.create_injector(injection_type)
            feature_desc = data_loader.feature_descriptions.get(feature, injector_factory.get_feature_description(feature))

            family_samples = []
            for strength_text in strengths:
                target_ts, mask_gt, injection_config = _build_family_variant(
                    injection_type=injection_type,
                    strength_text=strength_text,
                    base_ts=base_ts,
                    start_step=start_step,
                    duration=duration,
                    shared_state=shared_state,
                )
                intent = injector.get_edit_intent(injection_config)
                normalized_direction = _normalize_direction(intent.get("direction", "neutral"))
                attr_strategy = _resolve_attr_strategy(injection_type, normalized_direction)
                family_semantic_tag = _family_semantic_tag(injection_type, normalized_direction, attr_strategy)
                task_id = _task_id_from_family_semantics(injection_type, normalized_direction)
                prompt_text = _prompt_text(
                    tool_name=injection_type,
                    feature_desc=feature_desc,
                    effect_family=str(intent.get("effect_family", "unknown")),
                    shape=str(intent.get("shape", "unknown")),
                    duration_bucket=duration_bucket,
                    strength_text=strength_text,
                    direction=normalized_direction,
                )
                end_step = int(injection_config["end_step"])
                sample_payload = {
                    "family_id": f"family_{family_idx + 1:03d}",
                    "sample_id": f"family_{family_idx + 1:03d}_{strength_text}",
                    "dataset_name": dataset_name,
                    "target_feature": feature,
                    "feature_description": feature_desc,
                    "source_ts": base_ts.astype(float).tolist(),
                    "target_ts": np.asarray(target_ts, dtype=float).tolist(),
                    "mask_gt": np.asarray(mask_gt, dtype=int).tolist(),
                    "instruction_text": prompt_text,
                    "region": [int(start_step), end_step],
                    "strength_text": strength_text,
                    "strength_label": STRENGTH_TEXT_TO_LABEL[strength_text],
                    "strength_scalar": STRENGTH_TEXT_TO_SCALAR[strength_text],
                    "task_id": task_id,
                    "tool_name": injection_type,
                    "effect_family": str(intent.get("effect_family", "unknown")),
                    "shape": str(intent.get("shape", "unknown")),
                    "direction": normalized_direction,
                    "duration_bucket": duration_bucket,
                    "attr_strategy": attr_strategy,
                    "family_semantic_tag": family_semantic_tag,
                    "target_energy_type": _target_energy_type(base_ts, target_ts, start_step, end_step),
                    "injection_config": injection_config,
                    "timestamp": datetime.now().isoformat(),
                }
                _ensure_region_fields(sample_payload, int(start_step), end_step, seq_len)
                family_samples.append(sample_payload)
            if injection_type != "hard_zero" or _hard_zero_family_has_strength_signal(family_samples):
                break
        else:
            raise RuntimeError(
                f"Failed to sample a valid hard_zero family after {max_attempts} attempts "
                f"for family_idx={family_idx}, duration_bucket={duration_bucket}"
            )
        flat_samples.extend(family_samples)

        family_effect_family = str(family_samples[0].get("effect_family", "unknown")) if family_samples else "unknown"
        family_shape = str(family_samples[0].get("shape", "unknown")) if family_samples else "unknown"
        family_direction = str(family_samples[0].get("direction", "neutral")) if family_samples else "neutral"
        family_task_id = int(family_samples[0]["task_id"]) if family_samples and family_samples[0].get("task_id") is not None else None
        family_instruction_text = str(family_samples[0].get("instruction_text", "")) if family_samples else ""
        family_attr_strategy = str(family_samples[0].get("attr_strategy", "neutral")) if family_samples else "neutral"
        family_semantic_tag = str(family_samples[0].get("family_semantic_tag", "unknown")) if family_samples else "unknown"
        family_injection_config = dict(family_samples[0].get("injection_config", {})) if family_samples else {}
        families.append(
            {
                "family_id": f"family_{family_idx + 1:03d}",
                "tool_name": injection_type,
                "effect_family": family_effect_family,
                "shape": family_shape,
                "direction": family_direction,
                "task_id": family_task_id,
                "instruction_text": family_instruction_text,
                "attr_strategy": family_attr_strategy,
                "family_semantic_tag": family_semantic_tag,
                "injection_config": family_injection_config,
                "target_feature": feature,
                "feature_description": feature_desc,
                "duration_bucket": duration_bucket,
                "region": [int(start_step), int(start_step + duration)],
                "source_ts": base_ts.astype(float).tolist(),
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
        "strength_axis": {
            "type": "continuous_scalar",
            "anchor_mapping": STRENGTH_TEXT_TO_SCALAR,
            "range": [0.0, 1.0],
        },
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
        f"- injection_types: {', '.join(chosen_types)}",
        "- semantic_authority: family-level benchmark records",
        "- step_change_default_attr_strategy: neutral",
        "",
        "## Strength Families",
        "",
        "- each family fixes source/tool/region/shape template and varies only weak/medium/strong",
        "- non-trend families keep neutral attrs and rely on explicit instruction_text, with task_id available as a gated auxiliary channel",
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
