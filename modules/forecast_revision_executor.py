from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import yaml

from tool.tedit_wrapper import get_tedit_instance
from tool.ts_editors import EDIT_TOOL_SPECS, execute_llm_tool


def _default_synthetic_tedit_paths() -> Tuple[str, str]:
    root = Path(__file__).resolve().parent.parent
    model_path = root / "TEdit-main" / "save" / "synthetic" / "pretrain_multi_weaver" / "0" / "ckpts" / "model_best.pth"
    config_path = root / "TEdit-main" / "save" / "synthetic" / "pretrain_multi_weaver" / "0" / "model_configs.yaml"
    return str(model_path), str(config_path)


def _load_tedit_seq_len(config_path: str) -> int:
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    seq_len = int(config.get("ts", {}).get("seq_len", 128))
    return max(8, seq_len)


def _zscore(values: np.ndarray) -> Tuple[np.ndarray, float, float]:
    arr = np.asarray(values, dtype=np.float32).flatten()
    mean = float(np.mean(arr)) if arr.size else 0.0
    std = float(np.std(arr)) if arr.size else 1.0
    std = max(std, 1e-3)
    normalized = ((arr - mean) / std).astype(np.float32)
    return normalized, mean, std


def _resample_series(values: np.ndarray, target_len: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32).flatten()
    if arr.size == target_len:
        return arr.copy()
    src_x = np.linspace(0.0, 1.0, arr.size, dtype=np.float32)
    dst_x = np.linspace(0.0, 1.0, target_len, dtype=np.float32)
    return np.interp(dst_x, src_x, arr).astype(np.float32)


def _scale_index(index: int, src_len: int, dst_len: int) -> int:
    if src_len <= 1:
        return 0
    ratio = float(index) / float(src_len)
    return int(round(ratio * dst_len))


def _editor_context_len(history_len: int, forecast_len: int, model_seq_len: int) -> int:
    if history_len <= 0:
        return 0
    target_context = max(
        model_seq_len // 2,
        forecast_len,
        max(0, model_seq_len - forecast_len),
    )
    return int(min(history_len, max(0, target_context)))


def _build_editor_window(
    history_ts: np.ndarray | None,
    base_forecast: np.ndarray,
    model_seq_len: int,
) -> Tuple[np.ndarray, int]:
    history_arr = np.asarray(history_ts, dtype=np.float32).flatten() if history_ts is not None else np.zeros(0, dtype=np.float32)
    base_arr = np.asarray(base_forecast, dtype=np.float32).flatten()
    context_len = _editor_context_len(len(history_arr), len(base_arr), model_seq_len)
    history_tail = history_arr[-context_len:] if context_len > 0 else np.zeros(0, dtype=np.float32)
    editor_input = np.concatenate([history_tail, base_arr], axis=0).astype(np.float32)
    future_offset = int(len(history_tail))
    return editor_input, future_offset


def _infer_tool_name(intent: Dict[str, Any], preferred_tool_name: str | None = None) -> str:
    effect_family = str(intent.get("effect_family", "none"))
    direction = str(intent.get("direction", "neutral"))
    shape = str(intent.get("shape", "none"))

    if effect_family == "none" or shape == "none":
        return "none"
    if effect_family == "shutdown" or shape == "flatline":
        return "hybrid_down"
    if effect_family == "level" and shape in {"step", "plateau"}:
        return "hybrid_down" if direction == "down" else "hybrid_up"
    if preferred_tool_name in EDIT_TOOL_SPECS:
        return str(preferred_tool_name)
    if effect_family == "impulse" or shape == "hump":
        return "spike_inject"
    if effect_family == "volatility" or shape == "irregular_noise":
        return "volatility_increase"
    return "step_shift"


def _taper_mask(length: int, ramp: int) -> np.ndarray:
    if length <= 0:
        return np.zeros(0, dtype=np.float32)
    mask = np.ones(length, dtype=np.float32)
    ramp = max(1, min(ramp, length // 2 if length > 1 else 1))
    if ramp > 0 and length > 1:
        phase = np.linspace(0.0, np.pi, ramp, dtype=np.float32)
        edge = 0.5 * (1.0 - np.cos(phase))
        mask[:ramp] = edge
        mask[-ramp:] = edge[::-1]
    return mask


def _local_support_mask(total_len: int, start: int, end: int, boundary: int) -> np.ndarray:
    mask = np.zeros(total_len, dtype=np.float32)
    start = max(0, min(start, total_len))
    end = max(start, min(end, total_len))
    if end <= start:
        return mask
    mask[start:end] = 1.0
    boundary = max(0, boundary)
    if boundary > 0:
        left_start = max(0, start - boundary)
        right_end = min(total_len, end + boundary)
        if start > left_start:
            phase = np.linspace(0.0, np.pi / 2.0, start - left_start, dtype=np.float32)
            mask[left_start:start] = np.sin(phase)
        if right_end > end:
            phase = np.linspace(np.pi / 2.0, 0.0, right_end - end, dtype=np.float32)
            mask[end:right_end] = np.sin(phase)
    return np.clip(mask, 0.0, 1.0)


def _build_level_envelope(region_len: int, target_shift: float, shape: str) -> np.ndarray:
    if region_len <= 0:
        return np.zeros(0, dtype=np.float32)
    if shape == "plateau":
        ramp = max(2, region_len // 4)
        mask = _taper_mask(region_len, ramp)
        return (target_shift * mask).astype(np.float32)
    ramp = max(2, region_len // 5)
    ramp = min(ramp, max(1, region_len - 1))
    envelope = np.full(region_len, target_shift, dtype=np.float32)
    if ramp > 0:
        phase = np.linspace(0.0, np.pi / 2.0, ramp, dtype=np.float32)
        rise = np.sin(phase)
        envelope[:ramp] = target_shift * rise
        if shape == "flatline":
            envelope[-ramp:] = target_shift
    return envelope


def _apply_xtraffic_flow_guard(
    *,
    params: Dict[str, Any],
    intent: Dict[str, Any],
    region: List[int],
    base_future: np.ndarray,
    sample_metadata: Dict[str, Any] | None,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    if not sample_metadata:
        return dict(params), {"applied": False, "reason": "no_sample_metadata"}
    dataset_name = str(sample_metadata.get("dataset_name", ""))
    if "xtraffic" not in dataset_name.lower():
        return dict(params), {"applied": False, "reason": "not_xtraffic"}

    channel_name = str(sample_metadata.get("channel_name", "")).lower()
    if not channel_name:
        op = sample_metadata.get("revision_operator_params", {}) or {}
        channel_name = str(op.get("channel_name", "")).lower()
    if channel_name != "flow":
        return dict(params), {"applied": False, "reason": "not_flow_channel", "channel_name": channel_name}

    direction = str(intent.get("direction", "neutral")).lower()
    if direction != "down":
        return dict(params), {"applied": False, "reason": "not_down_direction", "channel_name": channel_name}

    start = max(0, min(int(region[0]), len(base_future)))
    end = max(start + 1, min(int(region[1]), len(base_future)))
    local_base = np.asarray(base_future[start:end], dtype=np.float64)
    local_std = float(np.std(local_base)) if local_base.size else 0.0
    local_std = max(local_std, 1e-3)

    shape = str(intent.get("shape", "none")).lower()
    duration = str(intent.get("duration", "none")).lower()
    if shape == "step":
        cap_ratio = 0.35 if duration == "short" else 0.45
    elif shape == "hump":
        cap_ratio = 0.18
    else:
        cap_ratio = 0.40

    guarded = dict(params)
    original_amp = float(abs(guarded.get("amplitude", 0.0)))
    cap_amp = float(cap_ratio * local_std)
    if original_amp > 0.0:
        guarded["amplitude"] = float(min(original_amp, cap_amp))
    if shape == "hump":
        guarded["_force_tool_name"] = "hybrid_down"
    if "volatility_scale" in guarded:
        guarded["volatility_scale"] = float(min(float(guarded["volatility_scale"]), 1.4))

    return guarded, {
        "applied": True,
        "channel_name": channel_name,
        "shape": shape,
        "duration": duration,
        "local_std": local_std,
        "cap_ratio": cap_ratio,
        "amplitude_before": original_amp,
        "amplitude_after": float(guarded.get("amplitude", 0.0)),
    }


def _refine_tedit_future_segment(
    *,
    base_future: np.ndarray,
    edited_future: np.ndarray,
    intent: Dict[str, Any],
    region: List[int],
    params: Dict[str, Any],
) -> np.ndarray:
    refined = np.asarray(edited_future, dtype=np.float64).copy()
    start = max(0, min(int(region[0]), len(refined)))
    end = max(start, min(int(region[1]), len(refined)))
    if end <= start:
        return refined

    shape = str(intent.get("shape", "none"))
    direction = str(intent.get("direction", "neutral"))
    effect_family = str(intent.get("effect_family", "none"))
    if shape not in {"step", "plateau", "flatline"} and effect_family not in {"level", "shutdown"}:
        return refined

    region_len = end - start
    target_shift = float(abs(params.get("amplitude", 0.0)))
    if direction == "down":
        target_shift = -target_shift
    local_std = float(np.std(base_future[start:end]))
    texture_scale = max(0.05 * local_std, 0.08 * abs(target_shift), 1e-3)

    raw_delta = np.asarray(edited_future[start:end] - base_future[start:end], dtype=np.float32)
    texture = raw_delta - float(np.mean(raw_delta))
    texture = np.clip(texture, -texture_scale, texture_scale)
    taper = _taper_mask(region_len, max(2, region_len // 5))
    texture = texture * taper
    envelope = _build_level_envelope(region_len, target_shift=target_shift, shape=shape)
    delta_full = np.zeros(len(refined), dtype=np.float64)
    delta_full[start:end] = envelope.astype(np.float64) + texture.astype(np.float64)
    support = _local_support_mask(len(refined), start, end, boundary=max(2, region_len // 4))
    return np.asarray(base_future, dtype=np.float64) + delta_full * support.astype(np.float64)


def _build_editor_plan(
    *,
    intent: Dict[str, Any],
    region: List[int],
    params: Dict[str, Any],
    preferred_tool_name: str | None,
    future_offset: int,
    future_len: int,
    normalization_scale: float,
) -> Dict[str, Any]:
    forced_tool_name = str(params.get("_force_tool_name", "")).strip()
    if forced_tool_name in EDIT_TOOL_SPECS:
        tool_name = forced_tool_name
    else:
        tool_name = _infer_tool_name(intent, preferred_tool_name=preferred_tool_name)
    if tool_name == "none":
        return {"tool_name": "none", "parameters": {"region": [0, 1]}, "intent": dict(intent)}

    start_local = max(0, min(int(region[0]), future_len))
    end_local = max(start_local + 1, min(int(region[1]), future_len))
    start_idx = future_offset + start_local
    end_idx = future_offset + end_local
    plan_params: Dict[str, Any] = {"region": [start_idx, end_idx]}

    amplitude = float(abs(params.get("amplitude", 0.0))) / max(normalization_scale, 1e-3)
    amplitude = float(np.clip(amplitude, 0.05, 2.5)) if amplitude > 0.0 else 0.0
    if tool_name in {"hybrid_up", "hybrid_down"} and amplitude > 0.0:
        plan_params["math_shift"] = amplitude
    elif tool_name == "spike_inject":
        width = max(2.0, float(params.get("duration", max(2, end_idx - start_idx))) / 3.0)
        center = int((start_idx + end_idx) // 2)
        signed_amp = -amplitude if intent.get("direction") == "down" else amplitude
        plan_params.update({"amplitude": signed_amp, "width": width, "center": center})
    elif tool_name == "volatility_increase":
        plan_params["amplify_factor"] = max(1.5, float(params.get("volatility_scale", 1.8)))
    elif tool_name == "step_shift" and amplitude > 0.0:
        sign = -1.0 if intent.get("direction") == "down" else 1.0
        plan_params["math_shift"] = sign * amplitude

    return {
        "tool_name": tool_name,
        "parameters": plan_params,
        "intent": dict(intent),
        "execution": {
            "tool_name": tool_name,
            "parameters": dict(plan_params),
        },
        "localization": {"region": [start_idx, end_idx]},
    }


def apply_tedit_hybrid_revision(
    *,
    history_ts: np.ndarray | None,
    base_forecast: np.ndarray,
    intent: Dict[str, Any],
    region: List[int],
    params: Dict[str, Any],
    preferred_tool_name: str | None = None,
    tedit_model_path: str | None = None,
    tedit_config_path: str | None = None,
    tedit_device: str = "cuda:0",
    sample_metadata: Dict[str, Any] | None = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    model_path = tedit_model_path
    config_path = tedit_config_path
    if not model_path or not config_path:
        model_path, config_path = _default_synthetic_tedit_paths()

    base_arr = np.asarray(base_forecast, dtype=np.float32).flatten()
    if str(intent.get("effect_family", "none")) == "none" or str(intent.get("shape", "none")) == "none":
        return base_arr.astype(np.float64), np.zeros_like(base_arr, dtype=np.float64), {
            "executor": "tedit_hybrid",
            "tool_name": "none",
            "tedit_model_path": model_path,
            "tedit_config_path": config_path,
        }

    model_seq_len = _load_tedit_seq_len(config_path)
    editor_input, future_offset = _build_editor_window(history_ts, base_arr, model_seq_len)
    original_window_len = int(len(editor_input))
    if original_window_len != model_seq_len:
        editor_input = _resample_series(editor_input, model_seq_len)
    editor_input_norm, norm_mean, norm_std = _zscore(editor_input)
    scaled_future_offset = _scale_index(future_offset, original_window_len, len(editor_input))
    scaled_future_end = _scale_index(future_offset + len(base_arr), original_window_len, len(editor_input))
    scaled_future_len = max(1, scaled_future_end - scaled_future_offset)
    guarded_params, guard_metadata = _apply_xtraffic_flow_guard(
        params=params,
        intent=intent,
        region=region,
        base_future=base_arr.astype(np.float64),
        sample_metadata=sample_metadata,
    )
    editor_plan = _build_editor_plan(
        intent=intent,
        region=region,
        params=guarded_params,
        preferred_tool_name=preferred_tool_name,
        future_offset=scaled_future_offset,
        future_len=scaled_future_len,
        normalization_scale=norm_std,
    )

    tedit = get_tedit_instance(model_path=model_path, config_path=config_path, device=tedit_device)
    if not tedit.is_loaded:
        tedit = get_tedit_instance(model_path=model_path, config_path=config_path, device=tedit_device, force_reload=True)
    edited_norm, log = execute_llm_tool(editor_plan, editor_input_norm, tedit=tedit, use_soft_boundary=True)
    edited_window = (np.asarray(edited_norm, dtype=np.float32).flatten() * norm_std + norm_mean).astype(np.float64)
    if len(edited_window) != original_window_len:
        edited_window = _resample_series(edited_window, original_window_len).astype(np.float64)
    edited = edited_window[future_offset:future_offset + len(base_arr)].copy()

    start = max(0, min(int(region[0]), len(base_arr)))
    end = max(start, min(int(region[1]), len(base_arr)))
    raw_delta = edited - base_arr.astype(np.float64)
    support = _local_support_mask(len(base_arr), start, end, boundary=max(2, (end - start) // 4))
    edited = base_arr.astype(np.float64) + raw_delta * support.astype(np.float64)
    delta = edited - base_arr.astype(np.float64)

    metadata = {
        "executor": "tedit_hybrid",
        "requested_tool_name": preferred_tool_name,
        "tool_name": editor_plan["tool_name"],
        "editor_plan": editor_plan,
        "editor_region": editor_plan["parameters"]["region"],
        "editor_window_len": int(original_window_len),
        "editor_window_len_resampled": int(len(editor_input)),
        "history_context_len": int(future_offset),
        "future_offset": int(future_offset),
        "future_offset_resampled": int(scaled_future_offset),
        "normalization": {"mean": norm_mean, "std": norm_std},
        "editor_seq_len_hint": int(model_seq_len),
        "tedit_model_path": model_path,
        "tedit_config_path": config_path,
        "execution_log": log,
    }
    edited = _refine_tedit_future_segment(
        base_future=base_arr.astype(np.float64),
        edited_future=edited,
        intent=intent,
        region=region,
        params=guarded_params,
    )
    delta = edited - base_arr.astype(np.float64)
    metadata["postprocess"] = {
        "type": "level_refine" if str(intent.get("shape", "none")) in {"step", "plateau", "flatline"} else "none",
    }
    metadata["flow_guard"] = guard_metadata
    return edited, delta, metadata
