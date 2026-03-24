"""Time Series Editors - Tool Executor Layer.

This module provides high-level editing tools that combine TEdit diffusion model
with mathematical operations for event-driven time series editing.

## TEdit Attribute Space (synthetic model)
attr[0] = trend_types:    0=flat, 1=linear, 2=quadratic, 3=sinusoidal_trend
attr[1] = trend_directions: 0=downward, 1=upward
attr[2] = season_cycles:  0=none, 1=low(1-cycle), 2=medium(2-cycles), 3=high(4-cycles)

## Available Tools
Trend editing:
- hybrid_up:            Linear upward trend  [src→tgt=[1,1,*]]  + math_shift
- hybrid_down:          Linear downward trend [src→tgt=[1,0,*]] + math_shift
- trend_quadratic_up:   Quadratic accelerating up [src→tgt=[2,1,*]] + math parabola
- trend_quadratic_down: Quadratic accelerating down [src→tgt=[2,0,*]] + math parabola

Seasonality editing:
- season_enhance:  Intensify periodicity [src→tgt=[*,*,3]]
- season_reduce:   Suppress periodicity  [src→tgt=[*,*,0]]

Volatility / noise editing:
- ensemble_smooth:     Noise cancellation via multi-sample averaging
- volatility_increase: Amplify local fluctuations (math residual scaling)

Event editing:
- spike_inject: Inject a Gaussian pulse (impulse event simulation)
- step_shift:  Inject a local level shift / regime switch

Key Innovation: Soft-Boundary Temporal Injection
- All *_soft methods use latent space blending
- Eliminates "cliff effect" at region boundaries
- Training-free attention region injection
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from modules.pure_editing_volatility import (
    volatility_envelope_monotonic,
    volatility_global_scale,
    volatility_burst_local,
)
from tool.tedit_wrapper import TEditWrapper, get_tedit_instance


EDIT_TOOL_SPECS: Dict[str, Dict[str, Any]] = {
    "hybrid_up": {
        "canonical_tool": "trend_linear_up",
        "control_source": "hybrid",
        "tool_layer": "native_tedit",
        "effect_family": "trend",
        "direction": "up",
        "shape": "linear",
        "duration": "medium_or_long",
        "strength": "medium",
        "description": "Linear upward trend driven by TEdit attrs plus a math anchor.",
    },
    "hybrid_down": {
        "canonical_tool": "trend_linear_down",
        "control_source": "hybrid",
        "tool_layer": "native_tedit",
        "effect_family": "trend",
        "direction": "down",
        "shape": "linear",
        "duration": "medium_or_long",
        "strength": "medium",
        "description": "Linear downward trend driven by TEdit attrs plus a math anchor.",
    },
    "trend_quadratic_up": {
        "canonical_tool": "trend_quadratic_up",
        "control_source": "hybrid",
        "tool_layer": "native_tedit",
        "effect_family": "trend",
        "direction": "up",
        "shape": "quadratic",
        "duration": "medium_or_long",
        "strength": "strong",
        "description": "Accelerating upward trend using TEdit quadratic attrs plus a parabolic anchor.",
    },
    "trend_quadratic_down": {
        "canonical_tool": "trend_quadratic_down",
        "control_source": "hybrid",
        "tool_layer": "native_tedit",
        "effect_family": "trend",
        "direction": "down",
        "shape": "quadratic",
        "duration": "medium_or_long",
        "strength": "strong",
        "description": "Accelerating downward trend using TEdit quadratic attrs plus a parabolic anchor.",
    },
    "season_enhance": {
        "canonical_tool": "seasonality_enhance",
        "control_source": "native_tedit",
        "tool_layer": "native_tedit",
        "effect_family": "seasonality",
        "direction": "neutral",
        "shape": "periodic",
        "duration": "medium_or_long",
        "strength": "strong",
        "description": "Increase seasonal oscillation amplitude in a local region.",
    },
    "season_reduce": {
        "canonical_tool": "seasonality_reduce",
        "control_source": "native_tedit",
        "tool_layer": "native_tedit",
        "effect_family": "seasonality",
        "direction": "neutral",
        "shape": "flatten",
        "duration": "medium_or_long",
        "strength": "medium",
        "description": "Flatten or suppress periodicity in a local region.",
    },
    "ensemble_smooth": {
        "canonical_tool": "smooth_denoise",
        "control_source": "native_tedit",
        "tool_layer": "native_tedit",
        "effect_family": "volatility",
        "direction": "neutral",
        "shape": "flatten",
        "duration": "medium",
        "strength": "medium",
        "description": "Reduce local volatility by TEdit ensemble averaging.",
    },
    "volatility_increase": {
        "canonical_tool": "volatility_increase",
        "control_source": "math_only",
        "tool_layer": "derived",
        "effect_family": "volatility",
        "direction": "neutral",
        "shape": "residual_amplify",
        "duration": "short_or_medium",
        "strength": "medium",
        "description": "Increase local fluctuations without shifting the mean level.",
    },
    "volatility_global_scale": {
        "canonical_tool": "volatility_global_scale",
        "control_source": "math_only",
        "tool_layer": "derived",
        "effect_family": "volatility",
        "direction": "neutral",
        "shape": "irregular_noise",
        "duration": "short_or_medium",
        "strength": "medium",
        "description": "Replace a local region with low-baseline higher-variance noise.",
    },
    "volatility_local_burst": {
        "canonical_tool": "volatility_local_burst",
        "control_source": "math_only",
        "tool_layer": "derived",
        "effect_family": "volatility",
        "direction": "neutral",
        "shape": "irregular_noise",
        "duration": "short_or_medium",
        "strength": "strong",
        "description": "Inject a localized burst-shaped noisy segment.",
    },
    "volatility_envelope_monotonic": {
        "canonical_tool": "volatility_envelope_monotonic",
        "control_source": "math_only",
        "tool_layer": "derived",
        "effect_family": "volatility",
        "direction": "neutral",
        "shape": "irregular_noise",
        "duration": "medium",
        "strength": "strong",
        "description": "Inject monotonic envelope-shaped local volatility changes.",
    },
    "spike_inject": {
        "canonical_tool": "impulse_spike",
        "control_source": "math_only",
        "tool_layer": "derived",
        "effect_family": "impulse",
        "direction": "up_or_down",
        "shape": "transient",
        "duration": "short",
        "strength": "strong",
        "description": "Inject a short-lived local pulse or shock.",
    },
    "step_shift": {
        "canonical_tool": "level_step",
        "control_source": "math_only",
        "tool_layer": "derived",
        "effect_family": "level",
        "direction": "up_or_down",
        "shape": "step",
        "duration": "short_or_medium",
        "strength": "strong",
        "description": "Apply a local regime switch with a near-constant shifted level.",
    },
}

CANONICAL_TOOL_TO_TOOL_NAME: Dict[str, str] = {
    spec["canonical_tool"]: tool_name for tool_name, spec in EDIT_TOOL_SPECS.items()
}
CANONICAL_TOOL_TO_TOOL_NAME.update({
    "impulse_up": "spike_inject",
    "impulse_down": "spike_inject",
    "level_step_shift": "step_shift",
    "seasonality_to_none": "season_reduce",
    "trend_flatten": "ensemble_smooth",
})


def get_edit_tool_specs() -> Dict[str, Dict[str, Any]]:
    """Return the layered tool registry used by prompts and evaluators."""
    return deepcopy(EDIT_TOOL_SPECS)


def get_prompt_tool_catalog() -> list[Dict[str, Any]]:
    """Return a compact prompt-oriented catalog ordered by execution tool name."""
    catalog = []
    for tool_name, spec in EDIT_TOOL_SPECS.items():
        item = {
            "tool_name": tool_name,
            **spec,
        }
        catalog.append(item)
    return catalog


def normalize_llm_plan(plan: Dict[str, Any], ts_length: Optional[int] = None) -> Dict[str, Any]:
    """Normalize old and new LLM plan schemas into the execution format.

    Supported inputs:
    - legacy: {"tool_name": ..., "parameters": {...}}
    - intent-first:
      {
        "intent": {...},
        "localization": {"region": [...]},
        "execution": {"canonical_tool": ..., "tool_name": ..., "parameters": {...}}
      }
    """
    normalized = deepcopy(plan)

    execution = normalized.get("execution", {}) if isinstance(normalized.get("execution"), dict) else {}
    localization = normalized.get("localization", {}) if isinstance(normalized.get("localization"), dict) else {}
    intent = normalized.get("intent", {}) if isinstance(normalized.get("intent"), dict) else {}

    parameters = {}
    if isinstance(normalized.get("parameters"), dict):
        parameters.update(normalized["parameters"])
    if isinstance(execution.get("parameters"), dict):
        parameters.update(execution["parameters"])
    if "region" not in parameters and isinstance(localization.get("region"), list):
        parameters["region"] = localization["region"]

    canonical_tool = execution.get("canonical_tool") or normalized.get("canonical_tool")
    tool_name = execution.get("tool_name") or normalized.get("tool_name")
    if not tool_name and canonical_tool:
        tool_name = CANONICAL_TOOL_TO_TOOL_NAME.get(canonical_tool)
    if not canonical_tool and tool_name in EDIT_TOOL_SPECS:
        canonical_tool = EDIT_TOOL_SPECS[tool_name]["canonical_tool"]

    if tool_name:
        normalized["tool_name"] = tool_name
    if canonical_tool:
        normalized["canonical_tool"] = canonical_tool
    if parameters:
        normalized["parameters"] = parameters

    if "localization" not in normalized:
        normalized["localization"] = {}
    if isinstance(normalized["localization"], dict) and "region" not in normalized["localization"] and "region" in parameters:
        normalized["localization"]["region"] = parameters["region"]

    if "execution" not in normalized:
        normalized["execution"] = {}
    if isinstance(normalized["execution"], dict):
        normalized["execution"].setdefault("tool_name", tool_name)
        normalized["execution"].setdefault("canonical_tool", canonical_tool)
        normalized["execution"].setdefault("parameters", parameters)

    if tool_name in EDIT_TOOL_SPECS:
        spec = EDIT_TOOL_SPECS[tool_name]
        normalized["tool_layer"] = spec["tool_layer"]
        normalized["control_source"] = spec["control_source"]
        if not intent:
            normalized["intent"] = {
                "effect_family": spec["effect_family"],
                "direction": spec["direction"],
                "shape": spec["shape"],
                "duration": spec["duration"],
                "strength": spec["strength"],
            }
        normalized["execution"]["control_source"] = spec["control_source"]

    region = normalized.get("parameters", {}).get("region")
    if ts_length is not None and isinstance(region, list) and len(region) == 2:
        start_idx = max(0, int(region[0]))
        end_idx = min(int(region[1]), ts_length)
        if end_idx <= start_idx:
            end_idx = min(ts_length, start_idx + 1)
        normalized["parameters"]["region"] = [start_idx, end_idx]
        normalized["localization"]["region"] = [start_idx, end_idx]

    return normalized


def _resolve_math_shift(
    params: Dict[str, Any],
    ts: np.ndarray,
    start_idx: int,
    end_idx: int,
    default_shift_factor: float,
    intent: Optional[Dict[str, Any]] = None,
    sign: float = 1.0,
) -> float:
    """Resolve math_shift from explicit value or adaptive shift_factor * std(region).

    Priority: explicit math_shift > shift_factor * std > default.
    """
    if "math_shift" in params:
        raw_shift = float(params["math_shift"])
        return sign * abs(raw_shift)

    region_std = float(np.std(ts[start_idx:end_idx]))
    scale = max(region_std, 1e-3)

    amplitude = params.get("amplitude")
    if amplitude is not None:
        raw_amp = float(amplitude)
        if abs(raw_amp) <= 4.0:
            return raw_amp * scale
        return raw_amp

    level_shift = params.get("level_shift")
    if level_shift is not None:
        raw_level = float(level_shift)
        if abs(raw_level) <= 4.0:
            return raw_level * scale
        return raw_level

    strength_scale = {
        "weak": 0.75,
        "light": 0.8,
        "mild": 0.85,
        "medium": 1.0,
        "moderate": 1.0,
        "strong": 1.2,
        "large": 1.25,
        "sharp": 1.3,
    }
    strength = str((intent or {}).get("strength", "")).lower()
    factor = abs(float(params.get("shift_factor", default_shift_factor))) * strength_scale.get(strength, 1.0)
    return sign * factor * scale


def _resolve_spike_parameters(
    params: Dict[str, Any],
    start_idx: int,
    end_idx: int,
    ts: np.ndarray,
    intent: Optional[Dict[str, Any]] = None,
) -> Tuple[float, float, int]:
    """Resolve spike parameters into safe region-local values.

    LLM outputs sometimes express `center`/`width` as relative fractions or
    emit values outside the target region. This helper converts them into a
    bounded local pulse so the execution stage cannot destroy background
    fidelity through extreme Gaussian tails.
    """
    region_len = max(1, end_idx - start_idx)
    region_std = float(np.std(ts[start_idx:end_idx]))
    default_amp = 3.0 * region_std
    direction = (intent or {}).get("direction", "")
    shape = (intent or {}).get("shape", "")

    explicit_amp = params.get("amplitude")
    if explicit_amp is not None:
        raw_amp = float(explicit_amp)
        if abs(raw_amp) <= 4.0:
            amplitude = raw_amp * max(region_std, 1e-3)
        else:
            amplitude = raw_amp
    else:
        amplitude = default_amp

    if explicit_amp is None:
        if direction == "down":
            amplitude = -abs(amplitude)
        elif direction == "up":
            amplitude = abs(amplitude)

    if shape == "hump":
        center = start_idx + max(1, region_len // 3)
        width = max(2.0, region_len / 4.0)
    else:
        center = (start_idx + end_idx) // 2
        width = max(2.0, region_len / 6.0)

    explicit_center = params.get("center")
    if explicit_center is not None:
        raw_center = float(explicit_center)
        if 0.0 <= raw_center <= 1.0:
            center = start_idx + int(round(raw_center * max(region_len - 1, 1)))
        else:
            center = int(round(raw_center))

    explicit_width = params.get("width")
    if explicit_width is not None:
        raw_width = float(explicit_width)
        if 0.0 < raw_width <= 1.0:
            width = raw_width * max(region_len, 1)
        else:
            width = raw_width

    center = max(start_idx, min(center, end_idx - 1))
    width = max(2.0, min(width, max(3.0, region_len / 2.0)))

    return amplitude, width, center


def _resolve_shutdown_shift(
    ts: np.ndarray,
    start_idx: int,
    end_idx: int,
) -> float:
    """Estimate a downward shift that pushes the region toward a low baseline."""
    floor_value = float(np.percentile(ts, 5))
    region_mean = float(np.mean(ts[start_idx:end_idx]))
    return floor_value - region_mean


def _resolve_step_shift(
    params: Dict[str, Any],
    ts: np.ndarray,
    start_idx: int,
    end_idx: int,
    intent: Optional[Dict[str, Any]] = None,
) -> float:
    """Resolve the magnitude of a local level switch.

    For step-like edits we want a stable plateau, not a linear ramp. The shift
    is therefore inferred as a constant level offset with sign driven by
    explicit parameters or intent.direction.
    """
    if "math_shift" in params:
        return float(params["math_shift"])

    if "level_shift" in params:
        raw_level = float(params["level_shift"])
        region_std = float(np.std(ts[start_idx:end_idx]))
        if abs(raw_level) <= 4.0:
            return raw_level * max(region_std, 1e-3)
        return raw_level

    if "amplitude" in params:
        raw_amp = float(params["amplitude"])
        region_std = float(np.std(ts[start_idx:end_idx]))
        if abs(raw_amp) <= 4.0:
            return raw_amp * max(region_std, 1e-3)
        return raw_amp

    region_std = float(np.std(ts[start_idx:end_idx]))
    factor = abs(float(params.get("shift_factor", 2.5)))
    direction = (intent or {}).get("direction", "")
    sign = -1.0 if direction == "down" else 1.0
    return sign * factor * max(region_std, 1e-3)


def execute_llm_tool(
    plan: Dict[str, Any],
    ts: np.ndarray,
    tedit: TEditWrapper,
    n_ensemble: int = 15,
    use_soft_boundary: bool = True,
) -> Tuple[np.ndarray, str]:
    """Execute editing tool based on LLM plan.

    Args:
        plan: LLM-generated plan with keys: thought, tool_name, parameters
        ts: Input time series (shape: [L])
        tedit: TEditWrapper instance
        n_ensemble: Number of samples for ensemble methods
        use_soft_boundary: Whether to use soft-boundary injection (default: True)

    Returns:
        Tuple of (edited_ts, log_message)

    Supported tool_name values:
        hybrid_up, hybrid_down,
        trend_quadratic_up, trend_quadratic_down,
        season_enhance, season_reduce,
        ensemble_smooth, volatility_increase,
        spike_inject, step_shift
    """
    normalized_plan = normalize_llm_plan(plan, ts_length=len(ts))
    tool_name = normalized_plan.get("tool_name", "")
    params = normalized_plan.get("parameters", {})
    intent = normalized_plan.get("intent", {}) if isinstance(normalized_plan.get("intent"), dict) else {}
    region = params.get("region", [0, len(ts)])
    start_idx, end_idx = region[0], region[1]

    L = len(ts)
    if start_idx < 0 or end_idx > L or start_idx >= end_idx:
        raise ValueError(
            f"Invalid region [{start_idx}, {end_idx}) for sequence length {L}. "
            f"Constraints: 0 <= start < end <= {L}"
        )

    # ── pure-math tools (no TEdit call needed) ────────────────────────────────
    if tool_name == "volatility_increase":
        amplify = float(params.get("amplify_factor") or 2.0)
        edited_ts = volatility_increase(ts=ts, start_idx=start_idx, end_idx=end_idx, amplify_factor=amplify)
        log = f"[volatility_increase] region=[{start_idx},{end_idx}], amplify_factor={amplify}"
        return edited_ts, log

    if tool_name == "volatility_global_scale":
        edited_ts = volatility_global_scale(
            base_ts=ts,
            region=[start_idx, end_idx],
            base_noise_scale=float(params.get("base_noise_scale") or 1.0),
            local_std_target_ratio=float(params.get("local_std_target_ratio") or 2.0),
            baseline_offset_ratio=float(params.get("baseline_offset_ratio") or 0.05),
            trend_preserve=float(params.get("trend_preserve") or 0.0),
        )
        log = (
            f"[volatility_global_scale] region=[{start_idx},{end_idx}], "
            f"base_noise_scale={float(params.get('base_noise_scale') or 1.0):.2f}, "
            f"std_ratio={float(params.get('local_std_target_ratio') or 2.0):.2f}"
        )
        return edited_ts, log

    if tool_name == "volatility_local_burst":
        edited_ts = volatility_burst_local(
            base_ts=ts,
            region=[start_idx, end_idx],
            background_scale=float(params.get("background_scale") or 0.5),
            burst_center=float(params.get("burst_center") or 0.5),
            burst_width=float(params.get("burst_width") or 0.25),
            burst_amplitude=float(params.get("burst_amplitude") or 2.4),
            burst_envelope_sharpness=float(params.get("burst_envelope_sharpness") or 0.8),
            baseline_offset_ratio=float(params.get("baseline_offset_ratio") or 0.05),
        )
        log = (
            f"[volatility_local_burst] region=[{start_idx},{end_idx}], "
            f"center={float(params.get('burst_center') or 0.5):.2f}, "
            f"width={float(params.get('burst_width') or 0.25):.2f}, "
            f"amplitude={float(params.get('burst_amplitude') or 2.4):.2f}"
        )
        return edited_ts, log

    if tool_name == "volatility_envelope_monotonic":
        edited_ts = volatility_envelope_monotonic(
            base_ts=ts,
            region=[start_idx, end_idx],
            base_noise_scale=float(params.get("base_noise_scale") or 1.0),
            start_scale=float(params.get("start_scale") or 0.3),
            end_scale=float(params.get("end_scale") or 2.0),
            baseline_offset_ratio=float(params.get("baseline_offset_ratio") or 0.05),
            trend_preserve=float(params.get("trend_preserve") or 0.0),
        )
        log = (
            f"[volatility_envelope_monotonic] region=[{start_idx},{end_idx}], "
            f"start_scale={float(params.get('start_scale') or 0.3):.2f}, "
            f"end_scale={float(params.get('end_scale') or 2.0):.2f}"
        )
        return edited_ts, log

    if tool_name == "spike_inject":
        amplitude, width, center = _resolve_spike_parameters(
            params=params,
            start_idx=start_idx,
            end_idx=end_idx,
            ts=ts,
            intent=intent,
        )
        edited_ts = spike_inject(ts=ts, start_idx=start_idx, end_idx=end_idx, center=center, amplitude=amplitude, width=width)
        log = f"[spike_inject] region=[{start_idx},{end_idx}], center={center}, amplitude={amplitude:.2f}, width={width}"
        return edited_ts, log

    if tool_name == "step_shift":
        level_shift = _resolve_step_shift(
            params=params,
            ts=ts,
            start_idx=start_idx,
            end_idx=end_idx,
            intent=intent,
        )
        edited_ts = step_shift(
            ts=ts,
            start_idx=start_idx,
            end_idx=end_idx,
            level_shift=level_shift,
        )
        log = f"[step_shift] region=[{start_idx},{end_idx}], level_shift={level_shift:.3f}"
        return edited_ts, log

    # ── TEdit-backed tools ─────────────────────────────────────────────────────
    if use_soft_boundary:
        if tool_name == "hybrid_up":
            math_shift = _resolve_math_shift(
                params,
                ts,
                start_idx,
                end_idx,
                default_shift_factor=2.0,
                intent=intent,
                sign=1.0,
            )
            edited_ts = hybrid_up_soft(ts=ts, start_idx=start_idx, end_idx=end_idx, math_shift=math_shift, tedit=tedit)
            log = f"[hybrid_up_soft] region=[{start_idx},{end_idx}], math_shift={math_shift:.3f}"

        elif tool_name == "hybrid_down":
            if intent.get("effect_family") == "shutdown" or intent.get("shape") == "flatline":
                math_shift = (
                    _resolve_math_shift(
                        params,
                        ts,
                        start_idx,
                        end_idx,
                        default_shift_factor=3.5,
                        intent=intent,
                        sign=-1.0,
                    )
                    if ("math_shift" in params or "shift_factor" in params)
                    else _resolve_shutdown_shift(ts, start_idx, end_idx)
                )
            else:
                math_shift = _resolve_math_shift(
                    params,
                    ts,
                    start_idx,
                    end_idx,
                    default_shift_factor=2.0,
                    intent=intent,
                    sign=-1.0,
                )
            edited_ts = hybrid_down_soft(ts=ts, start_idx=start_idx, end_idx=end_idx, math_shift=math_shift, tedit=tedit)
            log = f"[hybrid_down_soft] region=[{start_idx},{end_idx}], math_shift={math_shift:.3f}"

        elif tool_name == "trend_quadratic_up":
            math_shift = _resolve_math_shift(
                params,
                ts,
                start_idx,
                end_idx,
                default_shift_factor=2.5,
                intent=intent,
                sign=1.0,
            )
            edited_ts = trend_quadratic_up_soft(ts=ts, start_idx=start_idx, end_idx=end_idx, math_shift=math_shift, tedit=tedit)
            log = f"[trend_quadratic_up_soft] region=[{start_idx},{end_idx}], math_shift={math_shift:.3f}"

        elif tool_name == "trend_quadratic_down":
            math_shift = _resolve_math_shift(
                params,
                ts,
                start_idx,
                end_idx,
                default_shift_factor=2.5,
                intent=intent,
                sign=-1.0,
            )
            edited_ts = trend_quadratic_down_soft(ts=ts, start_idx=start_idx, end_idx=end_idx, math_shift=math_shift, tedit=tedit)
            log = f"[trend_quadratic_down_soft] region=[{start_idx},{end_idx}], math_shift={math_shift:.3f}"

        elif tool_name == "season_enhance":
            edited_ts = season_enhance_soft(ts=ts, start_idx=start_idx, end_idx=end_idx, tedit=tedit)
            log = f"[season_enhance_soft] region=[{start_idx},{end_idx}]"

        elif tool_name == "season_reduce":
            edited_ts = season_reduce_soft(ts=ts, start_idx=start_idx, end_idx=end_idx, tedit=tedit)
            log = f"[season_reduce_soft] region=[{start_idx},{end_idx}]"

        elif tool_name == "ensemble_smooth":
            edited_ts = ensemble_smooth_soft(ts=ts, start_idx=start_idx, end_idx=end_idx, tedit=tedit, n_samples=n_ensemble)
            log = f"[ensemble_smooth_soft] region=[{start_idx},{end_idx}], n_samples={n_ensemble}"

        else:
            raise ValueError(
                f"Unknown tool: '{tool_name}'. Valid tools: hybrid_up, hybrid_down, "
                "trend_quadratic_up, trend_quadratic_down, season_enhance, season_reduce, "
                "ensemble_smooth, volatility_increase, spike_inject, step_shift"
            )

        # Enforce strict background fidelity outside [start_idx, end_idx)
        edited_ts = np.asarray(edited_ts, dtype=np.float32).flatten()
        ts_orig = np.asarray(ts, dtype=np.float32).flatten()
        edited_ts[:start_idx] = ts_orig[:start_idx]
        edited_ts[end_idx:] = ts_orig[end_idx:]

    else:  # hard boundary (legacy)
        if tool_name == "hybrid_up":
            math_shift = _resolve_math_shift(
                params,
                ts,
                start_idx,
                end_idx,
                default_shift_factor=2.0,
                intent=intent,
                sign=1.0,
            )
            edited_ts = hybrid_up(ts=ts, start_idx=start_idx, end_idx=end_idx, math_shift=math_shift, tedit=tedit)
            log = f"[hybrid_up] region=[{start_idx},{end_idx}], math_shift={math_shift:.3f}"

        elif tool_name == "hybrid_down":
            if intent.get("effect_family") == "shutdown" or intent.get("shape") == "flatline":
                math_shift = (
                    _resolve_math_shift(
                        params,
                        ts,
                        start_idx,
                        end_idx,
                        default_shift_factor=3.5,
                        intent=intent,
                        sign=-1.0,
                    )
                    if ("math_shift" in params or "shift_factor" in params)
                    else _resolve_shutdown_shift(ts, start_idx, end_idx)
                )
            else:
                math_shift = _resolve_math_shift(
                    params,
                    ts,
                    start_idx,
                    end_idx,
                    default_shift_factor=2.0,
                    intent=intent,
                    sign=-1.0,
                )
            edited_ts = hybrid_down(ts=ts, start_idx=start_idx, end_idx=end_idx, math_shift=math_shift, tedit=tedit)
            log = f"[hybrid_down] region=[{start_idx},{end_idx}], math_shift={math_shift:.3f}"

        elif tool_name == "trend_quadratic_up":
            math_shift = _resolve_math_shift(
                params,
                ts,
                start_idx,
                end_idx,
                default_shift_factor=2.5,
                intent=intent,
                sign=1.0,
            )
            edited_ts = trend_quadratic_up(ts=ts, start_idx=start_idx, end_idx=end_idx, math_shift=math_shift, tedit=tedit)
            log = f"[trend_quadratic_up] region=[{start_idx},{end_idx}], math_shift={math_shift:.3f}"

        elif tool_name == "trend_quadratic_down":
            math_shift = _resolve_math_shift(
                params,
                ts,
                start_idx,
                end_idx,
                default_shift_factor=2.5,
                intent=intent,
                sign=-1.0,
            )
            edited_ts = trend_quadratic_down(ts=ts, start_idx=start_idx, end_idx=end_idx, math_shift=math_shift, tedit=tedit)
            log = f"[trend_quadratic_down] region=[{start_idx},{end_idx}], math_shift={math_shift:.3f}"

        elif tool_name == "season_enhance":
            edited_ts = season_enhance(ts=ts, start_idx=start_idx, end_idx=end_idx, tedit=tedit)
            log = f"[season_enhance] region=[{start_idx},{end_idx}]"

        elif tool_name == "season_reduce":
            edited_ts = season_reduce(ts=ts, start_idx=start_idx, end_idx=end_idx, tedit=tedit)
            log = f"[season_reduce] region=[{start_idx},{end_idx}]"

        elif tool_name == "ensemble_smooth":
            edited_ts = ensemble_smooth(ts=ts, start_idx=start_idx, end_idx=end_idx, tedit=tedit, n_samples=n_ensemble)
            log = f"[ensemble_smooth] region=[{start_idx},{end_idx}], n_samples={n_ensemble}"

        else:
            raise ValueError(
                f"Unknown tool: '{tool_name}'. Valid tools: hybrid_up, hybrid_down, "
                "trend_quadratic_up, trend_quadratic_down, season_enhance, season_reduce, "
                "ensemble_smooth, volatility_increase, spike_inject, step_shift"
            )

    return edited_ts, log


def hybrid_up_soft(
    ts: np.ndarray,
    start_idx: int,
    end_idx: int,
    math_shift: float,
    tedit: TEditWrapper,
    smooth_radius: float = 5.0,
) -> np.ndarray:
    """Hybrid upward editing with Latent Blending (State-Space Mixing).

    Implements Latent Blending for Ground-Truth Preservation:
    z_{t-1} = M ⊙ z_{t-1}^{pred} + (1-M) ⊙ z_{t-1}^{GT}

    Key Mechanism:
    - Foreground (z^{pred}): Denoised from TEdit conditioned on edit prompt.
    - Background (z^{GT}): Forward-diffused directly from original time series.
    - Math Anchor: M ⊙ anchor for smooth trend injection.

    Why Latent Blending (NOT Noise Blending):
    - Noise Blending: Background is "predicted", causes reconstruction error
    - Latent Blending: Background is "physical truth", ensures 100% fidelity

    Args:
        ts: Input time series
        start_idx: Start index of region
        end_idx: End index of region
        math_shift: Positive shift magnitude
        tedit: TEditWrapper instance
        smooth_radius: Radius for soft boundary smoothing

    Returns:
        Edited time series with smooth boundaries and 100% background fidelity
    """
    ts = np.asarray(ts, dtype=np.float32).copy()
    L = len(ts)

    src_attrs = np.array([0, 0, 0], dtype=np.int64)
    tgt_attrs = np.array([1, 1, 1], dtype=np.int64)

    total_steps = getattr(tedit.model, 'num_steps', 100) if tedit.model else 100
    edit_steps = int(total_steps * 0.4)
    tedit.set_edit_steps(edit_steps)

    edited_ts = tedit.edit_region_soft(
        ts=ts,
        start_idx=start_idx,
        end_idx=end_idx,
        src_attrs=src_attrs,
        tgt_attrs=tgt_attrs,
        n_samples=1,
        sampler="ddim",
        smooth_radius=smooth_radius,
    )

    from scipy.ndimage import gaussian_filter1d
    hard_mask = np.zeros(L, dtype=np.float32)
    hard_mask[start_idx:end_idx] = 1.0
    soft_mask = gaussian_filter1d(hard_mask, sigma=smooth_radius)

    region_len = end_idx - start_idx
    math_anchor_region = np.linspace(0, math_shift, region_len)
    
    math_anchor = np.zeros(L, dtype=np.float32)
    math_anchor[start_idx:end_idx] = math_anchor_region
    math_anchor = math_anchor * soft_mask

    result = edited_ts + math_anchor

    return result


def hybrid_down_soft(
    ts: np.ndarray,
    start_idx: int,
    end_idx: int,
    math_shift: float,
    tedit: TEditWrapper,
    smooth_radius: float = 5.0,
) -> np.ndarray:
    """Hybrid downward editing with Latent Blending (State-Space Mixing).

    Implements Latent Blending for Ground-Truth Preservation:
    z_{t-1} = M ⊙ z_{t-1}^{pred} + (1-M) ⊙ z_{t-1}^{GT}

    Key Mechanism:
    - Foreground (z^{pred}): Denoised from TEdit conditioned on edit prompt.
    - Background (z^{GT}): Forward-diffused directly from original time series.
    - Math Anchor: M ⊙ anchor for smooth trend injection.

    Why Latent Blending (NOT Noise Blending):
    - Noise Blending: Background is "predicted", causes reconstruction error
    - Latent Blending: Background is "physical truth", ensures 100% fidelity

    Args:
        ts: Input time series
        start_idx: Start index of region
        end_idx: End index of region
        math_shift: Negative shift magnitude (e.g., -15.0)
        tedit: TEditWrapper instance
        smooth_radius: Radius for soft boundary smoothing

    Returns:
        Edited time series with smooth boundaries and 100% background fidelity
    """
    ts = np.asarray(ts, dtype=np.float32).copy()
    L = len(ts)

    src_attrs = np.array([0, 0, 0], dtype=np.int64)
    tgt_attrs = np.array([1, 0, 1], dtype=np.int64)

    total_steps = getattr(tedit.model, 'num_steps', 100) if tedit.model else 100
    edit_steps = int(total_steps * 0.4)
    tedit.set_edit_steps(edit_steps)

    edited_ts = tedit.edit_region_soft(
        ts=ts,
        start_idx=start_idx,
        end_idx=end_idx,
        src_attrs=src_attrs,
        tgt_attrs=tgt_attrs,
        n_samples=1,
        sampler="ddim",
        smooth_radius=smooth_radius,
    )

    from scipy.ndimage import gaussian_filter1d
    hard_mask = np.zeros(L, dtype=np.float32)
    hard_mask[start_idx:end_idx] = 1.0
    soft_mask = gaussian_filter1d(hard_mask, sigma=smooth_radius)

    region_len = end_idx - start_idx
    math_anchor_region = np.linspace(0, math_shift, region_len)
    
    math_anchor = np.zeros(L, dtype=np.float32)
    math_anchor[start_idx:end_idx] = math_anchor_region
    math_anchor = math_anchor * soft_mask

    result = edited_ts + math_anchor

    return result


def ensemble_smooth_soft(
    ts: np.ndarray,
    start_idx: int,
    end_idx: int,
    tedit: TEditWrapper,
    n_samples: int = 15,
    smooth_radius: float = 5.0,
) -> np.ndarray:
    """Ensemble smoothing with Soft-Boundary Temporal Injection.

    Combines ensemble averaging with soft-boundary blending for
    maximum smoothness at region edges.

    Args:
        ts: Input time series
        start_idx: Start index of region
        end_idx: End index of region
        tedit: TEditWrapper instance
        n_samples: Number of samples for ensemble
        smooth_radius: Radius for soft boundary smoothing

    Returns:
        Smoothed time series with soft boundaries
    """
    ts = np.asarray(ts, dtype=np.float32).copy()

    src_attrs = np.array([0, 0, 0], dtype=np.int64)
    tgt_attrs = np.array([0, 0, 0], dtype=np.int64)

    samples = []
    for seed in range(n_samples):
        torch.manual_seed(seed)
        np.random.seed(seed)

        edited_ts = tedit.edit_region_soft(
            ts=ts,
            start_idx=start_idx,
            end_idx=end_idx,
            src_attrs=src_attrs,
            tgt_attrs=tgt_attrs,
            n_samples=1,
            sampler="ddpm",
            smooth_radius=smooth_radius,
        )
        samples.append(edited_ts)

    result = np.mean(samples, axis=0)

    return result


def hybrid_up(
    ts: np.ndarray,
    start_idx: int,
    end_idx: int,
    math_shift: float,
    tedit: TEditWrapper,
    edit_steps_ratio: float = 0.4,
) -> np.ndarray:
    """Hybrid upward editing: Math anchor + AI texture (Legacy hard boundary).

    Combines mathematical linear shift with TEdit diffusion texture.
    Uses 40% edit steps to preserve math guidance while adding AI texture.

    Note: This method uses hard array splicing which may cause "cliff effect"
    at boundaries. Consider using hybrid_up_soft() for smoother results.

    Args:
        ts: Input time series
        start_idx: Start index of region
        end_idx: End index of region
        math_shift: Positive shift magnitude
        tedit: TEditWrapper instance
        edit_steps_ratio: Ratio of edit steps (default 0.4 for hybrid)

    Returns:
        Edited time series
    """
    ts = np.asarray(ts, dtype=np.float32).copy()
    region_len = end_idx - start_idx

    slope = math_shift / region_len
    math_anchor = np.linspace(0, math_shift, region_len)

    src_attrs = np.array([0, 0, 0], dtype=np.int64)
    tgt_attrs = np.array([1, 1, 1], dtype=np.int64)

    total_steps = getattr(tedit.model, 'num_steps', 100) if tedit.model else 100
    edit_steps = int(total_steps * edit_steps_ratio)
    tedit.set_edit_steps(edit_steps)

    edited_region = tedit.edit_time_series(
        ts=ts[start_idx:end_idx],
        src_attrs=src_attrs,
        tgt_attrs=tgt_attrs,
        n_samples=1,
        sampler="ddim",
    )[0]

    ai_texture = edited_region - ts[start_idx:end_idx]
    ai_texture = ai_texture - np.mean(ai_texture)

    result = ts.copy()
    result[start_idx:end_idx] = ts[start_idx:end_idx] + math_anchor + ai_texture

    return result


def hybrid_down(
    ts: np.ndarray,
    start_idx: int,
    end_idx: int,
    math_shift: float,
    tedit: TEditWrapper,
    edit_steps_ratio: float = 0.4,
) -> np.ndarray:
    """Hybrid downward editing: Math anchor + AI texture (Legacy hard boundary).

    Combines mathematical linear drop with TEdit diffusion texture.
    Uses 40% edit steps to preserve math guidance while adding AI texture.

    Note: This method uses hard array splicing which may cause "cliff effect"
    at boundaries. Consider using hybrid_down_soft() for smoother results.

    Args:
        ts: Input time series
        start_idx: Start index of region
        end_idx: End index of region
        math_shift: Negative shift magnitude (e.g., -15.0)
        tedit: TEditWrapper instance
        edit_steps_ratio: Ratio of edit steps (default 0.4 for hybrid)

    Returns:
        Edited time series
    """
    ts = np.asarray(ts, dtype=np.float32).copy()
    region_len = end_idx - start_idx

    math_anchor = np.linspace(0, math_shift, region_len)

    src_attrs = np.array([0, 0, 0], dtype=np.int64)
    tgt_attrs = np.array([1, 0, 1], dtype=np.int64)

    total_steps = getattr(tedit.model, 'num_steps', 100) if tedit.model else 100
    edit_steps = int(total_steps * edit_steps_ratio)
    tedit.set_edit_steps(edit_steps)

    edited_region = tedit.edit_time_series(
        ts=ts[start_idx:end_idx],
        src_attrs=src_attrs,
        tgt_attrs=tgt_attrs,
        n_samples=1,
        sampler="ddim",
    )[0]

    ai_texture = edited_region - ts[start_idx:end_idx]
    ai_texture = ai_texture - np.mean(ai_texture)

    result = ts.copy()
    result[start_idx:end_idx] = ts[start_idx:end_idx] + math_anchor + ai_texture

    return result


def ensemble_smooth(
    ts: np.ndarray,
    start_idx: int,
    end_idx: int,
    tedit: TEditWrapper,
    n_samples: int = 15,
) -> np.ndarray:
    """Ensemble smoothing through multi-sample noise cancellation (Legacy hard boundary).

    Generates multiple samples using DDPM sampler and averages them.
    The stochastic noise cancels out (zero-mean property) while
    the deterministic structure is preserved.

    Note: This method uses hard array splicing which may cause "cliff effect"
    at boundaries. Consider using ensemble_smooth_soft() for smoother results.

    Args:
        ts: Input time series
        start_idx: Start index of region
        end_idx: End index of region
        tedit: TEditWrapper instance
        n_samples: Number of samples for ensemble (default 15)

    Returns:
        Smoothed time series
    """
    ts = np.asarray(ts, dtype=np.float32).copy()

    src_attrs = np.array([0, 0, 0], dtype=np.int64)
    tgt_attrs = np.array([0, 0, 0], dtype=np.int64)

    samples = []
    for seed in range(n_samples):
        torch.manual_seed(seed)
        np.random.seed(seed)

        edited_region = tedit.edit_time_series(
            ts=ts[start_idx:end_idx],
            src_attrs=src_attrs,
            tgt_attrs=tgt_attrs,
            n_samples=1,
            sampler="ddpm",
        )[0]
        samples.append(edited_region)

    avg_region = np.mean(samples, axis=0)

    result = ts.copy()
    result[start_idx:end_idx] = avg_region

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# New Tools — Soft-boundary versions
# ═══════════════════════════════════════════════════════════════════════════════

def trend_quadratic_up_soft(
    ts: np.ndarray,
    start_idx: int,
    end_idx: int,
    math_shift: float,
    tedit: TEditWrapper,
    smooth_radius: float = 5.0,
) -> np.ndarray:
    """Quadratic accelerating upward trend with Latent Blending.

    Uses TEdit attr tgt=[2,1,1] (quadratic, upward, low-season) combined with a
    parabolic math anchor to produce an accelerating rise effect.

    Suitable for: sudden demand explosions, panic-buying acceleration, exponential
    event propagation.

    Args:
        ts: Input time series
        start_idx: Start index of region
        end_idx: End index of region
        math_shift: Total positive shift at end of region (adaptive: 2.5 * std)
        tedit: TEditWrapper instance
        smooth_radius: Boundary smoothing radius

    Returns:
        Edited time series
    """
    from scipy.ndimage import gaussian_filter1d

    ts = np.asarray(ts, dtype=np.float32).copy()
    L = len(ts)

    src_attrs = np.array([0, 0, 0], dtype=np.int64)
    tgt_attrs = np.array([2, 1, 1], dtype=np.int64)  # quadratic, upward, low-season

    total_steps = getattr(tedit.model, 'num_steps', 100) if tedit.model else 100
    tedit.set_edit_steps(int(total_steps * 0.4))

    edited_ts = tedit.edit_region_soft(
        ts=ts, start_idx=start_idx, end_idx=end_idx,
        src_attrs=src_attrs, tgt_attrs=tgt_attrs,
        n_samples=1, sampler="ddim", smooth_radius=smooth_radius,
    )

    # Parabolic anchor: f(t) = math_shift * (t / region_len)^2
    hard_mask = np.zeros(L, dtype=np.float32)
    hard_mask[start_idx:end_idx] = 1.0
    soft_mask = gaussian_filter1d(hard_mask, sigma=smooth_radius)

    region_len = end_idx - start_idx
    t_norm = np.linspace(0, 1, region_len)
    math_anchor_region = math_shift * (t_norm ** 2)

    math_anchor = np.zeros(L, dtype=np.float32)
    math_anchor[start_idx:end_idx] = math_anchor_region
    math_anchor = math_anchor * soft_mask

    return edited_ts + math_anchor


def trend_quadratic_down_soft(
    ts: np.ndarray,
    start_idx: int,
    end_idx: int,
    math_shift: float,
    tedit: TEditWrapper,
    smooth_radius: float = 5.0,
) -> np.ndarray:
    """Quadratic accelerating downward trend with Latent Blending.

    Uses TEdit attr tgt=[2,0,1] (quadratic, downward, low-season) combined with a
    parabolic math anchor to produce an accelerating drop effect.

    Suitable for: market crashes, supply chain collapses, cascading failures.

    Args:
        ts: Input time series
        start_idx: Start index of region
        end_idx: End index of region
        math_shift: Total negative shift at end of region (adaptive: -2.5 * std)
        tedit: TEditWrapper instance
        smooth_radius: Boundary smoothing radius

    Returns:
        Edited time series
    """
    from scipy.ndimage import gaussian_filter1d

    ts = np.asarray(ts, dtype=np.float32).copy()
    L = len(ts)

    src_attrs = np.array([0, 0, 0], dtype=np.int64)
    tgt_attrs = np.array([2, 0, 1], dtype=np.int64)  # quadratic, downward, low-season

    total_steps = getattr(tedit.model, 'num_steps', 100) if tedit.model else 100
    tedit.set_edit_steps(int(total_steps * 0.4))

    edited_ts = tedit.edit_region_soft(
        ts=ts, start_idx=start_idx, end_idx=end_idx,
        src_attrs=src_attrs, tgt_attrs=tgt_attrs,
        n_samples=1, sampler="ddim", smooth_radius=smooth_radius,
    )

    hard_mask = np.zeros(L, dtype=np.float32)
    hard_mask[start_idx:end_idx] = 1.0
    soft_mask = gaussian_filter1d(hard_mask, sigma=smooth_radius)

    region_len = end_idx - start_idx
    t_norm = np.linspace(0, 1, region_len)
    math_anchor_region = math_shift * (t_norm ** 2)  # math_shift is negative

    math_anchor = np.zeros(L, dtype=np.float32)
    math_anchor[start_idx:end_idx] = math_anchor_region
    math_anchor = math_anchor * soft_mask

    return edited_ts + math_anchor


def season_enhance_soft(
    ts: np.ndarray,
    start_idx: int,
    end_idx: int,
    tedit: TEditWrapper,
    smooth_radius: float = 5.0,
) -> np.ndarray:
    """Intensify periodicity in region with Latent Blending.

    Uses TEdit attr tgt=[1,1,3] (linear-up, 4-cycles) to inject strong
    seasonal oscillation into the target region.

    Suitable for: seasonal demand amplification, cyclical volatility increase,
    policy-driven periodic effects.

    Args:
        ts: Input time series
        start_idx: Start index of region
        end_idx: End index of region
        tedit: TEditWrapper instance
        smooth_radius: Boundary smoothing radius

    Returns:
        Edited time series with enhanced seasonality
    """
    ts = np.asarray(ts, dtype=np.float32).copy()

    src_attrs = np.array([0, 0, 0], dtype=np.int64)
    tgt_attrs = np.array([1, 1, 3], dtype=np.int64)  # linear-up, 4-cycles (high season)

    total_steps = getattr(tedit.model, 'num_steps', 100) if tedit.model else 100
    tedit.set_edit_steps(int(total_steps * 0.5))

    return tedit.edit_region_soft(
        ts=ts, start_idx=start_idx, end_idx=end_idx,
        src_attrs=src_attrs, tgt_attrs=tgt_attrs,
        n_samples=1, sampler="ddim", smooth_radius=smooth_radius,
    )


def season_reduce_soft(
    ts: np.ndarray,
    start_idx: int,
    end_idx: int,
    tedit: TEditWrapper,
    smooth_radius: float = 5.0,
) -> np.ndarray:
    """Suppress periodicity in region with Latent Blending.

    Uses TEdit attr tgt=[1,1,0] (linear, no seasonal cycles) to flatten
    oscillations in the target region.

    Suitable for: central bank intervention, market stabilization, regulatory
    dampening of seasonal fluctuations.

    Args:
        ts: Input time series
        start_idx: Start index of region
        end_idx: End index of region
        tedit: TEditWrapper instance
        smooth_radius: Boundary smoothing radius

    Returns:
        Edited time series with reduced seasonality
    """
    ts = np.asarray(ts, dtype=np.float32).copy()

    src_attrs = np.array([0, 0, 2], dtype=np.int64)  # assume medium-season source
    tgt_attrs = np.array([1, 1, 0], dtype=np.int64)  # linear, no seasonal cycles

    total_steps = getattr(tedit.model, 'num_steps', 100) if tedit.model else 100
    tedit.set_edit_steps(int(total_steps * 0.5))

    return tedit.edit_region_soft(
        ts=ts, start_idx=start_idx, end_idx=end_idx,
        src_attrs=src_attrs, tgt_attrs=tgt_attrs,
        n_samples=1, sampler="ddim", smooth_radius=smooth_radius,
    )


# ── Pure-math tools (no TEdit call) ──────────────────────────────────────────

def volatility_increase(
    ts: np.ndarray,
    start_idx: int,
    end_idx: int,
    amplify_factor: float = 2.0,
) -> np.ndarray:
    """Amplify local fluctuations by scaling the mean-zero residual.

    Decomposes the region as: x = trend + residual.
    Scales the residual: x_new = trend + amplify_factor * residual.
    This preserves the overall trend while increasing noise/volatility.

    Suitable for: market turmoil, increased uncertainty, heightened volatility
    events that do NOT shift the level/trend.

    Args:
        ts: Input time series
        start_idx: Start index of region
        end_idx: End index of region
        amplify_factor: Residual scaling factor (>1 increases volatility, default 2.0)

    Returns:
        Edited time series with amplified fluctuations in region
    """
    ts = np.asarray(ts, dtype=np.float32).copy()
    region = ts[start_idx:end_idx].copy()

    # Estimate trend as a linear fit, residual = region - trend
    region_len = end_idx - start_idx
    x = np.arange(region_len, dtype=np.float32)
    coeffs = np.polyfit(x, region, 1)
    trend = np.polyval(coeffs, x)
    residual = region - trend

    region_new = trend + amplify_factor * residual

    result = ts.copy()
    result[start_idx:end_idx] = region_new
    return result


def spike_inject(
    ts: np.ndarray,
    start_idx: int,
    end_idx: int,
    center: Optional[int] = None,
    amplitude: Optional[float] = None,
    width: Optional[float] = None,
) -> np.ndarray:
    """Inject a Gaussian impulse pulse into the region.

    Models a sudden, short-lived shock event (flash crash, supply disruption,
    breaking news impulse) as a Gaussian bump.

    Args:
        ts: Input time series
        start_idx: Start index of region
        end_idx: End index of region
        center: Peak position index (default: midpoint of region)
        amplitude: Peak height of the impulse (default: 3 * std(region);
                   negative for a downward spike)
        width: Standard deviation of Gaussian kernel in timesteps
               (default: region_len / 6, giving a narrow pulse)

    Returns:
        Edited time series with impulse injected
    """
    ts = np.asarray(ts, dtype=np.float32).copy()
    L = len(ts)
    region_len = end_idx - start_idx

    if center is None:
        center = (start_idx + end_idx) // 2
    if amplitude is None:
        amplitude = 3.0 * float(np.std(ts[start_idx:end_idx]))
    if width is None:
        width = max(2.0, region_len / 6.0)

    t = np.arange(L, dtype=np.float32)
    pulse = amplitude * np.exp(-0.5 * ((t - center) / width) ** 2)

    # Zero out pulse outside [start_idx, end_idx)
    pulse[:start_idx] = 0.0
    pulse[end_idx:] = 0.0

    result = ts.copy()
    result += pulse
    return result


def step_shift(
    ts: np.ndarray,
    start_idx: int,
    end_idx: int,
    level_shift: float,
    left_ramp_steps: Optional[int] = None,
    right_ramp_steps: Optional[int] = None,
) -> np.ndarray:
    """Apply a local step-style level shift with short in-window ramps.

    The profile stays flat for most of the window, which is a better match for
    regime-switch tasks than the existing linear hybrid trend tools.
    """
    ts = np.asarray(ts, dtype=np.float32).copy()
    region_len = max(1, end_idx - start_idx)
    if region_len <= 2:
        result = ts.copy()
        result[start_idx:end_idx] += level_shift
        return result

    if left_ramp_steps is None:
        left_ramp_steps = max(1, min(2, region_len // 6))
    if right_ramp_steps is None:
        right_ramp_steps = max(1, min(3, region_len // 5))

    left_ramp_steps = min(left_ramp_steps, region_len - 1)
    right_ramp_steps = min(right_ramp_steps, max(1, region_len - left_ramp_steps))
    plateau_len = max(0, region_len - left_ramp_steps - right_ramp_steps)

    left = np.linspace(0.0, level_shift, left_ramp_steps + 1, dtype=np.float32)[1:]
    plateau = np.full(plateau_len, level_shift, dtype=np.float32)
    right = np.linspace(level_shift, 0.0, right_ramp_steps + 1, dtype=np.float32)[:-1]
    profile = np.concatenate([left, plateau, right]).astype(np.float32)

    if len(profile) < region_len:
        pad = np.full(region_len - len(profile), level_shift, dtype=np.float32)
        profile = np.concatenate([left, plateau, pad, right]).astype(np.float32)
    elif len(profile) > region_len:
        profile = profile[:region_len]

    result = ts.copy()
    result[start_idx:end_idx] = ts[start_idx:end_idx] + profile
    return result


# ── New Tools — Hard-boundary legacy versions ─────────────────────────────────

def trend_quadratic_up(
    ts: np.ndarray,
    start_idx: int,
    end_idx: int,
    math_shift: float,
    tedit: TEditWrapper,
    edit_steps_ratio: float = 0.4,
) -> np.ndarray:
    """Quadratic accelerating upward trend (Legacy hard boundary)."""
    ts = np.asarray(ts, dtype=np.float32).copy()
    region_len = end_idx - start_idx

    t_norm = np.linspace(0, 1, region_len)
    math_anchor = math_shift * (t_norm ** 2)

    src_attrs = np.array([0, 0, 0], dtype=np.int64)
    tgt_attrs = np.array([2, 1, 1], dtype=np.int64)

    total_steps = getattr(tedit.model, 'num_steps', 100) if tedit.model else 100
    tedit.set_edit_steps(int(total_steps * edit_steps_ratio))

    edited_region = tedit.edit_time_series(
        ts=ts[start_idx:end_idx], src_attrs=src_attrs, tgt_attrs=tgt_attrs, n_samples=1, sampler="ddim",
    )[0]

    ai_texture = edited_region - ts[start_idx:end_idx]
    ai_texture = ai_texture - np.mean(ai_texture)

    result = ts.copy()
    result[start_idx:end_idx] = ts[start_idx:end_idx] + math_anchor + ai_texture
    return result


def trend_quadratic_down(
    ts: np.ndarray,
    start_idx: int,
    end_idx: int,
    math_shift: float,
    tedit: TEditWrapper,
    edit_steps_ratio: float = 0.4,
) -> np.ndarray:
    """Quadratic accelerating downward trend (Legacy hard boundary)."""
    ts = np.asarray(ts, dtype=np.float32).copy()
    region_len = end_idx - start_idx

    t_norm = np.linspace(0, 1, region_len)
    math_anchor = math_shift * (t_norm ** 2)  # math_shift is negative

    src_attrs = np.array([0, 0, 0], dtype=np.int64)
    tgt_attrs = np.array([2, 0, 1], dtype=np.int64)

    total_steps = getattr(tedit.model, 'num_steps', 100) if tedit.model else 100
    tedit.set_edit_steps(int(total_steps * edit_steps_ratio))

    edited_region = tedit.edit_time_series(
        ts=ts[start_idx:end_idx], src_attrs=src_attrs, tgt_attrs=tgt_attrs, n_samples=1, sampler="ddim",
    )[0]

    ai_texture = edited_region - ts[start_idx:end_idx]
    ai_texture = ai_texture - np.mean(ai_texture)

    result = ts.copy()
    result[start_idx:end_idx] = ts[start_idx:end_idx] + math_anchor + ai_texture
    return result


def season_enhance(
    ts: np.ndarray,
    start_idx: int,
    end_idx: int,
    tedit: TEditWrapper,
    edit_steps_ratio: float = 0.5,
) -> np.ndarray:
    """Intensify periodicity (Legacy hard boundary)."""
    ts = np.asarray(ts, dtype=np.float32).copy()

    src_attrs = np.array([0, 0, 0], dtype=np.int64)
    tgt_attrs = np.array([1, 1, 3], dtype=np.int64)

    total_steps = getattr(tedit.model, 'num_steps', 100) if tedit.model else 100
    tedit.set_edit_steps(int(total_steps * edit_steps_ratio))

    edited_region = tedit.edit_time_series(
        ts=ts[start_idx:end_idx], src_attrs=src_attrs, tgt_attrs=tgt_attrs, n_samples=1, sampler="ddim",
    )[0]

    result = ts.copy()
    result[start_idx:end_idx] = edited_region
    return result


def season_reduce(
    ts: np.ndarray,
    start_idx: int,
    end_idx: int,
    tedit: TEditWrapper,
    edit_steps_ratio: float = 0.5,
) -> np.ndarray:
    """Suppress periodicity (Legacy hard boundary)."""
    ts = np.asarray(ts, dtype=np.float32).copy()

    src_attrs = np.array([0, 0, 2], dtype=np.int64)
    tgt_attrs = np.array([1, 1, 0], dtype=np.int64)

    total_steps = getattr(tedit.model, 'num_steps', 100) if tedit.model else 100
    tedit.set_edit_steps(int(total_steps * edit_steps_ratio))

    edited_region = tedit.edit_time_series(
        ts=ts[start_idx:end_idx], src_attrs=src_attrs, tgt_attrs=tgt_attrs, n_samples=1, sampler="ddim",
    )[0]

    result = ts.copy()
    result[start_idx:end_idx] = edited_region
    return result
