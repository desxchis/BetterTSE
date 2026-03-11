"""Time Series Editors - Tool Executor Layer.

This module provides high-level editing tools that combine TEdit diffusion model
with mathematical operations for event-driven time series editing.

Tools:
- hybrid_up: Upward trend editing with math anchor + AI texture
- hybrid_down: Downward trend editing with math anchor + AI texture  
- ensemble_smooth: Noise cancellation through multi-sample averaging

Key Innovation: Soft-Boundary Temporal Injection
- All hybrid_*_soft methods use latent space blending
- Eliminates "cliff effect" at region boundaries
- Training-free attention region injection
"""

from __future__ import annotations

import numpy as np
import torch
from typing import Dict, Any, Optional, Tuple

from tool.tedit_wrapper import TEditWrapper, get_tedit_instance


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
    """
    tool_name = plan.get("tool_name", "")
    params = plan.get("parameters", {})
    region = params.get("region", [0, len(ts)])
    start_idx, end_idx = region[0], region[1]
    
    L = len(ts)
    if start_idx < 0 or end_idx > L or start_idx >= end_idx:
        raise ValueError(
            f"Invalid region [{start_idx}, {end_idx}) for sequence length {L}. "
            f"Constraints: 0 <= start < end <= {L}"
        )

    if use_soft_boundary:
        if tool_name == "hybrid_up":
            math_shift = params.get("math_shift", 15.0)
            edited_ts = hybrid_up_soft(
                ts=ts,
                start_idx=start_idx,
                end_idx=end_idx,
                math_shift=math_shift,
                tedit=tedit,
            )
            log = f"[hybrid_up_soft] region=[{start_idx},{end_idx}], math_shift={math_shift}"

        elif tool_name == "hybrid_down":
            math_shift = params.get("math_shift", -15.0)
            edited_ts = hybrid_down_soft(
                ts=ts,
                start_idx=start_idx,
                end_idx=end_idx,
                math_shift=math_shift,
                tedit=tedit,
            )
            log = f"[hybrid_down_soft] region=[{start_idx},{end_idx}], math_shift={math_shift}"

        elif tool_name == "ensemble_smooth":
            edited_ts = ensemble_smooth_soft(
                ts=ts,
                start_idx=start_idx,
                end_idx=end_idx,
                tedit=tedit,
                n_samples=n_ensemble,
            )
            log = f"[ensemble_smooth_soft] region=[{start_idx},{end_idx}], n_samples={n_ensemble}"

        else:
            raise ValueError(f"Unknown tool: {tool_name}")
    else:
        if tool_name == "hybrid_up":
            math_shift = params.get("math_shift", 15.0)
            edited_ts = hybrid_up(
                ts=ts,
                start_idx=start_idx,
                end_idx=end_idx,
                math_shift=math_shift,
                tedit=tedit,
            )
            log = f"[hybrid_up] region=[{start_idx},{end_idx}], math_shift={math_shift}"

        elif tool_name == "hybrid_down":
            math_shift = params.get("math_shift", -15.0)
            edited_ts = hybrid_down(
                ts=ts,
                start_idx=start_idx,
                end_idx=end_idx,
                math_shift=math_shift,
                tedit=tedit,
            )
            log = f"[hybrid_down] region=[{start_idx},{end_idx}], math_shift={math_shift}"

        elif tool_name == "ensemble_smooth":
            edited_ts = ensemble_smooth(
                ts=ts,
                start_idx=start_idx,
                end_idx=end_idx,
                tedit=tedit,
                n_samples=n_ensemble,
            )
            log = f"[ensemble_smooth] region=[{start_idx},{end_idx}], n_samples={n_ensemble}"

        else:
            raise ValueError(f"Unknown tool: {tool_name}")

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
