# BetterTSE Pipeline Diagnosis

> Date: 2026-03-15

## Scope

This note summarizes the experiments run after the tool/prompt refactor, the CUDA/TEdit runtime fixes applied, and the main failure modes observed in the current BetterTSE pipeline.

## Runtime Fixes Applied

### 1. Removed hardcoded CUDA mask allocation

Files changed:

- `TEdit-main/models/diffusion/diff_csdi_multipatch.py`
- `TEdit-main/models/diffusion/diff_csdi_multipatch_weaver.py`

Fix:

- `get_mask(..., device="cuda:0")` was changed to infer device dynamically.
- This prevents CPU runs from crashing immediately when TEdit builds attention masks.

### 2. Fixed attention mask construction in MultiPatch Weaver

File changed:

- `TEdit-main/models/diffusion/diff_csdi_multipatch_weaver.py`

Fix:

- The attention mask in the weaver path now uses the actual patch-sequence length rather than incorrectly mixing in attribute-token length.
- This resolved the transformer mask shape mismatch that appeared after the first device fix.

## Experiments Run

### A. Single-sample end-to-end run in `tedit` environment

Sample:

- `001 / sensor_offline`

Result:

- LLM selected `volatility_increase`
- Predicted region: `[96, 144]`
- GT region: `[101, 117]`
- `t-IoU = 0.347`
- `editability = 0.663`
- `preservability = 0.988`

Interpretation:

- The planning layer can understand at least some volatility-style prompts.
- The region still drifts wide, but not catastrophically.

### B. LLM-only planning check with new prompt

Representative samples:

- `001 / sensor_offline`
- `002 / heatwave_overload`
- `004 / device_switch`

Observed behavior:

- `sensor_offline`:
  - Tool choice was reasonable: `volatility_increase`
  - Region was broad, but partially overlapped GT
- `heatwave_overload`:
  - Tool choice improved semantically to `hybrid_up`
  - Region still missed the GT event almost completely
- `device_switch`:
  - Planned as `spike_inject`
  - Region missed GT entirely

Interpretation:

- The new prompt helps semantic tool selection.
- The dominant remaining planning problem is localization, not pure intent classification.

### C. Oracle editor run with GT region and all current tools

Representative task types:

- `sensor_offline`
- `heatwave_overload`
- `market_trend`
- `device_switch`
- `facility_shutdown`

Top tools found under GT region:

- `sensor_offline`:
  - best: `hybrid_down` / `trend_quadratic_down`
- `heatwave_overload`:
  - best among current tools still had extremely poor error
- `market_trend`:
  - best: `hybrid_down`
- `device_switch`:
  - best: `hybrid_down`
- `facility_shutdown`:
  - best: `hybrid_down`

Interpretation:

- Several tasks are not naturally matched by the current tool inventory.
- In practice, many samples prefer a downward hybrid tool because the current target transformations are closer to local regime shifts than to clean impulses.
- `heatwave_overload` remains especially badly supported.

### D. GPU-backed 3-sample end-to-end run after TEdit fix

Samples:

- `001 / sensor_offline`
- `002 / heatwave_overload`
- `004 / device_switch`

Results:

- `001 / sensor_offline`
  - tool: `volatility_increase`
  - region: `[80, 140]`
  - `t-IoU = 0.279`
  - `editability = 0.630`
  - `preservability = 0.973`

- `002 / heatwave_overload`
  - tool: `hybrid_up`
  - region: `[80, 140]`
  - `t-IoU = 0.239`
  - `editability = -1.784`
  - `preservability = 0.893`

- `004 / device_switch`
  - tool: `spike_inject`
  - region: `[90, 110]`
  - `t-IoU = 0.0`
  - `editability = 0.442`
  - `preservability = 0.998`

Interpretation:

- The full pipeline now runs on GPU in the current environment.
- The remaining poor performance is no longer a CUDA/runtime issue.
- The dominant errors are now in task grounding, tool coverage, and region selection.

## Main Diagnoses

### 1. Localization is the largest consistent failure mode

The planning layer often picks a broadly plausible effect family, but not the correct time window.

Likely reason:

- Generated prompts use expressions like `今天中午`, `今晚深夜`, `今日清晨`
- But the model is not given an explicit temporal anchor from those natural phrases to sequence indices

Consequence:

- Even when the event type is understood, `t-IoU` remains low

### 2. Tool coverage is still incomplete for several benchmark tasks

Current task-to-tool mismatch is strongest for:

- `heatwave_overload`
- `device_switch`
- `facility_shutdown`

Reasons:

- No dedicated `shutdown_to_zero` tool
- No dedicated `step_change` / regime-switch tool
- No clean `level_shift` operator
- `heatwave_overload` is currently forced into trend-like or impulse-like approximations

### 3. Historical end-to-end runs overused fallback math tools

From the existing ETTh1 validation results:

- `spike_inject` was used most often
- `volatility_increase` was also common
- TEdit-native or hybrid tools were selected much less often

Interpretation:

- The previous prompt/tooling stack was collapsing many distinct event types into a small fallback set.
- This is one reason the overall pipeline looked uniformly weak.

### 4. Background preservation is not the main problem

Preservability is generally strong.

Interpretation:

- The project's local-edit and mask-preservation mechanisms are functioning.
- The main issue is not leakage outside the region.
- The issue is selecting the correct region and performing the correct local transformation.

## Recommended Next Steps

### Priority 1. Fix prompt generation in the testset builder

The event prompts should expose a clearer temporal anchor, for example:

- event starts in the last quarter of the sequence
- event occurs shortly before the end
- event lasts for about 12 timesteps

This should improve region selection much more than another round of prompt polishing alone.

### Priority 2. Extend the editor tool inventory

Highest-value additions:

- `shutdown_to_zero`
- `step_change`
- `level_shift_up`
- `level_shift_down`

Without these, the benchmark includes targets that the current editor cannot express cleanly.

### Priority 3. Evaluate in three separated modes

- `LLM-only`
- `editor-oracle` with GT region
- `full pipeline`

This should become the standard debugging workflow so data issues, planning issues, and editor issues do not get conflated.

## Bottom Line

After the CUDA/runtime fixes, the pipeline is runnable again and the remaining failure modes are now clearer:

- prompt semantics improved somewhat
- localization is still weak
- the benchmark prompt style is under-anchored in time
- the editor tool space still does not fully cover the benchmark task space
