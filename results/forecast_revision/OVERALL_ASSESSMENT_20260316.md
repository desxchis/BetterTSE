# Forecast Revision Overall Assessment

## Summary

This document summarizes the current `Weather`-based forecast revision line after four iterations:

- `v1`: initial Weather rollout
- `v2`: benchmark + calibrator cleanup
- `v3`: `revision_needed` gate fix
- `v4`: operator-aware calibrator tuning

The current conclusion is:

- the `forecast revision` task is valid
- localized revision is consistently better than global revision on region grounding
- the full localized pipeline now shows positive `revision_gain` on both CPU-safe baselines
- the remaining bottleneck is calibration quality and operator-specific execution, not basic task viability

## Main Result Paths

- [naive v1](/root/autodl-tmp/BetterTSE-main/results/forecast_revision/runs/weather_naive_v1/suite_summary.json)
- [naive v2](/root/autodl-tmp/BetterTSE-main/results/forecast_revision/runs/weather_naive_v2/suite_summary.json)
- [naive v3 gate](/root/autodl-tmp/BetterTSE-main/results/forecast_revision/runs/weather_naive_v3_gate/suite_summary.json)
- [naive v4 calibrator](/root/autodl-tmp/BetterTSE-main/results/forecast_revision/runs/weather_naive_v4_cal_fast/suite_summary.json)
- [dlinear v1](/root/autodl-tmp/BetterTSE-main/results/forecast_revision/runs/weather_dlinear_v1/suite_summary.json)
- [dlinear v2](/root/autodl-tmp/BetterTSE-main/results/forecast_revision/runs/weather_dlinear_v2/suite_summary.json)
- [dlinear v3 gate](/root/autodl-tmp/BetterTSE-main/results/forecast_revision/runs/weather_dlinear_v3_gate/suite_summary.json)
- [dlinear v4 calibrator](/root/autodl-tmp/BetterTSE-main/results/forecast_revision/runs/weather_dlinear_v4_cal_fast/suite_summary.json)

Milestones:

- [Weather v1](/root/autodl-tmp/BetterTSE-main/results/forecast_revision/milestones/20260316_weather_v1/README.md)
- [Weather v2](/root/autodl-tmp/BetterTSE-main/results/forecast_revision/milestones/20260316_weather_v2/README.md)
- [Weather v4](/root/autodl-tmp/BetterTSE-main/results/forecast_revision/milestones/20260316_weather_v4_cal/README.md)

## Progress by Iteration

### `naive_last`

| version | localized future MAE | localized revision gain | localized t-IoU | magnitude error | revision_needed match |
|---|---:|---:|---:|---:|---:|
| v1 | 1.044695 | -0.103695 | 0.700000 | 1.187214 | 0.600000 |
| v2 | 0.929637 | -0.021778 | 0.683333 | 0.858497 | 0.666667 |
| v3 | 0.929637 | -0.021778 | 0.683333 | 0.858497 | 1.000000 |
| v4 | 0.963127 | 0.054328 | 0.683333 | 0.530218 | 1.000000 |

Interpretation:

- v1 -> v2: main gain came from fixing operator coverage and gross calibration mismatch
- v2 -> v3: gate issue was removed, but score stayed flat, so gate was not the main remaining blocker
- v3 -> v4: calibrator tuning flipped `revision_gain` positive

### `dlinear_like`

| version | localized future MAE | localized revision gain | localized t-IoU | magnitude error | revision_needed match |
|---|---:|---:|---:|---:|---:|
| v1 | 1.513846 | -0.036091 | 0.700000 | 1.111162 | 0.600000 |
| v2 | 1.407083 | 0.038068 | 0.683333 | 0.748269 | 0.666667 |
| v3 | 1.407083 | 0.038068 | 0.683333 | 0.748269 | 1.000000 |
| v4 | 1.392258 | 0.086617 | 0.683333 | 0.530218 | 1.000000 |

Interpretation:

- `dlinear_like` was the first baseline where localized full revision turned positive
- v4 strengthens the conclusion by further reducing calibration error

## What Is Established

### 1. The task itself is valid

The full localized revision pipeline now achieves positive `revision_gain` on both:

- `naive_last`
- `dlinear_like`

That is the main go/no-go criterion for this stage.

### 2. Localized revision is better than global revision

On both baselines, localized revision keeps a much stronger region signal:

- `global_revision_only`: `avg_future_t_iou ≈ 0.186`
- `localized_full_revision`: `avg_future_t_iou ≈ 0.683`

This result has been stable across iterations.

### 3. Error decomposition is meaningful

The oracle chain consistently improves revision-target quality. For example, on `dlinear_like v4`:

- `localized_full_revision`: `revision_gain = 0.086617`
- `oracle_region`: `revision_gain = 0.123587`
- `oracle_intent`: `revision_gain = 0.185551`

This supports the decomposition into:

- `where`
- `what`
- `how much`

### 4. `revision_needed` is no longer the dominant problem

After the gate fix:

- `avg_revision_needed_match = 1.0`
- `applicable_avg_revision_needed_match = 1.0`

The system is no longer failing because it misses positive revision cases in this benchmark.

## What Is Not Yet Established

### 1. Full localized revision is not yet beating base forecast on future MAE

Positive `revision_gain` is already enough to validate the revision task in the controlled benchmark, but:

- `localized_full_revision` still does not outperform `base_only` on `avg_edited_mae_vs_future_gt`

So the current system is better at matching the controlled revision target than at improving true future error.

This matters because:

- `revision_target` validates edit faithfulness
- `future_gt` validates forecasting utility

Both are useful, but they are not the same.

### 2. Oracle calibration is still not a monotonic future-GT winner

On the revision-target side, oracle stages help.
On the `future_gt` side, gains are not perfectly monotonic.

This means:

- the benchmark decomposition is working
- but the controlled revision target is not identical to the true best correction for `future_gt`

That is expected in a semi-synthetic setup and should be stated explicitly in later reporting.

### 3. Operator difficulty is uneven

Current operator-level diagnosis:

- `step` and `plateau` are already useful
- `hump` is modest but positive
- `flatline` remains the most damaging family
- `irregular_noise` is still unstable and should not be the immediate optimization target

## Recommended Breakpoint Judgment

At this breakpoint, the overall result should be evaluated as:

- **task viability**: confirmed
- **localized vs global advantage**: confirmed
- **oracle decomposition value**: confirmed
- **need to continue framework redesign**: no
- **need to continue method refinement**: yes, but narrowly on calibration/execution quality

## Recommended Next Step

Do not expand scope immediately.

The most disciplined next move is one of:

1. add a stronger baseline such as `PatchTST` only if it can be integrated cheaply, or
2. keep the current baselines and move to the first real-world complement `TraffiDent`

Given current progress, the safest path is:

- keep the current Weather synthetic line as the controlled benchmark
- use the current v4 result as the synthetic proof point
- start implementing the real-world `TraffiDent` complement
