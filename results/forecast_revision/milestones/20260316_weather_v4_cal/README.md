# Weather Forecast Revision v4 Calibration

## Summary

This milestone records the first calibrator-tuned Weather run where
`localized_full_revision` achieves positive `revision_gain` on both:

- `naive_last`
- `dlinear_like`

The change from the previous milestone is focused and narrow:

- no benchmark redesign
- no new baseline
- no new planner
- only operator-aware calibrator tuning

## Result Paths

- `results/forecast_revision/runs/weather_naive_v4_cal_fast/`
- `results/forecast_revision/runs/weather_dlinear_v4_cal_fast/`

Copied summaries:

- `weather_naive_suite_summary.json`
- `weather_dlinear_suite_summary.json`

## Key Numbers

### naive_last

- `localized_full_revision`
  - `avg_revision_gain = 0.054328`
  - `avg_future_t_iou = 0.683333`
  - `avg_magnitude_calibration_error = 0.530218`
- previous gate-only version:
  - `avg_revision_gain = -0.021778`
  - `avg_magnitude_calibration_error = 0.858497`

### dlinear_like

- `localized_full_revision`
  - `avg_revision_gain = 0.086617`
  - `avg_future_t_iou = 0.683333`
  - `avg_magnitude_calibration_error = 0.530218`
- previous gate-only version:
  - `avg_revision_gain = 0.038068`
  - `avg_magnitude_calibration_error = 0.748269`

## What Changed

The calibrator was moved closer to benchmark operator generation:

- `step / plateau / flatline` amplitudes now use stronger operator-aligned scaling
- `plateau` recovery is no longer overly damped
- `flatline` floor is anchored to forecast-level stats instead of local window stats
- `irregular_noise` remains conservative and is not yet the main optimization target

## Current Interpretation

This is the first clear positive signal that the v1 forecast-revision line is working:

- `localized_full_revision` now beats `base_only` on revision-target gain for both CPU-safe baselines
- `localized` still strongly beats `global` on region fidelity
- oracle chains remain monotonic

The next bottleneck is no longer basic viability. The likely next step is:

1. bring in `PatchTST`, or
2. add operator-level result tables and then move to `TraffiDent`
