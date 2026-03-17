# Weather Forecast Revision v2

## Summary

- Main synthetic dataset: `Weather`
- Horizon: `24`
- Context length: `96`
- Baselines:
  - `naive_last`
  - `dlinear_like`
- Operator families now covered:
  - `hump`
  - `step`
  - `plateau`
  - `flatline`
  - `irregular_noise`
  - plus `revision_applicable_gt = false` negatives

This milestone follows `20260316_weather_v1` and fixes two concrete issues:

1. `plateau` was unintentionally missing because negative samples replaced every 5th operator slot.
2. The rule-based calibrator was too aggressive for `flatline` and `irregular_noise`.

## Included Files

- `forecast_revision_Weather_naive_last_30.json`
- `forecast_revision_Weather_dlinear_like_30.json`
- `weather_naive_suite_summary.json`
- `weather_dlinear_suite_summary.json`

Full run directories remain under:

- `results/forecast_revision/runs/weather_naive_v2/`
- `results/forecast_revision/runs/weather_dlinear_v2/`

## Key Results

### naive_last

- `base_only`: `avg_edited_mae_vs_future_gt = 0.834861`
- `global_revision_only`: `avg_edited_mae_vs_future_gt = 0.834694`
- `localized_full_revision`: `avg_edited_mae_vs_future_gt = 0.929637`
- `oracle_region`: `avg_revision_gain = 0.013087`
- `oracle_intent`: `avg_revision_gain = 0.046347`
- `oracle_calibration`: `avg_revision_gain = 0.137891`

Compared with v1:

- `localized_full_revision` revision gain improved from `-0.103695` to `-0.021778`
- `localized_full_revision` future MAE improved from `1.044695` to `0.929637`
- `localized_full_revision` magnitude calibration error improved from `1.187214` to `0.858497`

### dlinear_like

- `base_only`: `avg_edited_mae_vs_future_gt = 1.246324`
- `global_revision_only`: `avg_edited_mae_vs_future_gt = 1.382822`
- `localized_full_revision`: `avg_edited_mae_vs_future_gt = 1.407083`
- `oracle_region`: `avg_revision_gain = 0.072933`
- `oracle_intent`: `avg_revision_gain = 0.106194`
- `oracle_calibration`: `avg_revision_gain = 0.170180`

Compared with v1:

- `localized_full_revision` revision gain improved from `-0.036091` to `0.038068`
- `localized_full_revision` edited-vs-target MAE improved from `0.332502` to `0.218076`
- `localized_full_revision` magnitude calibration error improved from `1.111162` to `0.748269`

## Current Diagnosis

What is now true:

- `localized_full_revision` clearly beats `global_revision_only` on region fidelity
- oracle chains improve monotonically on both baselines
- calibrator fixes now produce positive localized revision gain on `dlinear_like`

What is still not good enough:

- `localized_full_revision` still does not beat `base_only` on Weather future MAE
- `naive_last` remains especially sensitive to over-editing
- `revision_needed` recall on positive samples is still limited

## Recommended Next Step

Focus on one thing:

1. improve `revision_needed` and positive-sample triggering before bringing in `PatchTST`

Reason:

- localization is already usable
- oracle gains are clear
- current bottleneck is no longer only calibration; the gate still misses too many applicable revisions
