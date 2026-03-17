# Weather Forecast Revision v1

## Summary

- Dataset: `Weather`
- Forecast horizon: `24`
- Context window: `96`
- Baselines:
  - `naive_last`
  - `dlinear_like`
- Revision operators:
  - `hump`
  - `step`
  - `plateau`
  - `flatline`
  - `irregular_noise`
- Negative sample cadence: every 5th sample is `revision_applicable_gt = false`

This milestone is the first end-to-end `forecast revision` run on the planned main synthetic dataset. It validates the benchmark builder, baseline interface, suite runner, and oracle decomposition outputs.

## Included Files

- `forecast_revision_Weather_naive_last_30.json`
- `forecast_revision_Weather_dlinear_like_30.json`
- `weather_naive_suite_summary.json`
- `weather_dlinear_suite_summary.json`

Full run directories remain under:

- `results/forecast_revision/runs/weather_naive_v1/`
- `results/forecast_revision/runs/weather_dlinear_v1/`

## Key Results

### naive_last

- `base_only`: `avg_edited_mae_vs_future_gt = 0.834861`
- `global_revision_only`: `avg_edited_mae_vs_future_gt = 0.834675`
- `localized_full_revision`: `avg_edited_mae_vs_future_gt = 1.044695`
- `oracle_calibration`: `avg_edited_mae_vs_future_gt = 0.975141`

Editing-side signals:

- `localized_full_revision`: `avg_future_t_iou = 0.700000`
- `oracle_region`: `avg_future_t_iou = 0.800000`
- `oracle_calibration`: `avg_revision_gain = 0.073398`

Interpretation:

- localization is already meaningful
- current rule-based localized editor is too aggressive on this baseline
- calibration remains the main bottleneck

### dlinear_like

- `base_only`: `avg_edited_mae_vs_future_gt = 1.246324`
- `global_revision_only`: `avg_edited_mae_vs_future_gt = 1.425227`
- `localized_full_revision`: `avg_edited_mae_vs_future_gt = 1.513846`
- `oracle_calibration`: `avg_edited_mae_vs_future_gt = 1.411458`

Editing-side signals:

- `localized_full_revision`: `avg_future_t_iou = 0.700000`
- `oracle_region`: `avg_future_t_iou = 0.800000`
- oracle chain on `avg_revision_gain`:
  - `oracle_region = 0.020875`
  - `oracle_intent = 0.042870`
  - `oracle_calibration = 0.121989`

Interpretation:

- oracle stages improve monotonically
- the decomposition is valid
- current localized runtime still loses too much on `how much / how long`

## Current Diagnosis

What is working:

- Weather benchmark generation works on a clean mainline dataset
- baseline -> benchmark -> suite runner loop is stable
- localized revision strongly improves region fidelity over global revision
- oracle decomposition produces interpretable staged gains
- negative samples expose `revision_needed` behavior cleanly

What is not good enough yet:

- `localized_full_revision` does not yet beat `base_only`
- `global_revision_only` and `localized_full_revision` both still over-edit in positive cases
- current calibrator is too coarse for baseline-dependent scaling

## Next Steps

1. make calibrator baseline-aware using local forecast stats and operator-specific scaling
2. reduce localized edit magnitude for `naive_last` and `dlinear_like`
3. add operator-level summary tables to identify which families dominate the loss
4. after localized revision improves on Weather, bring in `PatchTST`
5. keep `TraffiDent` as the first real-world follow-up
