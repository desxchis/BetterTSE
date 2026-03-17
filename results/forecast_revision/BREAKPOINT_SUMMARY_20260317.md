# Forecast Revision Breakpoint Summary

## Breakpoint

Current work should be considered frozen at this breakpoint:

- `Weather v4`
- `XTraffic v2 narrowed`
- `XTraffic v2 nonapp`
- `MTBench finance v2`
- `MTBench finance v2 (100 samples)`

This is the current stable evidence chain for the forecast-revision line.

## Status Update 2026-03-17

The two immediate stabilization steps documented at this breakpoint are now completed:

1. unified case studies and visualizations across:
   - `Weather`
   - `XTraffic`
   - `MTBench`
2. small reproducibility reruns on the current best checkpoints

New reproducibility result:

- rerun set:
  - `Weather v4 controlled` (`dlinear_like`)
  - `XTraffic v2 narrowed` (`dlinear_like`)
  - `XTraffic v2 nonapp` (`dlinear_like`)
  - `MTBench finance v2 100` (`dlinear_like`)
- result:
  - all checkpoint reruns matched the saved `suite_summary.json` files exactly
  - `all_exact_match = True`
  - `max_abs_diff = 0.0`

References:

- [Unified case studies](/root/autodl-tmp/BetterTSE-main/results/forecast_revision/case_studies/20260317_unified/CASE_STUDIES_20260317.md)
- [Repro check report](/root/autodl-tmp/BetterTSE-main/results/forecast_revision/repro_checks/20260317_resume_check/repro_check_report.md)

## What Is Established

### 1. Controlled task validity

`Weather v4` establishes that the forecast-revision task is valid under controlled supervision.

Core result:

- `localized_full_revision` is positive on both `naive_last` and `dlinear_like`
- `localized` consistently beats `global`
- the oracle chain remains meaningful

Reference:

- [Weather v4 milestone](/root/autodl-tmp/BetterTSE-main/results/forecast_revision/milestones/20260316_weather_v4_cal/README.md)

### 2. Real structured-event transfer

`XTraffic v2 narrowed` establishes that the framework transfers to real traffic data when the real-data subset is narrowed to:

- `speed`
- high-confidence incident types
- observable future response

Core result:

- `localized_full_revision` beats `global_revision_only`
- real positive transfer is established on `dlinear_like`

Reference:

- [XTraffic v2 narrowed milestone](/root/autodl-tmp/BetterTSE-main/results/forecast_revision/milestones/20260317_xtraffic_v2_narrowed/README.md)

### 3. Real no-op gate behavior

`XTraffic v2 nonapp` establishes that the gate can abstain correctly on real non-applicable windows.

Core result:

- real `revision_needed_match = 1.0`
- real `over_edit_rate = 0.0`

Reference:

- [XTraffic v2 nonapp milestone](/root/autodl-tmp/BetterTSE-main/results/forecast_revision/milestones/20260317_xtraffic_v2_nonapp/README.md)

### 4. Real native-text transfer

`MTBench finance v2` establishes that the framework also transfers to native text + time-series data, provided the revision schema matches the domain.

Core finance schema:

- `repricing -> step + full_horizon`
- `drift_adjust -> plateau + full_horizon`
- `neutral -> no_revision`

Reference:

- [MTBench finance v2 milestone](/root/autodl-tmp/BetterTSE-main/results/forecast_revision/milestones/20260317_mtbench_v2_finance/README.md)

### 5. Native-text stability beyond smoke scale

`MTBench finance v2 (100 samples)` establishes that the native-text line is still positive beyond a tiny smoke subset.

Core result:

- `naive_last`
  - `localized_full_revision avg_revision_gain = 0.1515`
  - `global_revision_only avg_revision_gain = 0.0002`
- `dlinear_like`
  - `localized_full_revision avg_revision_gain = 0.3765`
  - `global_revision_only avg_revision_gain = 0.1146`

Reference:

- [MTBench finance v2 100-sample milestone](/root/autodl-tmp/BetterTSE-main/results/forecast_revision/milestones/20260317_mtbench_v2_100/README.md)

## What Should Not Change Now

Do not keep redesigning the main framework.

The following are already stable enough:

- task framing: `history + base_forecast + context -> edited_forecast`
- decomposition: `revision_needed / where / what / how much`
- outer vs inner structure
- synthetic + real + native-text evidence chain

Also do not expand the data pool immediately.

The line does not currently need:

- more domains right away
- new large baseline integrations
- another round of framework refactor

## Current Main Open Problems

The remaining headroom is now much narrower:

- stronger domain-specific calibration
- cleaner execution quality on real data
- larger-scale MTBench validation beyond the current 100-sample checkpoint
- a cleaner treatment of `how much to edit`

## Recommended Immediate Next Step

Do not open a new dataset line yet.

The next work should stay inside the current framework and focus on calibration:

1. keep the current checkpoint frozen:
   - `Weather v4`
   - `XTraffic v2 narrowed`
   - `XTraffic v2 nonapp`
   - `MTBench finance v2 (100)`
2. start a small `how much to edit` calibration line inside the existing revision pipeline
3. compare calibration variants under the same method ladder:
   - `base_only`
   - `global_revision_only`
   - `localized_full_revision`
   - `oracle_region`
   - `oracle_intent`
   - `oracle_calibration`
4. only after calibration ablations are stable, consider a new dataset line

## Candidate Future Dataset Priority

If a new dataset must be opened later, use this priority:

1. `CGTSF`
2. `Time-MMD`
3. `Time-IMM`

But none of them should be started before the current breakpoint is fully stabilized.

## Primary Indexes

- [Experiment summary](/root/autodl-tmp/BetterTSE-main/results/forecast_revision/EXPERIMENT_SUMMARY_20260317.md)
- [Results index](/root/autodl-tmp/BetterTSE-main/results/README.md)
