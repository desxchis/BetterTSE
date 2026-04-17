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

## Dataset Role Freeze

The paper-facing dataset roles should now be treated as fixed:

- `XTraffic`:
  - main benchmark / main empirical battlefield
- `MTBench`:
  - realism and native-text complement
- `CiK`:
  - protocol and benchmark-design template
- `Time-MMD` or `Time-IMM`:
  - optional later extension, not current mainline
- `CGTSF` and pure traffic-only data:
  - lightweight complement or appendix material

The main current risk is not lack of data.
It is benchmark-role sprawl that dilutes the paper narrative.

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
- multiple new main benchmarks opened in parallel

## Current Main Open Problems

The remaining headroom is now much narrower:

- stronger domain-specific calibration
- cleaner execution quality on real data
- larger-scale MTBench validation beyond the current 100-sample checkpoint
- a cleaner treatment of `how much to edit`

## Calibration Scaffold Status

A first explicit calibration scaffold is now wired into the repo.

Added code paths:

- `modules/forecast_revision.py`
  - explicit `edit_spec` prediction
  - `edit_spec -> executor params` projection
  - calibration metrics: `normalized_parameter_error`, `peak_delta_error`, `signed_area_error`, `duration_error`, `recovery_slope_error`
- `run_forecast_revision.py`
  - `calibration_strategy` switch
  - result payloads now store `edit_spec`, `edit_spec_gt`, and `calibration_metrics`
- `test_scripts/build_forecast_revision_benchmark.py`
  - synthetic samples now export `edit_spec_gt`
- `test_scripts/run_forecast_revision_calibration_benchmark.py`
  - dedicated oracle-region / oracle-intent calibration benchmark runner

Current verification status:

- syntax check passed
- 3-sample ETTh1 smoke test passed end to end
- `normalized_parameter_error` is now numerically stable after shape-aware GT spec extraction

This means the next stage is no longer abstract planning.
The repo now has a runnable calibration benchmark skeleton.

## Calibration Framework Status

The next step has now been narrowed further:

- do not prioritize more training while GPU mode is off
- prioritize framework assembly for calibration experiments

Added framework layer:

- config-driven planner:
  - [prepare_forecast_revision_calibration_framework.py](/root/autodl-tmp/BetterTSE-main/test_scripts/prepare_forecast_revision_calibration_framework.py)
- config examples:
  - [weather_dlinear_v2.json](/root/autodl-tmp/BetterTSE-main/configs/forecast_revision_calibration/weather_dlinear_v2.json)
  - [xtraffic_dlinear_v2.json](/root/autodl-tmp/BetterTSE-main/configs/forecast_revision_calibration/xtraffic_dlinear_v2.json)
  - [mtbench_dlinear_v2_100.json](/root/autodl-tmp/BetterTSE-main/configs/forecast_revision_calibration/mtbench_dlinear_v2_100.json)

What this framework layer does:

- writes a reusable `experiment_plan.json`
- writes a runnable `run_commands.sh`
- writes a per-experiment `README.md`
- keeps training, oracle benchmark, and semi-oracle suite wired through one config

What this framework layer does not do yet:

- it does not claim new empirical gains
- it does not require GPU execution
- it does not replace the current benchmark scripts

Its role is only to make the next calibration stage reproducible and easier to resume.

## Recommended Immediate Next Step

Do not open a new dataset line yet.

The next work should stay inside the current framework and focus on calibration:

1. keep the current checkpoint frozen:
   - `Weather v4`
   - `XTraffic v2 narrowed`
   - `XTraffic v2 nonapp`
   - `MTBench finance v2 (100)`
2. keep dataset roles fixed:
   - `XTraffic` as main benchmark
   - `MTBench` as realism complement
   - `CiK` as protocol template
3. start a small `how much to edit` calibration line inside the existing revision pipeline
4. keep calibration work config-driven and resumable while GPU training is deferred
5. only after calibration ablations are stable, consider one broader extension dataset

## Candidate Future Dataset Priority

If a new dataset must be opened later, use this priority:

1. `Time-MMD`
2. `Time-IMM`
3. `CGTSF`

But none of them should be started before the current breakpoint is fully stabilized.

## Status Update 2026-04-18

A small repo-level archival note is added here to freeze two pure-editing strength breakpoints without reopening the forecast-revision line itself.

Frozen conclusions:

1. beta-direction repair is validated:
   - the beta path no longer drives edit gain in the inverted direction
   - gain now follows the intended positive direction after the repair
2. P2 downstream diagnostics restoration is validated:
   - after the DDIM-forward conditioning fix, downstream stage diagnostics are populated again
   - observability is restored, but control quality is still not fully solved

This note is only a cross-line archival pointer.
It does not change the forecast-revision breakpoint, dataset roles, or established evidence chain above.

## Primary Indexes

- [Experiment summary](/root/autodl-tmp/BetterTSE-main/results/forecast_revision/EXPERIMENT_SUMMARY_20260317.md)
- [Results index](/root/autodl-tmp/BetterTSE-main/results/README.md)
