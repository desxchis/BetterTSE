# Forecast Revision Experiment Summary

## Summary

This document consolidates the current forecast-revision evidence chain into one place:

- `Weather v4` as the controlled proof point
- `XTraffic v2 narrowed` as the best pure real-data positive-transfer checkpoint and main real benchmark
- `XTraffic v2 nonapp` as the real-data gate evaluation checkpoint
- `MTBench finance v2` as the native-text real complement
- `CiK` as the benchmark-design template rather than the main empirical battlefield

The current overall judgment is:

- the task is valid
- localized revision is better than global revision
- the framework transfers to real data
- real `revision_needed` behavior is now verified on real no-op samples
- the framework also transfers to native text + time-series data once the revision schema matches the domain
- the benchmark plan should now stay role-disciplined rather than expanding broadly

## Primary References

- [Weather overall assessment](/root/autodl-tmp/BetterTSE-main/results/forecast_revision/OVERALL_ASSESSMENT_20260316.md)
- [XTraffic overall assessment](/root/autodl-tmp/BetterTSE-main/results/forecast_revision/OVERALL_ASSESSMENT_XTRAFFIC_20260317.md)
- [Weather v4 milestone](/root/autodl-tmp/BetterTSE-main/results/forecast_revision/milestones/20260316_weather_v4_cal/README.md)
- [XTraffic v2 narrowed milestone](/root/autodl-tmp/BetterTSE-main/results/forecast_revision/milestones/20260317_xtraffic_v2_narrowed/README.md)
- [XTraffic v2 nonapp milestone](/root/autodl-tmp/BetterTSE-main/results/forecast_revision/milestones/20260317_xtraffic_v2_nonapp/README.md)
- [MTBench finance v2 milestone](/root/autodl-tmp/BetterTSE-main/results/forecast_revision/milestones/20260317_mtbench_v2_finance/README.md)
- [MTBench finance v2 100-sample milestone](/root/autodl-tmp/BetterTSE-main/results/forecast_revision/milestones/20260317_mtbench_v2_100/README.md)

## Role Summary

The paper-facing dataset split should now be read as:

1. `XTraffic`
   - main benchmark
   - main real empirical battlefield for localized forecast revision
2. `MTBench`
   - realism / native-text complement
3. `CiK`
   - benchmark-design and evaluation template
4. `Time-MMD` or `Time-IMM`
   - optional later extension, not current core evidence
5. `CGTSF` and pure traffic-only benchmarks
   - lightweight complement or appendix material

This role split is important because the main current risk is paper dilution, not insufficient data diversity.

## Controlled Benchmark: Weather v4

### `naive_last`

| mode | revision gain | future t-IoU | magnitude error |
|---|---:|---:|---:|
| localized_full_revision | 0.0543 | 0.6833 | 0.5302 |
| oracle_region | 0.0913 | 0.8000 | 0.5343 |
| oracle_intent | 0.1533 | 0.8000 | 0.2841 |

### `dlinear_like`

| mode | revision gain | future t-IoU | magnitude error |
|---|---:|---:|---:|
| localized_full_revision | 0.0866 | 0.6833 | 0.5302 |
| oracle_region | 0.1236 | 0.8000 | 0.5343 |
| oracle_intent | 0.1856 | 0.8000 | 0.2841 |

Takeaway:

- controlled forecast revision is established
- localized region grounding is strong
- oracle chain remains useful and mostly monotonic

## Real Data: XTraffic v2 Narrowed

Subset:

- `speed`
- high-confidence incident types
- observable future response only

### `dlinear_like`

| mode | revision gain | future t-IoU | magnitude error |
|---|---:|---:|---:|
| base_only | 0.0000 | 0.0000 | 4.3061 |
| global_revision_only | 0.1761 | 0.1103 | 3.8697 |
| localized_full_revision | 0.4065 | 0.5044 | 2.7274 |
| oracle_region | 0.2509 | 1.0000 | 2.6632 |
| oracle_intent | 0.2539 | 1.0000 | 1.9464 |

Takeaway:

- real-data positive transfer is established on `dlinear_like`
- localized revision clearly beats global revision
- current best pure real-data positive checkpoint is still `XTraffic v2 narrowed`
- this is the main real benchmark that should anchor the empirical story

## Real Data Gate Evaluation: XTraffic v2 Non-Applicable

Sample mix:

- total: `23`
- applicable: `14`
- non-applicable: `9`

### `dlinear_like`

| mode | overall gain | applicable gain | future t-IoU | non-app revision_needed match | non-app over-edit |
|---|---:|---:|---:|---:|---:|
| base_only | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 |
| global_revision_only | 0.0577 | 0.0948 | 0.0429 | 1.0000 | 0.0000 |
| localized_full_revision | 0.1259 | 0.2068 | 0.2572 | 1.0000 | 0.0000 |
| oracle_region | 0.0540 | 0.0887 | 0.6087 | 1.0000 | 0.0000 |

Important caveat:

- `oracle_intent` forces GT intent for all samples
- once real non-applicable samples are mixed in, its `revision_needed_match` is no longer a meaningful gate metric

Takeaway:

- real no-op detection is now verified
- on real non-applicable windows:
  - `revision_needed_match = 1.0`
  - `over_edit_rate = 0.0`
- localized revision remains positive overall even after adding real no-op samples

## Real Text Complement: MTBench Finance v2

The finance line uses domain-matched revision semantics:

- `repricing -> step + full_horizon`
- `drift_adjust -> plateau + full_horizon`
- `neutral -> no_revision`

### `naive_last`

| mode | overall gain | applicable gain | future t-IoU |
|---|---:|---:|---:|
| global_revision_only | 0.0003 | 0.0005 | 0.7000 |
| localized_full_revision | 0.6043 | 0.8632 | 0.3218 |
| oracle_region | 0.6488 | 0.9268 | 0.7000 |

### `dlinear_like`

| mode | overall gain | applicable gain | future t-IoU |
|---|---:|---:|---:|
| global_revision_only | 0.1757 | 0.2510 | 0.7000 |
| localized_full_revision | 0.2396 | 0.3423 | 0.3218 |
| oracle_region | 0.3164 | 0.4520 | 0.7000 |

Takeaway:

- MTBench becomes positive once the revision schema matches the finance domain
- this confirms the framework can extend beyond structured incident context to native long-text financial news
- MTBench should be read as realism support, not as the main benchmark replacing `XTraffic`

## Real Text Complement: MTBench Finance v2 (100 Samples)

This is the first non-smoke stability check on the finance line.

Sample mix:

- total: `100`
- applicable: `71`
- non-applicable: `29`

### `naive_last`

| mode | overall gain | applicable gain | future t-IoU | non-app gate match |
|---|---:|---:|---:|---:|
| global_revision_only | 0.0002 | 0.0003 | 0.7100 | 1.0000 |
| localized_full_revision | 0.1515 | 0.2133 | 0.2941 | 1.0000 |
| oracle_region | 0.0722 | 0.1017 | 0.7100 | 1.0000 |

### `dlinear_like`

| mode | overall gain | applicable gain | future t-IoU | non-app gate match |
|---|---:|---:|---:|---:|
| global_revision_only | 0.1146 | 0.1614 | 0.7100 | 1.0000 |
| localized_full_revision | 0.3765 | 0.5303 | 0.2941 | 1.0000 |
| oracle_region | 0.4748 | 0.6687 | 0.7100 | 1.0000 |

Takeaway:

- the MTBench finance line remains positive beyond the 10-sample smoke subset
- `localized_full_revision` continues to beat `global_revision_only` on both baselines
- the native-text complement is now stable enough to treat as part of the main evidence chain

## Current Evidence Chain

The current evidence chain is now:

1. `Weather v4`
   - proves the controlled benchmark task is valid
2. `XTraffic v2 narrowed`
   - proves localized revision can transfer to real traffic data
3. `XTraffic v2 nonapp`
   - proves the gate can abstain correctly on real no-op windows
4. `MTBench finance v2`
   - proves the framework can also transfer to native text + time-series data once the domain schema is adapted
5. `MTBench finance v2 (100 samples)`
   - proves the native-text complement is stable beyond a tiny smoke subset

## Calibration Progress

A first runnable calibration scaffold is now in place.

What has already been added:

- explicit `edit_spec` outputs and GT extraction
- calibration metrics in the main runner
- a dedicated calibration benchmark script for oracle-region / oracle-intent analysis

Current interpretation:

- the main open problem remains `how much to edit`
- but it is now isolated as a measurable subproblem rather than a vague planner field
- the next experiments should use this scaffold on `Weather`, then `XTraffic`, then `MTBench`

## Calibration Framework Progress

The repo now also has a config-driven experiment assembly layer for calibration work.

Primary additions:

- framework runner:
  - [prepare_forecast_revision_calibration_framework.py](/root/autodl-tmp/BetterTSE-main/test_scripts/prepare_forecast_revision_calibration_framework.py)
- config examples:
  - [weather_dlinear_v2.json](/root/autodl-tmp/BetterTSE-main/configs/forecast_revision_calibration/weather_dlinear_v2.json)
  - [xtraffic_dlinear_v2.json](/root/autodl-tmp/BetterTSE-main/configs/forecast_revision_calibration/xtraffic_dlinear_v2.json)
  - [mtbench_dlinear_v2_100.json](/root/autodl-tmp/BetterTSE-main/configs/forecast_revision_calibration/mtbench_dlinear_v2_100.json)

This layer standardizes:

- stage naming
- output directory layout
- learned-calibrator train/eval dependency wiring
- dry-run planning before actual execution

This is the right immediate emphasis when compute is not the bottleneck to solve.

## Recommended Breakpoint

At this breakpoint, the line should be treated as established enough to stop redesigning the framework.

What is established:

- task viability
- localized over global advantage
- real-data viability
- real gate viability
- native-text real-data viability
- native-text real-data stability beyond smoke scale
- a clear dataset-role split that keeps the paper narrative focused

What remains open:

- stronger real-data calibration
- cleaner real supervision beyond weak-label incident alignment
- larger-scale MTBench evaluation beyond the current smoke subset
- a cleaner solution to `how much to edit`

## Recommended Next Step

Do not keep changing the framework.

The disciplined next move is:

1. keep `XTraffic` as the main benchmark
2. keep `MTBench` as the realism / native-text complement
3. keep `CiK` as the benchmark-design template rather than a parallel main benchmark
4. start calibration-focused work on `how much to edit`
5. only after that, consider one broader extension dataset such as `Time-MMD` or `Time-IMM`
