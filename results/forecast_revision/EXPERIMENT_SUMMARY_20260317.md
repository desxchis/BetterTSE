# Forecast Revision Experiment Summary

## Summary

This document consolidates the current forecast-revision evidence chain into one place:

- `Weather v4` as the controlled proof point
- `XTraffic v2 narrowed` as the best pure real-data positive-transfer checkpoint
- `XTraffic v2 nonapp` as the real-data gate evaluation checkpoint
- `MTBench finance v2` as the native-text real complement

The current overall judgment is:

- the task is valid
- localized revision is better than global revision
- the framework transfers to real data
- real `revision_needed` behavior is now verified on real no-op samples
- the framework also transfers to native text + time-series data once the revision schema matches the domain

## Primary References

- [Weather overall assessment](/root/autodl-tmp/BetterTSE-main/results/forecast_revision/OVERALL_ASSESSMENT_20260316.md)
- [XTraffic overall assessment](/root/autodl-tmp/BetterTSE-main/results/forecast_revision/OVERALL_ASSESSMENT_XTRAFFIC_20260317.md)
- [Weather v4 milestone](/root/autodl-tmp/BetterTSE-main/results/forecast_revision/milestones/20260316_weather_v4_cal/README.md)
- [XTraffic v2 narrowed milestone](/root/autodl-tmp/BetterTSE-main/results/forecast_revision/milestones/20260317_xtraffic_v2_narrowed/README.md)
- [XTraffic v2 nonapp milestone](/root/autodl-tmp/BetterTSE-main/results/forecast_revision/milestones/20260317_xtraffic_v2_nonapp/README.md)
- [MTBench finance v2 milestone](/root/autodl-tmp/BetterTSE-main/results/forecast_revision/milestones/20260317_mtbench_v2_finance/README.md)
- [MTBench finance v2 100-sample milestone](/root/autodl-tmp/BetterTSE-main/results/forecast_revision/milestones/20260317_mtbench_v2_100/README.md)

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

- MTBench becomes positive once the revision schema matches the finance domain.
- This confirms the framework can extend beyond structured incident context to native long-text financial news.

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

- The MTBench finance line remains positive beyond the 10-sample smoke subset.
- `localized_full_revision` continues to beat `global_revision_only` on both baselines.
- The native-text complement is now stable enough to treat as part of the main evidence chain.

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

## Recommended Breakpoint

At this breakpoint, the line should be treated as established enough to stop redesigning the framework.

What is established:

- task viability
- localized over global advantage
- real-data viability
- real gate viability
- native-text real-data viability
- native-text real-data stability beyond smoke scale

What remains open:

- stronger real-data calibration
- cleaner real supervision beyond weak-label incident alignment
- larger-scale MTBench evaluation beyond the current smoke subset

## Recommended Next Step

Do not keep changing the framework.

The disciplined next move is:

1. keep `Weather v4` as the controlled proof point
2. keep `XTraffic v2 narrowed` as the best real positive-transfer checkpoint
3. keep `XTraffic v2 nonapp` as the real gate checkpoint
4. keep `MTBench finance v2` as the native-text checkpoint
5. then either:
   - scale MTBench beyond smoke size, or
   - add MTBench case studies / visualizations
