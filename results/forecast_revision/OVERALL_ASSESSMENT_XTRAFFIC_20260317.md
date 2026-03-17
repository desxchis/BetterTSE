# XTraffic Real-Context Overall Assessment

## Summary

This document summarizes the current `XTraffic` real-context revision line after three iterations:

- `v1`: direct transfer from Weather-style semantics on real traffic data
- `v2`: narrowed traffic subset on `speed` with response-driven filtering
- `v3`: traffic-specific operator refinement on the same narrowed subset

The current conclusion is:

- real-data access and `incident -> node/time-window` alignment are verified
- the `forecast revision` framework does transfer to real traffic data, but not without domain-aware narrowing
- the current best real-data checkpoint is `v2 narrowed`
- the main remaining bottleneck is traffic-specific calibration and execution quality, not task definition or basic localization

## Main Result Paths

- [xtraffic v1 naive](/root/autodl-tmp/BetterTSE-main/results/forecast_revision/runs/xtraffic_p01_naive_v1_real_suite/suite_summary.json)
- [xtraffic v1 dlinear](/root/autodl-tmp/BetterTSE-main/results/forecast_revision/runs/xtraffic_p01_dlinear_v1_real_suite/suite_summary.json)
- [xtraffic v2 naive](/root/autodl-tmp/BetterTSE-main/results/forecast_revision/runs/xtraffic_p01_speed_naive_v2_real_suite/suite_summary.json)
- [xtraffic v2 dlinear](/root/autodl-tmp/BetterTSE-main/results/forecast_revision/runs/xtraffic_p01_speed_dlinear_v2_real_suite/suite_summary.json)
- [xtraffic v3 dlinear](/root/autodl-tmp/BetterTSE-main/results/forecast_revision/runs/xtraffic_p01_speed_dlinear_v3_real_suite/suite_summary.json)

Milestones:

- [XTraffic v1](/root/autodl-tmp/BetterTSE-main/results/forecast_revision/milestones/20260317_xtraffic_v1/README.md)
- [XTraffic v2 narrowed](/root/autodl-tmp/BetterTSE-main/results/forecast_revision/milestones/20260317_xtraffic_v2_narrowed/README.md)
- [XTraffic v2 case studies](/root/autodl-tmp/BetterTSE-main/results/forecast_revision/milestones/20260317_xtraffic_v2_narrowed/CASE_STUDIES.md)
- [XTraffic v3 refined](/root/autodl-tmp/BetterTSE-main/results/forecast_revision/milestones/20260317_xtraffic_v3_refined/README.md)

## Progress by Iteration

### `naive_last`

| version | data slice | localized future MAE | localized revision gain | localized t-IoU | magnitude error |
|---|---|---:|---:|---:|---:|
| v1 | flow, 20 real weak-label samples | 72.418383 | -13.015605 | 0.563347 | 88.361740 |
| v2 | speed, narrowed 9-sample subset | 5.338623 | -0.427281 | 0.504377 | 5.295700 |

Interpretation:

- `v1 -> v2` confirms that data narrowing matters a lot.
- The failure scale drops sharply once the benchmark is restricted to:
  - `speed`
  - clearer incident types
  - observable future response
- Even then, `naive_last` remains too weak to support positive localized revision on real traffic.

### `dlinear_like`

| version | data slice | global gain | localized gain | localized t-IoU | localized magnitude error |
|---|---|---:|---:|---:|---:|
| v1 | flow, 20 real weak-label samples | 1.656528 | -11.440244 | 0.563347 | 121.568473 |
| v2 | speed, narrowed 9-sample subset | 0.176098 | 0.406457 | 0.504377 | 2.727418 |
| v3 | speed, same 9-sample subset, refined operators | 0.171086 | 0.287652 | 0.518519 | 2.834570 |

Interpretation:

- `v1 -> v2` is the decisive step:
  - real-context localized revision turns positive
  - localized revision beats both `base_only` and `global_revision_only`
- `v2 -> v3` shows that operator refinement is directionally reasonable but not automatically beneficial:
  - `v3` remains positive
  - but aggregate gain is worse than `v2`

## What Is Established

### 1. Real-data access is no longer the blocker

The following chain is now verified on `XTraffic`:

- real incident records
- real sensor metadata
- real node ordering
- real traffic shard loading
- `incident -> node/time-window` alignment

So the real-data line is no longer speculative.

### 2. The framework can transfer to real traffic data

The best current checkpoint, `v2 narrowed`, shows on `dlinear_like`:

- `global_revision_only avg_revision_gain = 0.1761`
- `localized_full_revision avg_revision_gain = 0.4065`

This is enough to establish that the localized forecast-revision pipeline can produce positive gains on real data.

### 3. Localized revision still has a real advantage over global revision

On `v2 dlinear_like`:

- `global_revision_only avg_future_t_iou = 0.1103`
- `localized_full_revision avg_future_t_iou = 0.5044`

So the real-data result is not just a scalar gain story. The localized mechanism still carries the stronger region signal.

### 4. Domain-specific narrowing is necessary

The successful transfer required all of the following:

- switch from `flow` to `speed`
- remove broad incident mixtures
- keep only samples with real observable response

This is not a framework redesign. It is domain adaptation at the operator/calibration level.

## What Is Not Yet Established

### 1. Transfer is not baseline-agnostic

`naive_last` remains negative even on the narrowed real subset.

So the current real-data signal is established on:

- `dlinear_like`

but not yet on:

- `naive_last`

### 2. Weak-label oracle is still limited

On `v2 dlinear_like`:

- `localized_full_revision revision_gain = 0.4065`
- `oracle_region revision_gain = 0.2509`
- `oracle_intent revision_gain = 0.2539`

Unlike the synthetic Weather line, the real weak-label oracle does not give a clean monotonic upper bound.

This is expected because:

- `revision_target = future_gt`
- masks and intents are weak labels derived from incident metadata and observed response

So the current real-data oracle is useful for diagnosis, but it is not a clean gold-standard decomposition.

### 3. Operator refinement is still unstable

`v3` used more defensible traffic-specific semantics, but did not beat `v2`.

This means:

- traffic-aware refinement is necessary
- but the current refinement policy is not yet stable enough to replace `v2`

## Recommended Breakpoint Judgment

At this breakpoint, the overall result should be evaluated as:

- **real-data viability**: confirmed
- **localized vs global advantage on real data**: confirmed on `dlinear_like`
- **need to redesign the framework**: no
- **need to continue blind operator tweaking**: no
- **need to retain a best-known real checkpoint**: yes, `v2 narrowed`

## Recommended Next Step

Freeze the current real-data line at:

- [XTraffic v2 narrowed](/root/autodl-tmp/BetterTSE-main/results/forecast_revision/milestones/20260317_xtraffic_v2_narrowed/README.md)

Use it as the first real-world complement to the Weather controlled benchmark.

Do not keep broadening scope or blindly changing operators.

The next disciplined steps are:

1. keep `Weather v4` as the controlled proof point
2. keep `XTraffic v2 narrowed` as the current best real complement
3. only then consider:
   - adding real `non-applicable` traffic samples for `revision_needed`
   - or replacing weak labels with cleaner incident-specific supervision
