# MTBench Finance Real-Context v2

## Scope

- Dataset:
  - `GGLabYale/MTBench_finance_aligned_pairs_short`
- Local file:
  - `data/mtbench/finance_aligned_pairs_short/data/train-00000-of-00001.parquet`
- Benchmarks:
  - `results/forecast_revision/benchmarks/mtbench_finance_naive_v2_smoke`
  - `results/forecast_revision/benchmarks/mtbench_finance_dlinear_v2_smoke`
- Runs:
  - `results/forecast_revision/runs/mtbench_finance_naive_v2_smoke_suite`
  - `results/forecast_revision/runs/mtbench_finance_dlinear_v2_smoke_suite`

## What Changed from v1

The benchmark labeling and schema were switched from traffic-style local event semantics to finance-specific revision semantics:

- `repricing`
  - strong directional revaluation
  - mapped to `step + full_horizon`
- `drift_adjust`
  - smoother horizon-wide adjustment
  - mapped to `plateau + full_horizon`
- `neutral`
  - mapped to `no_revision`

Labeling changes:

- direction now follows `output_percentage_change`
- strength is based on the magnitude of `output_percentage_change`
- no-op samples are still retained for gate sanity checks

## Data Snapshot

- total rows in source split:
  - `750`
- smoke subset:
  - `10`
- `naive_last` smoke:
  - applicable `7`
  - non-applicable `3`
- `dlinear_like` smoke:
  - applicable `7`
  - non-applicable `3`

## Results

### `naive_last`

- `global_revision_only avg_revision_gain = 0.0003`
- `localized_full_revision avg_revision_gain = 0.6043`
- `localized_full_revision applicable_avg_revision_gain = 0.8632`
- `oracle_region avg_revision_gain = 0.6488`

Interpretation:

- The finance-specific schema fixes the negative-transfer issue from v1 smoke.
- Even the weak baseline now benefits from localized revision on the smoke subset.

### `dlinear_like`

- `global_revision_only avg_revision_gain = 0.1757`
- `localized_full_revision avg_revision_gain = 0.2396`
- `localized_full_revision applicable_avg_revision_gain = 0.3423`
- `oracle_region avg_revision_gain = 0.3164`

Interpretation:

- The positive signal is not limited to `naive_last`.
- The MTBench finance line now behaves like a valid second real text complement.

## Main Takeaways

- `MTBench` is now a working real text + time-series complement.
- The key lesson is that finance needs a different revision schema from traffic:
  - horizon-wide repricing
  - horizon-wide drift adjustment
  - explicit no-revision
- Reusing traffic-style `step/hump/flatline` semantics directly was the wrong abstraction.

## Recommended Use in the Project

- Keep `Weather v4` as the controlled proof point.
- Keep `XTraffic v2 narrowed` as the structured-event real complement.
- Add `MTBench finance v2` as the native-text real complement.
