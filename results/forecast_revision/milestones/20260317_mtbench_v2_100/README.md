# MTBench Finance Real-Context v2 (100-Sample Checkpoint)

## Scope

- Dataset:
  - `GGLabYale/MTBench_finance_aligned_pairs_short`
- Local file:
  - `data/mtbench/finance_aligned_pairs_short/data/train-00000-of-00001.parquet`
- Benchmarks:
  - `results/forecast_revision/benchmarks/mtbench_finance_naive_v2_100`
  - `results/forecast_revision/benchmarks/mtbench_finance_dlinear_v2_100`
- Runs:
  - `results/forecast_revision/runs/mtbench_finance_naive_v2_100_suite`
  - `results/forecast_revision/runs/mtbench_finance_dlinear_v2_100_suite`

## Sample Mix

- total: `100`
- applicable: `71`
- non-applicable: `29`

The benchmark keeps the v2 finance-specific schema:

- `repricing -> step + full_horizon`
- `drift_adjust -> plateau + full_horizon`
- `neutral -> no_revision`

## Results

### `naive_last`

| mode | overall gain | applicable gain | future t-IoU | non-app gate match | non-app over-edit |
|---|---:|---:|---:|---:|---:|
| base_only | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 |
| global_revision_only | 0.0002 | 0.0003 | 0.7100 | 1.0000 | 0.0000 |
| localized_full_revision | 0.1515 | 0.2133 | 0.2941 | 1.0000 | 0.0000 |
| oracle_region | 0.0722 | 0.1017 | 0.7100 | 1.0000 | 0.0000 |
| oracle_intent | 0.1488 | 0.2096 | 0.7100 | 0.0000 | 0.0000 |
| oracle_calibration | 0.1488 | 0.2096 | 0.7100 | 0.0000 | 0.0000 |

Interpretation:

- The positive signal from the 10-sample smoke subset survives at 100 samples.
- `localized_full_revision` remains clearly better than `global_revision_only`.
- Gate behavior on real non-applicable samples remains clean.

### `dlinear_like`

| mode | overall gain | applicable gain | future t-IoU | non-app gate match | non-app over-edit |
|---|---:|---:|---:|---:|---:|
| base_only | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 |
| global_revision_only | 0.1146 | 0.1614 | 0.7100 | 1.0000 | 0.0000 |
| localized_full_revision | 0.3765 | 0.5303 | 0.2941 | 1.0000 | 0.0000 |
| oracle_region | 0.4748 | 0.6687 | 0.7100 | 1.0000 | 0.0000 |
| oracle_intent | 0.5372 | 0.7566 | 0.7100 | 0.0000 | 0.0000 |
| oracle_calibration | 0.5372 | 0.7566 | 0.7100 | 0.0000 | 0.0000 |

Interpretation:

- The finance-specific revision schema is stable beyond smoke scale.
- `localized_full_revision` remains positive and clearly beats `global_revision_only`.
- The gap to oracle is still meaningful, so calibration and intent quality remain the main headroom.

## Takeaways

- `MTBench finance v2` is no longer just a smoke result.
- At 100 samples, the native-text complement still holds for both CPU-safe baselines.
- The framework now has:
  - controlled support (`Weather v4`)
  - structured-event real support (`XTraffic`)
  - native-text real support (`MTBench finance`)

## Caveat

- `oracle_intent` and `oracle_calibration` force GT intent on all samples.
- Once non-applicable samples are mixed in, their `revision_needed_match` should not be used as a gate metric.
