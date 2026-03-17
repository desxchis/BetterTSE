# MTBench Finance Real-Context v1 Smoke

## Scope

- Dataset:
  - `GGLabYale/MTBench_finance_aligned_pairs_short`
- Local file:
  - `data/mtbench/finance_aligned_pairs_short/data/train-00000-of-00001.parquet`
- Benchmark type:
  - real text + time-series aligned pairs
- Baseline:
  - `naive_last`
- Current benchmark builder:
  - `test_scripts/build_mtbench_real_revision_benchmark.py`
- Current suite:
  - `results/forecast_revision/runs/mtbench_finance_naive_smoke_suite`

## Data Facts

- total rows:
  - `750`
- input length:
  - mainly `390`, with some shorter/irregular windows
- output length:
  - mainly `78`
- alignment labels:
  - `"consistent"`: `606`
  - `"inconsistent"`: `144`

Trend / text metadata highlights:

- `output_bin_label`:
  - `-2% ~ +2%`: `509`
  - `+2% ~ +4%`: `145`
  - `>+4%`: `45`
  - `-2% ~ -4%`: `35`
  - `<-4%`: `16`
- direction counts from `output_percentage_change`:
  - `up`: `448`
  - `down`: `301`
  - `flat`: `1`
- `label_sentiment` is dominated by:
  - `Mixed Outlook`
  - `Bullish`
  - `Balanced/Informational`
- `label_type` is dominated by:
  - `Company-Specific News`
  - `Stock Recommendations`
  - `Fundamental Analysis`

## Smoke Benchmark

- benchmark output:
  - `results/forecast_revision/benchmarks/mtbench_finance_naive_smoke/forecast_revision_MTBench_naive_last_10.json`
- sample count:
  - `10`
- applicable:
  - `8`
- non-applicable:
  - `2`

Current weak labeling policy:

- use `trend.output_percentage_change` / `output_bin_label` to infer direction
- map applicable cases to:
  - `effect_family = level`
  - `shape = step`
  - `duration = full horizon`
- inject a small number of explicit no-op samples for gate sanity checks

## Smoke Results

- suite summary:
  - `results/forecast_revision/runs/mtbench_finance_naive_smoke_suite/suite_summary.json`

Key numbers:

- `global_revision_only avg_revision_gain = 0.000045`
- `localized_full_revision avg_revision_gain = -0.096784`
- `oracle_region avg_revision_gain = -0.400909`
- `localized_full_revision avg_future_t_iou = 0.195513`
- `non_applicable_avg_revision_needed_match = 1.0`
- `non_applicable_avg_over_edit_rate = 0.0`

## Interpretation

- MTBench is a valid second real complement:
  - real text
  - real time series
  - real forecasting target
- Data access and local parsing are now verified.
- But the current revision semantics do **not** transfer directly.

The main reason is structural:

- `XTraffic` is naturally local and event-window driven
- `MTBench finance` is mostly:
  - sentiment-driven
  - company-news-driven
  - horizon-wide drift / repricing

So a traffic-style localized operator family is a poor fit here.

## Recommended Next Step

Do **not** keep using the current traffic-style `step/hump/flatline` assumption for MTBench finance.

The next disciplined move is:

1. define a finance-specific revision schema, likely centered on:
   - horizon-wide drift revision
   - directional repricing
   - confidence / uncertainty-aware adjustment
2. then rebuild the MTBench benchmark with that schema
3. only after that, rerun `naive_last` and `dlinear_like`
