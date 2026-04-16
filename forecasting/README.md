# Forecasting Baselines

Boundary:
`forecasting/` is a support layer for the forecast-revision pipeline.
Baselines here are maintained as `base_forecast` providers, not as a standalone forecasting benchmark track.

## Baseline Layers

Engineering / debug baselines currently wired in repo:

- `naive_last`
- `seasonal_naive`
- `holt_linear`
- `dlinear_like`
- `dlinear_official`
- `patchtst`
- `lstm_official`

Paper baselines planned through the unified TSLib-style path:

- `dlinear_tslib`
- `patchtst_tslib`
- `itransformer_tslib`
- `timemixer_tslib`
- `autoformer_tslib`

The `*_tslib` entries now expose a stable adapter and metadata contract. `dlinear_tslib` and `autoformer_tslib` additionally materialize a locally runnable launcher against the vendored `LTSF-Linear` code. `patchtst_tslib` continues to use the paper-baseline contract path until its external training/export bridge is finalized.

## Standard Forecasting Dataset Pool

The first-tier numeric forecasting pool is based on standard LTSF datasets rather than a custom large-scale data pool.

Configured dataset IDs:

- `traffic`
- `weather`
- `etth1`
- `etth2`
- `ettm1`
- `ettm2`
- `electricity`

List the built-in dataset registry with:

```bash
python test_scripts/train_forecast_baseline.py \
  --list-standard-datasets
```

## Train Or Materialize A Baseline

Standard dataset mode using dataset IDs:

```bash
python test_scripts/train_forecast_baseline.py \
  --dataset-id weather \
  --baseline-name dlinear_tslib \
  --context-length 96 \
  --prediction-length 24 \
  --output-dir tmp/baselines/weather_dlinear_tslib_contract
```

This materializes the baseline contract and metadata. For `dlinear_tslib` and `autoformer_tslib`, the output directory also contains a local launcher script that is ready to start training against the vendored `LTSF-Linear` code.

Existing local baseline training path remains available:

```bash
python test_scripts/train_forecast_baseline.py \
  --dataset-kind csv \
  --csv-path data/Weather.csv \
  --baseline-name seasonal_naive \
  --context-length 96 \
  --prediction-length 24 \
  --output-dir tmp/baselines/weather_seasonal_naive
```

XTraffic (multi-node training for better cross-node generalization):

```bash
python test_scripts/train_forecast_baseline.py \
  --dataset-kind xtraffic \
  --xtraffic-data-dir data/xtraffic_minimal \
  --xtraffic-shard-name p01_done.npy \
  --xtraffic-node-indices 4892,12058,407 \
  --xtraffic-channel flow \
  --baseline-name lstm_official \
  --context-length 288 \
  --prediction-length 144 \
  --optimizer adam \
  --hidden-size 64 \
  --output-dir tmp/baselines/xtraffic_lstm_official_multi_nodes
```

## Metadata Contract

Baseline materialization / training now records:

- `dataset_id`
- `dataset_family`
- `split_policy`
- `training_split_id`
- `feature`
- `baseline_source`
- `paper_role`

This keeps the forecast-revision side dependent only on stable baseline artifacts rather than a specific forecasting training repo.

## Use In Forecast Revision Benchmark Builders

Existing local baseline example:

```bash
python test_scripts/build_forecast_revision_benchmark.py \
  --csv-path data/Weather.csv \
  --dataset-name Weather \
  --output-dir tmp/weather_revision_patchtst \
  --baseline-name patchtst \
  --baseline-model-dir tmp/baselines/weather_patchtst
```

For `*_tslib` paper baselines, benchmark builders should only be used after a trained/exported inference artifact is attached to the materialized baseline directory.
