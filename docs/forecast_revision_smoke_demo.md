# Forecast Revision Smoke Demo

## Goal

Provide one reproducible, CPU-safe path that proves the current forecast-revision task mode can run end to end without depending on GPU training or parameter tuning.

This demo belongs to the **forecast-revision application line**, not the pure-editing mainline. Use it to sanity-check the downstream application path after the core editing method is already in place.

This demo intentionally fixes the setup to one narrow application configuration:

- data source: `data/Weather.csv`
- benchmark builder: synthetic forecast-revision benchmark
- forecast baseline: `patchtst` (loaded from checkpoint directory)
- task mode: `localized_full_revision`
- calibration strategy: `rule_local_stats`
- output location: `tmp/forecast_revision_smoke_demo`

## Command

```bash
python test_scripts/run_forecast_revision_smoke_demo.py \
  --output-dir tmp/forecast_revision_smoke_demo \
  --baseline-name patchtst \
  --baseline-model-dir tmp/baselines/weather_patchtst_smoke_small
```

To verify the old TEdit-backed executor on the forecast-revision path, run:

```bash
OMP_NUM_THREADS=1 conda run -p /root/autodl-tmp/conda_envs/tedit python run_forecast_revision.py \
  --benchmark results/forecast_revision/benchmarks/weather_dlinear_v2/forecast_revision_Weather_dlinear_like_30.json \
  --output tmp/forecast_revision_tedit_check/weather_tedit_localized_2.json \
  --mode localized_full_revision \
  --calibration-strategy rule_local_stats \
  --revision-executor tedit_hybrid \
  --tedit-model TEdit-main/save/synthetic/pretrain_multi_weaver/0/ckpts/model_best.pth \
  --tedit-config TEdit-main/save/synthetic/pretrain_multi_weaver/0/model_configs.yaml \
  --tedit-device cuda:0 \
  --max-samples 2 \
  --vis-dir tmp/forecast_revision_tedit_check/weather_vis_2
```

## What It Produces

- `benchmark/forecast_revision_WeatherSmoke_patchtst_3.json`
- `revision_results.json`
- `visualizations/*.png`
- `README.md`

## Task Example

The generated benchmark contains small revision cases such as:

- `在预测窗口前段，相关指标预计会短时冲高后逐步回落。`
- `在预测窗口中段，系统状态预计会突然切换到更低位并维持一段时间。`
- `当前没有新的外部信号会改变这段未来预测，整体可维持原预测。`

This is enough to check three critical paths:

- revision-needed detection
- localized edit execution
- visualization export

## Notes

- This demo is for pipeline sanity only, not for model quality claims.
- It should not be cited as evidence that forecast revision replaces the pure-editing task.
- It writes all outputs under `tmp/` so it does not touch published experiment artifacts in `results/`.
- `--revision-executor tedit_hybrid` is optional and is intended for execution-layer verification only.
