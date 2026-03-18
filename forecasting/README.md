# Forecasting Baselines

Current revision-benchmark forecasting baselines:

- `naive_last`
- `dlinear_like`
- `holt_linear`
- `lstm_official`
- `seasonal_naive`
- `patchtst`

Primary baseline for forecast-revision experiments:

- `patchtst`

## Train Or Materialize A Baseline

Use:

```bash
python test_scripts/train_forecast_baseline.py \
  --dataset-kind csv \
  --csv-path data/Weather.csv \
  --baseline-name seasonal_naive \
  --context-length 96 \
  --prediction-length 24 \
  --output-dir tmp/baselines/weather_seasonal_naive
```

XTraffic (single node/channel training series):

```bash
python test_scripts/train_forecast_baseline.py \
  --dataset-kind xtraffic \
  --xtraffic-data-dir data/xtraffic_minimal \
  --xtraffic-shard-name p01_done.npy \
  --xtraffic-node-index 0 \
  --xtraffic-channel speed \
  --baseline-name patchtst \
  --context-length 288 \
  --prediction-length 144 \
  --output-dir tmp/baselines/xtraffic_patchtst_node0_speed
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

MTBench (concatenated aligned windows):

```bash
python test_scripts/train_forecast_baseline.py \
  --dataset-kind mtbench \
  --mtbench-path data/mtbench/finance_aligned_pairs_short/data/train-00000-of-00001.parquet \
  --mtbench-limit 200 \
  --baseline-name patchtst \
  --context-length 96 \
  --prediction-length 24 \
  --output-dir tmp/baselines/mtbench_patchtst
```

LSTM aligned to PyTorch official example:

```bash
python test_scripts/train_forecast_baseline.py \
  --dataset-kind csv \
  --csv-path data/Weather.csv \
  --baseline-name lstm_official \
  --context-length 96 \
  --prediction-length 24 \
  --epochs 3 \
  --batch-size 32 \
  --optimizer adam \
  --hidden-size 64 \
  --output-dir tmp/baselines/weather_lstm_official
```

For trainable deep baselines such as `patchtst`, point the benchmark builder to the
saved directory via `--baseline-model-dir`.

## Use In Forecast Revision Benchmark Builders

Example:

```bash
python test_scripts/build_forecast_revision_benchmark.py \
  --csv-path data/Weather.csv \
  --dataset-name Weather \
  --output-dir tmp/weather_revision_patchtst \
  --baseline-name patchtst \
  --baseline-model-dir tmp/baselines/weather_patchtst
```

## LangGraph Agent Revision Runner

Use LangGraph agent planning (`what/where`) while keeping forecast-revision execution
on the stable `tedit_hybrid` path:

```bash
python test_scripts/run_forecast_revision_langgraph.py \
  --benchmark tmp/longrun_lstm_official/bench/weather_lstm_official_adam_e5/forecast_revision_WeatherLSTMOfficial_lstm_official_120.json \
  --output tmp/longrun_lstm_official/revision/weather_lstm_official_langgraph_agent_smoke.json \
  --max-samples 1 \
  --llm-name deepseek-chat \
  --source OpenAI \
  --base-url https://api.deepseek.com/v1 \
  --api-key $DEEPSEEK_API_KEY \
  --tedit-model TEdit-main/save/synthetic/pretrain_multi_weaver/0/ckpts/model_best.pth \
  --tedit-config TEdit-main/save/synthetic/pretrain_multi_weaver/0/model_configs.yaml \
  --tedit-device cuda:0
```
