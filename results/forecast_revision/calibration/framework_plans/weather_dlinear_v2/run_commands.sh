#!/usr/bin/env bash
set -euo pipefail

# train_learned_linear
/root/miniconda3/bin/python test_scripts/train_forecast_revision_calibrator.py --benchmark results/forecast_revision/benchmarks/weather_dlinear_v2/forecast_revision_Weather_dlinear_like_30.json --output-dir results/forecast_revision/calibration/framework_plans/weather_dlinear_v2/train_learned_linear --train-ratio 0.8 --seed 7 --alpha 1.0

# oracle_benchmark_heldout
/root/miniconda3/bin/python test_scripts/run_forecast_revision_calibration_benchmark.py --benchmark results/forecast_revision/calibration/framework_plans/weather_dlinear_v2/train_learned_linear/heldout_benchmark.json --output-dir results/forecast_revision/calibration/framework_plans/weather_dlinear_v2/oracle_benchmark_heldout --methods text_direct_numeric discrete_strength_table rule_local_stats learned_linear oracle_calibration --calibration-model results/forecast_revision/calibration/framework_plans/weather_dlinear_v2/train_learned_linear/learned_linear_calibrator.json

# semi_oracle_rule
/root/miniconda3/bin/python test_scripts/run_forecast_revision_semi_oracle_suite.py --benchmark results/forecast_revision/benchmarks/weather_dlinear_v2/forecast_revision_Weather_dlinear_like_30.json --output-dir results/forecast_revision/calibration/framework_plans/weather_dlinear_v2/semi_oracle_rule --calibration-strategy rule_local_stats

# semi_oracle_learned_heldout
/root/miniconda3/bin/python test_scripts/run_forecast_revision_semi_oracle_suite.py --benchmark results/forecast_revision/calibration/framework_plans/weather_dlinear_v2/train_learned_linear/heldout_benchmark.json --output-dir results/forecast_revision/calibration/framework_plans/weather_dlinear_v2/semi_oracle_learned_heldout --calibration-strategy learned_linear --calibration-model results/forecast_revision/calibration/framework_plans/weather_dlinear_v2/train_learned_linear/learned_linear_calibrator.json
