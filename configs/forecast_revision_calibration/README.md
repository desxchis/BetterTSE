# Forecast Revision Calibration Configs

This directory stores config files for the config-driven calibration experiment framework.

Primary entrypoint:

- `test_scripts/prepare_forecast_revision_calibration_framework.py`

Default behavior:

- reads one JSON config
- resolves benchmark / held-out / model dependencies
- writes:
  - `experiment_plan.json`
  - `run_commands.sh`
  - `README.md`
- does not execute experiments unless `--execute` is passed

Current example configs:

- `weather_dlinear_v2.json`
- `xtraffic_dlinear_v2.json`
- `mtbench_dlinear_v2_100.json`

Recommended usage:

```bash
python test_scripts/prepare_forecast_revision_calibration_framework.py \
  --config configs/forecast_revision_calibration/weather_dlinear_v2.json
```

This keeps framework assembly separate from GPU-heavy execution.
