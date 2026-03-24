# Experiment Preparation

`test_scripts/prepare_mainline_experiments.py` is the non-executing staging entrypoint for the current BetterTSE experiment plan.

It prepares:

- unified schema files for pure editing, forecast revision, and result logging
- staged paper-backbone contracts for the requested forecasting datasets and models
- a manifest of runnable commands for the experiment stages already supported by the repo
- a readiness report that distinguishes `ready_now` from `needs more implementation`

Current runnable entrypoints:

- `run_pipeline.py --mode full_bettertse|direct_edit|wo_localization|wo_canonical_layer`
- `run_forecast_revision.py --mode base_only|heuristic_revision|direct_delta_regression|wo_parameter_calibration|localized_full_revision`
- `run_forecast_revision.py --calibration-strategy teacher_search_oracle|teacher_distilled_linear|teacher_distilled_shrunk`
- `run_forecast_revision.py --calibration-strategy teacher_search_oracle|teacher_distilled_linear|teacher_distilled_shrunk|teacher_distilled_family_affine|teacher_distilled_family_duration_affine`
- `test_scripts/train_forecast_revision_calibrator.py --label-source gt|teacher_search --model-kind linear|family_affine|family_duration_affine`
- `test_scripts/run_multibackbone_forecast_revision.py --dataset-kind timemmd`
- `test_scripts/aggregate_revision_how_much_protocol.py --root-dir <run_root> --output-dir <aggregate_dir>`
- `test_scripts/run_pure_editing_teacher_search.py --testset <event_driven_json> --output <teacher_json>`
- `test_scripts/build_pure_editing_how_much_stress_benchmark.py --csv-path <csv> --output-dir <stress_dir>`

It does not start:

- baseline training
- benchmark generation
- revision evaluation
- pipeline runs

Typical usage:

```bash
python test_scripts/prepare_mainline_experiments.py
```

Optional narrowing:

```bash
python test_scripts/prepare_mainline_experiments.py \
  --pure-dataset-id etth1 \
  --revision-dataset-ids traffic,weather \
  --revision-backbones dlinear_tslib,patchtst_tslib
```

Output layout:

- `tmp/experiment_preparation/mainline_<timestamp>/experiment_plan.json`
- `tmp/experiment_preparation/mainline_<timestamp>/README.md`
- `tmp/experiment_preparation/mainline_<timestamp>/schemas/*.json`
- `tmp/experiment_preparation/mainline_<timestamp>/prepared_backbones/<dataset>/<baseline>/...`
- `tmp/experiment_preparation/mainline_<timestamp>/run_ready_now.sh`
- `tmp/experiment_preparation/mainline_<timestamp>/run_after_backbone_training.sh`

Current scope notes:

- Pure-editing benchmark build and full-pipeline smoke commands are already wired.
- Pure-editing ablation runners now have explicit CLI modes in `run_pipeline.py`.
- Forecast-revision DLinear/PatchTST Time-MMD runs can now be staged through `run_multibackbone_forecast_revision.py`.
- Forecast-revision how-much calibration now supports teacher-search pseudo labels and teacher-distilled runtime aliases.
- Forecast-revision how-much now has a locked multi-seed evaluation protocol documented in `docs/how_much_protocol.md`.
- Forecast-revision how-much calibrator artifacts are now model-kind specific and emit `group_coverage.json` for split auditing.
- Forecast-revision runtime now hard-validates calibration strategy against the saved calibrator `model_type`.
- The multi-seed protocol aggregate now includes `teacher_distilled_family_affine` in the main comparison table and dedicated oracle-gap outputs.
- Pure-editing now has a standalone tool-conditioned teacher-search prototype for the parameter layer, separated from the main pipeline runner.
- Pure-editing teacher protocol and bucket definitions are documented in `docs/pure_editing_how_much_protocol.md`.
- Pure-editing now also has a tool-balanced stress benchmark builder for how-much diagnosis, independent of slow event-prompt generation.
- Controlled synthetic forecast-revision builders are ready after backbone artifacts exist.
- Time-MMD projected revision is now exposed as a runnable benchmark path.
- A single generic projected-target builder across the full standard LTSF dataset pool is still not exposed as one CLI entrypoint.
