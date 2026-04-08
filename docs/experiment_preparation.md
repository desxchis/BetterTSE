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
- `test_scripts/build_tedit_strength_discrete_benchmark.py --csv-path <csv> --output-dir <strength_dir> --num-families <n>`
- `test_scripts/build_tedit_strength_trend_family_dataset.py --csv-path <csv> --output-dir <family_dir> --train-families <n> --valid-families <n> --test-families <n>`
- `test_scripts/build_pure_editing_volatility_closure_benchmark.py --csv-path <csv> --output-dir <closure_dir>`
- `test_scripts/build_pure_editing_volatility_route_closure_benchmark.py --csv-path <csv> --output-dir <route_closure_dir>`
- `test_scripts/build_event_driven_volatility_subtype_v2.py --benchmark <event_json> --output-dir <v2_dir> [--csv-path <csv> --target-monotonic-count <n>]`
- `test_scripts/run_pure_editing_volatility_audit.py --testset <volatility_json> --output <audit_json>`
- `test_scripts/run_pure_editing_volatility_split_validation.py --testset <volatility_json> --output <split_json>`
- `test_scripts/run_pure_editing_volatility_route_closure.py --testset <volatility_json> --output <route_json> --routing-source planner_llm|text_guard_only`
- `test_scripts/analyze_pure_editing_volatility_routing.py --testset <event_json> --result <pipeline_json> --output <routing_json>`

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
- Pure-editing teacher protocol remains historical reference only; it is no longer the active strength-control benchmark.
- Pure-editing now has a discrete `weak / medium / strong` family benchmark builder for the TEdit strength-control mainline.
- Pure-editing now also has a trend-only family training dataset builder for the new strength-injection line; this replaces the old `trend_types` attribute-gap supervision as the recommended Phase 1 training source.
- Pure-editing now has a volatility-only operator audit path with volatility-structure metrics and operator ablations.
- Pure-editing now also has an audit-only validator for minimal volatility canonical split hypotheses; it was used to decide the current registry split.
- Pure-editing now also has a volatility closure benchmark builder for split stability retests.
- The validated volatility split tools are now connected back to the pure-editing routing and execution path, while `non_monotonic_envelope` remains outside the main route.
- Pure-editing now also has a subtype-aware volatility route-closure benchmark and a route-only closure runner with `planner_llm` and `text_guard_only` modes.
- The volatility planner schema now includes an explicit `volatility_subtype` field and records `proposed_subtype / guarded_subtype / final_subtype` through the route path.
- A post-integration routing analyzer is now available to diagnose whether volatility samples are being sent to the correct split sub-tool.
- Pure-editing now also has a mainline benchmark refresh script that rewrites only volatility samples into subtype-aware prompts and labels, and can optionally supplement monotonic cases so the mainline pipeline can exercise `global_scale / local_burst / envelope_monotonic / preview_non_monotonic` semantics directly.
- The subtype-aware mainline refresh has now been validated on both the 56-sample ETTh1 sign-off rerun and a frozen second-distribution ETTm1 rerun; current volatility work is considered system-level stable rather than still exploratory.
- Pure-editing no longer treats teacher-search or tool-conditioned student override as the active `how much` mainline.
- The locked pure-editing mainline is now the TEdit internal strength-control path documented in `docs/pure_editing_how_much_protocol.md`.
- Controlled synthetic forecast-revision builders are ready after backbone artifacts exist.
- Time-MMD projected revision is now exposed as a runnable benchmark path.
- A single generic projected-target builder across the full standard LTSF dataset pool is still not exposed as one CLI entrypoint.
