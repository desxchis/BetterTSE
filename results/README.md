# Results Index

This directory centralizes generated artifacts, benchmark outputs, validation runs, and visualization files.

## Layout

- `testsets/event_driven/`
  - Event-driven benchmark datasets and their reports.
  - `etth1_100_ultimate/`: main ETTh1 100-sample event-driven set.
  - `traffic_20/`: main Traffic 20-sample event-driven set.
  - `generated/`: default output target for newly generated event-driven testsets.

- `validation/`
  - End-to-end validation runs with plots and aggregated metrics.
  - `etth1_5sample/`: first 5 ETTh1 samples. Generated on `2026-03-14 16:15 +08:00`.
  - `etth1_20sample/`: ETTh1 samples 6-25 plus analysis report and logs. Generated on `2026-03-14 16:21 +08:00`.
  - `traffic_20sample/`: Traffic validation outputs. Generated on `2026-03-14 17:11 +08:00`.

- `experiments/schema_v2/`
  - Small diagnostic runs created during schema/localizer refactors.
  - `etth1_3sample/`: schema-v2 ETTh1 quick run. Latest pipeline result on `2026-03-15 16:44 +08:00`.
  - `traffic_3sample/`: schema-v2 Traffic quick run. Latest pipeline result on `2026-03-15 16:40 +08:00`.

- `milestones/`
  - Best-known snapshots worth retaining as standalone checkpoints.
  - `20260315_1837_v1/`: current Traffic 12-sample step-localizer milestone with copied testset, best result JSON, and matched visualizations.

- `forecast_revision/`
  - Forecast-revision line, including synthetic Weather proof points and real XTraffic complements.
  - `BREAKPOINT_SUMMARY_20260317.md`: current frozen breakpoint, stabilization update, and next-step priority.
  - `EXPERIMENT_FRAMEWORK_20260317.md`: fixed dataset roles, baselines, metrics, and experiment matrix.
  - `EXPERIMENT_SUMMARY_20260317.md`: consolidated summary across Weather, XTraffic, and MTBench.
  - `case_studies/20260317_unified/CASE_STUDIES_20260317.md`: unified controlled + real-data case-study pack.
  - `repro_checks/20260317_resume_check/repro_check_report.md`: reproducibility rerun report for the canonical checkpoints.
  - `OVERALL_ASSESSMENT_20260316.md`: current Weather controlled-benchmark summary.
  - `OVERALL_ASSESSMENT_XTRAFFIC_20260317.md`: current XTraffic real-context summary.
  - `milestones/20260316_weather_v4_cal/`: current best Weather controlled checkpoint.
  - `milestones/20260317_xtraffic_v2_narrowed/`: current best XTraffic real-data checkpoint.
  - `milestones/20260317_xtraffic_v2_nonapp/`: current XTraffic real-data gate checkpoint.
  - `milestones/20260317_mtbench_v2_finance/`: current MTBench finance native-text checkpoint.
  - `milestones/20260317_mtbench_v2_100/`: 100-sample MTBench finance stability checkpoint.

- `pipeline/`
  - Generic `run_pipeline.py` outputs.

- `pipeline_inputs/`
  - Inputs prepared for generic pipeline runs.

- `figures/`
  - Standalone plots and example-generated images.
  - `root_outputs/`: formerly `outputs/`
  - `examples_outputs/`: formerly `examples/outputs/`
  - `examples/`: default output for `examples/llm_tedit_pipeline.py`

- `legacy/`
  - Older or ad hoc result dumps retained for reference.

## Notes

- Scripts that generate new outputs should prefer paths inside `results/`.
- This folder is for generated artifacts only; design notes and architecture docs remain at repository root.
- `validation/` is the canonical place for retained end-to-end result batches.
- Early standalone figure dumps that are not part of a validation batch are intentionally removed instead of archived.
