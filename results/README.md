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
