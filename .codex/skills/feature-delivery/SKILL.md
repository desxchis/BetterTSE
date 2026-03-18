---
name: feature-delivery
description: implement a BetterTSE feature while preserving architecture, benchmark contracts, and reproducible validation. use when the task requires a behavior change, integration into an existing pipeline, test updates, and final verification.
---

# BetterTSE Feature Delivery Workflow

Follow this sequence:

1. Confirm which pipeline the feature belongs to: event-driven editing or forecast revision.
2. Identify the owning entry point, core module, and expected output schema.
3. Implement the smallest architecture-consistent change.
4. Add or update regression coverage, smoke checks, or a reproducible validation command.
5. Run the narrowest useful validation first.
6. Review the diff for compatibility, result-schema stability, and scope control.

## Implementation rules

- Match the existing layer split across `agent/`, `tool/`, `modules/`, `forecasting/`, and `test_scripts/`.
- Prefer extending existing tool registries, planner logic, and calibration helpers over creating parallel paths.
- Keep output JSON fields stable unless the task explicitly changes consumer expectations.
- If a change affects metrics, localization, editability, or preservability, say so explicitly.
- Update docs when CLI usage, workflow, or schema changes.

## Validation guidance

Use the path-specific command that actually exercises the new behavior.

- syntax baseline: `python -m py_compile ...`
- pipeline feature: `python run_pipeline.py --testset <testset_json> --output tmp/pipeline_results.json --max-samples 3`
- forecast revision feature: `python run_forecast_revision.py --benchmark <benchmark_json> --output tmp/revision_results.json --max-samples 20`
- suite-level change: `python run_forecast_revision_suite.py --benchmark <benchmark_json> --output-dir tmp/revision_suite --max-samples 20`

## Final output format

### Feature implemented

### Files changed

### Validation

### Notes
