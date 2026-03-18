---
name: bugfix
description: fix a BetterTSE bug with minimal, scoped changes. use when the task is to reproduce an issue, locate the root cause, implement a targeted fix, validate the changed path, and summarize remaining risk.
---

# BetterTSE Bugfix Workflow

Follow this sequence:

1. Identify whether the bug belongs to `agent/`, `tool/`, `modules/`, `forecasting/`, or `test_scripts/`.
2. Reproduce the issue with the smallest available command, sample, or script.
3. Add or update a regression check first when practical.
4. Implement the narrowest root-cause fix.
5. Run targeted validation for the changed path.
6. Re-review the diff for schema drift, boundary mistakes, and unrelated file edits.

## Working rules

- Prefer root-cause fixes over prompt-only or fallback-only patches unless the failure is actually prompt-driven.
- Do not edit generated outputs under `results/` as a substitute for fixing source logic.
- Prefer wrappers and adapters over modifying `TEdit-main/`.
- Preserve existing CLI arguments, JSON field names, and output contracts unless the task explicitly changes them.

## Validation guidance

Start narrow:

1. `python -m py_compile ...` on touched modules
2. the smallest reproduction command for the affected path
3. adjacent smoke tests if the bug crosses module boundaries

Examples:

- planning / tool execution: `python run_pipeline.py --testset <testset_json> --output tmp/pipeline_results.json --max-samples 3`
- forecast revision: `python run_forecast_revision.py --benchmark <benchmark_json> --output tmp/revision_results.json --max-samples 20`
- builder issues: `python test_scripts/build_event_driven_testset.py --csv-path data/ETTh1.csv --dataset-name ETTh1 --output-dir tmp/event_driven_smoke --num-samples 3 --seq-len 192`

## Final output format

### Root cause

### Fix

### Validation

### Risk
