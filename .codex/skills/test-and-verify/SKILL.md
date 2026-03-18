---
name: test-and-verify
description: verify BetterTSE code changes with targeted syntax checks, smoke tests, and path-specific validation. use when behavior changed or when completion depends on credible self-validation.
---

# BetterTSE Test And Verify Workflow

Use this workflow whenever behavior changed or correctness is uncertain.

1. Identify the exact changed path.
2. Run `python -m py_compile` on the touched Python files when feasible.
3. Run the narrowest behavior-level smoke test.
4. Fix failures before widening scope.
5. Run a broader validation command only if the touched path justifies it.
6. Re-review the final diff after validation.

## Rules

- Start narrow, then widen.
- Prefer deterministic test inputs and small sample counts.
- Treat missing API keys, missing model checkpoints, and GPU absence as environment blockers, not silent passes.
- Separate unrelated pre-existing failures from regressions caused by the current change.

## Suggested commands

- syntax: `python -m py_compile config.py run_pipeline.py run_forecast_revision.py run_forecast_revision_suite.py agent/*.py modules/*.py tool/*.py forecasting/*.py test_scripts/*.py`
- builder smoke: `python test_scripts/build_event_driven_testset.py --csv-path data/ETTh1.csv --dataset-name ETTh1 --output-dir tmp/event_driven_smoke --num-samples 3 --seq-len 192`
- pipeline smoke: `python run_pipeline.py --testset <testset_json> --output tmp/pipeline_results.json --max-samples 3`
- forecast revision smoke: `python run_forecast_revision.py --benchmark <benchmark_json> --output tmp/revision_results.json --max-samples 20`
- suite smoke: `python run_forecast_revision_suite.py --benchmark <benchmark_json> --output-dir tmp/revision_suite --max-samples 20`

## Final output format

### Tests added or updated

### Commands run

### Results

### Remaining blockers
