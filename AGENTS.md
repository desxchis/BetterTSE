# BetterTSE Project AGENTS

## Project overview

This repository mixes research workflows, benchmark builders, and production-style time-series editing / forecast-revision code.
Prioritize reproducibility, minimal-risk changes, and validation that matches the path you touched.

## First-principles reasoning

- Use first-principles reasoning.
- Do not assume the user is always fully clear about the exact goal or the best path to achieve it.
- Start from the original problem, desired outcome, and hard constraints instead of jumping directly to an implementation pattern.
- Stay cautious about hidden assumptions, especially when the request is underspecified or the stated solution may not match the actual objective.
- If the motivation, target behavior, or business intent is unclear, stop and discuss it with the user before making code or design decisions.

## Repository map

- `agent/` - LangGraph control plane, prompts, and planner/editor workflow nodes
- `tool/` - deterministic editing/composer/describer functions plus the TEdit wrapper
- `modules/` - shared utilities, LLM access, localization helpers, forecast revision core
- `forecasting/` - baseline forecasters and forecast data helpers
- `test_scripts/` - benchmark builders, evaluators, calibration tooling, and reproducible experiment scripts
- `TEdit-main/` - vendored upstream diffusion model code; treat as high-risk and modify only when the task explicitly requires it
- `results/` - generated outputs, reports, and experiment artifacts; do not casually edit generated files
- `tmp/` - scratch outputs for smoke tests and temporary validation runs

## Task framing

For non-trivial work, internally frame the task as:

1. Goal
2. Relevant modules
3. Constraints
4. Validation plan
5. Done when

Infer conservatively from existing scripts and architecture docs.
Do not invent a new workflow that bypasses the current pipeline structure.

## Working rules

- Read the relevant entry point and the owning module before editing.
- Prefer the narrowest correct fix location.
- Match existing naming, Chinese/English doc style, and logging patterns in nearby files.
- Keep research code and stable pipeline code separated.
- Preserve backward compatibility for JSON schemas, CLI arguments, and evaluation outputs unless the task explicitly changes them.
- Do not mix broad refactors with behavior changes unless required.
- Do not edit unrelated result artifacts just because they are dirty in the worktree.

## TypeScript rule

- If any TypeScript code must be written, you must use the `typescript-project-specifications` skill.

## High-risk areas

- `tool/tedit_wrapper.py`, `tool/ts_editors.py`, and `agent/nodes.py` are stability-critical because they sit on the LLM-planning to editor-execution boundary.
- `modules/llm.py` and `config.py` are API-sensitive; never hardcode secrets or silently change provider defaults.
- `modules/forecast_revision.py` is the core calibration / revision logic for the forecast-revision pipeline; preserve schema compatibility and metric semantics.
- `TEdit-main/` is vendor-style model code; prefer fixing wrappers, adapters, or callers outside it before editing upstream model internals.

## File selection rules

Before editing:

1. Identify whether the issue belongs to planning, localization, tool execution, calibration, data building, or visualization.
2. Prefer fixing the root module for that responsibility.
3. Prefer adapters and wrappers over modifying `TEdit-main/`.
4. Prefer `tmp/` for new smoke-test outputs instead of writing into `results/` unless the task is explicitly about reports or published artifacts.

## Validation commands

Use the narrowest useful validation first, then widen scope if the touched path justifies it.

### Environment

- main dependencies: `pip install -r requirements.txt`
- extra benchmark dependencies when needed: `pip install -r test_scripts/requirements.txt`

### Baseline syntax check

- `python -m py_compile config.py run_pipeline.py run_forecast_revision.py run_forecast_revision_suite.py run_5sample_validation.py agent/*.py modules/*.py tool/*.py forecasting/*.py test_scripts/*.py`

### Event-driven builder smoke test

- `python test_scripts/build_event_driven_testset.py --csv-path data/ETTh1.csv --dataset-name ETTh1 --output-dir tmp/event_driven_smoke --num-samples 3 --seq-len 192`

### Pipeline smoke test

- `python run_pipeline.py --testset <testset_json> --output tmp/pipeline_results.json --max-samples 3`

Add TEdit paths only if the change depends on real model execution:

- `python run_pipeline.py --testset <testset_json> --tedit-model TEdit-main/save/synthetic/pretrain_multi_weaver/0/ckpts/model_best.pth --tedit-config TEdit-main/save/synthetic/pretrain_multi_weaver/0/model_configs.yaml --output tmp/pipeline_results.json --max-samples 3`

### Forecast revision validation

- targeted mode: `python run_forecast_revision.py --benchmark <benchmark_json> --output tmp/revision_results.json --mode localized_full_revision --max-samples 20`
- multi-mode suite: `python run_forecast_revision_suite.py --benchmark <benchmark_json> --output-dir tmp/revision_suite --max-samples 20`

## Testing policy

- When behavior changes, add or update the narrowest useful regression coverage.
- If no unit-test harness exists for the touched path, use a reproducible script-based smoke test and document the exact command.
- For bug fixes, prefer adding a failing reproduction or minimal sample before fixing.
- For prompt / localization / calibration changes, record whether the expected impact is on intent classification, region accuracy, editability, preservability, or calibration metrics.
- Do not claim success without at least one validation signal that exercises the changed code path.

## Change constraints

- Do not modify `.env`, API keys, or credential-loading behavior unless explicitly asked.
- Do not change default model paths or dataset paths without explaining why.
- Do not change JSON field names used by testsets, pipeline outputs, or benchmark outputs without updating all readers/writers.
- Do not rewrite large generated result files to accompany code changes unless the task explicitly requests regenerated artifacts.
- Do not edit `TEdit-main/` for convenience if the issue can be solved in `tool/` or `agent/`.

## Solution design rules

- When proposing a modification or refactor plan, do not provide compatibility-oriented or patch-style solutions.
- Do not over-design. Use the shortest correct path that satisfies the real requirement and does not violate the first-principles rule above.
- Do not introduce fallback, downgrade, hedging, or side-route solutions that the user did not ask for.
- Do not add alternative business behaviors beyond the stated requirement, because that can shift business logic away from the true target.
- Any proposed solution must be logically correct and must be checked through the full end-to-end flow before it is presented as the plan.

## Documentation policy

Update docs when you change:

- CLI usage
- output schema
- benchmark generation workflow
- calibration workflow
- review / retrospective rules

Also follow `docs/code_review.md` when reviewing or self-reviewing changes.

## Completion standard

A task is complete only when all of the following hold:

- requested behavior or documentation is updated
- changed scope is validated with the narrowest relevant command available
- high-risk boundary changes are called out explicitly
- the diff stays scoped to the request
- the final summary states:
  - what changed
  - what validation ran
  - what could not be validated
  - any remaining risks or follow-up work
