# BetterTSE Retrospective Template

## Incident title

Short and specific.

## Date

YYYY-MM-DD

## Triggering task

What was Codex or the engineer trying to complete?

## What went wrong

Describe the repeated mistake clearly.

Examples:

- edited the wrong layer
- changed `TEdit-main/` when a wrapper fix was sufficient
- changed behavior without validating the relevant pipeline path
- modified generated artifacts instead of source logic
- broke a JSON schema or CLI contract
- changed region boundaries or metric semantics unintentionally

## Root cause

Why did the mistake happen?

Examples:

- ownership between `agent/`, `tool/`, and `modules/` was unclear
- validation command for the touched path was not documented
- benchmark output schema was not treated as stable
- completion criteria were too vague

## Corrective rule

Write the exact rule that should be added to `AGENTS.md`, a module-level `AGENTS.md`, or a skill workflow.

Example:

"Before editing `TEdit-main/`, verify that the failure cannot be fixed in `tool/tedit_wrapper.py`, `tool/ts_editors.py`, or the calling module."

## Validation improvement

What new validation step should be added?

Examples:

- always run `python -m py_compile` on touched Python modules
- always run a `run_pipeline.py --max-samples 3` smoke test after changing plan-to-tool execution
- always run a forecast revision smoke test after touching `modules/forecast_revision.py`

## Preventive update applied

- [ ] `AGENTS.md` updated
- [ ] module-level `AGENTS.md` updated
- [ ] `docs/code_review.md` updated
- [ ] skill workflow updated
- [ ] regression coverage added

## Expected prevention effect

State how this process change should stop the same class of mistake from happening again.
