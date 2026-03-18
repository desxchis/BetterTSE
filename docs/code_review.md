# BetterTSE Code Review Rules

Review every change with a risk-first mindset.

## 1. Correctness

- Does the change solve the actual problem rather than masking a symptom?
- Is the logic placed in the correct layer: planner, localization, tool execution, calibration, data building, or visualization?
- Are region boundaries, inclusivity, and empty-region cases handled correctly?
- Are fallback paths, API failures, and missing-model cases considered?

## 2. Scope control

- Is the diff limited to the task?
- Were generated outputs under `results/` changed only when the task required it?
- Does the change mix feature work, refactor, and cleanup unnecessarily?

## 3. Architecture fit

- Does the change preserve the current split between `agent/`, `tool/`, `modules/`, `forecasting/`, and `test_scripts/`?
- Is vendor code in `TEdit-main/` touched only when a wrapper-level fix is insufficient?
- Does the change preserve existing CLI contracts and JSON schemas?

## 4. Research and reproducibility

- Are seeds, dataset assumptions, and benchmark semantics preserved?
- Does the change avoid hidden schema drift in generated benchmark files?
- If metrics may shift, is the expected direction explained?

## 5. API, secrets, and runtime safety

- Are API keys and provider defaults still environment-driven?
- Are new external calls or model requirements justified?
- Is CPU-safe behavior preserved where the script previously supported it?

## 6. Testing and validation

- Is there a regression test, smoke test, or reproducible command for the changed path?
- Were the narrowest relevant validation commands run first?
- If validation could not run, is the blocker stated clearly?

## 7. Code quality

- Are names precise and consistent with nearby modules?
- Is the code simpler or at least not more confusing?
- Are comments explaining intent instead of restating code?
- Is there dead code, duplication, or hidden coupling?

## Review output format

### Summary

One short paragraph covering the overall assessment.

### Findings

For each issue report:

- Severity: high / medium / low
- File
- Problem
- Why it matters
- Suggested fix

### Positive notes

Mention good design choices worth preserving.

### Final verdict

- approve
- approve with minor changes
- needs revision
