---
name: review-changes
description: review BetterTSE changes for correctness, regression risk, architecture fit, schema stability, and missing validation. use when asked to inspect local diffs, commits, or generated changes and report findings by severity.
---

# BetterTSE Review Changes Workflow

Review the current changes using a risk-first approach.

1. Understand the intended change.
2. Inspect the diff for wrong-layer edits and scope creep.
3. Check correctness, edge cases, and compatibility.
4. Check whether JSON schemas, CLI arguments, and benchmark outputs remain stable.
5. Check validation coverage and missing tests.
6. Report findings by severity.

## Review rules

Also follow `docs/code_review.md`.

Pay special attention to:

- edits to `tool/` and `modules/forecast_revision.py`
- changes touching `TEdit-main/`
- region-boundary logic
- hidden changes to metrics or evaluation semantics
- generated `results/` artifacts being modified without a source change justification

## Output format

### Summary

### Findings

For each issue:

- Severity
- File
- Problem
- Why it matters
- Suggested fix

### Positive notes

### Verdict
