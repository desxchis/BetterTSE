---
name: retrospective-update
description: turn a repeated BetterTSE mistake into a durable process improvement by writing a short retrospective and updating AGENTS or skill rules. use when the same error happens multiple times and the workflow needs tightening.
---

# BetterTSE Retrospective Update Workflow

Use this workflow after repeated mistakes.

1. Describe the repeated failure concretely.
2. Identify whether the failure came from missing ownership rules, validation gaps, or unclear completion standards.
3. Draft a narrow, enforceable corrective rule.
4. Update the relevant `AGENTS.md`, `docs/code_review.md`, or local skill document.
5. Keep the rule specific enough to prevent the real mistake without blocking unrelated work.

## Good corrective rules

A good rule is:

- specific
- testable
- tied to a real BetterTSE failure mode
- easy to apply during future edits

## Common BetterTSE failure classes

- edited the wrong layer
- changed `TEdit-main/` unnecessarily
- skipped path-specific validation
- changed result schema or metric semantics implicitly
- changed boundary behavior without checking outside-region preservation

## Final output format

### Repeated mistake

### Root cause

### Corrective rule

### Files to update

### Expected prevention effect
