---
name: bettertse-skill-catalog
description: local skill index for BetterTSE. use this file to discover the repository-specific skill workflows stored under .codex/skills/.
---

# BetterTSE Skill Catalog

This repository keeps task-specific Codex workflows under `.codex/skills/`.

## Available local skills

- `bugfix`
  - Fix a bug with the smallest correct change, validate the affected pipeline path, and summarize residual risk.
- `feature-delivery`
  - Deliver a new capability while preserving architecture, schema compatibility, and reproducible validation.
- `test-and-verify`
  - Run syntax checks, smoke tests, and path-specific validation for changed behavior.
- `review-changes`
  - Review local changes with a risk-first focus on correctness, architecture, schema drift, and missing validation.
- `retrospective-update`
  - Convert repeated mistakes into durable AGENTS or workflow improvements.

## When to use which skill

- Use `bugfix` for root-cause repairs.
- Use `feature-delivery` for new behavior or workflow additions.
- Use `test-and-verify` after behavior changes or when confidence is low.
- Use `review-changes` when asked to inspect diffs or provide review findings.
- Use `retrospective-update` when the same class of mistake happens repeatedly.

## Related repository rules

- Project-wide guardrails live in `AGENTS.md`.
- High-risk tooling rules live in `tool/AGENTS.md`.
- Review format lives in `docs/code_review.md`.
- Process improvements live in `docs/retrospective.md`.
