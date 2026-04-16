# BetterTSE Schema Design Note

## Purpose

This note defines the schema boundary between the two BetterTSE task lines:
- pure time-series editing
- forecast revision as a downstream application

The design goal is to share one semantic editing ontology without forcing both tasks to use the same executor or numeric control path.

## Shared Core Schema

Both task lines should produce or consume the same semantic core:
- `intent`
- `localization`
- `canonical_tool`
- `tool_name -> canonical_tool -> control_source -> tool_layer`
- confidence signals attached to the plan when available

This shared layer answers one question: what does the instruction mean as an editing request.

## Position Bucket Enum

The shared `position_bucket` enum is:
- `early`
- `mid`
- `late`
- `full`
- `none`

Legacy aliases are still accepted when reading historical benchmarks or plans:
- `middle -> mid`
- `early_horizon -> early`
- `mid_horizon -> mid`
- `late_horizon -> late`
- `full_horizon -> full`

New outputs should emit the shared enum rather than the legacy aliases.

## Task-Specific Layers

Pure editing and forecast revision diverge below the semantic layer.

Pure editing path:
- prompt
- semantic intent
- canonical tool
- editor parameters
- editor execution

Forecast revision path:
- prompt
- semantic intent
- canonical tool
- revision calibration spec
- revision parameters
- revision executor

`edit_spec` therefore belongs to the forecast revision calibration path. It should not be presented as a universal schema for every time-series editing task.

## Source Of Truth Rules

For forecast revision samples, the primary source-of-truth fields are:
- `history_ts`
- `future_gt`
- `base_forecast`
- `revision_target`
- `edit_mask_gt`
- `edit_intent_gt`
- `revision_applicable_gt`
- `revision_operator_params`

The following fields are denormalized cache fields kept for convenience in filtering, aggregation, and reporting:
- `effect_family_gt`
- `direction_gt`
- `shape_gt`
- `strength_bucket_gt`
- `duration_bucket_gt`
- `revision_operator_family`
- `edit_spec_gt`

When these two layers disagree, the source-of-truth fields win.

## Intent Structure Rule

`edit_intent_gt` is the canonical ground-truth semantic intent field for benchmark samples.

Other intent-like structures should be interpreted as follows:
- `plan.intent`: model-produced or heuristic-produced semantic interpretation
- `intent_struct`: extended or auxiliary structure carried by some benchmark builders

`intent_struct` should not replace `edit_intent_gt` as the benchmark source of truth.

## Design Boundary

The current BetterTSE schema is designed for:
- single-variable series
- one dominant edit region
- one dominant edit intent
- point-forecast revision
- controlled benchmark targets

It is not yet a universal schema for:
- multi-region composite edits
- multi-variate coupled edits
- probabilistic forecast revision
- arbitrary temporal transformations

That boundary should stay explicit in project docs and experiments.
