# Revision How-Much Protocol

This document records the stricter evaluation protocol for the forecast-revision `how much` layer.

## Scope

- task line: forecast revision only
- benchmark family: `Time-MMD` projected revision benchmark
- text source: `report`
- backbones: `dlinear_official`, `patchtst`
- current status: `teacher_search_oracle` is an upper-bound tool; `teacher_distilled_shrunk` is the formal runtime method

## Fixed comparison set

The protocol should be reported with the following methods:

- `teacher_search_oracle`
- `teacher_distilled_shrunk`
- `teacher_distilled_family_affine`
- `teacher_distilled_family_duration_affine`
- `heuristic_revision`
- `rule_local_stats`
- `direct_delta_regression`

Do not re-expand the main comparison table to older mixed modes unless a new method directly targets the same `how much` question.

## Heldout protocol

Use a larger benchmark than the earlier 20-sample smoke runs and keep the split procedure fixed across backbones:

- benchmark size: prefer `60+` samples before claiming stable behavior
- split: `train_ratio = 0.8`
- seeds: `7, 11, 19`
- teacher-label source: `teacher_search`

The current linear calibrator is deterministic once the train split is fixed, so the seed axis mainly stress-tests split sensitivity rather than optimizer noise.

When `teacher_distilled_shrunk` already shows positive trend but not enough backbone-robust margin, the only approved capacity upgrade is:

- `effect_family` conditioned affine shrink
- `effect_family x duration_bucket` conditioned affine shrink

Do not replace the teacher, benchmark, or comparison set when making this upgrade.

## Core questions

The protocol is meant to answer only three questions:

1. Does the teacher-distilled student beat strong heuristic calibration on heldout revision?
2. How much of the oracle gap does the distilled student close?
3. Is the win broad across effect families and duration buckets, or concentrated in a few easy patterns?

## Required metrics

Main method table:

- `edited_mae_vs_revision_target`
- `revision_gain`
- `edited_mae_vs_future_gt`

How-much decomposition:

- `magnitude_calibration_error`
- `signed_area_error`
- `duration_error`
- `recovery_slope_error`

Recommended auxiliary metrics:

- `future_t_iou`
- `normalized_parameter_error`
- `peak_delta_error`

## Required breakdowns

At minimum, aggregate by:

- `effect_family_gt`
- `duration_bucket_gt`

When sample count permits, also inspect:

- `shape_gt`
- `strength_bucket_gt`

## Recommended run layout

Store multi-seed protocol runs under:

- `tmp/how_much/revision/<dataset>/<backbone>/seed_<seed>/`

Recommended output file names:

- `teacher_search_oracle.json`
- `teacher_distilled_shrunk.json`
- `heuristic_revision.json`
- `rule_local_stats.json`
- `direct_delta_regression.json`

## Breakpoint rule

Do not describe the `how much` problem as solved.

The strongest acceptable current claim is:

- `revision how-much has a credible main route`
- the route is `teacher_search_oracle -> teacher_distilled_shrunk`
- the remaining question is whether that gain remains stable under larger heldout evaluation and bucketed decomposition
