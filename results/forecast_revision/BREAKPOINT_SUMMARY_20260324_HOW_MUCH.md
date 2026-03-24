# Breakpoint Summary 2026-03-24: Revision How-Much

Current work should be considered frozen at this breakpoint:

- main task line: `forecast revision`
- benchmark regime: `future_guided_projected_revision`
- dataset slice: `Time-MMD / Energy / report`
- backbones in active protocol: `dlinear_official`, `patchtst`
- current upper bound: `teacher_search_oracle`
- current formal runtime methods:
  - `teacher_distilled_shrunk` as the simple distilled baseline
  - `teacher_distilled_family_affine` as the conditioned-student mainline

## What Is Already Established

The repository has already crossed four technical milestones:

1. `teacher_search_oracle` now implements the intended constrained-search teacher for the parameter layer instead of only serving as a benchmark-construction idea.
2. `teacher_distilled_shrunk` is now exposed as a formal runtime calibration strategy rather than an informal learned alias.
3. `teacher_distilled_family_affine` is now exposed as a formal conditioned-student runtime strategy, with model-kind specific artifacts and hard runtime validation against mismatched calibrator types.
4. On small heldout smoke runs, the teacher-distilled student beats both `heuristic_revision` and `rule_local_stats` on `dlinear_official` and `patchtst`.

This is enough to say the revision `how much` line is no longer a placeholder.

## Current 100-Sample Multi-Seed Result

Locked protocol run:

- benchmark build size: `100`
- train/heldout split per seed: `80 / 20`
- seeds: `7, 11, 19`
- aggregate root:
  - `tmp/how_much/revision/timemmd_energy_report_100/aggregate`

Main aggregate summary:

- `dlinear_official`
  - `teacher_distilled_family_affine`: target MAE `0.0194`, revision gain `0.0347`, future MAE `0.2608`
  - `teacher_distilled_shrunk`: target MAE `0.0224`, revision gain `0.0317`, future MAE `0.2626`
  - `heuristic_revision`: `0.0250 / 0.0291 / 0.2642`
  - `rule_local_stats`: `0.0239 / 0.0302 / 0.2638`
  - interpretation:
    - `family_affine` beats shrunk, heuristic, and the light rule baseline on the main three metrics
    - shrunk still beats heuristic on the same slice, so the conditioned head is improving an already valid route
    - oracle remains far ahead, so the route is validated but not saturated

- `patchtst`
  - `teacher_distilled_family_affine`: target MAE `0.0226`, revision gain `0.0364`, future MAE `0.2760`
  - `teacher_distilled_shrunk`: target MAE `0.0245`, revision gain `0.0345`, future MAE `0.2782`
  - `heuristic_revision`: `0.0264 / 0.0326 / 0.2798`
  - `rule_local_stats`: `0.0252 / 0.0338 / 0.2790`
  - interpretation:
    - `family_affine` now beats shrunk, heuristic, and rule on the main three metrics
    - the margin is still modest, so this is a protocol-level positive result rather than a saturated endpoint
    - oracle headroom is still large here as well

Anchor control:

- `direct_delta_regression` stays clearly weakest on both backbones
- `teacher_search_oracle` remains dramatically better than every runtime method, which confirms there is still substantial learnable headroom in `how much`

Gap-to-oracle reading:

- `dlinear_official`
  - the shrunk student closes a positive but still modest fraction of the oracle gap relative to heuristic
  - shrunk target/gain gap-closed ratios are about `0.06` to `0.19` across the three seeds
  - `family_affine` expands that to about `0.21` to `0.42`, with future-MAE gap closure around `0.17` to `0.44`
- `patchtst`
  - the shrunk student consistently closes oracle gap relative to heuristic
  - shrunk target/gain gap-closed ratios are about `0.05` to `0.12` across the three seeds
  - `family_affine` expands that to about `0.14` to `0.27`, with future-MAE gap closure around `0.22` to `0.28`

This means the strongest defensible internal statement is now:

- `teacher search -> family-conditioned distilled student` is a credible main route for revision how-much
- the route now beats shrunk, heuristic, and rule on both `dlinear_official` and `patchtst`
- the remaining issue is oracle headroom, not whether the route works at all

## What Is Not Yet Established

The project should not yet claim that the global BetterTSE `how much` problem is solved.

The missing evidence is:

- bucket-level evidence across `effect_family` and `duration_bucket`
- gap analysis showing how much oracle headroom the student actually closes
- evidence that the student is learning transferable `how much` structure rather than only matching one target-construction mechanism

The first two items are now substantially addressed by the 100-sample, 3-seed protocol above.
The remaining issue is not whether the route works at all, but whether it generalizes broadly enough across buckets and backbones to support a stronger claim.

## Protocol Locked At This Breakpoint

The stricter protocol from this point onward is:

- benchmark size: `60+` samples where possible
- seeds: `7, 11, 19`
- train ratio: `0.8`
- teacher-label source: `teacher_search`
- comparison set:
  - `teacher_search_oracle`
  - `teacher_distilled_family_affine`
  - `teacher_distilled_shrunk`
  - `heuristic_revision`
  - `rule_local_stats`
  - `direct_delta_regression`

Infrastructure rules that are now part of the locked protocol:

- calibrator artifact names must match `model_kind`
- runtime strategy and calibrator `model_type` must match exactly
- each calibrator run must emit `group_coverage.json`
- `teacher_distilled_family_duration_affine` stays preview-only until it has the same multi-seed coverage as the main methods

Required summaries:

- overall method table
- oracle/distilled/heuristic gap table
- `effect_family` bucket table
- `duration_bucket` bucket table
- how-much decomposition on:
  - `magnitude_calibration_error`
  - `signed_area_error`
  - `duration_error`
  - `recovery_slope_error`

Current bucket reading from the locked protocol:

- `dlinear_official`
  - strongest gains are on `impulse`
  - on `level`, distilled is roughly tied or slightly behind heuristic
  - on `long` duration, distilled is clearly ahead of heuristic
- `patchtst`
  - on `impulse`, distilled is clearly better than heuristic
  - on `level`, distilled is slightly better than heuristic
  - on `short` duration, distilled improves over heuristic
  - on `long` duration, distilled also improves over heuristic

So the current evidence supports a narrower conclusion:

- the student is not just memorizing one trivial pattern, because it improves on multiple active slices
- but the win is still modest and oracle headroom remains large

## Recommended Immediate Next Step

The locked `100-sample` protocol has now been completed on `dlinear_official` and `patchtst`, and `family_affine` has been integrated into the formal aggregate.

The next step should be one of these, in order:

1. Keep `teacher_distilled_family_affine` as the mainline conditioned student and stop expanding student variants by default.
2. Treat `teacher_distilled_family_duration_affine` as preview-only until it has full `3-seed x 2-backbone` coverage.
3. Use the current aggregate as the decision point for whether to stay inside revision or migrate the same teacher-distill pattern back to pure editing.

The shortest next move is no longer “try family_affine”; it is to decide whether the remaining oracle gap is better spent on:

- more revision-side student absorption, or
- migrating the now-credible teacher-distill parameterization back into pure editing
