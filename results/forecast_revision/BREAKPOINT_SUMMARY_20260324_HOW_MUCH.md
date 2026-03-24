# Breakpoint Summary 2026-03-24: Revision How-Much

Current work should be considered frozen at this breakpoint:

- main task line: `forecast revision`
- benchmark regime: `future_guided_projected_revision`
- dataset slice: `Time-MMD / Energy / report`
- backbones in active protocol: `dlinear_official`, `patchtst`
- current upper bound: `teacher_search_oracle`
- current formal runtime method: `teacher_distilled_shrunk`

## What Is Already Established

The repository has already crossed three technical milestones:

1. `teacher_search_oracle` now implements the intended constrained-search teacher for the parameter layer instead of only serving as a benchmark-construction idea.
2. `teacher_distilled_shrunk` is now exposed as a formal runtime calibration strategy rather than an informal learned alias.
3. On small heldout smoke runs, the teacher-distilled student beats both `heuristic_revision` and `rule_local_stats` on `dlinear_official` and `patchtst`.

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
  - `teacher_distilled_shrunk`: target MAE `0.0224`, revision gain `0.0317`, future MAE `0.2626`
  - `heuristic_revision`: `0.0250 / 0.0291 / 0.2642`
  - `rule_local_stats`: `0.0239 / 0.0302 / 0.2638`
  - interpretation:
    - distilled now beats heuristic on the main three metrics
    - distilled also beats the light rule baseline on the same slice
    - oracle remains far ahead, so the route is validated but not saturated

- `patchtst`
  - `teacher_distilled_shrunk`: target MAE `0.0245`, revision gain `0.0345`, future MAE `0.2782`
  - `heuristic_revision`: `0.0264 / 0.0326 / 0.2798`
  - `rule_local_stats`: `0.0252 / 0.0338 / 0.2790`
  - interpretation:
    - distilled is no longer a pure tie; it now beats heuristic on the main three metrics
    - the margin is still small, so this is a positive trend rather than a finished result
    - oracle headroom is still large here as well

Anchor control:

- `direct_delta_regression` stays clearly weakest on both backbones
- `teacher_search_oracle` remains dramatically better than every runtime method, which confirms there is still substantial learnable headroom in `how much`

Gap-to-oracle reading:

- `dlinear_official`
  - the distilled student closes a positive but still modest fraction of the oracle gap relative to heuristic
  - target/gain gap-closed ratios are about `0.06` to `0.19` across the three seeds
- `patchtst`
  - the student is now consistently closing oracle gap relative to heuristic
  - target/gain gap-closed ratios are about `0.05` to `0.12` across the three seeds

This means the strongest defensible internal statement is now:

- `teacher search -> distilled student` is a credible main route for revision how-much
- the route is already stable enough to beat heuristic on `dlinear_official`
- the route now also shows small but consistent positive gain on `patchtst`
- the remaining issue is margin size, not whether the route works at all

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
  - `teacher_distilled_shrunk`
  - `heuristic_revision`
  - `rule_local_stats`
  - `direct_delta_regression`

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

The locked `100-sample` protocol has now been completed on `dlinear_official` and `patchtst`.

The next step should be one of these, in order:

```bash
python test_scripts/train_forecast_revision_calibrator.py \
  --benchmark tmp/how_much/revision/timemmd_energy_report_100/benchmarks/patchtst/forecast_revision_TimeMMD_Energy_report_patchtst_100.json \
  --output-dir tmp/how_much/revision/timemmd_energy_report_100/patchtst/seed_7/calibrator_family_affine \
  --train-ratio 0.8 \
  --seed 7 \
  --alpha 1.0 \
  --label-source teacher_search \
  --model-kind family_affine
```

Then expand the same conditioned-student check to:

- `patchtst` seeds `11` and `19`
- `family_duration_affine`
- `dlinear_official` as a regression guard

Current preview signal already exists on `patchtst seed_7`:

- baseline `teacher_distilled_shrunk`: target MAE `0.02785`, revision gain `0.03745`, future MAE `0.25524`
- `teacher_distilled_family_affine`: `0.02589 / 0.03941 / 0.25324`
- `teacher_distilled_family_duration_affine`: `0.02611 / 0.03919 / 0.25343`

So the shortest next step is to widen the conditioned-student check across all three seeds before considering any larger redesign.
