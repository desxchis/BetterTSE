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

## Current 60-Sample Multi-Seed Result

Locked protocol run:

- benchmark build size: `60`
- train/heldout split per seed: `48 / 12`
- seeds: `7, 11, 19`
- aggregate root:
  - `tmp/how_much/revision/timemmd_energy_report_60/aggregate`

Main aggregate summary:

- `dlinear_official`
  - `teacher_distilled_shrunk`: target MAE `0.0238`, revision gain `0.0394`, future MAE `0.2044`
  - `heuristic_revision`: `0.0271 / 0.0361 / 0.2062`
  - `rule_local_stats`: `0.0248 / 0.0384 / 0.2047`
  - interpretation:
    - distilled now beats heuristic on the main three metrics
    - distilled also edges past the light rule baseline on the same slice
    - oracle remains far ahead, so the route is validated but not saturated

- `patchtst`
  - `teacher_distilled_shrunk`: target MAE `0.0227`, revision gain `0.0357`, future MAE `0.1830`
  - `heuristic_revision`: `0.0228 / 0.0356 / 0.1824`
  - `rule_local_stats`: `0.0226 / 0.0358 / 0.1829`
  - interpretation:
    - distilled is now essentially tied with heuristic and rule
    - it improves some calibration components slightly, but not enough to claim stable superiority
    - oracle headroom is still large here as well

Anchor control:

- `direct_delta_regression` stays clearly weakest on both backbones
- `teacher_search_oracle` remains dramatically better than every runtime method, which confirms there is still substantial learnable headroom in `how much`

Gap-to-oracle reading:

- `dlinear_official`
  - the distilled student closes a positive but still modest fraction of the oracle gap relative to heuristic
  - target/gain gap-closed ratios are about `0.07` to `0.18` across the three seeds
- `patchtst`
  - the student is not yet consistently closing oracle gap relative to heuristic
  - two seeds are slightly negative and one seed is mildly positive

This means the strongest defensible internal statement is now:

- `teacher search -> distilled student` is a credible main route for revision how-much
- the route is already stable enough to beat heuristic on `dlinear_official`
- the route is not yet stable enough to claim backbone-robust superiority on `patchtst`

## What Is Not Yet Established

The project should not yet claim that the global BetterTSE `how much` problem is solved.

The missing evidence is:

- bucket-level evidence across `effect_family` and `duration_bucket`
- gap analysis showing how much oracle headroom the student actually closes
- evidence that the student is learning transferable `how much` structure rather than only matching one target-construction mechanism

The first two items are now partially addressed by the 60-sample, 3-seed protocol above.
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
  - on `impulse`, distilled is slightly better than heuristic
  - on `level`, distilled is roughly flat to slightly worse
  - on `short` duration, distilled improves over heuristic
  - on `long` duration, distilled is effectively tied with heuristic

So the current evidence supports a narrower conclusion:

- the student is not just memorizing one trivial pattern, because it improves on multiple active slices
- but the win is still pattern-dependent rather than uniformly dominant

## Recommended Immediate Next Step

The locked protocol has now been completed on `dlinear_official` and `patchtst`.

The next step should be one of these, in order:

```bash
python test_scripts/build_timemmd_projected_revision_benchmark.py \
  --timemmd-root data/Time-MMD \
  --domain Energy \
  --text-source report \
  --output-dir tmp/how_much/revision/timemmd_energy_report_100/benchmarks/dlinear_official \
  --baseline-name dlinear_official \
  --baseline-model-dir tmp/timemmd_dlinear_mainline \
  --seq-len 96 \
  --pred-len 24 \
  --max-samples 100
```

Then rerun the same `3-seed` protocol and check whether:

- the `dlinear_official` win stays positive
- `patchtst` turns from tie to stable positive gain
- the bucket-level advantage remains visible beyond the current 60-sample slice
