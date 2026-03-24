# Pure Editing How-Much Teacher Protocol

This document records the current protocol for the pure-editing `how much` layer.

## Scope

- task line: `pure editing`
- benchmark family: `event-driven controlled physical injection`
- current phase: `teacher benchmark`, not student distillation
- current goal:
  - determine whether tool-conditioned teacher search systematically beats heuristic parameters
  - determine which tool families are already learnable
  - determine which tool families need tool-side redesign before student training

## Locked comparison set

The pure-editing teacher protocol currently compares only:

1. `heuristic parameter layer`
2. `tool-conditioned teacher_search`

Do not introduce student variants into the main pure-editing table until the teacher benchmark is large enough to diagnose weak tool families.

## Current teacher design

The teacher is tool-conditioned and searches directly in each tool's native parameter space.

Current search families:

- `spike_inject`
  - `center / width / amplitude`
- `step_shift`
  - `level_shift / left_ramp_steps / right_ramp_steps`
- `hybrid_up`
  - `math_shift`
- `hybrid_down`
  - `math_shift`
- `volatility_increase`
  - `amplify_factor`

Current principle:

- fix `GT tool + GT region`
- search only the `how much` parameters
- do not rewrite planner / localization when running this protocol

## Required metrics

Main metrics:

- `mae_vs_target`
- `mse_vs_target`
- `preservation_mae`

Current teacher diagnostics:

- `peak_delta_error`
- `signed_area_error`
- `teacher_better_rate`

`t-IoU` stays part of the pure-editing main pipeline, but it is not a teacher-protocol discriminator because the current teacher benchmark fixes region to GT.

## Required buckets

At minimum, report:

- `tool_name`
- `effect_family`
- `shape`
- `duration_bucket`
- `strength_bucket`
- `region_length_bucket`
- `target_energy_type`

`target_energy_type` is currently:

- `peak_dominant`
- `area_dominant`
- `mixed`

## Current 20-sample checkpoint

Reference artifact:

- `tmp/how_much/pure_editing/teacher_protocol_20.json`

Current reading:

- overall:
  - teacher target MAE `0.5052`
  - heuristic target MAE `0.5716`
  - teacher MSE `11.2716`
  - heuristic MSE `12.7835`
  - teacher better rate `0.80`
- strong positive tools:
  - `spike_inject`
  - `hybrid_up`
  - `hybrid_down`
- positive but weaker:
  - `step_shift`
  - `volatility_increase`

Current interpretation:

- the teacher route is valid
- most main tool families already show positive signal
- `volatility_increase` is still a weak tool family and should be audited before student distillation

## Current 50-sample stress checkpoint

Reference artifacts:

- `tmp/how_much/pure_editing/stress50/pure_editing_how_much_stress_ETTh1_50.json`
- `tmp/how_much/pure_editing/teacher_protocol_stress50.json`

This stress benchmark is tool-balanced and parameter-coverage oriented. It is the preferred diagnostic benchmark when the goal is to study `how much`, not prompt diversity.

Current reading:

- overall:
  - teacher target MAE `0.9882`
  - heuristic target MAE `1.1541`
  - teacher MSE `36.3692`
  - heuristic MSE `41.5909`
  - teacher better rate `0.86`
- tool-level:
  - `hybrid_down`: better rate `0.90`
  - `hybrid_up`: better rate `1.00`
  - `spike_inject`: better rate `0.90`
  - `step_shift`: better rate `0.90`
  - `volatility_increase`: better rate `0.60`

Current interpretation:

- the teacher route remains strong under tool-balanced stress sampling
- three main tool families are now clearly stable: `spike_inject`, `hybrid_up/down`, `step_shift`
- `volatility_increase` remains the only clearly weak family
- this means the next step is tool audit, not student distillation

## Volatility Audit Checkpoint

Reference artifacts:

- `tmp/how_much/pure_editing/volatility24/pure_editing_how_much_stress_ETTh1_24.json`
- `tmp/how_much/pure_editing/volatility24_audit.json`

Audit setup:

- benchmark: volatility-only stress subset
- comparison anchor: current heuristic volatility operator
- ablations:
  - `global_subwindow`
  - `burst_local`
  - `envelope_noise`

Additional volatility-specific metrics:

- `local_std_error`
- `roughness_error`
- `windowed_energy_profile_error`

Current reading:

- `global_subwindow`
  - teacher better rate `0.29`
  - reduces variance/energy diagnostics, but usually not enough on target MAE
- `burst_local`
  - teacher better rate `0.50`
  - helps `medium` windows and some burst cases, but remains unstable
- `envelope_noise`
  - teacher better rate `0.58`
  - best current operator
  - clearly improves `local_std_error`, `roughness_error`, and `windowed_energy_profile_error`
  - still fails the current go threshold
- `piecewise_envelope_noise`
  - teacher better rate `0.42`
  - strongest current operator on structural volatility metrics
  - `local_std_error`, `roughness_error`, and `windowed_energy_profile_error` all improve sharply over heuristic
  - but target MAE still lags, so the operator is structurally better but not yet a stable replacement

Pattern reading for `envelope_noise`:

- `local_burst`: better rate `0.75`
- `uniform_variance`: better rate `0.67`
- `time_varying_envelope`: better rate `0.50`

Pattern reading for `piecewise_envelope_noise`:

- `monotonic_envelope`: better rate `1.00`
- `non_monotonic_envelope`: better rate `0.50`
- `local_burst`: better rate `0.00`
- `uniform_variance`: better rate `0.33`

Current interpretation:

- the weakness is not only search-space width; the original global operator is too coarse
- moving from global variance amplification to envelope-shaped synthetic noise is directionally correct
- `piecewise_envelope_noise` confirms the missing part is now objective/operator alignment, not pure expressivity
- even the best current redesign is not strong enough to unlock student distillation

## Volatility Canonical Split Validation

Reference artifacts:

- `tmp/how_much/pure_editing/volatility24_split_validation.json`
- `tmp/how_much/pure_editing/volatility24_split_validation_v2.json`
- `tmp/how_much/pure_editing/volclosure24_seed29/pure_editing_volatility_closure_ETTh1_24.json`
- `tmp/how_much/pure_editing/volclosure24_seed29_split_validation_v3.json`

Validation setup:

- benchmark: same volatility-only stress subset used by the audit checkpoint
- goal: test whether `volatility_increase` should remain one canonical family or be split into smaller sub-tools
- candidate set:
  - legacy generic operator: `legacy_generic_envelope`
  - split candidates:
    - `volatility_global_scale`
    - `volatility_local_burst`
    - `volatility_envelope_monotonic`
- routing policy:
  - `uniform_variance -> volatility_global_scale`
  - `local_burst -> volatility_local_burst`
  - `monotonic_envelope -> volatility_envelope_monotonic`
  - `non_monotonic_envelope` is kept outside the routed policy on purpose

Teacher objective specialization:

- `volatility_global_scale`
  - prioritize `local_std_error`, then MAE
- `volatility_local_burst`
  - prioritize `windowed_energy_profile_error`, then `roughness_error`, then MAE
- `volatility_envelope_monotonic`
  - prioritize `roughness_error + windowed_energy_profile_error`, then MAE

Current reading:

- first-pass split validation (`volatility24_split_validation.json`):
  - `legacy_generic_envelope`: better rate `0.58`
  - `volatility_global_scale`: `0.12`
  - `volatility_local_burst`: `0.46`
  - `volatility_envelope_monotonic`: `0.38`
  - `routed_split_policy`: `0.50`
- after tool-side redesign (`volatility24_split_validation_v2.json`):
  - `legacy_generic_envelope`: better rate `0.58`
  - `volatility_global_scale`: `0.88`
  - `volatility_local_burst`: `0.88`
  - `volatility_envelope_monotonic`: `0.38`
  - `routed_split_policy`: `0.92`
- structure alignment after redesign:
  - `volatility_global_scale` now materially improves both target MAE and `local_std_error`
  - `volatility_local_burst` now materially improves target MAE and `windowed_energy_profile_error`
  - `volatility_envelope_monotonic` remains the strongest structural match for monotonic envelopes
- subpattern-level routed reading after redesign:
  - `uniform_variance`: better rate `0.83`
  - `local_burst`: better rate `1.00`
  - `monotonic_envelope`: better rate `1.00`
  - `non_monotonic_envelope`: not routed, kept as preview
- preservation:
  - current split operators keep preservation unchanged on this audit because edits remain confined to the GT region

Closure retest:

- random-seed-balanced closure benchmark:
  - `uniform_variance`, `local_burst`, `monotonic_envelope` each collected with quota `8`
- closure retest (`volclosure24_seed29_split_validation_v3.json`):
  - `volatility_global_scale`: better rate `0.75`
  - `volatility_local_burst`: better rate `0.875`
  - `volatility_envelope_monotonic`: better rate `0.875`
  - routed split policy: `0.75`
- routed subpattern reading:
  - `uniform_variance`: `0.875`
  - `local_burst`: `0.75`
  - `monotonic_envelope`: `0.625`
- note:
  - the closure gate is now satisfied at operator level
  - routed monotonic still looks weaker than the other two, but no longer fails the closure standard at the operator level

Current interpretation:

- the volatility family should not be treated as one monolithic canonical tool
- the split hypothesis is now empirically validated on both the audit set and the closure retest
- `uniform_variance`, `local_burst`, and `monotonic_envelope` now each have a viable dedicated operator path
- `non_monotonic_envelope` remains a preview case and should not block the main conclusion
- the split is ready to be connected back to the pure-editing registry, with `non_monotonic_envelope` kept outside the main route

Current next step:

1. keep `non_monotonic_envelope` in preview
2. connect the validated split tools back to the BetterTSE canonical/hybrid registry
3. run a small full-pipeline sanity check to ensure the new routing does not damage other tool families

## Go / No-Go

Go:

- expand the teacher benchmark
- add more tool-level and pattern-level diagnostics
- audit weak tool families such as `volatility_increase`

No-Go:

- do not jump directly to a unified-spec student
- do not treat pure editing as already solved just because revision has a stronger student line
- do not blame student capacity before confirming the teacher and tool family are strong enough
- do not start volatility student work unless the operator audit reaches the target gate

## Recommended next step

The next step is:

1. scale the pure-editing teacher benchmark beyond smoke size
2. keep the comparison set fixed to `heuristic vs teacher`
3. use the resulting tool-level gaps to decide whether student learning should stay tool-conditioned
4. for `volatility_increase`, run tool-side ablations before any student work:
   - broader search space
   - burst-local noise operator
   - simple envelope-based noise operator

Current volatility gate:

- target:
  - best operator teacher better rate `>= 0.75`
  - `local_std_error` clearly below heuristic
  - `preservation_mae` not materially worse than heuristic
- current status:
  - best MAE operator is still `envelope_noise`
  - strongest structure-matching operator is `piecewise_envelope_noise`
  - neither crosses the gate
  - therefore volatility remains in tool-audit mode
