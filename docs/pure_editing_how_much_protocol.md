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

Pattern reading for `envelope_noise`:

- `local_burst`: better rate `0.75`
- `uniform_variance`: better rate `0.67`
- `time_varying_envelope`: better rate `0.50`

Current interpretation:

- the weakness is not only search-space width; the original global operator is too coarse
- moving from global variance amplification to envelope-shaped synthetic noise is directionally correct
- even the best current redesign is not strong enough to unlock student distillation

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
  - best operator is `envelope_noise`
  - teacher better rate is only `0.58`
  - therefore volatility remains in tool-audit mode
