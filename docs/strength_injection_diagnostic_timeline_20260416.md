# Strength Injection Diagnostic Timeline

## Purpose

This note records the April 15-16 diagnostic mainline for TEdit strength control in plain language.
The goal is not to restate the method proposal, but to pin down what has already been checked and what the current failure mode actually looks like.

## Current Bottom Line

The repository is no longer at the stage of asking whether a strength-control path exists.

The current state is:

- strength signals do enter the model
- internal modulation changes with requested strength
- but the final edited output is still often flat or reversed

So the main bottleneck has narrowed to:

- why the internal strength signal does not become the correct final edit-amplitude ordering

## Timeline

### 2026-04-15 Morning: first GPU validation variants

Result directories:

- `results/strength_first_gpu_validation`
- `results/strength_second_gpu_validation_lrfix`
- `results/strength_second_gpu_validation_initfix`
- `results/strength_second_gpu_validation_multboost`
- `results/strength_second_gpu_validation_scalargate`
- `results/strength_second_gpu_validation_stageprobe`
- `results/strength_second_gpu_validation_stageprobe_rerun`
- `results/strength_stageprobe_emitcheck`

What was being checked:

- whether the new strength path was actually active during training
- whether small training/configuration repairs changed the behavior at all
- whether stage-level debug outputs could be emitted reliably

Main takeaway:

- this stage did not support the idea that the strength path was completely dead
- it motivated a more targeted probe of internal responses instead of another broad retraining round

### 2026-04-15 Midday to Afternoon: gain-multiplier ablation

Result directory:

- `results/strength_gainmult4_cuda0_validation`

Variants that were compared:

- baseline
- freeze backbone
- text dropout
- freeze backbone + text dropout

Why this was done:

- to test whether the wrong behavior came from the backbone dominating the new strength branch
- to test whether text was overwhelming the numeric strength signal

Observed result:

- all variants still showed the wrong direction overall
- the final strong-minus-weak gain stayed negative
- removing text did not fix the issue
- freezing the backbone did not fix the issue

Interpretation:

- the problem is not explained by a simple “text branch dominates everything” story
- the problem is also not solved by only protecting the strength branch from the backbone

### 2026-04-15 Evening: semantic-split rerun and direct scalar probes

Result directory:

- `results/strength_second_gpu_validation_semantic_split`

What was added:

- direct evaluation in `both`, `label_only`, and related probe modes
- forward scalar probe
- reversed scalar probe
- label-swap checks

Key observed patterns:

1. Some earlier runs were perfectly flat

- weak, medium, and strong produced nearly identical final edit gains
- this looked like the output path had flattened the strength difference away

2. Reruns showed a stronger pattern than “flat”

- once the probe path was cleaned up, many samples were no longer exactly identical
- instead, the direction was often wrong:
  - higher requested strength produced slightly smaller edits

3. Reversing the scalar often flipped the behavior

- when the probe fed `1.0 -> weak` and `0.0 -> strong`, the output trend often flipped as well

Interpretation:

- the model is not simply ignoring the scalar
- the scalar axis appears to be usable, but is often mapped in the wrong direction

### 2026-04-16 Early Morning: beta-flip probe

Key artifacts:

- `results/strength_second_gpu_validation_semantic_split/beta_flip_baseline.json`
- `results/strength_second_gpu_validation_semantic_split/beta_flip_probe.json`

What was tested:

- keep the model weights fixed
- only flip the sign of the beta-side modulation during inference

Observed result:

- the local probe changed from decreasing-with-strength to increasing-with-strength

Interpretation:

- the most suspicious part of the current failure is now concentrated around the beta path
- this is not full proof that beta is the only issue, but it is the strongest localized signal found so far

### 2026-04-16 Early Morning: beta-only repair runs

Result directories:

- `results/beta_only_repair_short`
- `results/beta_only_repair_longer`

Why this was done:

- to test a narrower repair path instead of another general retraining loop

Observed result:

- without beta flip, the probe remained reversed
- with beta flip, the same probe turned positive

Interpretation:

- the reversal is stable enough to reproduce across repair attempts
- the issue is not a random seed accident

### 2026-04-16 Morning to Afternoon: Pass 2A and provenance repair

Result directories:

- `results/beta_direction_pass2`
- `results/beta_direction_pass2_w1`
- `results/provenance_smoke`
- `results/provenance_smoke_eval_only`
- `results/provenance_smoke_eval_existing`

What changed:

- `beta_direction_loss` was wired into the training path and actually run
- runtime provenance artifacts were added so that each run can be audited from the exact merged execution config

Main result:

- local evidence still suggested the direction issue was real
- but Gate 1 / Gate 2 main evaluations still did not show a clean monotonic repair

Conservative conclusion:

- the repository cannot yet claim that the baseline no longer needs beta-flip
- the mainline failure has been narrowed, but not solved

## What Has Been Confirmed

The following points now have strong support from code and recent experiment outputs:

- the strength control path is present and active
- modulation deltas change with strength
- final outputs are not reliably monotonic
- recent evidence points more strongly to a reversed or mis-mapped output direction than to a missing signal
- provenance tracking is now necessary because evaluation conclusions depend on the exact resolved runtime configuration

## What Has Not Been Solved Yet

These statements would still be too strong:

- “strength control is already fixed”
- “the model now reliably produces weak < medium < strong”
- “beta-direction training has already removed the need for inference-time beta flip”

Those claims are not yet supported by the current Gate 2 results.
