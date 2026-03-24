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

## Student Kickoff Status

An experimental pure-editing student path now exists, but it is not yet promoted into the locked main protocol.

Current student design:

- `tool-conditioned heads`, not unified spec
- supported heads:
  - `spike_inject`
  - `step_shift`
  - `hybrid_up`
  - `hybrid_down`
  - `volatility_global_scale`
  - `volatility_local_burst`
  - `volatility_envelope_monotonic`
- `preview_non_monotonic` remains outside the student path on purpose

Current repo entrypoints:

- teacher dump + train:
  - `test_scripts/train_pure_editing_student.py --testset <event_json> --output-dir <student_dir>`
- runtime injection:
  - `run_pipeline.py --how-much-student-model <student_json>`

Current reading:

- ETTh1 24-sample kickoff:
  - heldout target MAE:
    - student `0.4292`
    - teacher `0.4219`
    - heuristic `0.5248`
  - student better rate vs heuristic: `1.00`
- ETTh1 56-sample kickoff:
  - heldout target MAE:
    - student `0.4962`
    - teacher `0.4218`
    - heuristic `0.5025`
  - student better rate vs heuristic: `0.64`
- ETTh1 + ETTm1 combined kickoff:
  - heldout target MAE:
    - student `0.5408`
    - teacher `0.4257`
    - heuristic `0.5265`
  - student better rate vs heuristic: `0.44`

Current interpretation:

- the teacher-label dump, tool-conditioned head training, and runtime override path are all functional
- the current lightweight linear student is not yet stable enough to replace teacher search or enter the main pure-editing result table
- the next student work should focus on model adequacy and per-tool calibration quality, not on reopening volatility taxonomy or routing

## Student Capacity Ablation

The current student question is no longer whether a student path can run, but what the minimum adequate capacity is.

Current ablation set:

1. `linear`
2. `quadratic`
3. `mixed_capacity`
   - `volatility_global_scale`, `volatility_envelope_monotonic`: tiny MLP heads
   - `spike_inject`, `hybrid_up`, `volatility_local_burst`: quadratic heads
   - `step_shift`, `hybrid_down`: linear heads

Current ETTh1 56-sample heldout reading:

- `linear`
  - student MAE `0.5053`
  - heuristic MAE `0.5024`
  - teacher gap closed `0.0712`
- `quadratic`
  - student MAE `0.7004`
  - heuristic MAE `0.5027`
  - teacher gap closed `-1.5313`
- `mixed_capacity`
  - student MAE `0.4843`
  - heuristic MAE `0.5025`
  - teacher gap closed `0.4563`

Current interpretation:

- `mixed_capacity` is the best current student variant and materially improves over the plain linear head
- a naive global `quadratic` upgrade overfits and should not be treated as the default next step
- the remaining bottleneck is per-tool adequacy, especially:
  - `volatility_global_scale`
  - `volatility_envelope_monotonic`
  - cross-distribution robustness

Current runtime note:

- 3-sample ETTh1 smoke confirms `mixed_capacity` runtime override works and is less damaging than the initial linear kickoff
- it is still worse than the frozen teacher-backed `full_bettertse` path, so student remains experimental rather than promoted

## Runtime-Safe Student Variants

To reduce runtime mismatch, the current student path now supports:

1. `v1`
2. `clip`
   - clip each head to its teacher-label training quantile band
3. `clip_guard`
   - apply clipping
   - if the sample is in low-support space or the prediction is clipped too aggressively, fallback to heuristic how-much
4. `clip_softguard`
   - apply clipping
   - use per-tool risk calibration instead of a single global guard
   - only raise fallback pressure when low support and large support-domain deviation appear together
   - blend student and heuristic parameters with a risk-dependent soft fallback before any hard fallback is allowed

Current ETTh1 56-sample heldout reading for `mixed_capacity`:

- `v1`
  - student MAE `0.4843`
  - teacher gap closed `0.4563`
  - fallback rate `0.00`
- `clip`
  - student MAE `0.4614`
  - teacher gap closed `0.3083`
  - fallback rate `0.00`
- `clip_guard`
  - student MAE `0.4906`
  - teacher gap closed `0.3028`
  - fallback rate `0.82`
- `clip_softguard`
  - student MAE `0.4594`
  - teacher gap closed `0.1513`
  - fallback rate `0.27`
  - softguard rate `0.45`
  - avg guard weight `0.3836`

Current ETTh1+ETTm1 combined heldout reading for `mixed_capacity clip_softguard`:

- student MAE `0.5324`
- heuristic MAE `0.4877`
- teacher gap closed `-3.8722`
- fallback rate `0.00`
- softguard rate `0.125`

Current ETTh1 20-sample runtime smoke:

- frozen teacher-backed full chain
  - target MAE `1.1404`
  - preservation MAE `0.6114`
- `mixed_capacity v1`
  - target MAE `1.3280`
  - preservation MAE `0.6657`
- `mixed_capacity clip`
  - target MAE `1.3673`
  - preservation MAE `0.7071`
- `mixed_capacity clip_guard`
  - target MAE `0.6552`
  - preservation MAE `0.4189`
  - note: this was only validated on the earlier 3-sample smoke because the global guard was too conservative
- `mixed_capacity clip_softguard`
  - target MAE `1.1805`
  - preservation MAE `0.6181`

Current interpretation:

- the best current student is still `mixed_capacity`
- naive clipping helps heldout MAE but hurts runtime
- per-tool softguard materially improves deployment safety relative to raw `v1` and plain `clip`
- however, the student path still does not beat the frozen teacher-backed full chain, and the cross-distribution heldout result remains below heuristic
- the current bottleneck is therefore deployment calibration for the hardest heads, not taxonomy, routing, or volatility split design

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

## Post-Integration Formal Rerun

Reference artifacts:

- `tmp/pipeline_full_mainline20_volsplit.json`
- `tmp/pipeline_direct_mainline20_volsplit.json`
- `tmp/pipeline_woloc_mainline20_volsplit.json`
- `tmp/pipeline_wocanonical_mainline20_volsplit.json`
- `tmp/pipeline_full_mainline20_volsplit_routing.json`

Current reading on the 20-sample pure-editing mainline benchmark:

- `full_bettertse`
  - target MAE `0.9853`
  - target MSE `16.3463`
  - t-IoU `0.3284`
  - preservation MAE `0.4568`
- `direct_edit`
  - `1.6157 / 35.3121 / 0.1593 / 0.8957`
- `wo_localization`
  - `6.6399 / 115.8596 / 0.0852 / 6.4923`
- `wo_canonical_layer`
  - `1.2079 / 22.8725 / 0.2841 / 0.6637`

Current interpretation:

- after volatility split integration, the full mainline remains clearly stronger than `direct_edit`
- localization is still the dominant contributor
- the canonical layer remains positive after integration
- there is no sign that routing the volatility family through split tools damaged the non-volatility tool families in this small formal rerun

Initial volatility routing diagnosis:

- `total_volatility_cases`: `6`
- `preview_case_count`: `3`
- `fallback_or_unsupported_count`: `0`
- `overall_route_correct_rate`: `0.33`
- routed tools in the full rerun:
  - all 6 volatility cases were executed by `volatility_global_scale`
- subpattern reading:
  - `uniform_variance`: `1/1` routed correctly
  - `local_burst`: `0/2` routed correctly
  - `non_monotonic_envelope`: `3/3` kept on the preview-side generic path

Current conclusion:

- the volatility split itself is now integrated and does not break the pure-editing mainline
- the new primary bottleneck is no longer operator adequacy; it is route correctness from prompt semantics into the split volatility sub-tools

## Volatility Subtype Schema And Route Closure

Reference artifacts:

- `tmp/how_much/pure_editing/volroute24_seed37/pure_editing_volatility_route_closure_ETTh1_24.json`
- `tmp/how_much/pure_editing/volroute24_seed37_route_closure_text_v2.json`
- `tmp/how_much/pure_editing/volroute24_seed37_route_closure_llm4.json`
- `tmp/pipeline_full_mainline20_volroutefix.json`
- `tmp/pipeline_direct_mainline20_volroutefix.json`
- `tmp/pipeline_full_mainline20_volroutefix_routing.json`

What changed:

- `volatility_subtype` is now an explicit planner schema field
- routing is recorded as `proposed_subtype -> guarded_subtype -> final_subtype`
- supported subtype set is:
  - `global_scale`
  - `local_burst`
  - `envelope_monotonic`
- `preview_non_monotonic` remains outside the supported route and maps to the preview-side generic path

Stable route-only closure result on the subtype-aware benchmark:

- benchmark:
  - `uniform_variance`, `local_burst`, `monotonic_envelope`, `non_monotonic_envelope`
  - each collected with quota `6`
- `text_guard_only` closure:
  - `supported_route_accuracy = 1.00`
  - `preview_not_misrouted_rate = 1.00`
- `planner_llm` smoke:
  - `max_samples = 4`
  - `supported_route_accuracy = 1.00`
  - `preview_not_misrouted_rate = 1.00`

Important boundary:

- the subtype-aware route closure benchmark is now the authoritative routing benchmark
- legacy event-driven mainline prompts still often describe volatility only as generic `持续异常 / 杂乱跳变 / 无规律波动`
- those prompts are often too coarse to fully determine subtype semantics

Post-route-fix formal rerun:

- `full_bettertse`
  - target MAE `1.1026`
  - target MSE `21.2810`
  - t-IoU `0.2993`
  - preservation MAE `0.5854`
- `direct_edit`
  - `1.5485 / 30.8101 / 0.1593 / 0.8214`

Updated volatility routing diagnosis on the rerun:

- `total_volatility_cases`: `6`
- `preview_case_count`: `3`
- `fallback_or_unsupported_count`: `0`
- `overall_route_correct_rate`: `0.67`
- `overall_subtype_correct_rate`: `0.33`
- subpattern reading:
  - `local_burst`: `2/2` routed correctly
  - `uniform_variance`: `0/1` because planner still proposed `local_burst`
  - `non_monotonic_envelope`: `0/3` because legacy generic prompts still under-specify preview semantics

Current interpretation:

- the route closure itself is solved on the correct subtype-aware benchmark
- the remaining mainline volatility routing errors come from coarse prompt semantics, not from subtool inadequacy
- this means the current bottleneck has shifted from `route correctness` in the closure sense to `upstream subtype expressivity` in legacy generic prompts

## Mainline Volatility-Subtype-Aware Benchmark v2

Reference artifacts:

- `tmp/event_driven_mainline20_volsubtype_v2/event_driven_testset_ETTh1_20_volsubtype_v2.json`
- `tmp/pipeline_full_mainline20_volsubtype_v2.json`
- `tmp/pipeline_direct_mainline20_volsubtype_v2.json`
- `tmp/pipeline_full_mainline20_volsubtype_v2_routing.json`
- `tmp/event_driven_mainline24_volsubtype_v2_monotonic/event_driven_testset_ETTh1_20_volsubtype_v2.json`
- `tmp/pipeline_full_mainline24_volsubtype_v2_monotonic.json`
- `tmp/pipeline_direct_mainline24_volsubtype_v2_monotonic.json`
- `tmp/pipeline_full_mainline24_volsubtype_v2_monotonic_routing.json`
- `tmp/event_driven_protocol50_volsubtype_v2/event_driven_testset_ETTh1_50_volsubtype_v2.json`
- `tmp/pipeline_full_mainline50_volsubtype_v2.json`
- `tmp/pipeline_direct_mainline50_volsubtype_v2.json`
- `tmp/pipeline_full_mainline50_volsubtype_v2_routing.json`
- `tmp/event_driven_ettm1_36_volsubtype_v2/event_driven_testset_ETTm1_36_volsubtype_v2.json`
- `tmp/pipeline_full_ettm1_42_volsubtype_v2.json`
- `tmp/pipeline_direct_ettm1_42_volsubtype_v2.json`
- `tmp/pipeline_full_ettm1_42_volsubtype_v2_routing.json`

What changed:

- the mainline benchmark now refreshes only volatility samples
- non-volatility samples remain unchanged
- volatility prompts and labels are rewritten to expose explicit subtype semantics:
  - `global_scale`
  - `local_burst`
  - `envelope_monotonic`
  - `preview_non_monotonic`
- `preview_non_monotonic` remains a preview-side route target and is not forced into the supported tool path

Current volatility distribution on the 20-sample mainline v2 benchmark:

- `global_scale`: `1`
- `local_burst`: `2`
- `envelope_monotonic`: `0`
- `preview_non_monotonic`: `3`

Formal rerun on mainline v2:

- `full_bettertse`
  - target MAE `1.1490`
  - target MSE `20.9370`
  - t-IoU `0.3080`
  - preservation MAE `0.6095`
- `direct_edit`
  - target MAE `1.3092`
  - target MSE `23.6511`
  - t-IoU `0.1682`
  - preservation MAE `0.5664`

Volatility routing diagnosis on mainline v2:

- `total_volatility_cases`: `6`
- `preview_case_count`: `3`
- `fallback_or_unsupported_count`: `0`
- `overall_route_correct_rate`: `1.00`
- `overall_subtype_correct_rate`: `1.00`
- `preview_not_misrouted_rate`: `1.00`

Per-subpattern routed reading:

- `local_burst`
  - `2/2` routed to `volatility_local_burst`
- `uniform_variance`
  - `1/1` routed to `volatility_global_scale`
- `non_monotonic_envelope`
  - `3/3` kept on preview-side routing with `final_subtype = preview_non_monotonic`

Current interpretation:

- the subtype schema and split routing are now fully exercised by a mainline benchmark that actually exposes subtype semantics
- the previous mainline routing failures were benchmark-prompt granularity failures, not routing-system failures
- once benchmark volatility prompts are made subtype-aware, the route bottleneck disappears on the current 20-sample rerun

## Mainline Volatility-Subtype-Aware Benchmark v2 With Monotonic Coverage

Reference artifacts:

- `tmp/event_driven_mainline24_volsubtype_v2_monotonic/event_driven_testset_ETTh1_20_volsubtype_v2.json`
- `tmp/pipeline_full_mainline24_volsubtype_v2_monotonic.json`
- `tmp/pipeline_direct_mainline24_volsubtype_v2_monotonic.json`
- `tmp/pipeline_full_mainline24_volsubtype_v2_monotonic_routing.json`

What changed:

- the v2 refresh path can now supplement the mainline with additional `noise_injection` samples until a requested monotonic coverage target is met
- the generated monotonic samples reuse the same subtype-aware prompt template as the mainline v2 refresh
- this closes the only known mainline-v2 coverage hole from the earlier 20-sample rerun

Current volatility distribution on the 22-sample monotonic-augmented mainline v2 benchmark:

- `global_scale`: `1`
- `local_burst`: `2`
- `envelope_monotonic`: `2`
- `preview_non_monotonic`: `3`

Formal rerun on the monotonic-augmented mainline v2 benchmark:

- `full_bettertse`
  - target MAE `1.1591`
  - target MSE `20.0611`
  - t-IoU `0.2959`
  - preservation MAE `0.6212`
- `direct_edit`
  - target MAE `1.2752`
  - target MSE `22.1279`
  - t-IoU `0.1798`
  - preservation MAE `0.5518`

Volatility routing diagnosis on the monotonic-augmented rerun:

- `total_volatility_cases`: `8`
- `preview_case_count`: `3`
- `fallback_or_unsupported_count`: `0`
- `overall_route_correct_rate`: `1.00`
- `overall_subtype_correct_rate`: `1.00`
- `preview_not_misrouted_rate`: `1.00`

Per-subpattern routed reading:

- `uniform_variance`
  - `1/1` routed to `volatility_global_scale`
- `local_burst`
  - `2/2` routed to `volatility_local_burst`
- `monotonic_envelope`
  - `2/2` routed to `volatility_envelope_monotonic`
- `non_monotonic_envelope`
  - `3/3` kept on preview-side routing with `final_subtype = preview_non_monotonic`

Current interpretation:

- the subtype-aware mainline no longer has a volatility coverage hole; all three supported volatility subtypes are now exercised in the formal rerun
- once the benchmark text explicitly carries subtype semantics, the routed split policy remains stable in the live main pipeline, including `envelope_monotonic`
- the remaining pure-editing volatility question is no longer about routing correctness on the current mainline v2 path

## Mainline50 Volatility Split Stability Sign-Off

Reference artifacts:

- `tmp/event_driven_protocol50_volsubtype_v2/event_driven_testset_ETTh1_50_volsubtype_v2.json`
- `tmp/pipeline_full_mainline50_volsubtype_v2.json`
- `tmp/pipeline_direct_mainline50_volsubtype_v2.json`
- `tmp/pipeline_full_mainline50_volsubtype_v2_routing.json`

What changed:

- the subtype-aware mainline refresh was scaled from the earlier 20/22-sample reruns to the existing 50-sample event-driven protocol benchmark
- volatility coverage was explicitly supplemented until the supported routed subtypes all had non-trivial support:
  - `global_scale = 3`
  - `local_burst = 3`
  - `envelope_monotonic = 3`
  - `preview_non_monotonic = 8`
- this produced a 56-sample subtype-aware mainline benchmark for stability sign-off

Formal rerun on the 56-sample subtype-aware mainline benchmark:

- `full_bettertse`
  - target MAE `1.0931`
  - target MSE `19.1847`
  - t-IoU `0.3018`
  - preservation MAE `0.4493`
- `direct_edit`
  - target MAE `1.3892`
  - target MSE `22.2449`
  - t-IoU `0.1321`
  - preservation MAE `0.6937`

Volatility routing diagnosis on the 56-sample rerun:

- `total_volatility_cases`: `17`
- `preview_case_count`: `8`
- `fallback_or_unsupported_count`: `0`
- `overall_route_correct_rate`: `1.00`
- `overall_subtype_correct_rate`: `1.00`
- `preview_not_misrouted_rate`: `1.00`

Per-subpattern routed reading:

- `uniform_variance`
  - `3/3` routed to `volatility_global_scale`
- `local_burst`
  - `3/3` routed to `volatility_local_burst`
- `monotonic_envelope`
  - `3/3` routed to `volatility_envelope_monotonic`
- `non_monotonic_envelope`
  - `8/8` kept on preview-side routing with `final_subtype = preview_non_monotonic`

Current interpretation:

- the volatility split now has both system-closure evidence and mainline-scale stability evidence
- supported subtype routing remains perfect on the current 56-sample sign-off rerun
- `full_bettertse` remains clearly stronger than `direct_edit`, and preservation does not become a new blocking issue at this scale
- at the current protocol scope, the volatility split can be treated as resolved at the system level

## Cross-Distribution Frozen Recheck

Reference artifacts:

- `tmp/event_driven_ettm1_36/event_driven_testset_ETTm1_36.json`
- `tmp/event_driven_ettm1_36_volsubtype_v2/event_driven_testset_ETTm1_36_volsubtype_v2.json`
- `tmp/pipeline_full_ettm1_42_volsubtype_v2.json`
- `tmp/pipeline_direct_ettm1_42_volsubtype_v2.json`
- `tmp/pipeline_full_ettm1_42_volsubtype_v2_routing.json`

What changed:

- the same frozen volatility split design was replayed on a second data distribution, `ETTm1`, without changing:
  - operator definitions
  - subtype schema
  - route guard logic
  - planner/runtime method set
- the baseline 36-sample event-driven benchmark was refreshed into a 42-sample subtype-aware v2 benchmark with explicit coverage:
  - `global_scale = 3`
  - `local_burst = 4`
  - `envelope_monotonic = 3`
  - `preview_non_monotonic = 5`

Formal rerun on the second-distribution subtype-aware benchmark:

- `full_bettertse`
  - target MAE `0.9843`
  - target MSE `22.4546`
  - t-IoU `0.2490`
  - preservation MAE `0.3703`
- `direct_edit`
  - target MAE `1.1407`
  - target MSE `26.0659`
  - t-IoU `0.1059`
  - preservation MAE `0.5159`

Volatility routing diagnosis on the second-distribution rerun:

- `total_volatility_cases`: `15`
- `preview_case_count`: `5`
- `fallback_or_unsupported_count`: `0`
- `overall_route_correct_rate`: `1.00`
- `overall_subtype_correct_rate`: `1.00`
- `preview_not_misrouted_rate`: `1.00`

Per-subpattern routed reading:

- `uniform_variance`
  - `3/3` routed to `volatility_global_scale`
- `local_burst`
  - `4/4` routed to `volatility_local_burst`
- `monotonic_envelope`
  - `3/3` routed to `volatility_envelope_monotonic`
- `non_monotonic_envelope`
  - `5/5` kept on preview-side routing with `final_subtype = preview_non_monotonic`

Current interpretation:

- the volatility split no longer looks specific to the original ETTh1 sign-off distribution
- under a frozen method stack, the subtype-aware benchmark and split routing stay stable on a second data condition
- `full_bettertse` continues to beat `direct_edit` while preservation remains better, so the volatility split no longer needs active development before opening pure-editing student work

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

The next step is still not operator redesign or student work.

It is:

1. treat the subtype-aware mainline benchmark v2, with subtype-coverage supplementation when needed, as the current volatility-aware pure-editing mainline
2. keep legacy generic volatility prompts as historical artifacts, not as the authoritative routing benchmark
3. do not reopen volatility operator or routing work unless a larger future benchmark reveals a new failure mode
4. when pure-editing student work starts, treat the current volatility split as a frozen tool-conditioned taxonomy rather than a still-moving target
