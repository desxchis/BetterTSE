# Oracle Calibration Benchmark v1

## Scope

This directory records the first explicit oracle-calibration comparison after the calibration scaffold was added.

Benchmarks:

- `Weather dlinear v2`
- `XTraffic dlinear v2 narrowed`
- `MTBench finance dlinear v2 100`

Compared methods:

- `text_direct_numeric`
- `discrete_strength_table`
- `rule_local_stats`
- `oracle_calibration`

## Result Paths

- `weather_dlinear_v2/calibration_benchmark_summary.json`
- `xtraffic_dlinear_v2/calibration_benchmark_summary.json`
- `mtbench_dlinear_v2_100/calibration_benchmark_summary.json`

## Main Findings

### 1. Controlled calibration signal is real on `Weather`

`oracle_calibration` is clearly stronger than the heuristic strategies on the controlled benchmark.

Key pattern:

- `avg_normalized_parameter_error`: `0.24 ~ 0.26 -> 0.0`
- `avg_signed_area_error`: `3.07 ~ 3.34 -> 0.90`
- `avg_duration_error`: `0.97 -> 0.0`
- `avg_revision_gain`: `0.087 ~ 0.099 -> 0.160`

Interpretation:

- the calibration problem is measurable and non-trivial
- the current metrics are sensitive enough to capture meaningful improvements
- controlled synthetic supervision is usable for calibration-focused development

### 2. `XTraffic` does not yet provide a trustworthy oracle-calibration target

On the narrowed real benchmark, heuristic calibration stays mildly positive, but `oracle_calibration` is worse than the heuristics.

Key pattern:

- `rule_local_stats avg_revision_gain = 0.2579`
- `oracle_calibration avg_revision_gain = -0.7044`
- `oracle_calibration avg_signed_area_error = 184.15`

Interpretation:

- this is not evidence that calibration is unimportant
- it is evidence that the current XTraffic pseudo-parameter target is not a clean oracle for executor-space calibration
- real-data `revision_target` may be usable as outcome supervision, but current stored operator params should not yet be treated as gold numeric executor parameters

### 3. `MTBench` behaves like a realistic but weakly supervised calibration target

On `MTBench finance dlinear v2 100`, all heuristic strategies are very similar and `oracle_calibration` does not beat them.

Key pattern:

- `text_direct_numeric avg_revision_gain = 0.6076`
- `rule_local_stats avg_revision_gain = 0.5866`
- `oracle_calibration avg_revision_gain = 0.5598`
- `duration_error` and `recovery_slope_error` are already small for all methods
- `signed_area_error` remains large

Interpretation:

- the main remaining issue is not duration/recovery on this dataset
- the harder problem is cumulative revision magnitude / area calibration
- as with XTraffic, the real-data target is useful for evaluation but not yet a clean executor-parameter oracle

## Shape-Level Notes

### `Weather`

- `step` and `plateau` are the easiest families for the current system
- `flatline` is still hard and often negative under heuristic calibration
- `hump` and `irregular_noise` remain weakly calibrated

### `XTraffic`

- current narrowed subset is effectively all `step`
- the main failure is not family diversity but numeric grounding of disruption magnitude and duration

### `MTBench`

- `plateau` dominates the sample mix and is easier than `step`
- `step` has worse `normalized_parameter_error` and much larger signed-area error
- this supports the view that repricing-like edits are harder to ground numerically than drift-like edits

## XTraffic Executor Refit Sanity Check

A small executor-manifold sanity check was run on the canonical `XTraffic dlinear v2 narrowed` benchmark:

- script: `test_scripts/analyze_xtraffic_executor_refit.py`
- output: `xtraffic_dlinear_v2/executor_refit_analysis.json`

Comparison:

- `pseudo` = residual-derived pseudo-spec projected into executor params
- `heuristic` = current `rule_local_stats` calibrator
- `best_fit` = step-family executor refit against the revision target

Summary:

- `pseudo avg_revision_gain = -0.7044`
- `heuristic avg_revision_gain = 0.2579`
- `best_fit avg_revision_gain = 0.3721`
- `pseudo avg_signed_area_error = 184.15`
- `heuristic avg_signed_area_error = 47.45`
- `best_fit avg_signed_area_error = 1.46`

Interpretation:

- this strongly supports the executor-manifold mismatch hypothesis
- the current XTraffic pseudo target is not just noisy; it is systematically misaligned with the current step executor family
- average parameter pattern on the 9 step samples is:
  - `pseudo`: larger amplitude, much longer duration
  - `best_fit`: moderate amplitude, much shorter duration
- `best_fit` beats `pseudo` on `7 / 9` samples and beats `heuristic` on `8 / 9` samples

This means the next semi-oracle XTraffic experiments should be interpreted as `proxy-target stress tests`, not as clean oracle-calibration experiments.

## Immediate Next Step

Do not move directly to learned calibration on all real datasets.

Use the following order instead:

1. use `Weather` as the primary calibration development benchmark
2. keep `XTraffic` and `MTBench` as outcome-level transfer checks
3. build a semi-oracle experiment next:
   - `GT region + predicted intent`
   - `GT region + predicted tool`
4. for real datasets, treat current pseudo-parameter supervision as provisional rather than oracle-clean
