# Semi-Oracle Calibration Suite v1

## Scope

This directory records the first semi-oracle degradation analysis after the explicit calibration scaffold and oracle-calibration benchmark were added.

Benchmarks:

- `Weather dlinear v2`
- `MTBench finance dlinear v2 100`
- `XTraffic dlinear v2 narrowed`

Compared modes:

- `localized_full_revision`
- `oracle_region`
- `oracle_tool`
- `oracle_intent`
- `oracle_calibration`

Important implementation note:

- `oracle_tool` is a minimal tool-family override in the current executor
- it is not yet a full hybrid-tool execution layer
- its purpose is to make tool mismatch observable in the current codebase, not to claim full tool-layer faithfulness

## Result Paths

- `weather_dlinear_v2/semi_oracle_summary.json`
- `mtbench_dlinear_v2_100/semi_oracle_summary.json`
- `xtraffic_dlinear_v2/semi_oracle_summary.json`

## Main Findings

### 1. `Weather` shows a clean degradation curve

Key pattern:

- `localized_full_revision avg_revision_gain = 0.0006`
- `oracle_region avg_revision_gain = 0.0330`
- `oracle_tool avg_revision_gain = 0.0846`
- `oracle_intent avg_revision_gain = 0.0870`
- `oracle_calibration avg_revision_gain = 0.1597`

Interpretation:

- fixing region helps immediately
- fixing intent/tool family helps again
- the largest remaining gain still comes from numeric calibration itself
- this is the cleanest evidence that `where`, `what`, and `how much` are separable sources of error

### 2. `MTBench` shows that semantic drift hurts mainly through cumulative magnitude

Key pattern:

- `localized_full_revision avg_revision_gain = 0.3801`
- `oracle_region avg_revision_gain = 0.4672`
- `oracle_tool avg_revision_gain = 0.5315`
- `oracle_intent avg_revision_gain = 0.5866`
- `oracle_calibration avg_revision_gain = 0.5598`

Important calibration pattern:

- `duration_error` drops from `11.15` to `0.11` already at `oracle_tool`
- `recovery_slope_error` stays small across all semi-oracle settings
- `signed_area_error` remains very large even after region and intent are fixed

Interpretation:

- MTBench again supports the conclusion that the hard problem is cumulative magnitude / area grounding
- once region is fixed, semantic improvements help steadily
- fully forcing pseudo-oracle calibration does not help further, which is consistent with the earlier oracle-calibration result

### 3. `XTraffic` remains dominated by target-space mismatch, not semi-oracle semantic fragility

Key pattern:

- `localized_full_revision avg_revision_gain = 0.4180`
- `oracle_region avg_revision_gain = 0.2566`
- `oracle_tool avg_revision_gain = 0.2579`
- `oracle_intent avg_revision_gain = 0.2579`
- `oracle_calibration avg_revision_gain = -0.7044`

Interpretation:

- once region is fixed, `oracle_tool` and `oracle_intent` are nearly identical on the current narrowed subset
- the dominant issue is still not semantic drift inside the semi-oracle ladder
- the dominant issue is that the current real-data pseudo target is misaligned with the executor manifold
- this confirms that XTraffic should be treated as a `proxy-target stress test`, not as a clean oracle-calibration benchmark

## Immediate Use

These results support the following next step order:

1. use `Weather` for calibration-method development and ablation
2. use `MTBench` for realistic semi-oracle robustness analysis
3. keep `XTraffic` as an outcome-level stress test plus executor-refit diagnostic
4. do not interpret current XTraffic pseudo parameters as clean oracle supervision
