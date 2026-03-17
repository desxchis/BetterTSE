# XTraffic Real-Context v2 with Non-Applicable Samples

## Scope

- Real-context XTraffic follow-up on top of `v2 narrowed`.
- Channel:
  - `speed`
- Positive sample filter:
  - allowed incident types:
    - `Hazard`
    - `NoInj`
    - `AHazard`
    - `1141`
    - `CarFire`
  - response filter:
    - `drop_z >= 0.2`
- Added real `non-applicable` windows:
  - same shard
  - same node family
  - far from aligned incident windows
  - weak response only:
    - `abs(drop_z) <= 0.15`

## Inputs

- candidate source:
  - `results/forecast_revision/xtraffic_candidates/p01_speed_smoke/xtraffic_candidates_p01_done_speed.json`
- benchmark outputs:
  - `results/forecast_revision/benchmarks/xtraffic_p01_speed_naive_v2_nonapp`
  - `results/forecast_revision/benchmarks/xtraffic_p01_speed_dlinear_v2_nonapp`
- run outputs:
  - `results/forecast_revision/runs/xtraffic_p01_speed_naive_v2_nonapp_suite`
  - `results/forecast_revision/runs/xtraffic_p01_speed_dlinear_v2_nonapp_suite`

## Sample Mix

- total samples:
  - `23`
- applicable:
  - `14`
- non-applicable:
  - `9`

The non-applicable samples use real no-incident windows and explicit no-op context such as:

- `无新增影响`
- `暂无额外冲击`
- `维持原预测`
- `没有新的修正信号`

## Results

### `naive_last`

- `localized_full_revision avg_revision_gain = -0.1261`
- `localized_full_revision applicable_avg_revision_gain = -0.2072`
- `localized_full_revision non_applicable_avg_revision_needed_match = 1.0`
- `localized_full_revision non_applicable_avg_over_edit_rate = 0.0`

Interpretation:

- Real no-op detection works.
- But `naive_last` is still too weak to support positive localized revision on the real traffic subset.

### `dlinear_like`

- `global_revision_only avg_revision_gain = 0.0577`
- `localized_full_revision avg_revision_gain = 0.1259`
- `localized_full_revision applicable_avg_revision_gain = 0.2068`
- `localized_full_revision avg_future_t_iou = 0.2572`
- `localized_full_revision non_applicable_avg_revision_needed_match = 1.0`
- `localized_full_revision non_applicable_avg_over_edit_rate = 0.0`

Interpretation:

- The real-data line remains positive after adding real no-op windows.
- Localized revision still beats global revision overall.
- The gate now behaves correctly on real `non-applicable` samples.

## Important Caveat

- `oracle_intent` forces the GT intent for all samples, so:
  - its `revision_needed_match` is not meaningful once non-applicable samples are included
- For real gate evaluation, use:
  - `global_revision_only`
  - `localized_full_revision`
  - `base_only`

## Main Takeaways

- Real `revision_needed` evaluation is now available on XTraffic.
- On the current real subset:
  - `non_applicable_avg_revision_needed_match = 1.0`
  - `non_applicable_avg_over_edit_rate = 0.0`
- The best pure positive-transfer checkpoint is still:
  - `results/forecast_revision/milestones/20260317_xtraffic_v2_narrowed`
- The best checkpoint for real gate evaluation is now:
  - `results/forecast_revision/milestones/20260317_xtraffic_v2_nonapp`
