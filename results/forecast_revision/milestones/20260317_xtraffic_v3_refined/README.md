# XTraffic Real-Context v3 Refined

## Scope

- Domain-refined real-context benchmark on XTraffic.
- Data subset unchanged from v2:
  - `speed` channel
  - `drop_z >= 0.2`
  - allowed incident types:
    - `Hazard`
    - `NoInj`
    - `AHazard`
    - `1141`
    - `CarFire`
- Sample count:
  - `9`

## What Changed from v2

- `Hazard` / `AHazard`:
  - from `step` to `hump`
  - text changed to short disruption + gradual recovery
- `NoInj` / `1141` / `CarFire`:
  - keep `step`
  - keep short-duration default
- strength bucket now comes from real response:
  - `weak / medium / strong` based on `drop_z`
- duration is refined by operator family:
  - hump-like events are capped to short windows
  - step-like events keep short windows unless response is stronger

## Inputs

- benchmark:
  - `results/forecast_revision/benchmarks/xtraffic_p01_speed_dlinear_v3`
- run:
  - `results/forecast_revision/runs/xtraffic_p01_speed_dlinear_v3_real_suite`

## Results

### dlinear_like

- `base_only avg_revision_gain = 0.0000`
- `global_revision_only avg_revision_gain = 0.1711`
- `localized_full_revision avg_revision_gain = 0.2877`
- `oracle_region avg_revision_gain = 0.1339`
- `oracle_intent avg_revision_gain = 0.1649`

Additional metrics:

- `localized_full_revision avg_future_t_iou = 0.5185`
- `localized_full_revision avg_magnitude_calibration_error = 2.8346`
- `oracle_intent avg_magnitude_calibration_error = 2.1436`

## Comparison to v2

- v2 localized:
  - `avg_revision_gain = 0.4065`
  - `avg_magnitude_calibration_error = 2.7274`
- v3 localized:
  - `avg_revision_gain = 0.2877`
  - `avg_magnitude_calibration_error = 2.8346`

Interpretation:

- The new operator semantics are more defensible.
- But on this 9-sample subset, the refined labels did not improve the aggregate score over v2.
- The framework remains positive on real data, but the best current real checkpoint is still `v2`.

## Takeaway

- Traffic-specific narrowing is necessary.
- Operator refinement is directionally correct, but this specific v3 update does not beat v2 yet.
- Current best real-data checkpoint remains:
  - `results/forecast_revision/milestones/20260317_xtraffic_v2_narrowed`
