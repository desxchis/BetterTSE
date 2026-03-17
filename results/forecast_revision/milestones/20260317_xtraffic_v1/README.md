# XTraffic Real-Context v1

## Scope

- Real-world complement to the Weather controlled benchmark.
- Data source:
  - `p01_done.npy`
  - `incidents_y2023.csv`
  - `sensor_meta_feature.csv`
  - `node_order.npy`
- Baselines:
  - `naive_last`
  - `dlinear_like`
- Modes:
  - `base_only`
  - `global_revision_only`
  - `localized_full_revision`
  - `oracle_region`
  - `oracle_intent`

## Data Alignment

- `p01_done.npy` shape: `(8928, 16972, 3)`
- `sensor_meta_feature.csv` rows: `16972`
- `node_order.npy` rows: `16972`
- `station_id` and `node_order` are fully aligned.
- Real incident candidates built from January shard:
  - `results/forecast_revision/xtraffic_candidates/p01_smoke`
  - first 100 incidents -> 66 aligned candidates

## Benchmark

- Real-context benchmark outputs:
  - `results/forecast_revision/benchmarks/xtraffic_p01_naive_v1`
  - `results/forecast_revision/benchmarks/xtraffic_p01_dlinear_v1`
- Current benchmark type:
  - `real_context_weak_labels`
- Important limitation:
  - `revision_target` is set to `future_gt`
  - event mask and intent are weak labels derived from incident metadata
  - `oracle_calibration` is not defined for this version

## Results

### naive_last

- run dir:
  - `results/forecast_revision/runs/xtraffic_p01_naive_v1_real_suite`
- summary:
  - `base_only avg_future_gt_mae = 59.4028`
  - `global_revision_only avg_revision_gain = 0.0002`
  - `localized_full_revision avg_revision_gain = -13.0156`
  - `localized_full_revision avg_future_t_iou = 0.5633`
  - `oracle_intent avg_revision_gain = -5.8850`

### dlinear_like

- run dir:
  - `results/forecast_revision/runs/xtraffic_p01_dlinear_v1_real_suite`
- summary:
  - `base_only avg_future_gt_mae = 68.6890`
  - `global_revision_only avg_revision_gain = 1.6565`
  - `localized_full_revision avg_revision_gain = -11.4402`
  - `localized_full_revision avg_future_t_iou = 0.5633`
  - `oracle_intent avg_revision_gain = -6.0949`

## Current Interpretation

- Real data access and alignment are now verified.
- The current localized revision stack does not transfer directly from Weather to XTraffic.
- The main failure mode is not region grounding:
  - `localized_full_revision` still gives `t-IoU ~= 0.56`
  - `oracle_region` and `oracle_intent` remain negative
- This points to a domain-shift problem in:
  - operator semantics
  - calibration scale
  - execution profile

## Next Steps

- Replace current generic revision semantics with traffic-specific operators:
  - congestion increase
  - disruption drop
  - noisy sensor corruption
- Rebuild weak labels from incident type and duration with traffic-specific calibration.
- Add a filtered candidate split:
  - drop ambiguous `Other`
  - prefer high-duration or high-confidence incident types
- Re-run `naive_last` and `dlinear_like` before adding stronger baselines.
