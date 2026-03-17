# XTraffic Real-Context v2 Narrowed

## Scope

- Narrowed real-context benchmark on XTraffic.
- Channel:
  - `speed`
- Candidate filtering:
  - allowed incident types:
    - `Hazard`
    - `NoInj`
    - `AHazard`
    - `1141`
    - `CarFire`
  - response filter:
    - `drop_z >= 0.2`
- Sample count:
  - `9`

## Inputs

- candidate source:
  - `results/forecast_revision/xtraffic_candidates/p01_speed_smoke/xtraffic_candidates_p01_done_speed.json`
- benchmark outputs:
  - `results/forecast_revision/benchmarks/xtraffic_p01_speed_naive_v2`
  - `results/forecast_revision/benchmarks/xtraffic_p01_speed_dlinear_v2`
- run outputs:
  - `results/forecast_revision/runs/xtraffic_p01_speed_naive_v2_real_suite`
  - `results/forecast_revision/runs/xtraffic_p01_speed_dlinear_v2_real_suite`

## Results

### naive_last

- `base_only avg_revision_gain = 0.0000`
- `global_revision_only avg_revision_gain = -0.0002`
- `localized_full_revision avg_revision_gain = -0.4273`
- `oracle_region avg_revision_gain = -0.2842`
- `oracle_intent avg_revision_gain = -0.1202`

Interpretation:
- Narrowing improved scale mismatch a lot relative to v1.
- But `naive_last` remains too weak for positive localized revision on this real subset.

### dlinear_like

- `base_only avg_revision_gain = 0.0000`
- `global_revision_only avg_revision_gain = 0.1761`
- `localized_full_revision avg_revision_gain = 0.4065`
- `oracle_region avg_revision_gain = 0.2509`
- `oracle_intent avg_revision_gain = 0.2539`
- `localized_full_revision avg_future_t_iou = 0.5044`

Interpretation:
- Real-context XTraffic is now positive on `dlinear_like`.
- Localized revision beats global revision on this narrowed subset.
- Real transfer is therefore not uniformly failing; it depends strongly on domain-aligned operator/channel/sample selection.

## Main Takeaways

- Domain-specific narrowing was necessary and effective.
- The key narrowing choices were:
  - use `speed` instead of `flow`
  - filter to incident types with clearer disruption semantics
  - keep only samples with observable future drop
- This preserves the general forecast-revision framework while adapting the execution semantics to traffic.

## Next Steps

- Add visualized case studies for this 9-sample subset.
- Convert the current weak-label benchmark into a cleaner traffic-specific benchmark:
  - disruption drop
  - severe disruption / near-flatline
  - sensor corruption
- Add non-applicable traffic samples so `revision_needed` can be tested in real data.
