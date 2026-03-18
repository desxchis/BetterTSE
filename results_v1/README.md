# Results v1

This folder packages the full output of the current mainline consolidation run for:

- `lstm_official` as locked `base_forecast`
- `tedit_hybrid` executor
- `rule_local_stats` calibration
- `flow_guard(v2)` enabled

Contents:

- `mainline_consolidation/consolidation_report.md`: human-readable summary tables
- `mainline_consolidation/consolidation_report.json`: machine-readable report
- `mainline_consolidation/revision_runs/`: per-dataset/per-repeat mode outputs (`base_only`, `global_revision_only`, `localized_full_revision`)
- `mainline_consolidation/flow_diagnosis/`: XTraffic-flow diagnosis artifacts
- `mainline_consolidation/agent_smoke/`: LangGraph/Agent smoke outputs
- `mainline_consolidation/bench_slices/`: fixed benchmark slices used for repeats
