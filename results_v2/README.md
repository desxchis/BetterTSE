# Results v2

This package is the deliverable bundle for the latest mainline consolidation run.

Mainline lock:
- baseline: `lstm_official`
- executor: `tedit_hybrid`
- calibration: `rule_local_stats`
- guard: `flow_guard(v2)` with flow-step conservative no-op shrink

Localized gain (repeat-1 / repeat-2):
- weather: `0.0723 / 0.0700`
- mtbench: `0.2492 / 0.2881`
- xtraffic_speed: `0.1070 / 0.1069`
- xtraffic_flow: `1.0067 / 1.0047`

Flow diagnosis:
- hump bucket: clearly positive
- step bucket: pulled to near-zero (non-regressive)

Agent/LangGraph smoke:
- v4 consolidated smoke had one transient `xtraffic_speed` failure due API auth instability
- rerun added at `mainline_consolidation/agent_smoke/xtraffic_speed_smoke_rerun.json` and completed successfully

Contents:
- `mainline_consolidation/`: full raw run outputs (tables, repeats, flow diagnosis, agent smoke)
- `TABLES.md`: main report with Table 1/2/3 + one-page findings
- `CASES.md`: indexed case list (success + boundary) for each dataset
- `cases/`: copied case visualization images referenced by `CASES.md`
