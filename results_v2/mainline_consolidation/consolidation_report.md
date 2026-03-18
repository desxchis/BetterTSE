# Mainline Consolidation Report

- Generated (UTC): 2026-03-18T11:06:18.120488+00:00
- Baseline: lstm_official (locked)
- Executor: tedit_hybrid + rule_local_stats + flow_guard(v2)

## Table 1 — Mode Ladder (Repeat 1)

| Dataset | base_only gain | global gain | localized gain |
|---|---:|---:|---:|
| weather | 0.0000 | -0.4258 | 0.0723 |
| mtbench | 0.0000 | -0.5829 | 0.2492 |
| xtraffic_speed | 0.0000 | -0.3482 | 0.1070 |
| xtraffic_flow | 0.0000 | 6.1183 | 1.0067 |

## Table 2 — Stability Across Repeats (localized)

| Dataset | r1 gain | r2 gain |
|---|---:|---:|
| weather | 0.0723 | 0.0700 |
| mtbench | 0.2492 | 0.2881 |
| xtraffic_speed | 0.1070 | 0.1069 |
| xtraffic_flow | 1.0067 | 1.0047 |

## Table 3 — XTraffic-flow Diagnosis (by_shape)

| Repeat | Shape | revision_gain | edited_mae_vs_revision_target |
|---|---|---:|---:|
| r1 | hump | 2.2015 | 99.0983 |
| r1 | step | -0.0000 | 55.3019 |
| r1 | none | 0.0000 | 0.0000 |
| r2 | none | 0.0000 | 0.0000 |
| r2 | hump | 2.3609 | 98.9389 |
| r2 | step | -0.0000 | 55.3019 |

## One-Page Findings

- Stable positive lines: weather, mtbench, xtraffic_speed
- XTraffic-flow localized gains: r1=1.0067, r2=1.0047
- Agent smoke: success=['weather', 'mtbench', 'xtraffic_flow'], failed=['xtraffic_speed']
