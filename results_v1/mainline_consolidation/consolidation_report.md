# Mainline Consolidation Report

- Generated (UTC): 2026-03-18T10:16:37.274849+00:00
- Baseline: lstm_official (locked)
- Executor: tedit_hybrid + rule_local_stats + flow_guard(v2)

## Table 1 — Mode Ladder (Repeat 1)

| Dataset | base_only gain | global gain | localized gain |
|---|---:|---:|---:|
| weather | 0.0000 | -0.4258 | 0.0722 |
| mtbench | 0.0000 | -0.5828 | 0.2493 |
| xtraffic_speed | 0.0000 | -0.3483 | 0.1070 |
| xtraffic_flow | 0.0000 | 5.9102 | 0.9378 |

## Table 2 — Stability Across Repeats (localized)

| Dataset | r1 gain | r2 gain |
|---|---:|---:|
| weather | 0.0722 | 0.0700 |
| mtbench | 0.2493 | 0.2882 |
| xtraffic_speed | 0.1070 | 0.1068 |
| xtraffic_flow | 0.9378 | 0.9592 |

## Table 3 — XTraffic-flow Diagnosis (by_shape)

| Repeat | Shape | revision_gain | edited_mae_vs_revision_target |
|---|---|---:|---:|
| r1 | hump | 2.3931 | 98.9067 |
| r1 | step | -0.1925 | 55.4944 |
| r1 | none | 0.0000 | 0.0000 |
| r2 | none | 0.0000 | 0.0000 |
| r2 | hump | 2.3962 | 98.9036 |
| r2 | step | -0.1891 | 55.4909 |

## One-Page Findings

- Stable positive lines: weather, mtbench, xtraffic_speed
- XTraffic-flow localized gains: r1=0.9378, r2=0.9592
- Agent smoke: success=['weather', 'mtbench', 'xtraffic_speed', 'xtraffic_flow'], failed=[]
