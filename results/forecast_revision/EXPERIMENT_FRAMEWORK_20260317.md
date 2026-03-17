# Forecast Revision Experiment Framework

## Goal

Use the current datasets to make the forecast-revision experimental framework stable enough for sustained iteration, without opening new data lines.

The framework must answer four questions clearly:

1. Is the task itself valid?
2. Does localized revision beat global revision?
3. Can the framework transfer from controlled data to real data?
4. Can the framework abstain correctly when no revision is needed?

## Fixed Task Definition

Given:

- `history_ts`
- `base_forecast`
- `context_text`

predict:

- `edited_forecast`

with the interpretation:

- `base_forecast = F(history_ts)`
- `edited_forecast = R(history_ts, base_forecast, context_text)`

The problem is not end-to-end forecasting from text.
It is post-hoc forecast revision.

## Fixed Subproblem Decomposition

All experiments should be interpreted through these four subproblems:

1. `revision_needed`
2. `where`
3. `what`
4. `how much / how long`

This decomposition should stay fixed across all datasets.

## Calibration Framework Assembly

The current no-GPU priority is framework assembly, not more training.

Calibration experiments should now be organized through a config-driven runner:

- runner:
  - [prepare_forecast_revision_calibration_framework.py](/root/autodl-tmp/BetterTSE-main/test_scripts/prepare_forecast_revision_calibration_framework.py)
- example configs:
  - [weather_dlinear_v2.json](/root/autodl-tmp/BetterTSE-main/configs/forecast_revision_calibration/weather_dlinear_v2.json)
  - [xtraffic_dlinear_v2.json](/root/autodl-tmp/BetterTSE-main/configs/forecast_revision_calibration/xtraffic_dlinear_v2.json)
  - [mtbench_dlinear_v2_100.json](/root/autodl-tmp/BetterTSE-main/configs/forecast_revision_calibration/mtbench_dlinear_v2_100.json)

Default behavior:

- read one config
- resolve benchmark / held-out / learned-model dependencies
- write:
  - `experiment_plan.json`
  - `run_commands.sh`
  - `README.md`
- do not execute by default

This keeps experiment assembly reproducible even when training is intentionally deferred.

Recommended command:

```bash
python test_scripts/prepare_forecast_revision_calibration_framework.py \
  --config configs/forecast_revision_calibration/weather_dlinear_v2.json
```

This layer is meant to stabilize:

- directory naming
- stage ordering
- train/eval dependency wiring
- benchmark-role discipline

## Dataset-Role Discipline

Do not treat every candidate multimodal time-series dataset as a main benchmark.

The current paper-facing role split should stay explicit:

- `CiK`:
  - benchmark-design template only
  - use for field organization, context philosophy, and evaluation framing
  - do not treat as the main empirical battlefield for this line
- `XTraffic`:
  - main real benchmark
  - use as the primary empirical battlefield for localized forecast revision
- `MTBench`:
  - real native-text complement
  - use as realism / transfer support, not as the main benchmark
- `Time-MMD` or `Time-IMM`:
  - optional later extension
  - use only one of them if a broader generalization section is opened later
- `CGTSF` and pure traffic baselines:
  - lightweight complement / appendix material
  - not part of the core evidence chain

Reason:

- the current risk is not insufficient data coverage
- the current risk is scope spread and a diluted paper narrative
- benchmark breadth must remain subordinate to the forecast-revision story

## Dataset Roles

### 1. `Weather v4`

Role:

- controlled proof point

Purpose:

- validate task definition
- validate oracle decomposition
- validate localized vs global comparison under clean supervision

What `Weather` is allowed to prove:

- the task is meaningful
- the synthetic benchmark is usable
- the decomposition has measurable signal

What `Weather` is not supposed to prove:

- real-world transfer
- real text understanding

Reference:

- [Weather v4 milestone](/root/autodl-tmp/BetterTSE-main/results/forecast_revision/milestones/20260316_weather_v4_cal/README.md)

### 2. `XTraffic v2 narrowed`

Role:

- real structured-event positive-transfer checkpoint
- primary real benchmark

Purpose:

- validate transfer to real event-conditioned forecasting
- validate localized revision on real data

Dataset interpretation:

- real time series
- real incident context
- structured event converted to text

What `XTraffic v2 narrowed` is allowed to prove:

- the framework transfers to real traffic data
- localized revision can beat global revision on real data
- the main revision story holds on a realistic event-conditioned time-series task

What `XTraffic v2 narrowed` is not supposed to prove:

- free-form long-text understanding
- domain-agnostic operator semantics

Reference:

- [XTraffic v2 narrowed milestone](/root/autodl-tmp/BetterTSE-main/results/forecast_revision/milestones/20260317_xtraffic_v2_narrowed/README.md)

### 3. `XTraffic v2 nonapp`

Role:

- real no-op gate checkpoint

Purpose:

- validate `revision_needed`
- validate abstention
- validate no-op preservation on real windows

What `XTraffic v2 nonapp` is allowed to prove:

- real non-applicable windows can be left untouched
- over-edit behavior is controllable

Reference:

- [XTraffic v2 nonapp milestone](/root/autodl-tmp/BetterTSE-main/results/forecast_revision/milestones/20260317_xtraffic_v2_nonapp/README.md)

### 4. `MTBench finance v2`

Role:

- real native-text complement
- realism / transfer complement

Purpose:

- validate transfer to native text + time-series
- validate that domain-specific schema adaptation is enough to make the framework work

Dataset interpretation:

- real financial time series
- real financial text
- finance-specific revision schema:
  - `repricing`
  - `drift_adjust`
  - `neutral`

What `MTBench` is allowed to prove:

- the framework is not limited to structured incident context
- native text works when revision semantics match the domain

What `MTBench` is not supposed to prove:

- the main benchmark story by itself
- fine-grained localized weak-label supervision quality comparable to `XTraffic`

Reference:

- [MTBench finance v2 milestone](/root/autodl-tmp/BetterTSE-main/results/forecast_revision/milestones/20260317_mtbench_v2_finance/README.md)
- [MTBench finance v2 100-sample milestone](/root/autodl-tmp/BetterTSE-main/results/forecast_revision/milestones/20260317_mtbench_v2_100/README.md)

### 5. `CiK`-style protocol

Role:

- benchmark-design reference only

Purpose:

- borrow field organization
- borrow task phrasing and context-centric forecasting philosophy
- borrow evaluation presentation discipline

What `CiK` is allowed to influence:

- schema design
- benchmark writing style
- evaluation grouping

What `CiK` is not supposed to do in the current line:

- replace `XTraffic` as the main empirical benchmark
- expand the current data scope during stabilization

## Fixed Method Versions

All datasets should be evaluated with the same method ladder where applicable:

1. `base_only`
2. `global_revision_only`
3. `localized_full_revision`
4. `oracle_region`
5. `oracle_intent`
6. `oracle_calibration`

Rules:

- `base_only` is always the no-revision baseline.
- `global_revision_only` is the non-localized revision baseline.
- `localized_full_revision` is the main method.
- oracle variants are used for error attribution, not for claiming deployable performance.

## Fixed Baselines

Current baseline ladder:

1. `naive_last`
2. `dlinear_like`

`patchtst` remains optional and should not block the framework.

Reason:

- the current framework question is about revision validity and transfer
- not about squeezing the strongest forecaster into the stack

## Calibration Benchmark Status

The framework now has an explicit calibration scaffold rather than treating `how much` as an implicit planner strength field.

Current code-level status:

- explicit dimensionless `edit_spec` prediction is implemented
- `edit_spec` is projected to executor parameters in a separate step
- synthetic benchmarks can carry `edit_spec_gt`
- a dedicated calibration benchmark runner now compares:
  - `text_direct_numeric`
  - `discrete_strength_table`
  - `rule_local_stats`
  - `oracle_calibration`

Immediate intended use:

- first run calibration under `oracle_region + oracle_intent`
- then reuse the same calibration metrics inside the full pipeline

## Fixed Metric Groups

### A. Forecast Quality

- `MAE`
- `MSE`
- `sMAPE`
- `Revision Gain`

Interpretation:

- these answer whether revision improves the forecast

### B. Editing Quality

- `future_t_iou`
- `direction accuracy`
- `shape accuracy`
- `duration error`
- `magnitude calibration error`

Interpretation:

- these answer whether the edit itself matches the intended correction

### C. Preservation / Gate

- `outside-region preservation`
- `over_edit_rate`
- `revision_needed_match`

Interpretation:

- these answer whether the method avoids unnecessary changes

## Fixed Reading Order for Results

Do not read all metrics at once.
Use this order:

1. `Revision Gain`
2. `localized_full_revision` vs `global_revision_only`
3. oracle gaps
4. `revision_needed_match` on non-applicable samples
5. `future_t_iou`
6. `magnitude_calibration_error`

This order prevents overreacting to secondary metrics.

## Experiment Matrix

### Controlled

- dataset: `Weather v4`
- baselines:
  - `naive_last`
  - `dlinear_like`
- modes:
  - full ladder

Primary question:

- does the task hold under clean supervision?

### Real Structured Event

- dataset: `XTraffic v2 narrowed`
- baseline:
  - `dlinear_like`
- modes:
  - `base_only`
  - `global_revision_only`
  - `localized_full_revision`
  - oracle ladder

Primary question:

- does localized revision transfer to real event-driven forecasting?

### Real Gate

- dataset: `XTraffic v2 nonapp`
- baseline:
  - `dlinear_like`
- modes:
  - `base_only`
  - `global_revision_only`
  - `localized_full_revision`
  - `oracle_region`

Primary question:

- can the gate abstain on real no-op windows?

### Real Native Text

- dataset: `MTBench finance v2`
- baselines:
  - `naive_last`
  - `dlinear_like`
- modes:
  - full ladder

Primary question:

- does the framework transfer to native text + time-series once schema matches the domain?

## Fixed Evidence Chain

The framework should be argued in this order:

1. `Weather v4`
   - controlled validity
2. `XTraffic v2 narrowed`
   - real structured-event transfer
3. `XTraffic v2 nonapp`
   - real abstention / no-op gate
4. `MTBench finance v2`
   - native-text transfer
5. `MTBench finance v2 (100)`
   - native-text stability

Do not reorder this chain unless a stronger real checkpoint is added.

## What Not To Do

At the current stage, do not:

- redesign the main framework again
- open new dataset lines immediately
- expand operator families aggressively
- make `PatchTST` a blocker
- mix dataset roles together
- promote every candidate benchmark into the core evidence chain

The current line is already broad enough.

## Immediate Next Steps

The next work should stay inside this framework:

1. unified case studies and visual summaries
2. small reproducibility reruns on current best checkpoints
3. freeze dataset roles around:
   - `XTraffic` as main benchmark
   - `MTBench` as realism complement
   - `CiK` as protocol template
4. only then consider a new dataset line

If a new dataset line is opened later, the tentative order is:

1. `Time-MMD`
2. `Time-IMM`
3. `CGTSF`

## Canonical References

- [Breakpoint summary](/root/autodl-tmp/BetterTSE-main/results/forecast_revision/BREAKPOINT_SUMMARY_20260317.md)
- [Experiment summary](/root/autodl-tmp/BetterTSE-main/results/forecast_revision/EXPERIMENT_SUMMARY_20260317.md)
- [Unified case studies](/root/autodl-tmp/BetterTSE-main/results/forecast_revision/case_studies/20260317_unified/CASE_STUDIES_20260317.md)
