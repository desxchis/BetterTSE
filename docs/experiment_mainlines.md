# BetterTSE Experiment Mainlines

## Purpose

This repository should be reported and executed as two connected experiment lines, not one replacement line.

## Mainline A: Pure Time-Series Editing

- goal: evaluate BetterTSE as a time-series editing method
- input: original series plus vague or complex natural-language instruction
- output: edited series
- primary entrypoint: `run_pipeline.py`
- typical evaluation focus:
  - localization accuracy
  - intent alignment
  - editability / target matching
  - outside-region preservation

This line is the method core. It should carry the main claim that BetterTSE can convert ambiguous instructions into concrete executable edits.

## Mainline B: Forecast Revision Application

- goal: evaluate BetterTSE editing capability in a forecasting scenario
- input: history window, base forecast from a forecasting backbone, contextual instruction
- output: revised forecast
- primary entrypoints:
  - baseline preparation: `test_scripts/train_forecast_baseline.py`
  - benchmark build: `test_scripts/build_forecast_revision_benchmark.py`
  - revision run: `run_forecast_revision.py`
  - multi-backbone orchestration: `test_scripts/run_multibackbone_forecast_revision.py`
- typical evaluation focus:
  - gain over `base_forecast`
  - quality against `revision_target`
  - quality against `future_gt`
  - robustness across multiple forecast backbones

This line should be presented as a downstream application of BetterTSE rather than a replacement for the editing task.

## Forecast Backbone Policy

The forecast-revision line should treat forecasting models as **replaceable base predictors**, not as a fixed bundled set.

Current repository support includes examples such as:

- `patchtst`
- `dlinear_official`
- `lstm_official`

These are implementation examples and convenient starting points, not a locked paper backbone set. The expected direction is to progressively reproduce and plug in newer forecasting models for the `base_forecast` stage while keeping the BetterTSE revision interface stable.

`dlinear_like` can still be used for smoke tests or low-cost debugging, but it should not be the only model used in substantive claims.

## Target Construction Scope

Current implemented forecast-revision benchmarks primarily use controlled synthetic target construction rooted in the base forecast:

- physical / rule-based revision injection
- future-aware projection in the corresponding benchmark utilities when needed

Monte Carlo style target simulation is not yet the repository mainline and should be treated as future extension work unless a dedicated implementation lands.

## Recommended Reporting Structure

For papers or reports, use two subsections:

1. `Pure Time-Series Editing`
2. `Forecast Revision as Downstream Application`

Do not describe the project as having shifted away from editing. The forecast-revision line exists to show that the editing method transfers to another application setting.
