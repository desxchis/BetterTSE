# BetterTSE Target Construction Policy

## Overview

BetterTSE now uses different primary target-construction regimes for its two task lines.

1. Pure Editing
- primary regime: `controlled_physical_injection`
- rationale: this task is about editing capability itself, so the target should be maximally interpretable and controllable

2. Forecast Revision
- primary regime: `future_guided_projected_revision`
- rationale: this task naturally has access to both `base_forecast` and `future_gt`, so the target should be constructed as an intent-constrained revision that moves the forecast toward the real future

3. Forecast Revision Auxiliary Benchmark
- auxiliary regime: `controlled_synthetic_revision`
- rationale: this regime remains useful for calibration studies, ablations, and checks that the revision follows textual intent rather than merely tracking `future_gt`

## Pure Editing Policy

Pure editing should continue to use controlled physical target construction as the main benchmark setting.

Why:
- operator family is explicit
- shape, duration, and strength are controllable
- target quality is clean enough for localization and preservation analysis
- the benchmark measures editing ability rather than downstream forecasting value

## Forecast Revision Policy

Forecast revision should primarily use future-guided projected targets.

Definition:
- begin with `base_forecast`
- parse textual intent into a constrained edit family
- search only inside that constrained family
- use `future_gt` to identify the best projected revision target within that family

This is intentionally not unconstrained fitting to `future_gt`.

The textual instruction still defines:
- editable region family
- effect family
- qualitative shape
- qualitative strength and duration priors

`future_gt` only selects the best target inside that constrained revision space.

## Metadata Conventions

Benchmark payloads should expose the target regime explicitly.

Recommended values:
- `target_regime = controlled_physical_injection`
- `target_regime = controlled_synthetic_revision`
- `target_regime = future_guided_projected_revision`

For forecast revision benchmarks, keep these top-level fields stable:
- `task_family = forecast_revision`
- `application_of = bettertse_editing`
- `target_construction_method`
- `target_regime`
- `baseline_name`

## Reporting Guidance

When writing papers or reports:
- treat pure editing as the method core
- treat forecast revision as a downstream application
- report controlled synthetic revision and future-guided projected revision as two forecast-revision target regimes with different purposes

Suggested wording:
- controlled synthetic revision is used for calibration-oriented and controlled analysis
- future-guided projected revision is used for application-oriented downstream evaluation against real future values
