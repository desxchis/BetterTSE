# XTraffic v2 Narrowed Case Studies

## Setup

- benchmark:
  - `results/forecast_revision/benchmarks/xtraffic_p01_speed_dlinear_v2/forecast_revision_XTraffic_dlinear_like_9.json`
- localized run with visualizations:
  - `results/forecast_revision/runs/xtraffic_p01_speed_dlinear_v2_casevis/pipeline_results_localized_full_revision.json`
- oracle intent reference:
  - `results/forecast_revision/runs/xtraffic_p01_speed_dlinear_v2_casevis/pipeline_results_oracle_intent.json`

## Overall

- `localized_full_revision avg_revision_gain = 0.4065`
- `oracle_intent avg_revision_gain = 0.2539`
- `localized_full_revision avg_future_t_iou = 0.5044`

The narrowed real subset is now positive, but gains are not uniform across cases.

## Positive Cases

### Sample 015

- figure:
  - [localized](/root/autodl-tmp/BetterTSE-main/results/forecast_revision/runs/xtraffic_p01_speed_dlinear_v2_casevis/visualizations/20260317_011026_015_forecast_step.png)
- incident:
  - `1141`
  - `1179-Trfc Collision-1141 Enrt`
- metrics:
  - `revision_gain = 1.4065`
  - `t-IoU = 0.8727`
  - `magnitude_error = 5.0188`
- interpretation:
  - Region localization is close to GT.
  - The event semantics are strong enough that the generic traffic-drop operator helps.

### Sample 014

- figure:
  - [localized](/root/autodl-tmp/BetterTSE-main/results/forecast_revision/runs/xtraffic_p01_speed_dlinear_v2_casevis/visualizations/20260317_011026_014_forecast_step.png)
- incident:
  - `NoInj`
  - `20002-Hit and Run No Injuries`
- metrics:
  - `revision_gain = 1.2129`
  - `t-IoU = 0.5000`
  - `magnitude_error = 1.7300`
- interpretation:
  - Even with imperfect duration, a small early-window downgrade is enough to improve the forecast.

### Sample 018

- figure:
  - [localized](/root/autodl-tmp/BetterTSE-main/results/forecast_revision/runs/xtraffic_p01_speed_dlinear_v2_casevis/visualizations/20260317_011026_018_forecast_step.png)
- incident:
  - `Hazard`
  - `1125-Traffic Hazard`
- metrics:
  - `revision_gain = 1.0188`
  - `t-IoU = 0.2500`
  - `magnitude_error = 3.2054`
- interpretation:
  - Localization is loose, but the direction is correct and the event still benefits from a localized early slowdown.

## Failure Cases

### Sample 016

- figure:
  - [localized](/root/autodl-tmp/BetterTSE-main/results/forecast_revision/runs/xtraffic_p01_speed_dlinear_v2_casevis/visualizations/20260317_011026_016_forecast_step.png)
- incident:
  - `NoInj`
  - `1182-Trfc Collision-No Inj`
- metrics:
  - `revision_gain = -0.4270`
  - `t-IoU = 0.3333`
  - `magnitude_error = 4.5041`
- interpretation:
  - The operator direction is correct.
  - The main issue is over-editing relative to a short GT impact window.

### Sample 020

- figure:
  - [localized](/root/autodl-tmp/BetterTSE-main/results/forecast_revision/runs/xtraffic_p01_speed_dlinear_v2_casevis/visualizations/20260317_011026_020_forecast_step.png)
- incident:
  - `NoInj`
  - `1182-Trfc Collision-No Inj`
- metrics:
  - `revision_gain = -0.4270`
  - `t-IoU = 0.7500`
  - `magnitude_error = 4.4021`
- interpretation:
  - Region overlap is acceptable.
  - The failure is mostly calibration scale rather than localization.

### Sample 005

- figure:
  - [localized](/root/autodl-tmp/BetterTSE-main/results/forecast_revision/runs/xtraffic_p01_speed_dlinear_v2_casevis/visualizations/20260317_011026_005_forecast_step.png)
- incident:
  - `AHazard`
  - `ANIMAL-Live or Dead Animal`
- metrics:
  - `revision_gain = -0.0891`
  - `t-IoU = 0.2500`
  - `magnitude_error = 2.6259`
- interpretation:
  - This is a weaker event type.
  - The current drop-style operator appears too coarse for animal-hazard cases.

## Takeaway

- Positive transfer on real data is now visible.
- Best cases come from:
  - stronger collision / hazard events
  - clear early-window disruption
  - speed channel instead of flow
- Remaining failure modes are mostly:
  - duration overshoot
  - magnitude overshoot
  - weak event semantics not matching the same operator family
