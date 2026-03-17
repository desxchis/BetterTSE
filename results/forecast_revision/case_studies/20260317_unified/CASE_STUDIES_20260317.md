# Unified Forecast Revision Case Studies

This file consolidates one controlled case and four real-data cases across the current stable checkpoint line.

## Weather Controlled Positive Case

- source: Weather v4 (`dlinear_like`)
- sample_id: `014`
- figure: [weather_controlled_014.png](results/forecast_revision/case_studies/20260317_unified/visualizations/weather_controlled_014.png)
- applicable: `True`
- GT intent: `step` / `down` / `strong`
- predicted intent: `step` / `down` / `strong`
- region: pred `[16, 24]` vs gt `[18, 24]`
- revision_gain: `0.4429`
- future_t_iou: `0.7500`
- magnitude_error: `0.0000`
- over_edit_rate: `0.1111`
- revision_needed_match: `1.0000`

Context:

> 在预测窗口后段，系统状态预计会突然切换到更低位并维持一段时间。

Notes:

- source metadata: `{'effect_family': 'level', 'direction': 'down', 'shape': 'step', 'duration': 'medium', 'strength': 'strong'}`

## XTraffic Real Positive Case

- source: XTraffic v2 nonapp (`dlinear_like`, applicable)
- sample_id: `014`
- figure: [xtraffic_positive_014.png](results/forecast_revision/case_studies/20260317_unified/visualizations/xtraffic_positive_014.png)
- applicable: `True`
- GT intent: `step` / `down` / `weak`
- predicted intent: `step` / `down` / `strong`
- region: pred `[0, 24]` vs gt `[0, 12]`
- revision_gain: `1.2129`
- future_t_iou: `0.5000`
- magnitude_error: `1.7300`
- over_edit_rate: `0.0909`
- revision_needed_match: `1.0000`

Context:

> 在预测窗口前段，North Sac附近发生NoInj事件（20002-Hit and Run No Injuries，位置：El Camino Ave E Onr / Sr51 N），相关流量预计会切换到更低位，并在短时间内维持较低水平。

Notes:

- source metadata: `{'effect_family': 'level', 'direction': 'down', 'shape': 'step', 'duration': 'short', 'strength': 'weak', 'label_source': 'weak_real_incident'}`

## XTraffic Real No-Op Case

- source: XTraffic v2 nonapp (`dlinear_like`, non-applicable)
- sample_id: `NA_001`
- figure: [xtraffic_noop_NA_001.png](results/forecast_revision/case_studies/20260317_unified/visualizations/xtraffic_noop_NA_001.png)
- applicable: `False`
- GT intent: `none` / `neutral` / `none`
- predicted intent: `none` / `neutral` / `none`
- region: pred `[0, 0]` vs gt `[0, 0]`
- revision_gain: `0.0000`
- future_t_iou: `0.0000`
- magnitude_error: `0.0000`
- over_edit_rate: `0.0000`
- revision_needed_match: `1.0000`

Context:

> 在预测窗口前段，San Gorgonio Pass附近无新增影响，暂无额外冲击，维持原预测，没有新的修正信号。

Notes:

- source metadata: `{'effect_family': 'none', 'direction': 'neutral', 'shape': 'none', 'duration': 'none', 'strength': 'none', 'label_source': 'real_no_incident_window'}`

## MTBench Native-Text Repricing Case

- source: MTBench finance v2 100 (`dlinear_like`, repricing)
- sample_id: `047`
- figure: [mtbench_repricing_047.png](results/forecast_revision/case_studies/20260317_unified/visualizations/mtbench_repricing_047.png)
- applicable: `True`
- GT intent: `step` / `down` / `strong`
- predicted intent: `step` / `down` / `medium`
- region: pred `[0, 78]` vs gt `[0, 78]`
- revision_gain: `11.0031`
- future_t_iou: `1.0000`
- magnitude_error: `2.7062`
- over_edit_rate: `0.0000`
- revision_needed_match: `1.0000`

Context:

> Why Wex Stock Crashed and Burned Today 市场可能对这条消息进行向下重新定价，带动整个预测窗口的价格路径下修。 文本标签参考：sentiment=['[3][b] Risk & Warning']; type=['[1][c] Company-Specific News', '[1][b] Stock Market Updates']; time=['[1][a] Short-Term Retrospective (≤ 3 months)', '[2][b] Recent Trends (Past Few Weeks – Ongoing)'].

Notes:

- source metadata: `{'effect_family': 'level', 'direction': 'down', 'shape': 'step', 'duration': 'long', 'strength': 'strong', 'label_source': 'mtbench_trend_heuristic'}`

## MTBench Native-Text Drift-Adjust Case

- source: MTBench finance v2 100 (`dlinear_like`, drift_adjust)
- sample_id: `059`
- figure: [mtbench_drift_059.png](results/forecast_revision/case_studies/20260317_unified/visualizations/mtbench_drift_059.png)
- applicable: `True`
- GT intent: `plateau` / `up` / `weak`
- predicted intent: `step` / `up` / `medium`
- region: pred `[26, 43]` vs gt `[0, 69]`
- revision_gain: `2.4624`
- future_t_iou: `0.2464`
- magnitude_error: `12.5822`
- over_edit_rate: `0.0000`
- revision_needed_match: `1.0000`

Context:

> 3 Things About Adobe That Smart Investors Know 这条消息更像中短期偏多驱动，未来价格路径可能整体缓慢上修。 文本标签参考：sentiment=['[1][a] Bullish']; type=['[2][a] Fundamental Analysis', '[1][c] Company-Specific News', '[1][b] Stock Market Updates']; time=['[3][a] Short-Term Outlook (Next 3–6 months)', '[1][c] Long-Term Retrospective (> 1 year)', '[2][b] Recent Trends (Past Few Weeks – Ongoing)'].

Notes:

- source metadata: `{'effect_family': 'level', 'direction': 'up', 'shape': 'plateau', 'duration': 'long', 'strength': 'weak', 'label_source': 'mtbench_trend_heuristic'}`
