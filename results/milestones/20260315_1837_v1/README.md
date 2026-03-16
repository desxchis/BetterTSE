# 20260315_1837_v1

## Snapshot

- 时间: `2026-03-15 18:37 UTC`
- 数据集: `Traffic`
- 样本数: `12`
- 当前最佳结果文件: `pipeline_results_step_focus_v3.json`
- 对应测试集: `event_driven_testset_Traffic_12.json`
- 对应可视化: `visualizations/`

## 本节点改动

- 文件: `modules/region_localizer.py`
- 改动 1: `step + onset` 的候选窗口打分改为边界优先，不再主要依赖窗口中心。
- 改动 2: 将 `预计在...` 这类时间表达纳入 onset 判断，覆盖 `预计在今晚深夜` 这类晚段切换事件。
- 改动 3: 保留前一阶段的 `step_shift` 执行器与 `step` 形态映射，使定位改进可以直接传导到编辑执行。

## 改善指标

对照基线文件:
- 基线: `results/experiments/localization_diagnostics/traffic_12sample_v1/pipeline_results_tedit_v2_midfix.json`
- 当前: `pipeline_results_step_focus_v3.json`

整体指标:
- `avg t-IoU`: `0.3627 -> 0.4186` `(+0.0558)`
- `avg Editability`: `0.5938 -> 0.5794` `(-0.0144)`
- `avg Preservability`: `0.9840 -> 0.9850` `(+0.0010)`
- `avg Direction Match`: `1.0000 -> 0.9167`

关键任务:
- `regime_switch` 平均 `t-IoU`: `0.2345 -> 0.7000`

关键样本:
- `sample 003`: `GT [127,138]`, 预测 ` [138,162] -> [121,140]`, `t-IoU 0.0278 -> 0.6000`
- `sample 009`: `GT [49,64]`, 预测 ` [31,63] -> [45,64]`, `t-IoU 0.4412 -> 0.8000`

## 当前判断

- `step/regime_switch` 已从主要病灶变成可用状态。
- 本轮提升主要来自定位层，不是工具层。
- `transient_hump`、`signal_corruption`、`shutdown_flatline` 基本未被这轮修改拖坏。
- `sustained_gain` 仍然是后续主攻方向，尤其是 `plateau/elevated level` 的稳定定位与执行。

## 日志留存情况

- 该里程碑目录下未保留独立的运行日志文件；检索范围内没有找到对应 `20260315_1837_v1` 的 `.log`。
- 当前仍可回溯的内容是结果产物本身，以及源实验目录中的文件时间线。
- 里程碑快照来源目录: `results/experiments/localization_diagnostics/traffic_12sample_v1/`
- 可确认的关键时间点:
  - `event_driven_testset_Traffic_12.json`: `2026-03-15 17:48 UTC`
  - `pipeline_results_tedit_v2_midfix.json` (README 中基线): `2026-03-15 17:57 UTC`
  - `pipeline_results_step_focus.json`: `2026-03-15 18:27 UTC`
  - `pipeline_results_step_focus_v2.json`: `2026-03-15 18:32 UTC`
  - `pipeline_results_step_focus_v3.json`: `2026-03-15 18:37 UTC`
- 与本次最佳结果匹配的可视化文件名前缀为 `20260315_103357_*`，说明这组图像来自同一批次导出。
- 结论: 原始 stdout/stderr 级别的 run log 基本没有单独存档，但该 milestone 的结果链路和主要时间线仍可从现有产物恢复。

## 后续方向

- 优先继续做 `sustained_gain / plateau` 的定位与评测映射。
- 单独检查 `Direction Match` 从 `1.0` 降到 `0.9167` 的那一个样本，确认是 prompt 歧义还是 planner 回退。
- 如果后续继续扩展 benchmark，优先沿现有跨数据集 schema 推进，不再回到 `device_switch` 这类数据集特定标签。
