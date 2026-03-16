# BetterTSE 工具分层与 Prompt 重构说明

## 改动目标

这次改动围绕两个核心问题展开：

1. 当前编辑工具命名和 TEdit 原生控制空间之间的关系不够清晰。
2. 当前 event-driven prompt 让 LLM 直接从事件跳到具体工具，容易把事件理解、时序形态判断、执行器选择混在一起。

本次调整的目标是：

- 将编辑工具整理成更接近 TEdit 的 canonical task 空间
- 给工具增加分层与控制来源元信息
- 将 prompt 从 `tool-first` 改为 `intent-first -> tool-second`
- 保持现有 `run_pipeline.py` 和验证脚本的兼容性

## 具体代码改动

### 1. `tool/ts_editors.py`

新增了工具注册与兼容层：

- `EDIT_TOOL_SPECS`
  - 为每个现有工具声明：
    - `canonical_tool`
    - `control_source`
    - `tool_layer`
    - `effect_family`
    - `direction`
    - `shape`
    - `duration`
    - `strength`
    - `description`

- `CANONICAL_TOOL_TO_TOOL_NAME`
  - 负责 canonical task 到现有执行工具名的映射

- `get_edit_tool_specs()`
  - 返回完整工具注册表，供后续评估或文档使用

- `get_prompt_tool_catalog()`
  - 返回适合给 LLM prompt 使用的简化 catalog

- `normalize_llm_plan()`
  - 支持两种 plan 格式：
    - 旧格式：`tool_name + parameters`
    - 新格式：`intent + localization + execution`
  - 自动补全：
    - `tool_name`
    - `canonical_tool`
    - `parameters.region`
    - `tool_layer`
    - `control_source`
    - `intent`

同时，`execute_llm_tool()` 已改为先走 `normalize_llm_plan()`，因此旧脚本无需联动修改即可继续运行。

### 2. `agent/prompts.py`

重写了 `get_event_driven_agent_prompt()` 的设计思路：

- 从旧版的“九个工具直接选一”改成“两阶段、单次输出”
- 第一阶段要求 LLM 输出 canonical intent：
  - `effect_family`
  - `direction`
  - `shape`
  - `duration`
  - `strength`
- 第二阶段再把 intent 映射到执行工具

Prompt 中新增了：

- layered tool catalog 注入
- canonical mapping guide
- 明确的反歧义规则
  - transient 优先判为 `impulse`
  - 均值不变、波动变化优先判为 `volatility`
  - 周期变化优先判为 `seasonality`
  - 只有持续漂移才判为 `trend`

新的输出 schema 为：

```json
{
  "thought": "...",
  "intent": {...},
  "localization": {...},
  "execution": {...}
}
```

同时保留对现有执行层的兼容要求：

- `execution.tool_name` 仍必须是当前系统能直接执行的工具名

### 3. `modules/llm.py`

更新了 `get_event_driven_plan()`：

- 给 user message 补充了 effect family 和 region hint
- 对 LLM 返回 JSON 统一调用 `normalize_llm_plan()`

结果是：

- 上层开始可以拿到新的 `intent/localization/execution`
- 下层继续可以直接用 `tool_name/parameters`

## 当前工具分层

### Native TEdit 对齐任务

这些任务优先代表 TEdit 原生可表达的控制空间：

- `trend_linear_up` -> `hybrid_up`
- `trend_linear_down` -> `hybrid_down`
- `trend_quadratic_up` -> `trend_quadratic_up`
- `trend_quadratic_down` -> `trend_quadratic_down`
- `seasonality_enhance` -> `season_enhance`
- `seasonality_reduce` -> `season_reduce`
- `smooth_denoise` -> `ensemble_smooth`

### Derived 任务

这些任务仍然保留，但被标记为扩展层：

- `volatility_increase` -> `volatility_increase`
- `impulse_spike` -> `spike_inject`

## 为什么这样改

### 改之前的问题

- LLM 直接对着工具名选，容易被描述相似性误导
- 工具名和 TEdit attr 空间没有形成稳定中间层
- 后续如果替换执行器，prompt 和评测都要一起大改

### 改之后的收益

- 事件理解和工具执行被拆成两个层次
- 更方便做 intent accuracy、tool accuracy、region accuracy 的拆分评估
- 更容易把测试集注入器后续也对齐到 canonical task schema
- 更清楚地区分：
  - TEdit 原生能力
  - hybrid 能力
  - pure math 能力

## 兼容性说明

本次改动没有要求立刻修改现有 pipeline 调用方。

原因：

- `get_event_driven_plan()` 已统一输出兼容格式
- `execute_llm_tool()` 已支持新旧 plan 自动归一化

所以现有这些脚本理论上可以继续工作：

- `run_pipeline.py`
- `run_5sample_validation.py`
- `run_20sample_analysis.py`
- `run_traffic_validation.py`

## 建议的后续工作

下一阶段建议继续做三件事：

1. 把测试集注入器也切到 canonical task schema
2. 在评估中新增：
   - `intent_accuracy`
   - `canonical_tool_accuracy`
   - `control_source_breakdown`
3. 为 prompt 增加少量高质量 few-shot examples
   - impulse vs trend
   - volatility vs seasonality
   - seasonality enhance vs reduce

## 一句话总结

这次改动不是简单“换一套 prompt 文案”，而是把 BetterTSE 的事件理解层、任务语义层、工具执行层明确拆开，并让工具空间开始向 TEdit 原生控制空间收敛。
