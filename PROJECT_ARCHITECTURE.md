# BetterTSE 项目架构文档

> 更新时间：2026-03-12

---

## 一、项目概述

**BetterTSE** (Better Time Series Editing) 是一个结合**扩散模型**与**大语言模型（LLM）**的时间序列智能编辑框架，核心目标是：

> 给定一段真实时间序列（Base TS）和一条模糊的事件性自然语言指令（Vague Prompt），让系统自动理解编辑意图、定位时间区间、调用编辑工具，产出符合预期的 Target TS，并以多维指标量化评估效果。

### 核心技术

| 组件 | 技术方案 |
|------|---------|
| 时间序列编辑模型 | TEdit — NeurIPS 2024 扩散模型，Training-free 软边界潜空间融合 |
| LLM 意图解析 | DeepSeek Chat（OpenAI 兼容接口），支持 vLLM 本地部署 |
| Agent 工作流 | LangGraph `StateGraph`，4 节点有向图 |
| 测试集范式 | CiK (Context is Key) 物理注入 + LLM 反向生成模糊 Prompt |
| 评估指标 | t-IoU · Editability MSE/MAE · Preservability MSE/MAE · Feature Accuracy |

### 潜空间融合公式（软边界编辑核心）

```
z_{t-1} = M ⊙ z_{t-1}^{pred} + (1 - M) ⊙ z_{t-1}^{GT}
```

M 为高斯平滑软掩码，消除编辑边界的"悬崖效应"。

---

## 二、目录结构

```
BetterTSE-main/
│
├── config.py                       # 全局配置（API Key、模型路径）
├── run_pipeline.py                 # 端到端 Pipeline 入口脚本
├── requirements.txt                # Python 依赖
├── .env / .env.example             # 环境变量（API Key 等）
│
├── agent/                          # LangGraph Agent 模块
│   ├── agent.py                    #   A1 类：主 Agent + AgentState 定义
│   ├── nodes.py                    #   4 个工作流节点函数
│   ├── prompts.py                  #   Prompt 模板（含 EVENT_DRIVEN_AGENT_PROMPT）
│   ├── llm_instruction_decomposer.py  # LLM 编辑意图结构化分解器
│   └── instruction_decomposer.py   #   规则型分解器（备用）
│
├── modules/                        # 通用工具模块
│   ├── llm.py                      #   CustomLLMClient + get_event_driven_plan()
│   └── utils.py                    #   工具函数（绘图、解析、格式化）
│
├── tool/                           # 时间序列编辑工具
│   ├── tedit_wrapper.py            #   TEditWrapper：TEdit 模型封装 + 单例
│   ├── ts_editors.py               #   execute_llm_tool() + 3 种编辑工具实现
│   ├── ts_composers.py             #   合成工具
│   ├── ts_describers.py            #   描述工具
│   ├── ts_processor.py             #   预处理工具
│   ├── ts_synthesizer.py           #   合成器
│   ├── region_selector.py          #   时间区间选择器
│   └── tool_description/           #   工具文档字符串（供 LLM Prompt 使用）
│       ├── ts_editors.py
│       ├── ts_composers.py
│       ├── ts_describers.py
│       ├── tedit_tools.py
│       └── ts_processor.py
│
├── test_scripts/                   # 测试集构建与评估
│   ├── build_event_driven_testset.py  # ★ 主用：事件驱动测试集生成器
│   ├── bettertse_cik_official.py      #   CiK 范式完整 Pipeline + TSEditEvaluator
│   ├── result_evaluator.py            #   结果评估器（ResultEvaluator）
│   ├── build_mini_benchmark.py        #   小型基准集构建（旧版）
│   ├── data_loader.py                 #   数据加载模块
│   ├── config.py                      #   test_scripts 内部枚举（ChangeType 等）
│   ├── change_injector.py             #   物理注入器集合
│   ├── llm_interface.py               #   LLM 接口封装
│   ├── test_pipeline.py               #   测试 Pipeline
│   ├── main.py                        #   入口脚本
│   ├── data/ETTh1.csv                 #   本地测试数据
│   └── test_results/                  #   评估结果输出
│
├── event_driven_testset_ultimate/  # ★ 当前正式测试集（保留）
│   ├── event_driven_testset_ETTh1_5.json   # 5 条样本
│   ├── event_driven_testset_ETTh1_5.csv    # 数据摘要
│   └── event_driven_report.txt             # 生成报告
│
├── data/
│   └── ETTh1.csv                   # 原始时间序列数据
│
├── examples/                       # 保留中的示例与诊断脚本
│   ├── llm_tedit_pipeline_v9_full_validation.py  # 最新完整验证版
│   ├── llm_tedit_pipeline_v8_soft_boundary.py    # 软边界版
│   └── ...                         # 其它保留中的分析/诊断脚本
│
├── TEdit-main/                     # TEdit 核心模型代码（子项目）
│   ├── models/
│   │   ├── conditional_generator.py        # ConditionalGenerator（核心入口）
│   │   ├── diffusion/diff_csdi_multipatch_weaver.py  # 多Patch扩散模型
│   │   └── encoders/ · energy/             # 编码器 + 能量模型
│   ├── samplers.py                         # DDPM / DDIM 采样器
│   ├── save/                               # 预训练权重
│   │   └── synthetic/pretrain_multi_weaver/0/ckpts/model_best.pth
│   └── configs/                            # 数据集配置（synthetic/air/motor）
│
├── PROJECT_ARCHITECTURE.md         # 本文档
├── SKILL.md                        # 技术规范
├── TEDIT_CONFIG.md                 # TEdit 配置详解
└── coding plan.md                  # 开发计划
```

---

## 三、全流程 Pipeline

### 3.1 流程总览

```
┌─────────────────────────────────────────────────────────┐
│  Phase 0: 测试集构建（离线，一次性）                        │
│  build_event_driven_testset.py                          │
│    CSV数据 → 物理注入(CiK) → LLM反向生成Prompt → JSON     │
└────────────────────────┬────────────────────────────────┘
                         │ event_driven_testset_ultimate/*.json
                         ▼
┌─────────────────────────────────────────────────────────┐
│  Phase 1: LLM 意图解析                                    │
│  modules/llm.py :: get_event_driven_plan()              │
│    Vague Prompt → { tool_name, region:[s,e], math_shift }│
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│  Phase 2: 时间序列编辑                                    │
│  tool/ts_editors.py :: execute_llm_tool()               │
│    hybrid_up_soft / hybrid_down_soft / ensemble_smooth  │
│    → TEditWrapper.edit_region_soft() + 数学锚定          │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│  Phase 3: 指标评估                                        │
│  test_scripts/bettertse_cik_official.py :: TSEditEvaluator│
│    t-IoU · Editability · Preservability · Feature Acc   │
└─────────────────────────────────────────────────────────┘
                         │
                         ▼
             run_pipeline.py 汇总结果 → JSON 报告
```

### 3.2 测试集构建（`build_event_driven_testset.py`）

```
ETTh1.csv
    │
    ▼  CSVDataLoader.get_sequence()
真实1D时间序列 (Base TS, seq_len=192)
    │
    ▼  随机选取注入器（五选一）+ 随机区间
┌──────────────────────────────────────────────┐
│ MultiplierInjector   乘法放大 ×2.0~4.0        │
│ HardZeroInjector     强制归零                 │
│ NoiseInjector        底噪替换                 │
│ TrendInjector        上升/下降趋势             │
│ StepChangeInjector   阶跃跳变                 │
└──────────────────────────────────────────────┘
    │
    ▼  inject() → (base_ts, target_ts, gt_mask, gt_config)
    │
    ▼  DeepSeek LLM 反向生成 3 级模糊 Prompt
┌──────────────────────────────────────────────────┐
│ Level 1 (低模糊)  调度员视角：直接业务指令          │
│ Level 2 (中模糊)  新闻主播视角：宏观事件描述        │
│ Level 3 (高模糊)  社交媒体视角：无关联线索          │
└──────────────────────────────────────────────────┘
    │
    ▼  EventDrivenSample 序列化 → JSON
```

**EventDrivenSample 数据结构**：

```json
{
  "sample_id": "001",
  "target_feature": "OT",
  "task_type": "market_trend",
  "gt_start": 38,
  "gt_end": 58,
  "event_prompts": [
    { "level": 1, "perspective": "调度员", "prompt": "..." },
    { "level": 2, "perspective": "新闻主播", "prompt": "..." },
    { "level": 3, "perspective": "社交媒体路人", "prompt": "..." }
  ],
  "base_ts": [...],
  "target_ts": [...],
  "gt_mask": [...],
  "gt_config": { "injection_type": "trend_injection", "multiplier": null, "trend_slope": 0.03 }
}
```

---

## 四、核心模块详解

### 4.1 全局配置（`config.py`）

| 函数 | 说明 |
|------|------|
| `get_api_config()` | 返回 `{api_key, base_url, model_name}`，从环境变量读取 |
| `get_model_config(dataset)` | 返回 TEdit 模型权重与配置绝对路径 |
| `get_openai_client()` | 构造 OpenAI 兼容客户端 |
| `setup_environment()` | 将配置写入环境变量（兼容旧代码） |

> **注意**：API Key 通过 `.env` 文件或环境变量注入，不得硬编码。

---

### 4.2 Agent 工作流（`agent/`）

#### AgentState（节点间共享状态）

```python
class AgentState(TypedDict):
    context_messages: list[BaseMessage]   # 对话历史
    pipeline_outputs: list[dict]          # 工作流事件日志
    next_step: str | None                 # 路由控制
    step_content: dict | str | None       # 当前步骤解析结果
    input: dict | None                    # 用户输入
    forecast: dict | None                 # 预测结果
    cnt_retry_planner: int                # Planner 重试计数
    editing_mode: bool                    # 是否进入编辑模式
    edit_task: dict | None                # 编辑任务参数
```

#### LangGraph 工作流（`agent/agent.py :: A1`）

```
START
  │
  ▼
[describer]  ──────────────────────────────────────────────────→ END
  │  route_main()
  ▼
[planner]  ─── retry(cnt_retry_planner ≥ 3) ──────────────────→ END
  │  route_main()
  ├─ "compose" ──→ [composer] ─────────────────────────────────→ END
  └─ "edit"    ──→ [editor]  ─────────────────────────────────→ END
```

| 节点 | 文件 | 职责 |
|------|------|------|
| `node_describer` | `nodes.py:20` | 分析输入时序，生成统计摘要，判断任务类型 |
| `node_planner` | `nodes.py:147` | 调用 LLM 生成结构化编辑计划 |
| `node_composer` | `nodes.py:428` | 调用数学工具完成合成类任务 |
| `node_editor` | `nodes.py:770` | 调用 TEdit + 数学工具完成编辑任务 |

#### LLM 意图分解器（`agent/llm_instruction_decomposer.py`）

将自然语言编辑指令分解为结构化参数：
```python
{
    "operation": "increase",
    "region": [start, end],       # 时间步范围
    "magnitude": 2.5,             # 倍数/量级
    "smooth": True
}
```

---

### 4.3 LLM 模块（`modules/llm.py`）

#### `CustomLLMClient`

OpenAI 兼容客户端，支持 DeepSeek / vLLM 本地部署：
- 优先使用 `/responses` 端点，fallback 至 Chat Completions
- 支持 function calling（tools 参数）

#### `get_event_driven_plan(news_text, instruction_text, llm/client)`

Pipeline 核心解析函数，返回：
```python
{
    "thought": "分析过程...",
    "tool_name": "hybrid_up",          # hybrid_up | hybrid_down | ensemble_smooth
    "parameters": {
        "region": [38, 58],            # [start_step, end_step]
        "math_shift": 0.5              # 数学平移量
    }
}
```

---

### 4.4 编辑工具（`tool/`）

#### `execute_llm_tool(plan, ts, tedit, n_ensemble, use_soft_boundary)`（`ts_editors.py:26`）

统一编辑入口，根据 `plan["tool_name"]` 分发：

| tool_name | 函数 | 实现逻辑 |
|-----------|------|---------|
| `hybrid_up` | `hybrid_up_soft()` | TEdit 生成上升趋势 + 线性数学锚定 |
| `hybrid_down` | `hybrid_down_soft()` | TEdit 生成下降趋势 + 线性数学锚定 |
| `ensemble_smooth` | `ensemble_smooth_soft()` | 多样本 DDPM 平均 + 软边界融合 |

返回值：`(edited_ts: np.ndarray, edit_log: dict)`

#### `TEditWrapper`（`tool/tedit_wrapper.py:24`）

TEdit 扩散模型的 Python 封装：

| 方法 | 说明 |
|------|------|
| `edit_time_series(ts, src_attrs, tgt_attrs, mask)` | 基础编辑接口 |
| `edit_region(ts, start, end, operation)` | 区间编辑 |
| `edit_region_soft(ts, start, end, operation, smooth_width)` | 软边界编辑（推荐） |
| `_generate_soft_mask(hard_mask, smooth_width)` | 硬→软掩码（高斯平滑） |

全局单例通过 `get_tedit_instance(model_path, config_path, device)` 获取（非线程安全）。

---

### 4.5 评估器（`test_scripts/bettertse_cik_official.py`）

#### `TSEditEvaluator.evaluate(base_ts, target_ts, generated_ts, gt_mask, gt_config, llm_prediction)`

| 指标 | 计算方式 | 含义 |
|------|---------|------|
| `t_iou` | `|P∩G| / |P∪G|` | LLM 预测区间 vs GT 区间的时间重叠度 |
| `feature_accuracy` | 特征名匹配率 | LLM 是否识别出正确的目标特征 |
| `mse_edit_region` | MSE(generated, target) in mask | 编辑区域与目标的均方误差（越小越好） |
| `mae_edit_region` | MAE(generated, target) in mask | 编辑区域绝对误差 |
| `mse_preserve_region` | MSE(generated, source) in ~mask | 背景保留区域的保真度（越小越好） |
| `mae_preserve_region` | MAE(generated, source) in ~mask | 背景保留绝对误差 |
| `editability_score` | 1 / (1 + mse_edit) | 编辑能力综合分（0~1，越高越好） |
| `preservability_score` | 1 / (1 + mse_preserve) | 背景保真综合分（0~1，越高越好） |

`gt_config` 必须包含键：`start_step`、`end_step`、`target_feature`。

---

### 4.6 Pipeline 集成脚本（`run_pipeline.py`）

端到端评估入口，串联所有阶段：

```python
run_pipeline(
    testset_path,          # 测试集 JSON 路径
    tedit_model_path=None, # TEdit 权重路径（None = 纯数学模式）
    tedit_config_path=None,
    tedit_device="cpu",
    output_path="test_results/pipeline_results.json",
    max_samples=None
)
```

**格式兼容**：通过 `_normalize_gt_config()` 统一两种 gt_config 字段命名：
- `build_mini_benchmark.py` 产出：`gt_start` / `gt_end`
- `bettertse_cik_official.py` 产出：`start_step` / `end_step`

**数学-only fallback**（`_math_only_edit()`）：无 TEdit 时，基于 LLM 预测区间做线性平移，仅评估区间定位能力。

CLI 用法：
```bash
python run_pipeline.py \
    --testset event_driven_testset_ultimate/event_driven_testset_ETTh1_5.json \
    --tedit-model TEdit-main/save/synthetic/pretrain_multi_weaver/0/ckpts/model_best.pth \
    --tedit-config TEdit-main/save/synthetic/pretrain_multi_weaver/0/model_configs.yaml \
    --device cpu \
    --output results.json
```

---

## 五、模块依赖关系

```
run_pipeline.py
    ├── config.py                    # API Key、模型路径
    ├── modules/llm.py               # get_event_driven_plan()
    ├── tool/ts_editors.py           # execute_llm_tool()
    │       └── tool/tedit_wrapper.py    # TEditWrapper → TEdit-main/
    └── test_scripts/bettertse_cik_official.py  # TSEditEvaluator

agent/agent.py (A1)
    ├── agent/nodes.py               # node_* 函数
    │       ├── modules/llm.py
    │       ├── tool/ts_editors.py
    │       └── tool/tedit_wrapper.py
    ├── agent/prompts.py             # EVENT_DRIVEN_AGENT_PROMPT 等
    └── agent/llm_instruction_decomposer.py

test_scripts/build_event_driven_testset.py
    ├── test_scripts/data_loader.py  # CSVDataLoader
    ├── test_scripts/change_injector.py # 5 种注入器
    └── modules/llm.py               # DeepSeek LLM（反向生成 Prompt）
```

---

## 六、已知问题与注意事项

| 问题 | 位置 | 说明 |
|------|------|------|
| `sys.modules` 命名冲突 | `test_scripts/config.py` vs 根 `config.py` | `test_scripts/__init__.py` 已清空导入，各子模块自行管理 `sys.path` |
| 单例非线程安全 | `tedit_wrapper.py:415`、`llm_instruction_decomposer.py` | 单进程单线程使用无问题，并发场景需加锁 |
| `torch.load(weights_only=False)` | `tedit_wrapper.py:95` | 存在反序列化 RCE 风险，仅加载可信模型权重 |
| `plot_series()` 绘图 bug | `modules/utils.py` | `output_ts` 被重复绘制，`predicted_ts` 参数未使用 |
| `node_composer` 空参数 bug | `agent/nodes.py:428` | `spec.get("required_parameters")` 返回 None 时迭代报错 |
| LangGraph 图 PNG 副作用 | `agent/agent.py:218` | 每次 `_configure_workflow()` 写盘，网络不可用时崩溃 |

---

## 七、快速上手

### 7.1 环境配置

```bash
# 安装依赖
pip install -r requirements.txt

# 配置 API Key（复制 .env.example 并填写）
cp .env.example .env
# 编辑 .env，设置 DEEPSEEK_API_KEY=sk-...
```

### 7.2 构建测试集

```bash
python test_scripts/build_event_driven_testset.py \
    --csv data/ETTh1.csv \
    --num-samples 20 \
    --output event_driven_testset_ultimate/
```

### 7.3 运行完整评估 Pipeline

```bash
python run_pipeline.py \
    --testset event_driven_testset_ultimate/event_driven_testset_ETTh1_5.json \
    --output results.json
```

### 7.4 直接使用 Agent

```python
from agent.agent import A1

agent = A1(model_name="deepseek-chat", base_url="https://api.deepseek.com/v1")
agent.set_editing_mode(True)
result = agent.go("过去一周用电需求激增，变压器油温在午高峰时段出现明显上升")
```

---

## 八、评估结果格式

`run_pipeline.py` 输出的 JSON 结构：

```json
{
  "metadata": {
    "testset_path": "...",
    "total_samples": 5,
    "tedit_available": true,
    "timestamp": "2026-03-12T..."
  },
  "summary": {
    "valid_samples": 5,
    "avg_t_iou": 0.62,
    "avg_editability_score": 0.78,
    "avg_preservability_score": 0.91,
    "avg_feature_accuracy": 0.80
  },
  "results": [
    {
      "sample_id": "001",
      "metrics": {
        "t_iou": 0.71,
        "editability_score": 0.82,
        "preservability_score": 0.94,
        "feature_accuracy": 1.0,
        "mse_edit_region": 0.018,
        "mse_preserve_region": 0.003
      },
      "llm_prediction": { "tool_name": "hybrid_up", "region": [40, 56] }
    }
  ]
}
```

---

## 九、参考文献

1. **TEdit**: Jiang et al., *Time Series Editing via Diffusion Models*, NeurIPS 2024
2. **CSDI**: Tashiro et al., *Conditional Score-based Diffusion Models for Probabilistic Time Series Imputation*, NeurIPS 2021
3. **CiK**: Goswami et al., *Context is Key: A Benchmark for Forecasting with Essential Textual Information*, NeurIPS 2024
4. **LangGraph**: LangChain documentation, *Building Stateful Multi-Actor Applications*

---

*BetterTSE Team · 2026-03-12*
