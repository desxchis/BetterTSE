# TEdit模型权重配置说明

## 模型权重位置

TEdit模型权重已经存在于以下位置：

```
TEdit-main/checkpoints/
├── air/
│   └── pretrain_multi_weaver/
│       ├── ckpts/
│       │   └── model_best.pth          # Air数据集预训练模型
│       ├── model_configs.yaml          # 模型配置
│       ├── pretrain_configs.yaml       # 预训练配置
│       └── eval_configs.yaml           # 评估配置
├── motor/
│   └── pretrain_multi_weaver/
│       ├── ckpts/
│       │   └── model_best.pth          # Motor数据集预训练模型
│       ├── model_configs.yaml
│       ├── pretrain_configs.yaml
│       └── eval_configs.yaml
└── synthetic/
    └── pretrain_multi_weaver/
        ├── ckpts/
        │   └── model_best.pth          # Synthetic数据集预训练模型
        ├── model_configs.yaml
        ├── pretrain_configs.yaml
        └── eval_configs.yaml
```

## 如何使用TEdit模型

### 方法1: 使用BetterTSE Agent（推荐）

```python
from agent.agent import A1

# 创建agent并启用TEdit
agent = A1(
    llm_name="claude-sonnet-4-20250514",
    planner_context_mode="hybrid",
    enable_tedit=True,
    enable_instruction_decomposition=True,
    
    # TEdit配置 - 使用Air数据集模型
    tedit_model_path="TEdit-main/save/air/pretrain_multi_weaver/0/ckpts/model_best.pth",
    tedit_config_path="TEdit-main/save/air/pretrain_multi_weaver/0/model_configs.yaml",
    tedit_device="cuda:0"  # 或 "cpu"
)

# 设置编辑模式
agent.set_editing_mode(True)

# 执行编辑
for log, last_message, pipeline in agent.go(user_input):
    print(log)
```

### 方法2: 直接使用TEdit Wrapper

```python
from tool.tedit_wrapper import TEditWrapper
import numpy as np

# 初始化TEdit wrapper
tedit = TEditWrapper(
    model_path="TEdit-main/save/air/pretrain_multi_weaver/0/ckpts/model_best.pth",
    config_path="TEdit-main/save/air/pretrain_multi_weaver/0/model_configs.yaml",
    device="cuda:0"
)

# 准备时间序列数据
ts = np.random.randn(100)  # 你的时间序列数据

# 编辑时间序列
src_attrs = [0, 0]  # 源属性: [趋势类型, 季节性类型]
tgt_attrs = [1, 0]  # 目标属性: [趋势类型, 季节性类型]

edited_ts = tedit.edit_time_series(
    ts, 
    src_attrs, 
    tgt_attrs, 
    n_samples=1, 
    sampler="ddim"
)

# 编辑特定区域
edited_region = tedit.edit_region(
    ts,
    start_idx=20,
    end_idx=50,
    src_attrs=src_attrs,
    tgt_attrs=tgt_attrs
)
```

## 选择合适的模型

TEdit提供了三个预训练模型，针对不同类型的时间序列：

### 1. Air模型（推荐用于空气质量数据）
- **路径**: `TEdit-main/checkpoints/air/pretrain_multi_weaver/`
- **适用场景**: 空气质量监测、环境数据、周期性较强的数据
- **特点**: 对季节性和周期性变化敏感

### 2. Motor模型
- **路径**: `TEdit-main/checkpoints/motor/pretrain_multi_weaver/`
- **适用场景**: 电机运行数据、工业传感器数据、机械设备数据
- **特点**: 对设备运行模式和异常检测敏感

### 3. Synthetic模型
- **路径**: `TEdit-main/checkpoints/synthetic/pretrain_multi_weaver/`
- **适用场景**: 合成数据、通用时间序列、实验数据
- **特点**: 通用性强，适合多种类型的时间序列

## 属性编码说明

TEdit使用属性编码来控制编辑方向。属性通常包括：

### 趋势类型（trend_type）
- 0: 无趋势/平稳
- 1: 线性上升趋势
- 2: 线性下降趋势
- 3: 指数上升趋势
- 4: 指数下降趋势
- ...

### 季节性类型（seasonality_type）
- 0: 无季节性
- 1: 正弦波季节性
- 2: 方波季节性
- 3: 锯齿波季节性
- ...

### 波动性类型（volatility_type）
- 0: 低波动性
- 1: 中等波动性
- 2: 高波动性
- ...

**注意**: 具体的属性编码取决于模型的训练配置，请参考对应的`model_configs.yaml`文件。

## 配置文件说明

每个模型包含三个配置文件：

### model_configs.yaml
定义模型架构和参数：
- `num_steps`: 扩散步数（通常为50-1000）
- `edit_steps`: 编辑步数（通常为50-100）
- `bootstrap_ratio`: Bootstrap学习比率（0-1之间）
- `n_attrs`: 属性数量

### pretrain_configs.yaml
定义预训练参数：
- 学习率
- 批次大小
- 训练轮数
- 数据路径

### eval_configs.yaml
定义评估参数：
- 评估指标
- 评估数据集
- 结果保存路径

## 使用示例

### 示例1: 改变趋势

```python
from tool.tedit_wrapper import get_tedit_instance

# 获取TEdit实例
tedit = get_tedit_instance(
    model_path="TEdit-main/checkpoints/air/pretrain_multi_weaver/ckpts/model_best.pth",
    config_path="TEdit-main/checkpoints/air/pretrain_multi_weaver/model_configs.yaml",
    device="cuda:0"
)

# 改变趋势：从无趋势到上升趋势
ts = np.random.randn(100)
edited_ts = tedit.edit_time_series(
    ts,
    src_attrs=[0, 0],  # [无趋势, 无季节性]
    tgt_attrs=[1, 0],  # [线性上升, 无季节性]
    n_samples=1
)
```

### 示例2: 改变波动性

```python
# 增加波动性
edited_ts = tedit.edit_time_series(
    ts,
    src_attrs=[0, 0, 0],  # [无趋势, 无季节性, 低波动]
    tgt_attrs=[0, 0, 2],  # [无趋势, 无季节性, 高波动]
    n_samples=1
)
```

### 示例3: 两阶段编辑

```python
from tool.region_selector import get_selector
from tool.tedit_wrapper import get_tedit_instance

# 选择区域
selector = get_selector()
region = selector.select_region(ts, intent="volatility", method="semantic")

# 编辑选定区域
tedit = get_tedit_instance(...)
edited_ts = tedit.edit_region(
    ts,
    start_idx=region["start_idx"],
    end_idx=region["end_idx"],
    src_attrs=[0, 0],
    tgt_attrs=[0, 1]
)
```

## 性能优化建议

### 1. 设备选择
- **GPU**: 使用`device="cuda:0"`，速度快但需要GPU内存
- **CPU**: 使用`device="cpu"`，速度慢但不需要GPU

### 2. 采样方法
- **DDIM**: `sampler="ddim"`，快速且确定性，推荐用于生产环境
- **DDPM**: `sampler="ddpm"`，标准扩散采样，质量可能更好但速度慢

### 3. 编辑步数
- **快速编辑**: `edit_steps=20-50`，速度快
- **高质量编辑**: `edit_steps=100-200`，质量高但速度慢

### 4. 批量处理
```python
# 生成多个样本
edited_samples = tedit.edit_time_series(
    ts, src_attrs, tgt_attrs, 
    n_samples=10  # 生成10个候选
)

# 选择最佳结果
best_result = select_best(edited_samples)
```

## 常见问题

### Q1: 模型加载失败
**A**: 检查路径是否正确，确保模型文件存在：
```python
import os
model_path = "TEdit-main/checkpoints/air/pretrain_multi_weaver/ckpts/model_best.pth"
print(f"模型文件存在: {os.path.exists(model_path)}")
```

### Q2: GPU内存不足
**A**: 尝试以下方法：
1. 使用CPU: `device="cpu"`
2. 减少编辑步数: `edit_steps=20`
3. 减少样本数: `n_samples=1`

### Q3: 编辑效果不理想
**A**: 尝试以下方法：
1. 增加编辑步数: `edit_steps=100`
2. 尝试不同的属性组合
3. 使用更适合的预训练模型（air/motor/synthetic）

### Q4: 如何知道使用哪个属性编码？
**A**: 查看模型的配置文件和训练数据：
```python
import yaml
with open("TEdit-main/checkpoints/air/pretrain_multi_weaver/model_configs.yaml") as f:
    config = yaml.safe_load(f)
    print(f"属性数量: {config.get('n_attrs', 'N/A')}")
```

## 下一步

1. **测试模型加载**: 运行 `python test_tedit_loading.py` 验证模型是否能正常加载
2. **尝试编辑功能**: 使用示例代码测试不同的编辑操作
3. **集成到工作流**: 将TEdit集成到BetterTSE的编辑流程中

## 相关文件

- [tool/tedit_wrapper.py](../tool/tedit_wrapper.py) - TEdit模型封装
- [tool/tool_description/tedit_tools.py](../tool/tool_description/tedit_tools.py) - TEdit工具描述
- [agent/nodes.py](../agent/nodes.py) - 编辑节点实现
- [examples/tedit_integration_example.py](../examples/tedit_integration_example.py) - 完整示例
