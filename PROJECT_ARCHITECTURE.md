# BetterTSE 项目代码结构与方法说明文档

## 一、项目概述

**BetterTSE** (Better Time Series Editing) 是一个基于扩散模型的时间序列编辑框架，实现了 **Training-free** 的软边界编辑功能，解决了传统方法中的"悬崖效应"问题。

### 核心创新

1. **潜空间融合 (Latent Blending)**: 背景区域零重建误差
2. **注意力注入 (Attention Injection)**: 语义隔离，编辑指令不泄漏到背景
3. **软边界处理 (Soft Boundary)**: 平滑过渡，避免悬崖效应

---

## 二、项目目录结构

```
BetterTSE-main/
├── TEdit-main/                    # 核心模型代码
│   ├── models/                    # 模型模块
│   │   ├── conditional_generator.py    # 条件生成器（核心入口）
│   │   ├── diffusion/                  # 扩散模型
│   │   │   ├── diff_csdi.py                 # CSDI基础扩散模型
│   │   │   ├── diff_csdi_multipatch.py      # 多Patch扩散模型
│   │   │   ├── diff_csdi_multipatch_weaver.py # 多Patch编织器（核心）
│   │   │   └── diff_csdi_time_weaver.py     # 时间编织器
│   │   ├── encoders/                   # 编码器
│   │   │   ├── attr_encoder.py             # 属性编码器
│   │   │   └── side_encoder.py             # 侧信息编码器
│   │   └── energy/                     # 能量模型
│   │       ├── energy_model.py             # 能量预测模型
│   │       └── layers/                     # 网络层
│   │           ├── SelfAttention_Family.py     # 注意力层（含注入机制）
│   │           ├── Transformer_EncDec.py       # Transformer编解码器
│   │           └── Embed.py                    # 嵌入层
│   ├── samplers.py                     # DDPM/DDIM采样器
│   ├── evaluation/                     # 评估模块
│   │   ├── base_evaluator.py           # 基础评估器
│   │   ├── basic_metrics.py            # 基础指标
│   │   └── energy_model.py             # 能量模型评估
│   ├── configs/                        # 配置文件
│   │   ├── air/                        # 空气质量数据集配置
│   │   ├── motor/                      # 电机数据集配置
│   │   └── synthetic/                  # 合成数据集配置
│   ├── datasets/                       # 数据集
│   ├── data/                           # 数据加载模块
│   ├── run_pretrain.py                 # 预训练脚本
│   ├── run_finetune.py                 # 微调脚本
│   └── tests/                          # 测试文件
│       └── test_attention_injection.py # 注意力注入测试
│
├── test_scripts/                   # 测试集构建工具
│   ├── build_testset.py            # 测试集构建主脚本
│   └── bettertse_cik_official.py   # CiK范式实现
│
├── test_sets/                      # 生成的测试集
│   └── testset_ETTh1_*.json        # ETTh1测试集
│
├── data/                           # 原始数据
│   └── ETTh1.csv                   # ETTh1数据集
│
├── SKILL.md                        # 技术规范文档
├── TEDIT_CONFIG.md                 # 配置说明文档
└── README.md                       # 项目说明
```

---

## 三、核心模块详解

### 3.1 条件生成器 (conditional_generator.py)

**职责**: 整合编码器和扩散模型，提供统一的生成/编辑接口

#### 关键类: `ConditionalGenerator`

```python
class ConditionalGenerator(nn.Module):
    def __init__(self, configs: Dict)
```

**初始化参数**:
| 参数 | 类型 | 说明 |
|------|------|------|
| `configs` | Dict | 配置字典，包含device、attrs、side、diffusion等配置 |

**核心属性**:
| 属性 | 类型 | 说明 |
|------|------|------|
| `attr_en` | AttributeEncoder | 属性编码器 |
| `side_en` | SideEncoder | 侧信息编码器 |
| `diff_model` | Diff_CSDI_* | 扩散模型 |
| `ddpm` | DDPMSampler | DDPM采样器 |
| `ddim` | DDIMSampler | DDIM采样器 |
| `edit_steps` | int | 编辑步数 |
| `bootstrap_ratio` | float | 引导比例 |

#### 核心方法

##### 1. `predict_noise()` - 噪声预测

```python
def predict_noise(
    self, 
    xt: torch.Tensor,           # [B, K, L] 噪声潜在变量
    side_emb: torch.Tensor,     # 侧信息嵌入
    src_attr_emb: torch.Tensor, # 属性嵌入
    t: torch.Tensor,            # 扩散时间步
    soft_mask: torch.Tensor = None,     # [B, L] 软边界掩码
    keys_null: torch.Tensor = None,     # [B, L, D] 背景Key
    values_null: torch.Tensor = None    # [B, L, D] 背景Value
) -> torch.Tensor:              # [B, K, L] 预测噪声
```

**功能**: 预测当前时间步的噪声，支持注意力注入

##### 2. `edit_soft()` - 软边界编辑入口

```python
def edit_soft(
    self,
    batch: Dict,                # 数据批次
    n_samples: int,             # 生成样本数
    sampler: str = "ddim",      # 采样器类型
    soft_mask: np.ndarray = None,   # 软边界掩码
    hard_mask: np.ndarray = None,   # 硬边界掩码
    smooth_width: int = 5,      # 平滑宽度
    smooth_type: str = "gaussian"   # 平滑类型
) -> torch.Tensor:              # [n_samples, B, K, L]
```

**功能**: 执行软边界编辑，支持硬掩码自动转换

##### 3. `_edit_soft()` - 核心编辑逻辑

```python
def _edit_soft(
    self,
    src_x: torch.Tensor,        # [B, K, L] 源序列
    side_emb: torch.Tensor,     # 侧信息嵌入
    src_attr_emb: torch.Tensor, # 源属性嵌入
    tgt_attr_emb: torch.Tensor, # 目标属性嵌入
    sampler: str,               # 采样器类型
    soft_mask: np.ndarray       # 软边界掩码
) -> torch.Tensor:              # [B, K, L] 编辑后序列
```

**核心公式**:
```
潜空间融合: z_{t-1} = M ⊙ z_{t-1}^{pred} + (1-M) ⊙ z_{t-1}^{GT}
```

##### 4. `create_soft_mask()` - 软边界掩码创建

```python
@staticmethod
def create_soft_mask(
    hard_mask: np.ndarray,      # [L] 硬边界掩码
    smooth_width: int = 5,      # 平滑宽度
    smooth_type: str = "gaussian"   # 平滑类型
) -> np.ndarray:                # [L] 软边界掩码
```

**功能**: 将硬边界掩码转换为软边界掩码，实现平滑过渡

---

### 3.2 扩散模型 (diff_csdi_multipatch_weaver.py)

**职责**: 实现多分辨率Patch扩散模型，支持注意力注入

#### 关键类

##### 1. `AttentionInjectionLayer` - 注意力注入层

```python
class AttentionInjectionLayer(nn.Module):
    def __init__(
        self,
        d_model: int,           # 模型维度
        nhead: int,             # 注意力头数
        dim_feedforward: int = 64,  # FFN维度
        dropout: float = 0.1,   # Dropout率
        activation: str = "gelu"    # 激活函数
    )
```

**核心公式**:
```
A_inj = Λ ⊙ A(Q, K_edit) + (I - Λ) ⊙ A(Q, K_null)
V_inj = Λ ⊙ (A_edit @ V_edit) + (I - Λ) ⊙ (A_null @ V_null)
```

##### 2. `ResidualBlock` - 残差块

```python
class ResidualBlock(nn.Module):
    def __init__(
        self,
        side_dim: int,          # 侧信息维度
        attr_dim: int,          # 属性维度
        channels: int,          # 通道数
        diffusion_embedding_dim: int,  # 扩散嵌入维度
        nheads: int,            # 注意力头数
        is_linear: bool = False,    # 是否使用线性注意力
        is_attr_proj: bool = False  # 是否使用属性投影
    )
    
    def forward(
        self,
        x: torch.Tensor,        # [B, C, K, L]
        side_emb: torch.Tensor, # [B, side_dim, K, L]
        attr_emb: torch.Tensor, # [B, n_attrs, attr_dim]
        diffusion_emb: torch.Tensor,  # [B, diffusion_embedding_dim]
        attention_mask: torch.Tensor = None,
        soft_mask: torch.Tensor = None,   # [B, L]
        keys_null: torch.Tensor = None,   # [B, L, D]
        values_null: torch.Tensor = None  # [B, L, D]
    ) -> Tuple[torch.Tensor, torch.Tensor]
```

##### 3. `Diff_CSDI_MultiPatch_Weaver_Parallel` - 主扩散模型

```python
class Diff_CSDI_MultiPatch_Weaver_Parallel(nn.Module):
    def __init__(
        self,
        config: Dict,           # 模型配置
        inputdim: int = 2       # 输入维度
    )
    
    def forward(
        self,
        x_raw: torch.Tensor,        # [B, inputdim, K, L]
        side_emb_raw: torch.Tensor, # [B, side_dim, K, L]
        attr_emb_raw: torch.Tensor, # [B, n_attrs, attr_dim]
        diffusion_step: torch.Tensor,  # [B]
        soft_mask: torch.Tensor = None,    # [B, L]
        keys_null: torch.Tensor = None,    # [B, L, D]
        values_null: torch.Tensor = None   # [B, L, D]
    ) -> torch.Tensor:            # [B, K, L]
```

---

### 3.3 注意力层 (SelfAttention_Family.py)

**职责**: 实现多种注意力机制，包括软边界注入

#### 关键类

##### 1. `FullAttention` - 全注意力（含注入）

```python
class FullAttention(nn.Module):
    def forward(
        self,
        queries: torch.Tensor,  # [B, L, H, E]
        keys: torch.Tensor,     # [B, S, H, E]
        values: torch.Tensor,   # [B, S, H, D]
        attn_mask: torch.Tensor,
        tau: float = None,      # 去平稳因子
        delta: float = None,    # 去平稳偏移
        soft_mask: torch.Tensor = None,  # [B, L] 软边界掩码
        keys_null: torch.Tensor = None,  # [B, S, H, E] 背景Key
        values_null: torch.Tensor = None # [B, S, H, D] 背景Value
    ) -> Tuple[torch.Tensor, torch.Tensor]
```

**语义隔离公式**:
```python
# 编辑区域注意力
A_edit = softmax(Q @ K_edit^T / sqrt(d))
# 背景区域注意力
A_null = softmax(Q @ K_null^T / sqrt(d))
# 最终输出
V = Λ ⊙ (A_edit @ V_edit) + (I - Λ) ⊙ (A_null @ V_null)
```

##### 2. `DSAttention` - 去平稳注意力

```python
class DSAttention(nn.Module):
    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attn_mask: torch.Tensor,
        tau: float = None,
        delta: float = None,
        soft_mask: torch.Tensor = None,
        keys_null: torch.Tensor = None,
        values_null: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]
```

**特点**: 在注意力计算中加入去平稳因子 τ 和偏移 δ

##### 3. `AttentionLayer` - 注意力层封装

```python
class AttentionLayer(nn.Module):
    def __init__(
        self,
        attention: nn.Module,   # 内部注意力模块
        d_model: int,           # 模型维度
        n_heads: int,           # 注意力头数
        d_keys: int = None,     # Key维度
        d_values: int = None    # Value维度
    )
```

---

### 3.4 采样器 (samplers.py)

**职责**: 实现DDPM和DDIM采样过程

#### 关键类

##### 1. `DDPMSampler` - DDPM采样器

```python
class DDPMSampler:
    def __init__(
        self,
        num_steps: int,         # 扩散步数
        beta_start: float = 0.0001,  # β起始值
        beta_end: float = 0.02,      # β结束值
        schedule: str = "linear",    # 噪声调度
        device: str = "cuda"         # 设备
    )
    
    def forward(self, x: torch.Tensor, t: torch.Tensor, noise: torch.Tensor = None) -> torch.Tensor:
        """前向扩散：添加噪声"""
        
    def reverse(self, xt: torch.Tensor, pred_noise: torch.Tensor, t: torch.Tensor, noise: torch.Tensor = None) -> torch.Tensor:
        """反向去噪：去除噪声"""
```

##### 2. `DDIMSampler` - DDIM采样器

```python
class DDIMSampler:
    def forward(self, x: torch.Tensor, pred_noise: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """DDIM前向（确定性）"""
        
    def reverse(self, xt: torch.Tensor, pred_noise: torch.Tensor, t: torch.Tensor, noise: torch.Tensor = None, is_determin: bool = True) -> torch.Tensor:
        """DDIM反向去噪"""
```

---

### 3.5 编码器

#### 3.5.1 属性编码器 (attr_encoder.py)

```python
class AttributeEncoder(nn.Module):
    def __init__(self, configs: Dict)
    
    def forward(self, attrs: torch.Tensor) -> torch.Tensor:
        """
        将离散属性索引转换为嵌入向量
        
        Args:
            attrs: [B, n_attrs] 属性索引
            
        Returns:
            [B, n_attrs, emb_dim] 属性嵌入
        """
```

#### 3.5.2 侧信息编码器 (side_encoder.py)

```python
class SideEncoder(nn.Module):
    def __init__(self, configs: Dict)
    
    def forward(self, tp: torch.Tensor) -> torch.Tensor:
        """
        编码时间点信息
        
        Args:
            tp: [B, L, tp_dim] 时间点特征
            
        Returns:
            [B, total_emb_dim, K, L] 侧信息嵌入
        """
```

---

### 3.6 测试集构建工具 (build_testset.py)

**职责**: 基于真实数据集构建时间序列编辑测试集

#### 关键类

##### 1. `CSVDataLoader` - CSV数据加载器

```python
class CSVDataLoader:
    def __init__(self, csv_path: str, dataset_name: str = "ETTh1")
    
    def get_sequence(self, start_idx: int, seq_len: int, feature: str) -> Tuple[np.ndarray, List[str]]:
        """获取指定特征的1D时间序列"""
```

##### 2. 物理注入器

```python
class MultiplierInjector(PhysicalInjector):
    """乘法倍数注入器 - 数值放大/缩小"""

class HardZeroInjector(PhysicalInjector):
    """强制归零注入器 - 数值置零"""

class NoiseInjector(PhysicalInjector):
    """底噪替换注入器 - 替换为噪声"""

class TrendInjector(PhysicalInjector):
    """趋势注入器 - 添加上升/下降趋势"""

class StepChangeInjector(PhysicalInjector):
    """阶跃变化注入器 - 突然跳变"""
```

##### 3. `MultiLevelVaguePromptGenerator` - 多模糊程度提示词生成器

```python
class MultiLevelVaguePromptGenerator:
    def generate_multiple_prompts(
        self,
        feature_name: str,
        feature_desc: str,
        injection_type: str,
        start_step: int,
        end_step: int,
        seq_len: int,
        causal_scenario: str,
        injection_config: Dict,
        num_prompts: int = 4
    ) -> List[VaguePromptItem]
```

**模糊程度分级**:
| 等级 | 标签 | 特点 |
|------|------|------|
| 1 | 轻微模糊 | 保留技术细节，仅模糊数值 |
| 2 | 中度模糊 | 使用相对时间，模糊数值比例 |
| 3 | 高度模糊 | 完全自然语言，无技术参数 |
| 4 | 极度模糊 | 仅描述业务场景 |

##### 4. `TestSetBuilder` - 测试集构建器

```python
class TestSetBuilder:
    def build_single_sample(
        self,
        sample_id: str,
        start_idx: int,
        feature: str = None,
        injection_type: str = None
    ) -> TestSetSample:
        """构建单个测试样本"""
    
    def build_batch(
        self,
        num_samples: int = 10,
        features: List[str] = None,
        injection_types: List[str] = None
    ) -> List[TestSetSample]:
        """批量构建测试样本"""
    
    def save_testset(self, filename: str = None) -> str:
        """保存测试集为JSON"""
```

---

## 四、模块依赖关系

```
┌─────────────────────────────────────────────────────────────────┐
│                      ConditionalGenerator                        │
│                    (models/conditional_generator.py)             │
└─────────────────────────────────────────────────────────────────┘
         │                    │                    │
         ▼                    ▼                    ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────┐
│ AttributeEncoder│  │   SideEncoder   │  │ Diff_CSDI_MultiPatch│
│ (encoders/)     │  │ (encoders/)     │  │ _Weaver_Parallel    │
└─────────────────┘  └─────────────────┘  └─────────────────────┘
                                                    │
                    ┌───────────────────────────────┼───────────────────────────────┐
                    ▼                               ▼                               ▼
          ┌─────────────────┐             ┌─────────────────┐             ┌─────────────────┐
          │ ResidualBlock   │             │ DiffusionEmbed  │             │ AttrProjector   │
          │                 │             │                 │             │                 │
          └─────────────────┘             └─────────────────┘             └─────────────────┘
                    │
                    ▼
          ┌─────────────────────────────────────────────────────────────────────────┐
          │                    AttentionInjectionLayer                              │
          │                    (SelfAttention_Family.py)                            │
          │                                                                           │
          │  核心公式:                                                                │
          │  A_inj = Λ ⊙ A(Q, K_edit) + (I - Λ) ⊙ A(Q, K_null)                      │
          │  V_inj = Λ ⊙ (A_edit @ V_edit) + (I - Λ) ⊙ (A_null @ V_null)            │
          └─────────────────────────────────────────────────────────────────────────┘
```

---

## 五、数据交互流程

### 5.1 训练流程

```
输入数据 (batch)
    │
    ├── src_x: [B, K, L] ─────────────────────────────┐
    ├── tp: [B, L, tp_dim] ──► SideEncoder ──► side_emb │
    └── attrs: [B, n_attrs] ─► AttrEncoder ──► attr_emb │
                                                       │
                                                       ▼
                                              ┌───────────────┐
                                              │  DDPM.forward │
                                              │  (加噪声)      │
                                              └───────────────┘
                                                       │
                                                       ▼
                                              noisy_x: [B, K, L]
                                                       │
                                                       ▼
                                              ┌───────────────┐
                                              │ Diff_CSDI     │
                                              │ (预测噪声)     │
                                              └───────────────┘
                                                       │
                                                       ▼
                                              pred_noise: [B, K, L]
                                                       │
                                                       ▼
                                              MSE(pred, gt_noise)
```

### 5.2 编辑流程

```
输入: src_x, src_attrs, tgt_attrs, soft_mask
    │
    ├── Forward Diffusion (DDIM) ──► xt_gt (背景真值)
    │
    ├── Reverse Denoising Loop:
    │   │
    │   ├── predict_noise(xt, tgt_attr) ──► pred_noise_tgt (前景)
    │   │
    │   ├── predict_noise(xt, src_attr) ──► pred_noise_src (背景)
    │   │
    │   ├── DDIM.reverse(xt, pred_noise_tgt) ──► xt_pred
    │   │
    │   ├── DDIM.reverse(xt, pred_noise_src) ──► xt_gt_step
    │   │
    │   └── Latent Blending:
    │       xt = M ⊙ xt_pred + (1-M) ⊙ xt_gt_step
    │
    └── 输出: edited_x (编辑后序列)
```

### 5.3 测试集构建流程

```
CSV数据 (ETTh1.csv)
    │
    ▼
┌─────────────────┐
│  CSVDataLoader  │ ──► 加载真实时间序列
└─────────────────┘
    │
    ▼
┌─────────────────┐
│ PhysicalInjector│ ──► 注入确定的物理变化
│ (5种注入器)      │     - multiplier: 数值放大
└─────────────────┘     - hard_zero: 强制归零
    │                   - noise_injection: 底噪替换
    ▼                   - trend_injection: 趋势注入
┌─────────────────┐     - step_change: 阶跃变化
│  TestSetSample  │ ──► 四元组: {Base_TS, Target_TS, Mask_GT, Config}
└─────────────────┘
    │
    ▼
┌─────────────────────┐
│ LLMVaguePromptGen   │ ──► 生成3-5条不同模糊程度的指令
│ (DeepSeek API)      │
└─────────────────────┘
    │
    ▼
输出: testset_ETTh1_*.json
```

---

## 六、配置文件说明

### 6.1 模型配置 (model.yaml)

```yaml
diffusion:
  type: "CSDI_MultiPatch_Weaver_Parallel"
  channels: 64
  diffusion_embedding_dim: 128
  num_steps: 100
  layers: 4
  nheads: 8
  is_linear: false
  L_patch_len: 4
  multipatch_num: 3
  attention_mask_type: "full"
  edit_steps: 50
  bootstrap_ratio: 0.5

attrs:
  dim: 128
  num_attrs: 3

side:
  dim: 128
```

### 6.2 训练配置 (train.yaml)

```yaml
batch_size: 16
num_epochs: 100
learning_rate: 1e-4
weight_decay: 1e-6
```

---

## 七、使用示例

### 7.1 加载模型进行编辑

```python
from models.conditional_generator import ConditionalGenerator
import torch
import numpy as np

# 加载配置
configs = load_config("configs/synthetic/model_multi_weaver.yaml")

# 初始化模型
model = ConditionalGenerator(configs)
model.load_state_dict(torch.load("save/synthetic/pretrain_multi_weaver/0/ckpts/model_best.pth"))
model.eval()

# 准备数据
batch = {
    "src_x": torch.randn(1, 7, 192),  # [B, K, L]
    "src_attrs": torch.randint(0, 3, (1, 3)),  # [B, n_attrs]
    "tgt_attrs": torch.randint(0, 3, (1, 3)),
    "tp": torch.randn(1, 192, 4)
}

# 创建软边界掩码
hard_mask = np.zeros(192)
hard_mask[60:100] = 1  # 编辑区域

# 执行编辑
edited_samples = model.edit_soft(
    batch,
    n_samples=1,
    hard_mask=hard_mask,
    smooth_width=5,
    smooth_type="gaussian"
)
```

### 7.2 构建测试集

```python
from test_scripts.build_testset import TestSetBuilder

# 初始化构建器
builder = TestSetBuilder(
    csv_path="data/ETTh1.csv",
    dataset_name="ETTh1",
    api_key="your-deepseek-api-key",
    seq_len=192,
    num_prompts_per_sample=4
)

# 批量构建
builder.build_batch(
    num_samples=100,
    features=["HUFL", "OT"],
    injection_types=["multiplier", "hard_zero"]
)

# 保存测试集
builder.save_testset()
```

---

## 八、评估指标

| 指标 | 公式 | 说明 |
|------|------|------|
| **t-IoU** | $\frac{|P \cap G|}{|P \cup G|}$ | 时间区间交并比 |
| **Editability** | MSE(pred, target) in edit region | 编辑区域精度 |
| **Preservability** | MSE(pred, source) in preserve region | 保留区域保真度 |
| **Feature Accuracy** | 正确预测特征的比例 | 特征识别准确率 |

---

## 九、版本历史

| 版本 | 日期 | 更新内容 |
|------|------|----------|
| v2.0 | 2026-03 | 添加注意力注入机制、多模糊程度提示词生成 |
| v1.0 | 2025-12 | 初始版本，实现基础编辑功能 |

---

## 十、参考文献

1. TEdit: Time Series Editing via Diffusion Models
2. CSDI: Conditional Score-based Diffusion Models for Time Series Imputation
3. Context is Key (CiK): Benchmark for Time Series Editing

---

*文档生成时间: 2026-03-11*
*BetterTSE Team*
