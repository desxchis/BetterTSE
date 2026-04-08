下面这版可以直接复制给 Codex。
它只聚焦你们这次组会真正确定下来的主线：**先把 how-much 真正打进 TEdit 的 diffusion 内部，先验证“可控编辑幅度”这件事本身成立。** 这正对着老师指出的几个核心问题：当前 how-much 还没真正传给 diffusion；TEdit 原生不带强度控制；现有 teacher/student 路线有明显“自己教自己”的偏差风险；而“只要能控制编辑幅度，本身就是贡献”。 同时，它也和你们现有方法总定位一致：**主任务仍是 Pure Time-Series Editing，Forecast Revision 是下游应用，不是当前这一轮的主战场；整体流程仍保留 Intent → Localization → Canonical Tool → Hybrid Tool → How Much → Editing Execution。**  

---

# 给 Codex 的完整版任务书

## 任务标题

在 BetterTSE / TEdit 代码库中实现一个**最小可行、可验证的时序编辑强度控制版本**：
以 **TEdit** 为扩散编辑基座，引入 **Numeric Projector / Strength Adapter + Modulation Injection**，将离散强度条件（weak / medium / strong）真正注入 **diffusion 内部**，先在 **Pure Editing** 任务上验证编辑幅度的单调可控性与非编辑区稳定性。

---

## 1. 背景与问题定义

当前 BetterTSE 的高层语义链路已经相对清楚：复杂文本先拆成 `intent / localization / canonical tool / hybrid tool / how-much`，再执行编辑；其中主任务应仍然是时间序列编辑，Forecast Revision 只是下游应用验证。 

但当前最大的研究与工程断点在于：

1. **how-much 参数没有真正进入 diffusion 编辑内核**，还停留在外部逻辑层。
2. **TEdit 原生只支持离散属性编辑**，例如 trend type / trend direction / season cycle，并没有天然的“幅度控制”机制。
3. 现有 teacher/student 路线存在明显“自己教自己”的偏差放大风险，本轮先不继续扩展这条线。
4. 这轮最核心的贡献目标，不是 Forecast Revision 指标，而是：
   **能否让扩散模型真正学会“改多大”，并且做到可控、单调、稳定。** 

另外，NumeriKontrol 已经明确说明：
**仅把数字或强度写进文本 prompt，让模型直接学数值控制，通常是不可行的；更合理的是单独的 numeric projector / adapter，并结合 task ID 隔离不同单位/任务。** 同时它使用四元组训练样本 `(source, instruction, numeric parameter, target)`，而不是只有 source/instruction/target。

---

## 2. 本轮任务边界

### 2.1 这一轮必须做的

1. 基座仍使用 **TEdit**
2. 只做 **Pure Editing**
3. 只做 **离散强度**：

   * weak
   * medium
   * strong
4. 强度必须通过**独立条件通路**进入 TEdit
5. 注入位置优先放在 **多分辨率去噪网络的 modulation / conditional normalization 路径**
6. 数据从三元组升级为四元组
7. 重点实验是：

   * 可行性对照实验
   * 强度响应曲线实验
   * ablation

### 2.2 这一轮明确不做的

1. 不做 teacher/student 蒸馏主线
2. 不做 Forecast Revision 主实验
3. 不做 reward model / DPO / CRD
4. 不做连续强度主实验
5. 不做多变量 parameter slot 全量版
6. 不做大规模跨 backbone 横向比较

---

## 3. 成功标准

代码和实验至少要满足以下标准：

1. 对同一个 `source_ts + 同一编辑语义 + 同一区域`，输入强度满足：

   * `weak < medium < strong`
   * 实际编辑幅度单调增加

2. 在强度增加时：

   * 编辑区的目标变化增强
   * 非编辑区误差不能明显恶化

3. 完整方法必须优于下面两个对照：

   * 原始 TEdit
   * 仅把强度写进文本 prompt，但不做内部注入

4. 新模块 zero-init 时，模型应退化为原始 TEdit，保持向后兼容

---

## 4. 总体方法设计

### 4.1 方法一句话定义

在保留 BetterTSE 高层结构化语义链路的前提下，为 TEdit 增加一条**独立的强度条件通路**：
将 `strength_label + task_id + pooled text_context` 编码为 `strength_cond_vec`，再将其映射为各个去噪 block 的 `Δgamma / Δbeta`，以残差方式注入原始 modulation 参数，实现 diffusion 内部的时序编辑强度控制。

### 4.2 为什么这样改

TEdit 本身是一个**多分辨率噪声估计器**，不同属性在不同尺度上起作用，例如趋势更偏全局，季节性更偏局部。
因此强度控制不应该只放在输入头，而应进入**各个分辨率 block 的特征调制过程**。

### 4.3 借鉴来源

借鉴 NumeriKontrol 的三个关键点：

1. 单独的 **numeric projector**
2. **task ID / task-separated design**
3. 四元组监督样本 `(source, instruction, strength, target)` 

---

## 5. 代码改造总览

请优先检查并修改仓库中与以下职责等价的模块：

### 5.1 模型主干

优先搜索并修改：

* `TEdit-main/models/diffusion/diff_csdi_multipatch.py`
* `TEdit-main/models/diffusion/diff_csdi_multipatch_weaver.py`
* 所有 multi-resolution processing block / transformer block / residual block
* 所有生成 `attr embedding / timestep embedding / conditioning vector` 的模块

### 5.2 训练与推理入口

优先搜索并修改：

* `run_pipeline.py`
* `run_tedit*.py`
* `train*.py`
* `inference*.py`

### 5.3 数据集与 dataloader

优先搜索并修改：

* synthetic benchmark 构造脚本
* dataset class
* collate_fn
* prompt builder / sample generator

### 5.4 配置文件

优先搜索并修改：

* `configs/*.yaml`
* argparse/dataclass config

如果文件名和这里不完全一致，请修改**同职责模块**。

---

## 6. 具体架构改造

---

### 6.1 新增 batch 数据结构

所有 Pure Editing 样本从三元组升级为四元组，并保留结构化字段：

```python
{
    "source_ts": Tensor[L] or Tensor[C, L],
    "target_ts": Tensor[L] or Tensor[C, L],
    "instruction_text": str,
    "strength_label": int,   # 0=weak, 1=medium, 2=strong
    "task_id": int,          # effect_family + direction 联合映射
    "region": Tuple[int, int],   # [start_idx, end_idx)
    "effect_family": str,
    "direction": str,
}
```

要求：

1. `strength_label` 必须显式存在，不能只藏在文本里
2. `task_id` 必须显式存在
3. 不传 `strength_label` 时，模型必须 fallback 到原始 TEdit 路径

---

### 6.2 新增 StrengthProjector

新建文件：

* `models/conditioning/numeric_projector.py`

实现一个轻量级 projector，输入：

* `strength_label`
* `task_id`
* `pooled text_context`

输出：

* `strength_cond_vec`

参考实现风格：

```python
import torch
import torch.nn as nn
from typing import Optional

class StrengthProjector(nn.Module):
    def __init__(
        self,
        num_strength_bins: int = 3,
        num_tasks: int = 32,
        emb_dim: int = 64,
        hidden_dim: int = 128,
        out_dim: int = 128,
        use_text_context: bool = True,
        text_dim: int = 128,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.use_text_context = use_text_context

        self.strength_emb = nn.Embedding(num_strength_bins, emb_dim)
        self.task_emb = nn.Embedding(num_tasks, emb_dim)

        in_dim = emb_dim * 2
        if use_text_context:
            in_dim += text_dim

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

        # zero-init, preserve original behavior at step 0
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(
        self,
        strength_label: torch.Tensor,
        task_id: torch.Tensor,
        text_context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        s = self.strength_emb(strength_label)
        t = self.task_emb(task_id)

        feats = [s, t]
        if self.use_text_context:
            if text_context is None:
                raise ValueError("text_context is required")
            feats.append(text_context)

        x = torch.cat(feats, dim=-1)
        return self.mlp(x)
```

要求：

1. 第一版只支持离散 strength
2. 输出维度要可用于 modulation
3. 最后一层必须 zero-init
4. 后续可扩展为连续值版，但本轮不做

---

### 6.3 Text Context pooling 必须显式固定

不要让代码自己发明 pooling 逻辑。

要求：

1. 若 text encoder 输出是 `[B, T, D]`，统一用：

```python
text_context_pooled = text_context.mean(dim=1)
```

2. 若输出已是 `[B, D]`，直接使用

3. 在进入 `StrengthProjector` 前做 shape assert：

```python
assert text_context_pooled.dim() == 2
assert text_context_pooled.shape[0] == strength_label.shape[0]
```

4. 不允许新增独立的大型文本编码器；只能复用现有文本/intent 编码结果

建议新增一个工具函数：

```python
def pool_text_context(text_context: torch.Tensor) -> torch.Tensor:
    if text_context.dim() == 3:
        return text_context.mean(dim=1)
    elif text_context.dim() == 2:
        return text_context
    else:
        raise ValueError(f"Unexpected text_context shape: {text_context.shape}")
```

---

### 6.4 新增 StrengthModulationAdapter

新建文件：

* `models/conditioning/strength_modulation.py`

作用：
将 `strength_cond_vec` 映射为每个 block 的 `Δgamma, Δbeta`

参考实现：

```python
import torch
import torch.nn as nn

class StrengthModulationAdapter(nn.Module):
    def __init__(self, cond_dim: int, hidden_dim: int):
        super().__init__()
        self.to_gamma_beta = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
        )
        nn.init.zeros_(self.to_gamma_beta[-1].weight)
        nn.init.zeros_(self.to_gamma_beta[-1].bias)

    def forward(self, cond_vec: torch.Tensor):
        gb = self.to_gamma_beta(cond_vec)
        gamma, beta = gb.chunk(2, dim=-1)
        return gamma, beta
```

要求：

1. 最后一层必须 zero-init
2. 每个 block 自己维护匹配 hidden_dim 的 adapter，不允许一个固定 adapter 全局复用

---

### 6.5 在 TEdit block 内加入 strength-aware modulation

这是本轮最关键改动。

TEdit 本来是多分辨率扩散编辑模型，负责在不同尺度上对离散属性进行编辑。
现在要把强度控制也打进去。

#### 原则

对于每个 block，若原来有基础条件调制参数：

* `gamma_orig`
* `beta_orig`

则改成：

[
\gamma_{final} = \gamma_{orig} + \Delta\gamma_{strength}
]

[
\beta_{final} = \beta_{orig} + \Delta\beta_{strength}
]

再对 block hidden state 做调制。

#### 推荐模板

```python
class StrengthAwareBlock(nn.Module):
    def __init__(self, hidden_dim, cond_dim, ...):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.norm = nn.LayerNorm(hidden_dim)

        self.base_cond_proj = nn.Linear(cond_dim, hidden_dim * 2)
        self.strength_adapter = StrengthModulationAdapter(
            cond_dim=cond_dim,
            hidden_dim=hidden_dim,
        )

        self.main = ...   # original block body

    def forward(self, h, base_cond_vec, strength_cond_vec=None):
        base_gamma, base_beta = self.base_cond_proj(base_cond_vec).chunk(2, dim=-1)

        if strength_cond_vec is None:
            delta_gamma = torch.zeros_like(base_gamma)
            delta_beta = torch.zeros_like(base_beta)
        else:
            delta_gamma, delta_beta = self.strength_adapter(strength_cond_vec)

        h = self.norm(h)
        h = (1.0 + base_gamma + delta_gamma) * h + (base_beta + delta_beta)
        h = self.main(h)
        return h
```

要求：

1. 注入位置优先选 **conditional norm / modulation** 路径
2. 不要只在输入头拼接 strength
3. 不传 strength 时输出行为必须与原始 TEdit 一致

---

### 6.6 多分辨率 hidden_dim 必须动态匹配

TEdit 的核心是多分辨率噪声估计器，不同 resolution block 可能具有不同 hidden_dim。

要求：

1. `StrengthProjector` 可以全局共享一个 trunk
2. **每个 block 内部必须单独实例化自己的 `StrengthModulationAdapter(hidden_dim=block_hidden_dim)`**
3. 不允许一个写死 `hidden_dim=128` 的全局 adapter 强行给所有 block 用
4. 模型构建时打印每个 block 的绑定日志：

```python
print(f"[StrengthAdapter] block={block_name}, hidden_dim={hidden_dim}")
```

---

### 6.7 task_id 定义

第一版 task_id 采用 `effect_family + direction` 联合编码。

例如：

```python
TASK_ID_MAP = {
    ("level", "up"): 0,
    ("level", "down"): 1,
    ("trend", "up"): 2,
    ("trend", "down"): 3,
    ("seasonality", "up"): 4,
    ("seasonality", "down"): 5,
    ("volatility", "up"): 6,
    ("volatility", "down"): 7,
}
```

理由：

* 不同 family 的量纲几何不同
* 相同强度标签在不同 family 下不能共享完全相同的语义空间
* 这正是 NumeriKontrol 的 task-separated design 的关键思想。

---

## 7. 数据集与样本构造

---

### 7.1 总原则

这一轮只构造 **Pure Editing 的 controlled benchmark**，不要从真实文本里反推真实强度。

### 7.2 family 选择

第一版只做两个最容易测幅度的 family：

1. `level_step`
2. `trend_linear_up/down`

### 7.3 强度定义

对每条 source，生成三个 target：

* weak
* medium
* strong

建议定义：

#### level_step

* weak: 编辑区均值偏移 = `0.25 * local_std`
* medium: 编辑区均值偏移 = `0.50 * local_std`
* strong: 编辑区均值偏移 = `0.75 * local_std`

#### trend_linear_up/down

* weak: 编辑区斜率增量 = `0.25 * slope_ref`
* medium: 编辑区斜率增量 = `0.50 * slope_ref`
* strong: 编辑区斜率增量 = `0.75 * slope_ref`

要求：

1. 保存离散 `strength_label`
2. 同时保存实际数值强度，供响应曲线使用
3. 非编辑区尽量保持不变

### 7.4 文本模板

第一版模板要稳定，不要太复杂。

例如：

* “将中间这段轻微抬高”
* “将中间这段明显抬高”
* “将后半段趋势轻微上升”
* “将后半段趋势明显下降”

### 7.5 Dataset 接口

实现一个新的 dataset class，例如：

```python
class StrengthControlledEditingDataset(Dataset):
    def __getitem__(self, idx):
        return {
            "source_ts": source_ts,
            "target_ts": target_ts,
            "instruction_text": instruction,
            "strength_label": strength_label,
            "task_id": task_id,
            "region": region,
            "effect_family": effect_family,
            "direction": direction,
        }
```

---

## 8. 训练方案

---

### 8.1 config 新增字段

```yaml
model:
  use_strength_control: true
  strength_num_bins: 3
  strength_emb_dim: 64
  strength_hidden_dim: 128
  strength_cond_dim: 128
  strength_use_text_context: true
  strength_injection_mode: "modulation"
  strength_task_id: true

train:
  freeze_backbone: true
  unfreeze_strength_modules_only: true
  use_lora_near_modulation: false
  lr_strength: 1e-4
  lr_backbone_partial: 1e-5
```

---

### 8.2 参数冻结逻辑必须显式写在 train.py 里

不要只在 config 中写 `freeze_backbone: true`，必须真正冻结参数。

要求：

1. 默认冻结全部 backbone
2. 只训练：

   * `strength_projector`
   * `strength_adapter`
   * 紧邻 modulation 注入位置的少量 projection 层
3. 优化器只接收 `requires_grad=True` 的参数

参考实现：

```python
def set_trainable_params(model: torch.nn.Module):
    trainable_keywords = [
        "strength_projector",
        "strength_adapter",
        "strength_modulation",
        "base_cond_proj",
        "modulation",
    ]

    total_params = 0
    trainable_params = 0

    for name, param in model.named_parameters():
        total_params += param.numel()

        if any(k in name for k in trainable_keywords):
            param.requires_grad = True
            trainable_params += param.numel()
        else:
            param.requires_grad = False

    print(f"[Freeze] trainable params: {trainable_params}/{total_params}")
```

优化器：

```python
optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=config.train.lr_strength
)
```

---

### 8.3 loss 设计

这一轮主干不要复杂化。

**扩散主干仍然使用原始 denoising MSE。**

也就是：

* strength 是条件输入
* 主监督依然是 diffusion noise prediction loss
* 不要把 strength 当成主干的分类头去训

说明：

* 离散 strength 不等于主干用 classification loss
* parser 若单独做，可另用分类 loss
* 但 projector + TEdit 主干这里仍然是统一的 diffusion objective

---

### 8.4 forward 接口

训练和推理都要支持以下字段：

```python
pred = model(
    source_ts=batch["source_ts"],
    instruction_text=batch["instruction_text"],
    strength_label=batch["strength_label"],
    task_id=batch["task_id"],
    region=batch["region"],
    target_ts=batch["target_ts"],   # if needed
)
```

要求：

1. 若未提供 `strength_label`，自动 fallback 到原始路径
2. 若提供了 `strength_label` 但模型未启用强度控制，抛出清晰错误

---

## 9. 推理与 CLI 支持

新增或扩展推理参数：

```bash
--strength-label weak|medium|strong
--strength-id 0|1|2
--effect-family level|trend
--direction up|down
```

要求：

1. 未传 strength 时，走原始 TEdit
2. task_id 不在映射中时，日志 warning 并退回默认
3. 推理日志里必须打印：

   * 输入 strength_label
   * task_id
   * region
   * effect_family
   * direction

---

## 10. 评估脚本与指标

---

### 10.1 核心指标

Pure Editing 主表保留三类：

1. `target_mae / target_mse`
2. `t_iou`
3. `preservation_mae` 

但本轮必须新增强度控制指标：
4. `edit_region_mae`
5. `monotonicity_success_rate`
6. `mean_actual_magnitude_by_strength`

---

### 10.2 preservation_mae 的定义必须严格

这一点必须写死。

`preservation_mae` 必须只在**非编辑区**计算。
也就是基于 `region = [start_idx, end_idx)` 生成反向 mask。

参考实现：

```python
def build_non_edit_mask(length: int, region: tuple[int, int], device=None):
    start_idx, end_idx = region
    mask = torch.ones(length, dtype=torch.bool, device=device)
    mask[start_idx:end_idx] = False
    return mask


def compute_preservation_mae(pred, target, regions):
    if pred.dim() == 3:
        B, C, L = pred.shape
        maes = []
        for b in range(B):
            mask = build_non_edit_mask(L, regions[b], device=pred.device)
            mask = mask.unsqueeze(0).expand(C, L)
            err = torch.abs(pred[b] - target[b])[mask].mean()
            maes.append(err)
        return torch.stack(maes).mean()

    elif pred.dim() == 2:
        B, L = pred.shape
        maes = []
        for b in range(B):
            mask = build_non_edit_mask(L, regions[b], device=pred.device)
            err = torch.abs(pred[b] - target[b])[mask].mean()
            maes.append(err)
        return torch.stack(maes).mean()

    else:
        raise ValueError(f"Unexpected prediction shape: {pred.shape}")
```

同时新增 `edit_region_mae`，只在编辑区统计误差。

---

### 10.3 响应曲线实验必须实现

这是本轮最关键实验。

#### 目标

证明模型对强度条件存在**单调、可预测、稳定**的内部响应。

#### 做法

固定：

* 同一个 `source_ts`
* 同一个 `effect_family`
* 同一个 `region`
* 同一个 `direction`

分别输入：

* weak
* medium
* strong

记录：

1. 编辑区实际幅度

   * level: 编辑区均值偏移
   * trend: 编辑区斜率增量
2. 非编辑区 MAE

输出两张图：

* `strength_response_curve.png`
* `preservation_curve.png`

还要同时输出 csv：

```csv
sample_id,effect_family,direction,strength_label,actual_magnitude,preservation_mae
...
```

---

## 11. 实验设计

---

### 实验 1：最小可行性对照实验

#### 目的

证明 how-much 已经真正进入 diffusion 内部，而不是只是 prompt 层面或后处理伪控制。

#### 数据

* Pure Editing synthetic benchmark
* family 仅：

  * `level_step`
  * `trend_linear_up/down`

#### 对照组

A. 原始 TEdit
B. TEdit + 仅把 strength 写进 instruction 文本
C. TEdit + StrengthProjector + Modulation Injection（ours）

#### 指标

* target MAE / MSE
* edit_region_mae
* preservation_mae
* t-IoU

#### 预期

C 明显优于 A 和 B，尤其在：

* target similarity
* monotonicity
* preservation stability

---

### 实验 2：响应曲线实验

#### 目的

验证强度条件与实际编辑幅度之间的单调关系。

#### 通过标准

1. `weak < medium < strong`
2. 非编辑区误差不爆炸
3. 不同 family 下都成立

---

### 实验 3：消融实验

必须做以下消融：

1. **w/o projector**

   * 去掉 StrengthProjector，只靠文本 strength

2. **w/o task_id**

   * projector 不输入 task_id

3. **w/o modulation injection**

   * strength 只在输入头拼接，不进入 block 内调制

4. **freeze vs partial unfreeze**

   * 比较训练稳定性和性能

预期：
`w/o projector` 和 `w/o task_id` 都会明显退化，这与 NumeriKontrol 的结论一致。

---

## 12. 单元测试 / smoke test

必须新增至少两个测试：

### test 1：zero-init fallback

同一 batch：

* 不传 strength
* 传 strength，但 projector/adapter 仍为 zero-init

两次输出必须几乎一致：

```python
max_abs_diff < 1e-6   # 或合理容差
```

### test 2：strength forward smoke test

同一 batch 分别输入：

* weak
* medium
* strong

检查：

1. 前向不报错
2. shape 一致
3. 无 NaN
4. strength_label 日志正确打印

---

## 13. 需要新增或修改的脚本

请至少交付：

### 新增模块

* `models/conditioning/numeric_projector.py`
* `models/conditioning/strength_modulation.py`

### 新增评估脚本

* `tools/eval_strength_control.py`
* `tools/plot_strength_response.py`

### 修改脚本

* dataset builder
* dataloader / collate_fn
* train script
* eval script
* inference script
* config
* README

---

## 14. 交付物清单

Codex 最终必须交付：

1. 可运行的模型代码改动
2. 新增 conditioning 模块文件
3. 新 dataset / dataloader 支持
4. 新 config
5. 新训练命令示例
6. 新推理命令示例
7. 三组实验结果表
8. 两张核心曲线图
9. csv 统计文件
10. README 更新

---

## 15. 最终验收标准

代码验收以以下条件为准：

1. 新模型能正常训练与推理
2. 不传 strength 时，退化为原始 TEdit
3. 对 level_step / trend_linear 两个 family，响应曲线满足单调性
4. 完整方法优于原始 TEdit 与“文本强度版”
5. preservation_mae 明确基于非编辑区 mask 计算
6. 所有训练参数冻结逻辑真实生效
7. 每个 resolution block 的 adapter hidden_dim 对齐正确
8. README 足够让别人复现

---

# 可直接复制给 Codex 的最终 Prompt

```text
请在 BetterTSE / TEdit 代码库中实现一个最小可行、可验证的“时序编辑强度控制”版本。严格按照以下要求执行，不要自行扩展到蒸馏、Forecast Revision、连续强度或 reward 对齐。

【任务目标】
1. 基座仍然使用 TEdit。
2. 当前只做 Pure Editing。
3. 先支持离散强度：weak / medium / strong。
4. 核心目标是把 strength 真正注入 diffusion 内部，而不是仅写进文本 prompt。
5. 方法核心：新增 StrengthProjector / StrengthModulationAdapter，并将其注入 TEdit 的多分辨率 block 的 modulation / conditional normalization 路径。
6. 重点验证“编辑幅度是否真正可控、单调、稳定”。

【方法原则】
1. 保留 BetterTSE 现有高层结构：intent / localization / canonical tool / hybrid tool。
2. 只重做 how-much 的底层实现。
3. 参考 NumeriKontrol 的思路：
   - numeric projector
   - task-separated design / task id
   - 四元组训练样本
4. TEdit 原生只支持离散属性编辑，没有幅度控制，所以你需要新增一条独立的强度条件通路。

【必须完成的代码改造】
1. 新增 models/conditioning/numeric_projector.py
   - 输入：strength_label, task_id, pooled text_context
   - 输出：strength_cond_vec
   - 最后一层 zero-init

2. 新增 models/conditioning/strength_modulation.py
   - 输入：strength_cond_vec
   - 输出：delta_gamma, delta_beta
   - 最后一层 zero-init

3. 修改 TEdit 多分辨率主干
   - 在每个 relevant block 内部加入 strength-aware modulation：
     gamma_final = gamma_orig + delta_gamma
     beta_final = beta_orig + delta_beta
   - 不传 strength 时必须退化为原始 TEdit
   - 不允许只在输入头拼接 strength

4. Text context pooling 必须显式固定
   - 若 text encoder 输出为 [B, T, D]，统一做 mean(dim=1)
   - 若输出为 [B, D]，直接使用
   - 进入 projector 前做 shape assert
   - 不允许新增独立大型文本编码器

5. 多分辨率 adapter 必须按 block hidden_dim 动态匹配
   - StrengthProjector 可共享 trunk
   - StrengthModulationAdapter 必须在每个 block 内单独实例化
   - 不允许全局共用一个固定 hidden_dim 的 adapter

6. task_id 必须显式存在
   - 第一版用 effect_family + direction 联合映射
   - 同 strength_label 在不同 family 下不能共享完全相同语义空间

【数据改造】
1. Pure Editing 样本从三元组升级为四元组：
   (source_ts, instruction_text, strength_label, target_ts)
2. 同时保存：
   - task_id
   - region
   - effect_family
   - direction
3. 第一版 benchmark 只做两个 family：
   - level_step
   - trend_linear_up/down
4. 每条 source 生成三个强度 target：
   - weak
   - medium
   - strong
5. 需要同时保存真实连续幅度值，供响应曲线统计

【训练要求】
1. 主干损失仍为 diffusion denoising MSE。
2. 不要把 strength 当作主干分类头。
3. train.py 中必须显式冻结参数：
   - 默认冻结全部 backbone
   - 只训练 strength_projector / strength_adapter / 紧邻 modulation 的少量 projection 层
4. 优化器只能接收 requires_grad=True 的参数
5. 所有改动必须加 config 开关

【推理要求】
1. 支持 CLI 参数：
   --strength-label weak|medium|strong
   --effect-family level|trend
   --direction up|down
2. 不传 strength 时自动走原始 TEdit
3. task_id 不在映射中时给 warning

【评估要求】
实现以下指标：
1. target_mae / target_mse
2. edit_region_mae
3. preservation_mae
4. t-IoU
5. monotonicity_success_rate
6. mean_actual_magnitude_by_strength

其中 preservation_mae 必须严格基于 region=[start_idx, end_idx) 的反向 mask 计算，只统计非编辑区误差。

【必须实现的实验】
实验 1：最小可行性对照
A. 原始 TEdit
B. TEdit + 仅文本 strength
C. TEdit + StrengthProjector + Modulation Injection（ours）

实验 2：响应曲线实验
- 固定同一 source、同一 family、同一区域、同一 direction
- 分别输入 weak / medium / strong
- 记录 actual_magnitude 与 preservation_mae
- 输出：
  - strength_response_curve.png
  - preservation_curve.png
  - csv：sample_id,effect_family,direction,strength_label,actual_magnitude,preservation_mae

实验 3：消融
1. w/o projector
2. w/o task_id
3. w/o modulation injection
4. freeze vs partial unfreeze

【测试要求】
1. zero-init fallback test
   - 不传 strength 与传 strength 但 zero-init 时输出必须几乎一致
2. weak/medium/strong forward smoke test
   - shape 一致
   - 无 NaN
   - 前向不报错

【交付要求】
1. 修改主干模型代码
2. 新增 conditioning 模块代码
3. 修改 dataset / dataloader
4. 修改 train / eval / inference 脚本
5. 新增：
   - tools/eval_strength_control.py
   - tools/plot_strength_response.py
6. 更新 README，给出训练和推理命令
7. 输出实验结果表、曲线图、csv 统计文件

【实现约束】
- 保持 backward compatibility
- 代码必须可运行
- 对关键修改加注释
- 若仓库中文件名略有差异，请修改同职责模块
- 当前目标是“先跑通并验证可行有效”，不要扩展到蒸馏体系、Forecast Revision 或连续 control
```


