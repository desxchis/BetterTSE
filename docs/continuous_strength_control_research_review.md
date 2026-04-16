# BetterTSE 连续强度控制调研与科研 Debug 复盘

## 1. 目标与当前问题

你们现在想做的事情，本质上是把 KontinuousKontext / ContinuousStrengthControl 的核心思想迁移到 BetterTSE / TEdit：

> 在 instruction-based editing 里，把 `what to edit` 和 `how much to edit` 分开建模，并让 `strength` 成为一个真正进入模型内部、可连续调节、可验证单调响应的控制变量。

这个方向本身是对的。

但从当前仓库实现和已有现象看，问题不是“有没有加 strength 通道”，而是：

> **虽然代码里已经加了 strength projector / modulation / monotonic diagnostics，但模型输出层面仍然没有稳定体现出可感知、可校准、可单调的强度差异。**

这说明现在的失败不是单点 bug，更像是一个**系统性失配**。

---

## 2. KontinuousKontext 的真正关键点是什么

根据公开 repo 与训练脚本可见信息，KontinuousKontext 成功的关键，不是“多加了一个数字”，而是下面这 4 件事同时成立。

### 2.1 它把 strength 作为独立控制通道，而不是 prompt 修饰词

它不是仅靠：

- `slightly`
- `more`
- `very strong`

这种 prompt wording 去表达强度。

它做的是：

- 明确传入 `slider_value`
- 用小型 projector 把 scalar 映射成 embedding
- 把该 embedding 注入 backbone 内部 modulation path

这意味着 strength 不是语言描述的一部分，而是模型显式接收的内部控制变量。

### 2.2 它的 strength 注入位置是“模型内部调制”，不是后处理数值缩放

公开实现最关键的一点是：

- `slider_value -> SliderProjector -> modulation_embeddings`
- 再进入 transformer / denoiser

这是重要分水岭。

因为如果只是：

- 在输出后做一点比例缩放
- 在 tool 参数上做 heuristic 映射
- 在 prompt 中写“更强一点”

那么 strength 往往不会成为生成过程中的主导因素。

KontinuousKontext 的设计意味着：

> **strength 直接参与每一步 denoising / editing，而不是事后微调。**

### 2.3 它训练的是 source-instruction-strength-target 四元组，而不是散乱标签

它的训练单位本质上是：

- source
- instruction
- strength
- target

更重要的是，strength 不是任意抽象标签，而是和一条可排序 edit trajectory 对应。

也就是说，模型不是在学：

- weak/medium/strong 三分类词汇

而是在学：

- 同一个语义编辑方向下，输出如何沿着 strength 轴连续变化。

### 2.4 它解决的是“strength 被主语义淹没”的问题

在普通 instruction editing 里，模型最容易学会的是：

- 改不改
- 改什么属性

最难学会的是：

- 改多少

KontinuousKontext 的价值就在于，它把这个困难点拆出来，给了：

- 独立通道
- 独立数据组织方式
- 独立的强度调制入口

所以它解决的不是一般 controllability，而是：

> **语义编辑已成立后，如何让幅度差异真正进入生成路径。**

这和 BetterTSE 当前卡点高度同构。

---

## 3. 你们当前实现里，已经做对了哪些事

从 TEdit 当前代码看，你们其实已经完成了 KontinuousKontext 迁移里相当大的一半。

## 3.1 已经不是“只有 prompt strength”

当前你们并不是只在 prompt 里写 `weak / medium / strong`。

已经有：

- `StrengthProjector`
- `strength_modulation`
- `strength_input_projection`
- `strength_label`
- `task_id`
- `instruction_text`
- layer-level diagnostics
- monotonic / gain-match / background preservation losses

这比很多“连续控制”工作都更进一步。

### 相关代码位置

| 模块 | 位置 | 作用 |
|---|---|---|
| `StrengthProjector` | `TEdit-main/models/conditioning/numeric_projector.py:64` | strength / task / text -> conditioning vector |
| diffusion 注入 | `TEdit-main/models/diffusion/diff_csdi_multipatch.py:443-540` | 构造 `strength_cond` 并注入 residual layers |
| residual modulation | `TEdit-main/models/diffusion/diff_csdi_multipatch.py:213-221` | `base_modulation + strength residual modulation` |
| finetune strength losses | `TEdit-main/models/conditional_generator.py:293-379` | monotonic / gain match / locality supervision |
| diagnostics | `TEdit-main/train/finetuner.py:168-202` | projector / modulation / generator diagnostics |

## 3.2 已经意识到“内部注入”是必要的

当前 `Diff_CSDI_MultiPatch_Parallel` 里已经不是简单拼接标签，而是：

- `strength_projector(strength_label, task_id, text_context)`
- residual layers 接收 `strength_cond`
- `strength_modulation(strength_cond)` 产生 `delta_gamma / delta_beta`
- 最后对 backbone modulation 做 residual 调制

这一点和 KontinuousKontext 的思路是同方向的。

## 3.3 已经开始做“单调性”而不只是分类准确率

`ConditionalGenerator._strength_supervision_loss()` 里已经有：

- edit-region loss
- background loss
- monotonic loss
- gain-match loss

这说明你们已经意识到：

> 强度控制不是一个 label classification 问题，而是一个**响应曲线**问题。

这是正确方向。

---

## 4. 为什么你们现在还是出不来强度差异

这里是最核心部分。

我认为当前失败不是一个原因，而是至少 6 个原因叠加。

## 4.1 最致命的问题：strength supervision 的“真值”太弱

当前 synthetic finetune 数据里的 `strength_label`，主要来自属性差异的离散化：

- `weak / medium / strong`
- 基于 `src_attrs -> tgt_attrs` 的离散 bucket

相关代码：

- `TEdit-main/data/synthetic_finetune.py:22-35`
- `TEdit-main/data/synthetic_finetune.py:57-70`

这里的根本问题是：

> **你们的 strength label 更像“属性变化程度的粗分类”，而不是“编辑输出轨迹上的连续物理幅度”。**

这会直接带来两个后果：

### 后果 A：标签不对应真实编辑空间中的等距变化

例如：

- trend weak
- trend medium
- trend strong

在 attr 空间里可能只是几档离散跳变；
但在输出时间序列空间里：

- 并不一定对应平滑、单调、可区分的幅度差
- 甚至可能不同 task 下含义完全不同

### 后果 B：同一个 strength_label 跨 task 含义不一致

比如：

- seasonality 的 strong
- trend 的 strong
- volatility 的 strong

它们在实际输出空间中的能量、峰值、面积、频域变化都不是一个量纲。

所以模型看到的是：

- 同样 label=2
- 却对应完全不同几何含义

这会把 strength 通道训练成一个**模糊任务标签补丁**，而不是稳定幅度轴。

### 结论

KontinuousKontext 成功的核心之一，是 strength 对应同一 edit direction 下的连续轨迹。

而你们当前的 strength supervision 还更像：

> **粗任务标签 + 粗变化档位**

这会从源头上削弱可学性。

---

## 4.2 strength 通道初始化和主干竞争关系，极可能让它长期处于“可有可无”

当前几个关键 strength 模块都做了零初始化：

- `StrengthProjector.mlp[-1]` 全零
- `ResidualBlock.strength_modulation[-1]` 全零
- `strength_input_projection` 全零

相关代码：

- `numeric_projector.py:102-113`
- `diff_csdi_multipatch.py:111-117`
- `diff_csdi_multipatch.py:462-464`

零初始化本身不是错，很多 controllable generation 都会这么做，优点是稳定。

但在你们这里，它可能叠加了一个坏现象：

### 主干已经足够强，strength 支路没有被迫承担必要信息

如果 backbone 已经能根据：

- `src_x`
- `src_attr_emb`
- `tgt_attr_emb`
- `instruction_text`

大致完成编辑任务，
那么 strength 支路的梯度就会非常弱。

模型最容易学到的是：

- 先把 edit 做出来
- 至于 weak/medium/strong 的细分，能分一点算一点
- 分不出来也不影响主损失太多

这会导致：

> strength 模块在代码上存在，但在优化上始终是次要分支。

尤其当主损失仍以 diffusion denoising 为主时，这个问题更明显。

---

## 4.3 你们当前 strength 是 3 档离散 label，不是真连续控制

这是和 KontinuousKontext 的结构性差异。

KontinuousKontext 的核心是 continuous slider。

而你们当前主要还是：

- `strength_label ∈ {0,1,2}`
- embedding lookup
- 再 MLP 投影

这当然能做 ordinal control，但它很难自然逼出真正的连续响应。

### 这会导致什么

模型很容易学成：

- 3 个离散模式
- 而不是一条连续强度轴

所以你们现在看到的现象很可能是：

- `weak` 和 `medium` 很像
- `strong` 有时突然跳得很大
- 或三者几乎重叠

这正是“离散条件被 backbone 吸收/忽略”时的典型表现。

### 更严重的一点

你们现在希望论文故事接近 continuous strength control，
但训练实体却还主要是 3-bin categorical control。

这会导致方法叙事和机制本体之间出现错位。

---

## 4.4 当前 monotonic loss 约束的是“平均 edit gain”，不是“语义正确且局部保真的强度轨迹”

当前 `_strength_supervision_loss()` 做的核心量之一是：

- `gains = mean(|pred - src| in edit_mask)`
- `target_gains = mean(|tgt - src| in edit_mask)`
- 再做 monotonic 和 gain match

相关代码：

- `conditional_generator.py:312-345`

这个设计有价值，但也有明显局限。

### 问题 1：edit gain 太粗

`|pred - src|` 的均值变大，不等于：

- 方向对了
- 形状对了
- duration 对了
- 强度语义对了

模型完全可以通过以下方式“作弊”满足更大 gain：

- 扩大错误区域的改动
- 边界模糊外溢
- 把噪声放大而不是把目标属性放大
- 全局轻微漂移

### 问题 2：monotonic 约束的是量值次序，不是控制轴一致性

即便满足：

- weak < medium < strong

也不代表这条轴在不同 family / different prompt / different source pattern 下有同样含义。

所以你们可能学到了：

- “更强 = 平均改动更多”

而不是：

- “更强 = 同一语义编辑方向上的更大幅度版本”

这两者差别非常大。

---

## 4.5 你们的数据组织并不完全符合“同源 edit family trajectory”训练范式

KontinuousKontext 类方法最依赖的一点是：

> 同一个 source、同一个 instruction semantics、同一 edit family 下，存在多强度版本可排序样本。

你们当前已经有 `family_sizes`、`family_valid`、`family_order_valid` 等机制，说明你们在朝这个方向做。

但从现有结构判断，问题可能在于：

### 你们的 family 仍然不是足够严格的“同一语义轨迹”

如果 family 内部只是：

- 同一大类 task
- 但具体 target attr / target realization 并非严格同一路径

那 monotonic supervision 就会被污染。

因为模型看到的不是：

- 同一编辑方向的 3 个强度版本

而可能是：

- 相近但不完全同一目标的 3 个样本

这会让强度轴变得不干净。

### 结果

模型学到的可能是：

- 弱中强对应某种数据分组模式

而不是：

- 输出幅度的连续有序变化。

---

## 4.6 当前 BetterTSE 最大结构问题：强度含义被“多条控制链”分流了

这点非常关键，也是我认为你们和 KontinuousKontext 最大的差异。

你们当前仓库里，`how much` 并不是单一路径决定的，而是同时散落在：

- prompt 中的 strength words
- heuristic parser 中的强弱词 (`modules/llm.py`)
- forecast revision 的 `edit_spec`
- tool-level 参数投影
- teacher / rule / learned calibrator
- executor 参数（如 shift / amp / duration / recovery）
- TEdit 模型内部 strength projector

这意味着现在系统里至少存在两类强度来源：

### 路径 A：模型内部 strength conditioning

- TEdit backbone 接收 `strength_label`
- 希望输出不同强度轨迹

### 路径 B：外部参数/规则层强度控制

- `edit_spec`
- `project_edit_spec_to_params`
- rule-based calibration
- learned calibration
- reliability gate

在 pure editing / forecast revision 两条线混合存在时，很容易出现：

> **真正决定最终幅度的是外部参数链，而不是模型内部 strength 通道。**

一旦如此，模型内部强度就会失去可识别性：

- backbone 不需要认真学
- 学了也被外部规则覆盖
- 最后看输出时“强度差异不明显”

这和 KontinuousKontext 完全不同。

KontinuousKontext 的 slider 是核心控制变量之一，路径更单一。

你们这里 strength 目前更像：

- 一个新加支路
- 挂在一个本来就有很多控制层的系统上

所以它特别容易被淹没。

---

## 5. 当前代码里暴露出的几个高概率具体症状

结合你们代码结构，我认为现在很可能出现以下症状。

| 症状 | 高概率根因 |
|---|---|
| 弱中强输出几乎重叠 | projector / modulation 梯度太弱，或 supervision 不够区分 |
| strong 偶尔突然过大但不稳定 | 离散 label 触发模式切换，而不是连续幅度控制 |
| gain 能拉开，但 shape/duration 不同步 | 当前 loss 更偏 edit energy，而不是结构一致控制 |
| 不同任务下同一强度表现很不一致 | cross-family 强度语义不统一 |
| diagnostics 显示有 modulation 差异，但可视化输出差别不大 | 内部 modulation 变化没有有效穿透到最终 edit manifold |
| Weather 上还行，真实数据上差 | synthetic strength 标签比真实场景更干净，domain gap 放大 |

---

## 6. 为什么“简单改一下 TEdit 模型部分”一直效果不好

一句话说：

> **因为你们现在遇到的不是“模型少了一层强度输入”，而是“训练目标、数据组织、强度定义、控制链归因”四件事没有同时对齐。**

如果只是简单改模型，很容易遇到以下科研错觉：

### 错觉 1：有 projector = 已经做了 continuous control

不是。

有 projector 只说明：

- 你给模型留了接口

不说明：

- strength 真的是强监督信号
- 模型必须依赖它
- 输出真的沿 strength 轴变化

### 错觉 2：monotonic loss = 一定能学出强度

不是。

如果真值轨迹不干净，monotonic 只会逼模型学某种平均增益排序，
不一定学出语义一致的强度控制。

### 错觉 3：diagnostics 里有 pairwise L2 = 强度注入有效

也不是。

这最多说明：

- projector 输出不同
- modulation 参数不同

但不等于：

- 最终编辑结果真的不同
- 且这种不同是对的、可控的、单调的

### 错觉 4：只要把 backbone freeze / lr scale 调一调就会自然出来

这只能影响“有没有学到点东西”，
不能解决“学的到底是不是正确强度轴”这个更核心的问题。

---

## 7. 我对当前问题的核心诊断

如果要把当前问题压缩成一句科研 debug 结论，我会写成：

> **你们当前失败的根因，不在于 strength 没有进入 TEdit，而在于进入模型的 strength 信号本身还不是一个干净、同语义、可排序、对最终输出具有唯一归因权的控制变量。**

拆开就是四句话：

1. **信号不干净**：strength label 仍偏离散粗糙。
2. **语义不统一**：不同 edit family 下同一 strength 不同量纲。
3. **归因不唯一**：最终幅度还受外部规则/投影/工具链强烈影响。
4. **监督不够强**：loss 约束的是 edit gain，不是完整强度轨迹一致性。

---

## 8. 下一步该怎么做：最推荐的科研路线

这里我不写大而全，只给最值得做的路线排序。

## 路线 1：先把“强度控制”缩回 pure editing 受控场景，做干净闭环

这是我最推荐的第一优先级。

### 目标

先别让 forecast revision / edit_spec / rule calibrator / teacher search 一起掺进来。

先在 pure editing controlled regime 下证明：

- 同 source
- 同 region
- 同 edit family
- 同 instruction semantics
- 只改变 strength scalar
- 输出幅度能单调、局部、可视化明显变化

### 为什么必须先这么做

因为现在你们同时在 debug：

- prompt parsing
- region localization
- intent mapping
- calibration spec
- tool params
- editor backbone
- downstream forecast metrics

这样没法定位原因。

KontinuousKontext 风格的工作，第一步必须先证明：

> **在纯编辑层面，strength 通道本身是 alive 的。**

### 这个阶段该验证什么

不是先看 downstream gain，
而是看：

- peak delta vs strength
- signed area vs strength
- local energy vs strength
- outside-region leakage vs strength
- family monotonic hit rate
- 同一 source 样例的强度 sweep 可视化

也就是先证明“control axis exists”。

---

## 路线 2：把 strength 从 3 档类别改成 family-normalized continuous scalar

这是第二关键步。

### 当前问题

现在的 `0/1/2` 过于离散。

### 建议方向

对每个 edit family 单独定义 normalized strength 标度，例如：

| family | strength 的归一化含义 |
|---|---|
| trend/level | normalized signed area 或 normalized peak shift |
| seasonality | amplitude ratio |
| volatility | residual std ratio |
| impulse | peak amplitude / local energy |
| shutdown | floor drop ratio + duration ratio |

然后统一映射到 `[0,1]` 或某个 family-specific continuous range。

### 意义

这样你们学到的就不再是：

- 类别标签 weak/medium/strong

而是：

- 同一 family 内部真正的幅度轴

这才更接近 KontinuousKontext。

---

## 路线 3：把训练样本重组为“严格 family trajectory”而不是普通 pair

### 关键原则

每个训练 family 应该满足：

- 同一个 `src_x`
- 同一个 `region`
- 同一个 `edit family`
- 同一个方向 / shape / duration
- 只改变 strength

也就是说，train unit 不该只是 pair，
而该是 ordered trajectory / family pack。

### 为什么这一步重要

因为 monotonic / ranking / consistency 只有在 trajectory truly aligned 时才有意义。

否则模型会学偏。

### 和你们现有机制关系

你们已经有 `family_sizes` / `family_valid` / `family_order_valid`。

这很好，但建议把这件事从“辅助 batch 结构”提升成：

> **数据组织的第一原则。**

---

## 路线 4：明确谁才是最终幅度控制的 source of truth

这是和 forecast revision 结合时最重要的一点。

如果你们未来要把这条线接回 BetterTSE 主故事，必须先定：

### 方案 A：模型内部 strength 为主，外部 calibration 为辅

也就是：

- TEdit 内部学连续强度轴
- `edit_spec` / tool params 只是把 planner 输出映射到 strength target
- 不再让 rule/teacher 主导最终幅度

### 方案 B：外部 calibration 为主，模型内部只学基础 editability

也就是：

- 模型只负责语义编辑
- 幅度主要由 edit_spec / param projection 决定
- strength projector 不再承担论文核心创新

这两条都可以，但不能同时都想要。

因为如果两套链路都想当主控制源，最后就会互相稀释。

### 我的判断

如果你们真的想复现 KontinuousKontext 风格故事，应该选 **A**：

> **把模型内部 strength path 变成主角，外部 calibration 只做目标构造和解释层，而不是替代控制层。**

---

## 9. 一个更具体的三阶段实验建议

## 阶段 I：证明强度轴真的存在（pure editing only）

目标：

- 不看 forecasting
- 不看 teacher-student
- 不看复杂真实文本

只在 controlled synthetic pure editing 上做：

- fixed source + fixed family + strength sweep
- monotonicity curves
- leakage curves
- visualization panels

如果这一步都做不出来，就不要继续往 forecast revision 接。

## 阶段 II：证明强度轴可跨复杂模糊指令被解析

这里再引入你们的创新点：

- complex fuzzy instruction decomposition

但此时不要一上来直接接真实 forecast revision。

而应该先做：

- 指令拆解 -> semantic intent + continuous strength target
- 再看 pure editing backbone 是否 obey

这样你们就能回答：

> 我们不仅能 parse 模糊指令中的强弱语义，而且内部生成模型真的沿该强度轴变化。

## 阶段 III：把内部 strength axis 接回 forecast revision

这时再做：

- semantic intent + continuous strength target -> edited forecast
- 在 `oracle_region + oracle_intent` 下先看 magnitude calibration
- 再逐步恢复 full pipeline

这条顺序和当前文档里提出的 calibration reading order 是一致的。

---

## 10. 我最不建议你们现在继续做的事情

### 不建议 1：继续主要靠 teacher search / semi-oracle 堆 stronger labels

这条线可以做 baseline，但不适合当核心解法。

原因你们已经知道：

- self-distillation 风险
- 偏差再编码
- 论文创新点会虚

### 不建议 2：继续只调 projector、lr、freeze 策略 hoping it works

这些可以作为辅助手段，
但如果数据组织和强度定义不改，收益大概率有限。

### 不建议 3：直接在 full forecast revision pipeline 上 debug strength

因为变量太多，问题不可定位。

### 不建议 4：把弱中强 label 的分类/分离当作主要成功标准

你们真正要的不是：

- label separable

而是：

- output response monotone
- local and semantically aligned
- physically interpretable
- robust across families/prompts

---

## 11. 最终判断

### 11.1 对方向本身的判断

你们想借鉴 KontinuousKontext 到 BetterTSE，这个方向是**对的，而且很有研究价值**。

因为它和你们当前卡点完全一致：

- 不是 `what`
- 不是 `where`
- 正是 `how much`

而且还能自然结合你们已有的：

- 复杂模糊指令拆解
- intent/localization decomposition
- calibration benchmark

所以故事是连贯的。

### 11.2 对当前失败原因的判断

当前之所以一直效果不好，不是因为这个方向错了，而是因为：

> **你们现在做的是“把强度通道接进 TEdit”，但还没有同时把 strength 的数据定义、轨迹组织、监督目标、以及系统归因权一起改造成 KontinuousKontext 风格。**

换句话说：

- 架构接口已经有了
- 但训练世界观还没完全切换过去

### 11.3 我认为最关键的一句科研 debug 结论

> **现在最该解决的不是“怎样再改一层模型”，而是“怎样让 strength 变成同一编辑语义轨迹上的唯一主控变量，而不是系统中众多幅度信号之一”。**

只要这件事没做到，模型内部 strength 就会一直像“加了，但没真用上”。

---

## 12. 一句话行动建议

如果现在只能做一件最关键的事，我建议是：

> **先在 pure editing controlled family-trajectory 数据上，构造真正连续且 family-normalized 的 strength 标度，并只验证内部 TEdit backbone 的单调响应；不要一开始就在 full forecast revision pipeline 里调。**

这是最可能把当前困局真正打开的一步。
