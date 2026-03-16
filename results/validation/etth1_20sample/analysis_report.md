# BetterTSE 时序编辑 Pipeline 问题分析报告

**数据来源**：20 个样本（sid 006–025），运行日志 `run_20260314_110422.log`
**报告日期**：2026-03-14

---

## 执行摘要

本次 20 样本验证揭示了 BetterTSE pipeline 中存在的若干系统性问题。整体 avg t-IoU 仅为 **0.1735**，20 个样本中有 **13 个样本 t-IoU = 0**（完全错位率 65%），说明 LLM 时间定位能力是当前最大瓶颈。工具层面，`hybrid_down` 工具出现负 Editability，编辑方向存在根本性错误。Preservability 整体较好（均值 0.9497），但个别样本出现严重退化。

---

## 一、最核心问题汇总（优先级排序）

### P0（阻断性问题）

| 优先级 | 问题 | 影响样本数 | 关键数据 |
|--------|------|-----------|---------|
| P0-1 | LLM 时间区域定位严重偏移，大量完全错位 | 13/20 (65%) | t-IoU=0 的样本：006,007,010,012,013,015,016,017,018,019,023 及 020,017 |
| P0-2 | hybrid_down 工具 Editability 为负值，编辑方向错误 | 2/20 (10%) | sid-020: Edit=-0.970；sid-017: Edit 虽正但区域完全偏 |
| P0-3 | LLM 规划区域系统性过宽（平均是 GT 的 2.02 倍） | 影响全局 | avg LLM duration=32.4，avg GT duration=17.1 |

### P1（显著性能损失）

| 优先级 | 问题 | 影响样本数 | 关键数据 |
|--------|------|-----------|---------|
| P1-1 | heatwave_overload 任务平均 Edit 仅 0.010，几乎无效 | 3/3 该类型 | sid-014: Edit=-0.768；整组均值 0.010 |
| P1-2 | facility_shutdown 任务 t-IoU 最低之一（0.156），4/4 均偏 | 4/4 该类型 | 含 3 个 t-IoU=0 样本 |
| P1-3 | 编辑幅度偏大（avg edit/GT mag ratio=1.266） | 影响全局 | avg GT=3.508，avg Model=2.528，但比率失控 |

### P2（需关注）

| 优先级 | 问题 | 关键数据 |
|--------|------|---------|
| P2-1 | 部分样本 Preservability 低（<0.90） | sid-020: Pres=0.728；sid-006: Pres=0.850 |
| P2-2 | sensor_offline 仅 1 个样本，统计无意义，需补充测试 | t-IoU=0.000 |

---

## 二、LLM 时间定位问题（t-IoU 低的根本原因）

### 2.1 定位偏移程度

avg pre-t-IoU = **0.1667**，avg t-IoU = **0.1735**，两者接近，说明时间 IoU 低的主要来源是**区域位置偏移**，而非后处理损失。

**完全错位（t-IoU=0）的 13 个样本区域对比：**

```
sid-006: GT=[140,160]  LLM=[70,90]    偏移量: ~70步，错在序列前段
sid-007: GT=[111,128]  LLM=[40,70]    偏移量: ~70步，错在序列前段
sid-010: GT=[122,136]  LLM=[70,85]    偏移量: ~50步，错在序列前段
sid-012: GT=[105,123]  LLM=[50,80]    偏移量: ~55步，错在序列前段
sid-013: GT=[141,163]  LLM=[80,95]    偏移量: ~60步，错在序列前段
sid-015: GT=[117,125]  LLM=[40,70]    偏移量: ~77步，错在序列前段
sid-016: GT=[120,140]  LLM=[80,95]    偏移量: ~40步，错在序列前段
sid-017: GT=[65,80]    LLM=[0,30]     偏移量: ~65步，错在序列起始
sid-018: GT=[91,111]   LLM=[0,49]     偏移量: ~91步，错在序列起始
sid-019: GT=[25,42]    LLM=[80,99]    偏移量: ~55步，错在序列末段
sid-023: GT=[128,150]  LLM=[70,85]    偏移量: ~58步，错在序列前段
```

**规律总结**：
- 约 10/11 个完全错位的样本，LLM 规划的区域集中在 **序列前半段（0–99 步）**，而 GT 事件多位于 **序列后半段（91–163 步）**。
- 例外是 sid-019（GT=[25,42]，LLM=[80,99]），LLM 规划在末段而事件在前段。
- 这表明 LLM 在解析时序描述时存在**系统性的时间锚点偏差**：倾向于将事件定位在"早期"或"中段"，对序列长度的感知不足。

### 2.2 区域过宽问题

在有部分重叠的样本中，过宽问题显著：

```
sid-008: GT=[41,62](21步)  LLM=[20,80](60步)  LLM是GT的2.86倍  t-IoU=0.361
sid-009: GT=[32,43](11步)  LLM=[20,60](40步)  LLM是GT的3.64倍  t-IoU=0.293
sid-024: GT=[45,66](21步)  LLM=[20,80](60步)  LLM是GT的2.86倍  t-IoU=0.361
```

- avg LLM duration = **32.4** vs avg GT duration = **17.1**，LLM/GT 比率 = **2.02**。
- 即使 LLM 位置方向正确，过宽的区域导致大量"非事件区域"被纳入编辑范围，压低了 t-IoU 的上限。
- 即使中心对齐，宽度比为 2 倍时，理论 IoU 上限约 **0.50**，这解释了为何即便是"最优"样本 t-IoU 也只到 0.561（sid-022）。

### 2.3 根本原因分析

1. **Prompt 缺乏序列长度上下文**：LLM 在规划区域时，未被告知输入序列的总长度，导致无法进行相对位置的准确推理。
2. **时间描述歧义性**：任务描述（如"device_switch 发生在时序末段"）缺乏精确的步数锚点，LLM 只能靠语义模糊推断。
3. **区域边界倾向"整数/整十"**：LLM 规划的区域边界高度集中于 0, 20, 30, 40, 50, 60, 70, 80, 95, 99 等整十数，显示出 LLM 对边界的"取整偏好"，而非基于数据内容判断。
4. **缺乏区域宽度约束**：没有机制限制 LLM 输出的区域不能超过某个最大宽度（如 GT 均值的 1.5 倍）。

---

## 三、工具选择与幅度问题

### 3.1 hybrid_down 工具的负 Editability 问题

**受影响样本：**

| sid | task | GT_reg | LLM_reg | Edit | Pres |
|-----|------|--------|---------|------|------|
| 020 | facility_shutdown | [72,87] | [50,80] | **-0.970** | 0.728 |
| 017 | facility_shutdown | [65,80] | [0,30] | +0.767 | 0.873 |

- **sid-020** 是最严重的负 Editability 案例（-0.970），且 Preservability 仅 0.728，是全样本中最差的综合表现。
- **分析**：`hybrid_down` 工具设计用于模拟"向下跳变/下行趋势"类型的事件（如设备关停导致数值下降）。但在 sid-020 中，编辑效果**与期望方向完全相反**（负值）。

**可能原因：**
1. `hybrid_down` 的编辑方向硬编码与具体 feature（LULL）的数值范围不匹配——LULL 本身可能已处于低值区，继续"下压"导致溢出或方向倒转。
2. 工具调用时的幅度参数（magnitude）未做 feature 级别的归一化，绝对幅度过大导致数值穿越零点或边界。
3. LLM 在选择工具时可能误判了事件方向：facility_shutdown 应该是"关停"→数值下降，但工具参数配置错误导致反向。

### 3.2 hybrid_up 工具的 Editability 偏低

| sid | task | t-IoU | Edit |
|-----|------|-------|------|
| 006 | heatwave_overload | 0.000 | 0.212 |
| 014 | heatwave_overload | 0.340 | **-0.768** |
| 021 | heatwave_overload | 0.516 | 0.586 |
| 023 | market_trend | 0.000 | 0.885 |

- `hybrid_up` 在 heatwave_overload 任务上表现不稳定：sid-021（t-IoU=0.516）Edit=0.586 尚可，但 sid-014（t-IoU=0.340）Edit=-0.768，方向完全错误。
- sid-014 的 LLM_reg=[50,99]，几乎覆盖整个序列后半段，严重过宽，这可能导致 TEdit 模型在大范围编辑时失去方向感。
- 组均值：hybrid_up（n=4）Edit=0.229，拉低主因是 sid-014 的 -0.768。

### 3.3 ensemble_smooth 工具表现

- `ensemble_smooth`（n=14）avg Edit=0.702，是三个工具中最稳定的。
- 但其 avg t-IoU=0.170，定位问题同样存在。
- 在 t-IoU=0 的情况下依然能取得较高 Edit（如 sid-015: t-IoU=0, Edit=0.909），说明 **TEdit 模型的编辑能力本身是有效的**，核心瓶颈确实在 LLM 定位层。

### 3.4 编辑幅度偏大

- avg GT injection magnitude = **3.508**，avg Model edit magnitude = **2.528**，比率 = **1.266**。
- 模型编辑幅度系统性偏大（超出 GT 约 26.6%），可能导致过度编辑，在边界区域产生人工痕迹。
- 对于 hybrid_down/hybrid_up 这类方向性强的工具，幅度过大会放大方向错误的副作用。

---

## 四、Preservability 异常分析

### 4.1 低 Preservability 样本汇总

| sid | feat | task | tool | LLM_reg | Pres | 根因 |
|-----|------|------|------|---------|------|------|
| 020 | LULL | facility_shutdown | hybrid_down | [50,80] | **0.728** | 工具方向错误+幅度过大，编辑区域外溢 |
| 006 | MULL | heatwave_overload | hybrid_up | [70,90] | 0.850 | 完全错位，"编辑区域"实为非事件区 |
| 017 | OT | facility_shutdown | hybrid_down | [0,30] | 0.873 | 完全错位，在非事件区强制施加下行编辑 |
| 014 | HUFL | heatwave_overload | hybrid_up | [50,99] | 0.891 | LLM 区域过宽（49步），影响大量非事件区 |
| 018 | LULL | device_switch | ensemble_smooth | [0,49] | 0.901 | 完全错位，在序列起始段编辑 |

### 4.2 Preservability 异常的两类模式

**模式 A：因区域完全错位导致的保留性下降**
- 当 LLM 规划区域与 GT 完全不重叠时，TEdit 对"错误区域"的编辑必然破坏原始信号。
- sid-006（Pres=0.850）、sid-017（Pres=0.873）、sid-018（Pres=0.901）均属此类。
- 这些样本的 Edit 分数仍然较高（0.212–0.767），说明 TEdit 在错误区域上也"完成"了某种编辑，但代价是背景保真度受损。

**模式 B：因工具本身问题导致的保留性崩溃**
- sid-020（Pres=0.728）：hybrid_down 以 -0.970 的负方向强力编辑，导致编辑效果扩散至边界外，严重破坏非编辑区域。
- 这是 Preservability 最低的样本，且 Edit 也接近 -1，是双重失败。

### 4.3 背景保真度整体通过（20/20 PASS）与局部异常的矛盾

整体背景保真度 PASS 率为 20/20，但个别样本（sid-020: Pres=0.728）表明"PASS"的阈值设置可能过于宽松，或评估方法需要细化（区分"轻度破坏"和"严重破坏"）。

---

## 五、改进方案

### 5.1 改进 LLM 时间定位（针对 P0-1、P0-3）

#### 方案 A：在 Prompt 中注入序列元信息（立竿见影）

在发送给 LLM 的 Prompt 中明确包含：
```
序列总长度: 168 步
已知统计特征: 事件通常持续 10-25 步
当前时序数据的粗略分段描述（前段/中段/后段的统计摘要）
```
这解决了 LLM 无法感知"序列尺度"的核心问题。

#### 方案 B：引入区域宽度硬约束

在 LLM 输出解析阶段，增加后处理规则：
```python
max_region_width = 25  # 基于 GT 均值 17.1 的 1.5 倍
if (end - start) > max_region_width:
    center = (start + end) // 2
    start = center - max_region_width // 2
    end = center + max_region_width // 2
```
这可将 LLM/GT 宽度比从 2.02 压降到 1.5 以内，理论上将 t-IoU 上限从 ~0.5 提升到 ~0.67。

#### 方案 C：Two-stage 定位策略

第一阶段：让 LLM 判断事件发生在序列的哪个**相对位置**（前 1/3、中 1/3、后 1/3）；
第二阶段：在该粗定位范围内，用时序异常检测算法（如 BOCPD、Z-score 滑动窗口）精确确定边界。

这将 LLM 的角色从"精确定位"降级为"粗定位+语义理解"，符合其擅长语义、不擅长精确数值的特性。

#### 方案 D：Few-shot 示例优化

在 Prompt 的 few-shot 示例中，增加"事件在序列后半段"的正面案例，纠正 LLM 倾向于在"前段"定位的系统偏差。

---

### 5.2 修复 hybrid_down 工具的方向错误（针对 P0-2）

#### 方案 A：工具调用前的 feature 方向检测

在调用 `hybrid_down` 前，检测目标 feature 在 LLM 规划区域内的数值范围与均值：
```python
region_mean = ts[start:end].mean()
global_mean = ts.mean()
if region_mean < global_mean * 0.5:
    # feature 已处于低值，hybrid_down 可能导致方向反转
    # 改用 ensemble_smooth 或降低 magnitude
    magnitude = magnitude * 0.3
```

#### 方案 B：按 feature 归一化 magnitude

当前 avg edit/GT magnitude ratio = 1.266，需将幅度参数标准化为 feature 标准差的倍数，而非绝对值：
```python
feature_std = ts.std()
normalized_magnitude = target_magnitude / feature_std
```

#### 方案 C：编辑后方向验证

在 TEdit 执行编辑后，立即计算编辑区域的变化方向是否与预期一致。如果 Editability < 0，触发回退机制（rollback），使用 ensemble_smooth 替代。

---

### 5.3 改善 heatwave_overload 任务（针对 P1-1）

- heatwave_overload 的 avg Edit = **0.010**，近乎无效，是任务类型中最差的。
- 该任务使用 `hybrid_up` 工具，但 3 个样本中有 1 个负 Edit（sid-014=-0.768），拉低均值。
- **建议**：检查 heatwave_overload 的 GT 注入逻辑——是否存在 GT 事件方向定义不一致的问题（部分样本"热浪过载"表现为上升，部分为震荡），需对该任务类型进行 GT 审核。

---

### 5.4 提升 Preservability（针对 P2-1）

#### 方案 A：边界平滑处理

在 TEdit 编辑完成后，对编辑区域的两端（各 3–5 步）施加线性混合（crossfade）：
```python
blend_width = 4
for i in range(blend_width):
    alpha = i / blend_width
    ts_edited[start + i] = alpha * ts_edited[start + i] + (1 - alpha) * ts_original[start + i]
    ts_edited[end - i] = alpha * ts_edited[end - i] + (1 - alpha) * ts_original[end - i]
```
这可防止编辑区域边界的突变扩散到非编辑区域。

#### 方案 B：Preservability 实时监控与告警

在 pipeline 运行中，当 Pres < 0.90 时输出警告，并记录具体的 MSE_preserve 分布，便于定位高破坏区域。

---

### 5.5 数据集与评估层面的改进

1. **补充 sensor_offline 样本**：当前仅 1 个样本（sid-019），无法得出统计结论，建议增加至 ≥5 个。
2. **背景保真度阈值收紧**：建议将 PASS 判断阈值从当前设置收紧，至少对 Pres < 0.80 的样本标记为 FAIL，以避免 20/20 PASS 的假乐观。
3. **区分定位误差与编辑误差**：建议增加一个"假设定位完美时的 Edit 分数"指标（即在 GT 区域内强制运行 TEdit），以分离"LLM 定位失误"与"TEdit 编辑能力不足"两类问题的贡献。

---

## 六、问题优先级与预期收益总结

| 改进项 | 预期 t-IoU 提升 | 预期 Edit 提升 | 实施难度 |
|--------|----------------|---------------|---------|
| Prompt 注入序列元信息 | +0.15~0.25 | - | 低 |
| 区域宽度硬约束 | +0.05~0.10 | - | 低 |
| hybrid_down 方向验证+rollback | - | +0.3~0.5（针对失效样本）| 中 |
| feature 归一化 magnitude | - | +0.1~0.2 | 中 |
| Two-stage 定位策略 | +0.20~0.35 | - | 高 |
| 边界平滑处理 | - | - | +0.05~0.10 Pres | 低 |

**建议优先实施**：Prompt 元信息注入 + 区域宽度约束 + hybrid_down rollback，这三项改动代码量小、风险低，预计可将 avg t-IoU 从 0.1735 提升至 0.30+ 水平，并消除 hybrid_down 的负 Editability 问题。

---

*报告生成于 2026-03-14，基于 20 个样本的验证数据。*
