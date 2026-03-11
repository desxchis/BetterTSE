# BetterTSE 评估指标说明文档

## 一、评估指标体系概述

BetterTSE 采用三维度评估体系，全面衡量时间序列编辑模型的性能：

1. **时间边界定位** - t-IoU (Temporal Intersection over Union)
2. **编辑有效性** - Editability MSE/MAE
3. **全局保留度** - Preservability MSE/MAE

---

## 二、核心指标详解

### 2.1 时间边界定位指标：t-IoU

#### 评估目的
评估 LLM 对模糊指令的时间推理能力（即模型能否根据"下午那会儿"或"突然断电时"找到正确的起止时间步）。

#### 计算原理
t-IoU 是计算机视觉中目标检测 IoU 在时间序列（一维）上的变体。

设：
- 真实注入变化的时间区间为 $T_{gt} = [start_{gt}, end_{gt}]$
- LLM 根据模糊指令预测出的时间区间为 $T_{pred} = [start_{pred}, end_{pred}]$

#### 计算公式

$$tIoU = \frac{|T_{pred} \cap T_{gt}|}{|T_{pred} \cup T_{gt}|}$$

即：两个区间的交集长度，除以两个区间的并集长度。

#### 代码实现

```python
def compute_t_iou(pred_start: int, pred_end: int, gt_start: int, gt_end: int) -> float:
    intersection = max(0, min(pred_end, gt_end) - max(pred_start, gt_start) + 1)
    union = max(pred_end, gt_end) - min(pred_start, gt_start) + 1
    return intersection / union if union > 0 else 0.0
```

#### 解读
| t-IoU 值 | 含义 |
|----------|------|
| 0.0 | LLM 完全找错了位置 |
| 0.5 | 预测区间与真实区间有部分重叠 |
| 1.0 | LLM 的理解与真实注入窗口分毫不差 |

---

### 2.2 编辑有效性指标：Editability MSE / MAE

#### 评估目的
评估底层时序模型在目标编辑区域内生成指定物理变化的能力（即"该变的地方变到位了吗"）。

#### 计算原理
仅在真实掩码（Ground Truth Mask）为 1 的区间内计算预测值与理想目标值（Target TS）的误差。

#### 计算公式

**MSE (Mean Squared Error):**
$$MSE_{edit} = \frac{1}{N_{edit}} \sum_{i \in T_{gt}} (y_i - \hat{y}_i)^2$$

**MAE (Mean Absolute Error):**
$$MAE_{edit} = \frac{1}{N_{edit}} \sum_{i \in T_{gt}} |y_i - \hat{y}_i|$$

其中：
- $N_{edit}$ 是编辑区间的长度
- $y_i$ 是理想序列 Target TS 的值
- $\hat{y}_i$ 是模型实际生成的序列值

#### 代码实现

```python
def compute_editability_metrics(pred_ts: np.ndarray, target_ts: np.ndarray, mask_gt: np.ndarray) -> Dict[str, float]:
    edit_indices = mask_gt == 1
    if edit_indices.sum() == 0:
        return {"mse": float('inf'), "mae": float('inf')}
    
    pred_edit = pred_ts[edit_indices]
    target_edit = target_ts[edit_indices]
    
    mse = np.mean((pred_edit - target_edit) ** 2)
    mae = np.mean(np.abs(pred_edit - target_edit))
    
    return {"mse": mse, "mae": mae}
```

#### 解读
- 误差越小，说明模型越完美地执行了编辑指令
- 例如"负荷激增 3 倍"或"传感器归零"等操作

---

### 2.3 全局保留度指标：Preservability MSE / MAE

#### 评估目的
评估模型在非编辑区域的稳定性（即"不该变的地方是不是原封不动"）。

这对于 ETTh1 数据集中那些小数点后多位的高精度连续浮点数至关重要。

#### 计算原理
仅在真实掩码（Ground Truth Mask）为 0 的区间内计算误差。

#### 计算公式

$$MSE_{preserve} = \frac{1}{N_{preserve}} \sum_{i \notin T_{gt}} (x_i - \hat{y}_i)^2$$

$$MAE_{preserve} = \frac{1}{N_{preserve}} \sum_{i \notin T_{gt}} |x_i - \hat{y}_i|$$

其中：
- $N_{preserve}$ 是保留区间的长度
- $x_i$ 是原始序列 Base TS 的值

#### 代码实现

```python
def compute_preservability_metrics(pred_ts: np.ndarray, base_ts: np.ndarray, mask_gt: np.ndarray) -> Dict[str, float]:
    preserve_indices = mask_gt == 0
    if preserve_indices.sum() == 0:
        return {"mse": float('inf'), "mae": float('inf')}
    
    pred_preserve = pred_ts[preserve_indices]
    base_preserve = base_ts[preserve_indices]
    
    mse = np.mean((pred_preserve - base_preserve) ** 2)
    mae = np.mean(np.abs(pred_preserve - base_preserve))
    
    return {"mse": mse, "mae": mae}
```

#### 解读
- 在理想的编辑框架下，这个值应该无限趋近于 0
- 证明模型没有产生"幻觉"去篡改背景数据

---

## 三、综合评估流程

### 3.1 评估流程图

```
输入: pred_ts (模型预测), base_ts (原序列), target_ts (理想序列), 
      pred_start/end (预测区间), gt_start/end (真实区间), mask_gt (真实掩码)

Step 1: 计算 t-IoU
        t_iou = compute_t_iou(pred_start, pred_end, gt_start, gt_end)

Step 2: 计算 Editability
        edit_metrics = compute_editability_metrics(pred_ts, target_ts, mask_gt)

Step 3: 计算 Preservability
        preserve_metrics = compute_preservability_metrics(pred_ts, base_ts, mask_gt)

输出: {
    "t_iou": float,
    "editability_mse": float,
    "editability_mae": float,
    "preservability_mse": float,
    "preservability_mae": float
}
```

### 3.2 指标权重建议

| 指标 | 权重 | 说明 |
|------|------|------|
| t-IoU | 30% | 时间定位准确性 |
| Editability | 40% | 编辑区域执行精度 |
| Preservability | 30% | 背景保留能力 |

---

## 四、注意事项

1. **t-IoU 的局限性**：仅衡量时间区间定位，不衡量数值精度
2. **MSE vs MAE**：MSE 对大误差更敏感，MAE 对异常值更鲁棒
3. **数据精度**：ETTh1 数据为浮点数，需注意数值稳定性

---

*文档生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
*BetterTSE Team*
