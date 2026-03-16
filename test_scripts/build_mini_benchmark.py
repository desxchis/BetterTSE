"""
BetterTSE Mini-Benchmark 生成器
==============================

功能：生成小型测试集（100条样本），包含三元组：
- Base TS (原序列)
- Vague Prompt (模糊编辑指令)  
- Target TS (理想序列)

输出：
- CSV格式测试集
- 可视化图表
- 评估指标说明文档

作者: BetterTSE Team
"""

import os
import sys
import json
import logging
import argparse
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict, field
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.ndimage import gaussian_filter1d

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("MiniBenchmark")


DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("OPENAI_API_KEY", "")


@dataclass
class BenchmarkSample:
    """Mini-Benchmark样本 - 三元组格式"""
    sample_id: str
    dataset_name: str
    target_feature: str
    task_type: str
    gt_start: int
    gt_end: int
    vague_prompt: str
    base_ts: List[float]
    target_ts: List[float]
    mask_gt: List[int]
    injection_config: Dict[str, Any]
    seq_len: int
    timestamp: str = ""


class CSVDataLoader:
    """真实CSV数据加载器"""
    
    DATASET_INFO = {
        "ETTh1": {
            "freq": "hourly",
            "desc": "Electricity Transformer Temperature (hourly)",
            "features_desc": {
                "HUFL": "High Useful Load (高压侧有效负荷)",
                "HULL": "High Useless Load (高压侧无效负荷)",
                "MUFL": "Middle Useful Load (中压侧有效负荷)",
                "MULL": "Middle Useless Load (中压侧无效负荷)",
                "LUFL": "Low Useful Load (低压侧有效负荷)",
                "LULL": "Low Useless Load (低压侧无效负荷)",
                "OT": "Oil Temperature (变压器油温)"
            }
        },
        "ETTm1": {"freq": "15min", "desc": "Electricity Transformer Temperature (15-min)"},
        "Traffic": {"freq": "hourly", "desc": "California Traffic Flow"},
    }
    
    def __init__(self, csv_path: str, dataset_name: str = "ETTh1"):
        self.csv_path = Path(csv_path)
        self.dataset_name = dataset_name
        self.data = None
        self.features = []
        self.feature_to_idx = {}
        self.timestamps = []
        self.feature_descriptions = self.DATASET_INFO.get(dataset_name, {}).get("features_desc", {})
        
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV文件不存在: {csv_path}")
        self._load_data()
        
    def _load_data(self):
        logger.info(f"正在加载数据集: {self.csv_path}")
        self.data = pd.read_csv(self.csv_path)
        logger.info(f"数据形状: {self.data.shape}")
        
        if 'date' in self.data.columns:
            self.timestamps = self.data['date'].tolist()
            self.data = self.data.drop(columns=['date'])
        
        self.features = self.data.columns.tolist()
        self.feature_to_idx = {feat: idx for idx, feat in enumerate(self.features)}
        logger.info(f"特征列表 ({len(self.features)}): {self.features}")
            
    def get_sequence(self, start_idx: int, seq_len: int, feature: str) -> Tuple[np.ndarray, List[str]]:
        """获取指定特征的1D时间序列"""
        end_idx = start_idx + seq_len
        if end_idx > len(self.data):
            end_idx = len(self.data)
            start_idx = max(0, end_idx - seq_len)
            
        feat_idx = self.feature_to_idx[feature]
        sequence = self.data.iloc[start_idx:end_idx, feat_idx].values.astype(np.float64)
        
        timestamps = []
        if self.timestamps:
            timestamps = self.timestamps[start_idx:end_idx]
            
        return sequence, timestamps


class PhysicalInjector(ABC):
    """物理变化注入器基类"""
    
    TASK_TYPES = {
        "multiplier": "heatwave_overload",
        "hard_zero": "facility_shutdown",
        "noise_injection": "sensor_offline",
        "trend_injection": "market_trend",
        "step_change": "device_switch"
    }
    
    def __init__(self, rng: np.random.RandomState):
        self.rng = rng
        
    @abstractmethod
    def inject(self, base_ts: np.ndarray, start_step: int, duration: int) -> Tuple[np.ndarray, np.ndarray, Dict]:
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        pass
    
    def get_task_type(self) -> str:
        return self.TASK_TYPES.get(self.get_name(), "unknown")


class MultiplierInjector(PhysicalInjector):
    """乘法倍数注入器 - 热浪过载"""
    
    def __init__(self, rng: np.random.RandomState, multiplier_range: Tuple[float, float] = (2.0, 4.0)):
        super().__init__(rng)
        self.multiplier_range = multiplier_range
        
    def inject(self, base_ts: np.ndarray, start_step: int, duration: int) -> Tuple[np.ndarray, np.ndarray, Dict]:
        target_ts = base_ts.copy()
        mask_gt = np.zeros(len(base_ts))
        end_step = min(start_step + duration, len(base_ts))
        
        multiplier = self.rng.uniform(*self.multiplier_range)
        target_ts[start_step:end_step] *= multiplier
        mask_gt[start_step:end_step] = 1
        
        config = {
            "injection_type": "multiplier",
            "multiplier": round(multiplier, 2),
            "start_step": start_step,
            "end_step": end_step - 1,
            "duration": duration,
        }
        return target_ts, mask_gt, config
    
    def get_name(self) -> str:
        return "multiplier"


class HardZeroInjector(PhysicalInjector):
    """强制归零注入器 - 设施关闭"""
    
    def inject(self, base_ts: np.ndarray, start_step: int, duration: int) -> Tuple[np.ndarray, np.ndarray, Dict]:
        target_ts = base_ts.copy()
        mask_gt = np.zeros(len(base_ts))
        end_step = min(start_step + duration, len(base_ts))
        
        target_ts[start_step:end_step] = 0.0
        mask_gt[start_step:end_step] = 1
        
        config = {
            "injection_type": "hard_zero",
            "start_step": start_step,
            "end_step": end_step - 1,
            "duration": duration,
        }
        return target_ts, mask_gt, config
    
    def get_name(self) -> str:
        return "hard_zero"


class NoiseInjector(PhysicalInjector):
    """底噪替换注入器 - 传感器离线"""
    
    def __init__(self, rng: np.random.RandomState, noise_level: float = 0.1):
        super().__init__(rng)
        self.noise_level = noise_level
        
    def inject(self, base_ts: np.ndarray, start_step: int, duration: int) -> Tuple[np.ndarray, np.ndarray, Dict]:
        target_ts = base_ts.copy()
        mask_gt = np.zeros(len(base_ts))
        end_step = min(start_step + duration, len(base_ts))
        
        baseline = self.rng.uniform(3.0, 8.0)
        noise = self.rng.normal(loc=baseline, scale=self.noise_level, size=end_step - start_step)
        target_ts[start_step:end_step] = np.abs(noise)
        mask_gt[start_step:end_step] = 1
        
        config = {
            "injection_type": "noise_injection",
            "baseline": round(baseline, 2),
            "noise_level": self.noise_level,
            "start_step": start_step,
            "end_step": end_step - 1,
            "duration": duration,
        }
        return target_ts, mask_gt, config
    
    def get_name(self) -> str:
        return "noise_injection"


class TrendInjector(PhysicalInjector):
    """趋势注入器 - 市场趋势"""
    
    def inject(self, base_ts: np.ndarray, start_step: int, duration: int) -> Tuple[np.ndarray, np.ndarray, Dict]:
        target_ts = base_ts.copy()
        mask_gt = np.zeros(len(base_ts))
        end_step = min(start_step + duration, len(base_ts))
        
        direction = self.rng.choice(["upward", "downward"])
        data_range = np.max(base_ts) - np.min(base_ts)
        slope = data_range * self.rng.uniform(0.1, 0.3) / duration
        
        if direction == "downward":
            slope = -abs(slope)
        else:
            slope = abs(slope)
            
        trend = np.linspace(0, slope * duration, end_step - start_step)
        target_ts[start_step:end_step] += trend
        mask_gt[start_step:end_step] = 1
        
        config = {
            "injection_type": "trend_injection",
            "direction": direction,
            "slope": round(slope, 4),
            "start_step": start_step,
            "end_step": end_step - 1,
            "duration": duration,
        }
        return target_ts, mask_gt, config
    
    def get_name(self) -> str:
        return "trend_injection"


class StepChangeInjector(PhysicalInjector):
    """阶跃变化注入器 - 设备切换"""
    
    def __init__(self, rng: np.random.RandomState, step_range: Tuple[float, float] = (0.3, 0.7)):
        super().__init__(rng)
        self.step_range = step_range
        
    def inject(self, base_ts: np.ndarray, start_step: int, duration: int) -> Tuple[np.ndarray, np.ndarray, Dict]:
        target_ts = base_ts.copy()
        mask_gt = np.zeros(len(base_ts))
        end_step = min(start_step + duration, len(base_ts))
        
        data_range = np.max(base_ts) - np.min(base_ts)
        step_magnitude = data_range * self.rng.uniform(*self.step_range)
        direction = self.rng.choice(["up", "down"])
        
        if direction == "down":
            step_magnitude = -step_magnitude
            
        target_ts[start_step:end_step] += step_magnitude
        mask_gt[start_step:end_step] = 1
        
        config = {
            "injection_type": "step_change",
            "direction": direction,
            "magnitude": round(step_magnitude, 2),
            "start_step": start_step,
            "end_step": end_step - 1,
            "duration": duration,
        }
        return target_ts, mask_gt, config
    
    def get_name(self) -> str:
        return "step_change"


class InjectorFactory:
    """注入器工厂"""
    
    INJECTOR_REGISTRY = {
        "multiplier": MultiplierInjector,
        "hard_zero": HardZeroInjector,
        "noise_injection": NoiseInjector,
        "trend_injection": TrendInjector,
        "step_change": StepChangeInjector,
    }
    
    TASK_TYPES = {
        "multiplier": "heatwave_overload",
        "hard_zero": "facility_shutdown",
        "noise_injection": "sensor_offline",
        "trend_injection": "market_trend",
        "step_change": "device_switch"
    }
    
    CAUSAL_SCENARIOS = {
        "multiplier": ["热浪导致负荷激增", "极端天气影响", "用电高峰期过载"],
        "hard_zero": ["设备维护停机", "紧急断电", "设施关闭"],
        "noise_injection": ["传感器故障", "通信中断", "数据采集异常"],
        "trend_injection": ["市场变化导致趋势", "季节性因素", "供需关系变化"],
        "step_change": ["设备切换", "运行模式改变", "系统重置"],
    }
    
    FEATURE_DESCRIPTIONS = {
        "HUFL": "High Useful Load (高压侧有效负荷)",
        "HULL": "High Useless Load (高压侧无效负荷)",
        "MUFL": "Middle Useful Load (中压侧有效负荷)",
        "MULL": "Middle Useless Load (中压侧无效负荷)",
        "LUFL": "Low Useful Load (低压侧有效负荷)",
        "LULL": "Low Useless Load (低压侧无效负荷)",
        "OT": "Oil Temperature (变压器油温)",
    }
    
    def __init__(self, random_seed: Optional[int] = None):
        self.rng = np.random.RandomState(random_seed)
        
    def create_injector(self, injection_type: Optional[str] = None) -> PhysicalInjector:
        if injection_type is None:
            injection_type = self.rng.choice(list(self.INJECTOR_REGISTRY.keys()))
        injector_class = self.INJECTOR_REGISTRY[injection_type]
        return injector_class(self.rng)
    
    def get_task_type(self, injection_type: str) -> str:
        return self.TASK_TYPES.get(injection_type, "unknown")
    
    def get_causal_scenario(self, injection_type: str) -> str:
        scenarios = self.CAUSAL_SCENARIOS.get(injection_type, ["未知场景"])
        return self.rng.choice(scenarios)
    
    def get_feature_description(self, feature: str) -> str:
        return self.FEATURE_DESCRIPTIONS.get(feature, feature)
    
    def generate_random_config(self, seq_len: int) -> Tuple[int, int]:
        duration = self.rng.randint(max(5, seq_len // 20), max(10, seq_len // 8))
        start_step = self.rng.randint(seq_len // 8, seq_len - duration - seq_len // 8)
        return start_step, duration
    
    def _get_relative_position(self, start_step: int, end_step: int, seq_len: int) -> str:
        mid_point = (start_step + end_step) / 2
        ratio = mid_point / seq_len
        
        if ratio < 0.25:
            return "观测初期"
        elif ratio < 0.5:
            return "观测前中期"
        elif ratio < 0.75:
            return "观测后中期"
        else:
            return "观测后期"


class LLMPromptGenerator:
    """LLM模糊指令生成器 - 使用API反向推理生成高质量模糊指令"""
    
    VAGUENESS_LEVELS = {
        1: {"label": "轻微模糊", "desc": "保留较多技术细节，仅模糊具体数值"},
        2: {"label": "中度模糊", "desc": "使用相对时间，模糊数值和比例"},
        3: {"label": "高度模糊", "desc": "完全使用自然语言，无任何技术参数"},
        4: {"label": "极度模糊", "desc": "仅描述业务场景，完全不提及技术细节"},
    }
    
    def __init__(self, api_key: str, base_url: str = "https://api.deepseek.com", model_name: str = "deepseek-chat"):
        self.api_key = api_key
        self.base_url = base_url
        self.model_name = model_name
        self.client = None
        self._init_client()
        
    def _init_client(self):
        try:
            from openai import OpenAI
            import httpx
            http_client = httpx.Client(
                timeout=httpx.Timeout(60.0, connect=30.0),
                limits=httpx.Limits(max_connections=100, max_keepalive_connections=20)
            )
            self.client = OpenAI(
                api_key=self.api_key, 
                base_url=self.base_url,
                http_client=http_client
            )
            logger.info(f"LLM客户端初始化成功: {self.base_url}")
        except ImportError:
            raise ImportError("请安装openai库: pip install openai httpx")
    
    def generate_vague_prompt(
        self,
        feature_name: str,
        feature_desc: str,
        injection_type: str,
        start_step: int,
        end_step: int,
        seq_len: int,
        causal_scenario: str,
        injection_config: Dict,
        vagueness_level: int = 2,
    ) -> str:
        """使用LLM API生成单条模糊指令"""
        
        relative_position = self._get_relative_position(start_step, end_step, seq_len)
        change_description = self._get_change_description(injection_type, injection_config)
        
        level_instructions = {
            1: """【模糊程度：轻微模糊】
- 可以提及特征名称和大致变化类型
- 用"大约"、"左右"等词模糊具体数值
- 可以暗示时间范围（如"中间那段时间"）
- 保留因果场景的关键词""",
            
            2: """【模糊程度：中度模糊】
- 必须提及特征名称
- 只能用相对时间方位（如"观测中段"、"后半段"）
- 不能出现任何数值或比例
- 用定性描述代替定量描述（如"大幅上升"代替"上升3倍"）""",
            
            3: """【模糊程度：高度模糊】
- 必须提及特征名称
- 完全不出现时间相关词汇，只描述现象
- 用业务场景语言描述变化
- 像是在给不懂技术的同事讲故事""",
            
            4: """【模糊程度：极度模糊】
- 可以不提及具体特征名称，用业务术语代替
- 只描述业务场景和因果关系
- 完全不涉及任何技术细节
- 像是在描述一个业务问题，而非数据问题""",
        }
        
        system_prompt = """你是一个时间序列数据分析专家，擅长将技术性的数据变化描述转化为符合人类认知习惯的自然语言指令。

你的任务：根据给定的技术信息，生成一条符合指定模糊程度的编辑指令。

要求：
1. 指令必须简洁明了，一句话即可
2. 指令应该像是一个分析师在对同事说的话
3. 不同模糊程度要有明显的区分度
4. 必须符合中文表达习惯
5. 不要使用markdown格式，只返回纯文本"""

        user_prompt = f"""请根据以下技术信息，生成一条【{self.VAGUENESS_LEVELS[vagueness_level]['label']}】程度的编辑指令：

[技术信息]
- 特征: {feature_name} ({feature_desc})
- 变化类型: {injection_type}
- 变化详情: {change_description}
- 编辑区间: 第{start_step}步到第{end_step}步（共{seq_len}步序列中的{relative_position}）
- 因果场景: {causal_scenario}

{level_instructions[vagueness_level]}

只返回一句指令，不要任何多余的解释。"""

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=100,
            )
            
            prompt_text = response.choices[0].message.content.strip()
            prompt_text = prompt_text.strip('"\'')
            prompt_text = prompt_text.replace('**', '')
            
            return prompt_text
            
        except Exception as e:
            logger.error(f"LLM API调用失败: {e}")
            return f"[LLM生成失败] {feature_name}在{relative_position}发生了{injection_type}变化"
    
    def _get_relative_position(self, start_step: int, end_step: int, seq_len: int) -> str:
        mid_point = (start_step + end_step) / 2
        ratio = mid_point / seq_len
        
        if ratio < 0.25:
            return "观测初期"
        elif ratio < 0.5:
            return "观测前中期"
        elif ratio < 0.75:
            return "观测后中期"
        else:
            return "观测后期"
    
    def _get_change_description(self, injection_type: str, config: Dict) -> str:
        descriptions = {
            "multiplier": f"数值放大{config.get('multiplier', 2):.1f}倍",
            "hard_zero": "数值强制归零",
            "noise_injection": f"替换为基线{config.get('baseline', 5):.1f}附近的噪声",
            "trend_injection": f"添加{config.get('direction', 'upward')}趋势（斜率{config.get('slope', 0.1):.4f}）",
            "step_change": f"{config.get('direction', 'up')}跳变{abs(config.get('magnitude', 1)):.2f}",
        }
        return descriptions.get(injection_type, "未知变化")


class BenchmarkVisualizer:
    """测试集可视化工具"""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
    def plot_sample(self, sample: BenchmarkSample, save_path: Optional[str] = None):
        """绘制单个样本的可视化图"""
        fig, ax = plt.subplots(figsize=(14, 6))
        
        base_ts = np.array(sample.base_ts)
        target_ts = np.array(sample.target_ts)
        seq_len = len(base_ts)
        x = np.arange(seq_len)
        
        ax.plot(x, base_ts, 'b-', linewidth=1.5, label='Base TS (原序列)', alpha=0.8)
        ax.plot(x, target_ts, 'r-', linewidth=1.5, label='Target TS (理想序列)', alpha=0.8)
        
        ax.axvspan(sample.gt_start, sample.gt_end, alpha=0.3, color='green', label=f'编辑区域 [{sample.gt_start}, {sample.gt_end}]')
        
        ax.axvline(x=sample.gt_start, color='green', linestyle='--', linewidth=1, alpha=0.7)
        ax.axvline(x=sample.gt_end, color='green', linestyle='--', linewidth=1, alpha=0.7)
        
        ax.set_xlabel('时间步', fontsize=12)
        ax.set_ylabel('数值', fontsize=12)
        ax.set_title(f'Sample {sample.sample_id}: {sample.vague_prompt}\n'
                    f'特征: {sample.target_feature} | 任务类型: {sample.task_type}', 
                    fontsize=11)
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        textstr = f't-IoU待计算\n编辑区间: {sample.gt_end - sample.gt_start + 1}步'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', bbox=props)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"图表已保存: {save_path}")
        
        plt.close()
        return fig
    
    def plot_samples_grid(self, samples: List[BenchmarkSample], save_path: Optional[str] = None):
        """绘制多个样本的网格图"""
        n_samples = min(len(samples), 9)
        n_cols = 3
        n_rows = (n_samples + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 4 * n_rows))
        axes = axes.flatten() if n_samples > 1 else [axes]
        
        for i, sample in enumerate(samples[:n_samples]):
            ax = axes[i]
            
            base_ts = np.array(sample.base_ts)
            target_ts = np.array(sample.target_ts)
            x = np.arange(len(base_ts))
            
            ax.plot(x, base_ts, 'b-', linewidth=1, alpha=0.7)
            ax.plot(x, target_ts, 'r-', linewidth=1, alpha=0.7)
            ax.axvspan(sample.gt_start, sample.gt_end, alpha=0.3, color='green')
            
            ax.set_title(f'{sample.sample_id}: {sample.vague_prompt[:30]}...', fontsize=9)
            ax.grid(True, alpha=0.3)
        
        for i in range(n_samples, len(axes)):
            axes[i].axis('off')
        
        plt.suptitle('Mini-Benchmark 样本概览', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"网格图已保存: {save_path}")
        
        plt.close()


class EvaluationMetricsDoc:
    """评估指标说明文档生成器"""
    
    @staticmethod
    def generate_doc(output_path: str):
        """生成评估指标说明文档"""
        doc_content = """# BetterTSE 评估指标说明文档

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

$$tIoU = \\frac{|T_{pred} \\cap T_{gt}|}{|T_{pred} \\cup T_{gt}|}$$

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
$$MSE_{edit} = \\frac{1}{N_{edit}} \\sum_{i \\in T_{gt}} (y_i - \\hat{y}_i)^2$$

**MAE (Mean Absolute Error):**
$$MAE_{edit} = \\frac{1}{N_{edit}} \\sum_{i \\in T_{gt}} |y_i - \\hat{y}_i|$$

其中：
- $N_{edit}$ 是编辑区间的长度
- $y_i$ 是理想序列 Target TS 的值
- $\\hat{y}_i$ 是模型实际生成的序列值

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

$$MSE_{preserve} = \\frac{1}{N_{preserve}} \\sum_{i \\notin T_{gt}} (x_i - \\hat{y}_i)^2$$

$$MAE_{preserve} = \\frac{1}{N_{preserve}} \\sum_{i \\notin T_{gt}} |x_i - \\hat{y}_i|$$

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
"""
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(doc_content)
        
        logger.info(f"评估指标说明文档已保存: {output_path}")


class MiniBenchmarkBuilder:
    """Mini-Benchmark 构建器"""
    
    def __init__(
        self,
        csv_path: str,
        dataset_name: str = "ETTh1",
        output_dir: str = "mini_benchmark",
        random_seed: Optional[int] = None,
        seq_len: int = 192,
        api_key: Optional[str] = None,
        num_prompts_per_sample: int = 4,
    ):
        self.data_loader = CSVDataLoader(csv_path, dataset_name)
        self.injector_factory = InjectorFactory(random_seed)
        self.seq_len = seq_len
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.visualizer = BenchmarkVisualizer(self.output_dir / "visualizations")
        self.random_seed = random_seed
        if random_seed is not None:
            np.random.seed(random_seed)
            
        self.samples: List[BenchmarkSample] = []
        
        self.api_key = api_key or DEEPSEEK_API_KEY
        self.num_prompts_per_sample = num_prompts_per_sample
        
        self.llm_generator = None
        if self.api_key:
            try:
                self.llm_generator = LLMPromptGenerator(self.api_key)
                logger.info("LLM模糊指令生成器初始化成功")
            except Exception as e:
                logger.warning(f"LLM生成器初始化失败: {e}，将使用模板生成")
        
    def build_single_sample(
        self,
        sample_id: str,
        start_idx: int,
        feature: Optional[str] = None,
        injection_type: Optional[str] = None,
    ) -> BenchmarkSample:
        """构建单个测试样本"""
        
        if feature is None:
            feature = self.injector_factory.rng.choice(self.data_loader.features)
            
        base_ts, timestamps = self.data_loader.get_sequence(start_idx, self.seq_len, feature)
        
        injector = self.injector_factory.create_injector(injection_type)
        start_step, duration = self.injector_factory.generate_random_config(self.seq_len)
        
        target_ts, mask_gt, injection_config = injector.inject(base_ts, start_step, duration)
        
        causal_scenario = self.injector_factory.get_causal_scenario(injector.get_name())
        feature_desc = self.injector_factory.get_feature_description(feature)
        
        if self.llm_generator:
            try:
                vague_prompt = self.llm_generator.generate_vague_prompt(
                    feature_name=feature,
                    feature_desc=feature_desc,
                    injection_type=injector.get_name(),
                    start_step=injection_config['start_step'],
                    end_step=injection_config['end_step'],
                    seq_len=self.seq_len,
                    causal_scenario=causal_scenario,
                    injection_config=injection_config,
                    vagueness_level=self.injector_factory.rng.randint(1, 4),
                )
            except Exception as e:
                logger.warning(f"LLM生成失败: {e}，使用模板")
                vague_prompt = self._generate_template_prompt(
                    injection_type=injector.get_name(),
                    feature=feature,
                    start_step=injection_config['start_step'],
                    end_step=injection_config['end_step'],
                    seq_len=self.seq_len,
                    config=injection_config
                )
        else:
            vague_prompt = self._generate_template_prompt(
                injection_type=injector.get_name(),
                feature=feature,
                start_step=injection_config['start_step'],
                end_step=injection_config['end_step'],
                seq_len=self.seq_len,
                config=injection_config
            )
        
        sample = BenchmarkSample(
            sample_id=sample_id,
            dataset_name=self.data_loader.dataset_name,
            target_feature=feature,
            task_type=injector.get_task_type(),
            gt_start=injection_config['start_step'],
            gt_end=injection_config['end_step'],
            vague_prompt=vague_prompt,
            base_ts=base_ts.tolist(),
            target_ts=target_ts.tolist(),
            mask_gt=mask_gt.astype(int).tolist(),
            injection_config=injection_config,
            seq_len=self.seq_len,
            timestamp=datetime.now().isoformat(),
        )
        
        self.samples.append(sample)
        
        logger.info(f"样本 {sample_id} 构建: 特征={feature}, 类型={injector.get_name()}, 区间=[{sample.gt_start}, {sample.gt_end}]")
        
        return sample
    
    def _generate_template_prompt(
        self,
        injection_type: str,
        feature: str,
        start_step: int,
        end_step: int,
        seq_len: int,
        config: Dict
    ) -> str:
        """模板生成模糊指令（备用方案）"""
        position = self.injector_factory._get_relative_position(start_step, end_step, seq_len)
        direction_map = {"upward": "上升", "downward": "下降", "up": "向上", "down": "向下"}
        direction = direction_map.get(config.get("direction", ""), "")
        
        templates = {
            "multiplier": f"在{position}，{feature}突然飙升了",
            "hard_zero": f"在{position}，{feature}突然归零了",
            "noise_injection": f"在{position}，{feature}的传感器似乎离线了",
            "trend_injection": f"在{position}，{feature}出现了明显的{direction}趋势",
            "step_change": f"在{position}，{feature}突然{direction}跳变",
        }
        return templates.get(injection_type, f"在{position}，{feature}发生了变化")
        
    def build_batch(self, num_samples: int = 100) -> List[BenchmarkSample]:
        """批量构建测试样本"""
        
        logger.info(f"\n{'='*60}")
        logger.info(f"开始构建 Mini-Benchmark，共 {num_samples} 个样本")
        logger.info(f"数据集: {self.data_loader.dataset_name}")
        logger.info(f"序列长度: {self.seq_len}")
        logger.info(f"{'='*60}\n")
        
        max_start = len(self.data_loader.data) - self.seq_len
        step = max(1, max_start // num_samples)
        
        for i in range(num_samples):
            start_idx = (i * step) % max_start
            sample_id = f"{i+1:03d}"
            
            try:
                self.build_single_sample(
                    sample_id=sample_id,
                    start_idx=start_idx,
                )
            except Exception as e:
                logger.error(f"样本 {sample_id} 构建失败: {e}")
                continue
                
        return self.samples
        
    def save_csv(self, filename: Optional[str] = None) -> str:
        """保存为CSV格式"""
        
        if filename is None:
            filename = f"mini_benchmark_{self.data_loader.dataset_name}_{len(self.samples)}.csv"
            
        output_file = self.output_dir / filename
        
        csv_data = []
        for s in self.samples:
            csv_data.append({
                "sample_id": s.sample_id,
                "dataset_name": s.dataset_name,
                "target_feature": s.target_feature,
                "task_type": s.task_type,
                "gt_start": s.gt_start,
                "gt_end": s.gt_end,
                "vague_prompt": s.vague_prompt,
                "base_ts": json.dumps(s.base_ts),
                "target_ts": json.dumps(s.target_ts),
                "mask_gt": json.dumps(s.mask_gt),
                "injection_config": json.dumps(s.injection_config),
                "seq_len": s.seq_len,
                "timestamp": s.timestamp,
            })
        
        df = pd.DataFrame(csv_data)
        df.to_csv(output_file, index=False, encoding='utf-8')
        
        logger.info(f"CSV文件已保存: {output_file}")
        
        return str(output_file)
        
    def save_json(self, filename: Optional[str] = None) -> str:
        """保存为JSON格式"""
        
        if filename is None:
            filename = f"mini_benchmark_{self.data_loader.dataset_name}_{len(self.samples)}.json"
            
        output_file = self.output_dir / filename
        
        benchmark_data = {
            "metadata": {
                "dataset_name": self.data_loader.dataset_name,
                "total_samples": len(self.samples),
                "seq_len": self.seq_len,
                "features": self.data_loader.features,
                "created_at": datetime.now().isoformat(),
                "random_seed": self.random_seed,
            },
            "samples": [asdict(s) for s in self.samples],
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(benchmark_data, f, ensure_ascii=False, indent=2)
            
        logger.info(f"JSON文件已保存: {output_file}")
        
        return str(output_file)
        
    def generate_visualizations(self, num_samples: int = 20):
        """生成可视化图表"""
        
        logger.info(f"\n生成可视化图表...")
        
        for i, sample in enumerate(self.samples[:num_samples]):
            save_path = self.output_dir / "visualizations" / f"sample_{sample.sample_id}.png"
            self.visualizer.plot_sample(sample, str(save_path))
        
        grid_path = self.output_dir / "visualizations" / "samples_grid.png"
        self.visualizer.plot_samples_grid(self.samples[:9], str(grid_path))
        
        logger.info(f"可视化图表生成完成")
        
    def generate_metrics_doc(self):
        """生成评估指标说明文档"""
        doc_path = self.output_dir / "EVALUATION_METRICS.md"
        EvaluationMetricsDoc.generate_doc(str(doc_path))
        
    def generate_summary_report(self):
        """生成摘要报告"""
        report_file = self.output_dir / "benchmark_summary.txt"
        
        task_type_dist = {}
        feature_dist = {}
        duration_list = []
        
        for s in self.samples:
            task_type_dist[s.task_type] = task_type_dist.get(s.task_type, 0) + 1
            feature_dist[s.target_feature] = feature_dist.get(s.target_feature, 0) + 1
            duration_list.append(s.gt_end - s.gt_start + 1)
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write("BetterTSE Mini-Benchmark 摘要报告\n")
            f.write("=" * 70 + "\n\n")
            
            f.write(f"数据集: {self.data_loader.dataset_name}\n")
            f.write(f"样本数量: {len(self.samples)}\n")
            f.write(f"序列长度: {self.seq_len}\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("-" * 70 + "\n")
            f.write("任务类型分布:\n")
            f.write("-" * 70 + "\n")
            for tt, count in task_type_dist.items():
                f.write(f"  {tt}: {count}\n")
            f.write("\n")
            
            f.write("-" * 70 + "\n")
            f.write("特征分布:\n")
            f.write("-" * 70 + "\n")
            for fn, count in feature_dist.items():
                f.write(f"  {fn}: {count}\n")
            f.write("\n")
            
            f.write("-" * 70 + "\n")
            f.write("编辑区间统计:\n")
            f.write("-" * 70 + "\n")
            f.write(f"  平均长度: {np.mean(duration_list):.1f} 步\n")
            f.write(f"  最小长度: {min(duration_list)} 步\n")
            f.write(f"  最大长度: {max(duration_list)} 步\n")
            f.write("\n")
            
            f.write("-" * 70 + "\n")
            f.write("样本示例 (前5个):\n")
            f.write("-" * 70 + "\n")
            for i, sample in enumerate(self.samples[:5]):
                f.write(f"\n样本 {sample.sample_id}:\n")
                f.write(f"  特征: {sample.target_feature}\n")
                f.write(f"  任务类型: {sample.task_type}\n")
                f.write(f"  编辑区间: [{sample.gt_start}, {sample.gt_end}]\n")
                f.write(f"  模糊指令: {sample.vague_prompt}\n")
                
        logger.info(f"摘要报告已保存: {report_file}")


def main():
    parser = argparse.ArgumentParser(description='BetterTSE Mini-Benchmark 生成器')
    
    parser.add_argument('--csv-path', type=str, 
                        default=r'C:\Users\ghb\Desktop\李老师科研\BetterTSE-main\data\ETTh1.csv',
                        help='CSV数据文件路径')
    parser.add_argument('--dataset-name', type=str, default='ETTh1',
                        help='数据集名称')
    parser.add_argument('--output-dir', type=str, default='mini_benchmark',
                        help='输出目录')
    parser.add_argument('--num-samples', type=int, default=100,
                        help='构建样本数量')
    parser.add_argument('--seq-len', type=int, default=192,
                        help='序列长度')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    parser.add_argument('--visualize', type=int, default=20,
                        help='可视化样本数量 (0表示不生成)')
    parser.add_argument('--api-key', type=str, default=None,
                        help='DeepSeek API密钥 (不提供则使用环境变量或模板生成)')
    parser.add_argument('--num-prompts', type=int, default=4,
                        help='每个样本生成的模糊指令数量')
    
    args = parser.parse_args()
    
    logger.info("=" * 70)
    logger.info("BetterTSE Mini-Benchmark 生成器")
    logger.info("=" * 70)
    logger.info(f"CSV文件: {args.csv_path}")
    logger.info(f"数据集: {args.dataset_name}")
    logger.info(f"样本数量: {args.num_samples}")
    logger.info(f"序列长度: {args.seq_len}")
    logger.info(f"输出目录: {args.output_dir}")
    logger.info(f"使用LLM生成: {'是' if args.api_key or DEEPSEEK_API_KEY else '否'}")
    
    try:
        builder = MiniBenchmarkBuilder(
            csv_path=args.csv_path,
            dataset_name=args.dataset_name,
            output_dir=args.output_dir,
            random_seed=args.seed,
            seq_len=args.seq_len,
            api_key=args.api_key,
            num_prompts_per_sample=args.num_prompts,
        )
        
        builder.build_batch(num_samples=args.num_samples)
        
        builder.save_csv()
        builder.save_json()
        
        if args.visualize > 0:
            builder.generate_visualizations(num_samples=args.visualize)
        
        builder.generate_metrics_doc()
        builder.generate_summary_report()
        
        logger.info("\n" + "=" * 70)
        logger.info("Mini-Benchmark 构建完成!")
        logger.info(f"输出目录: {builder.output_dir}")
        logger.info("=" * 70)
        
    except Exception as e:
        logger.error(f"构建失败: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
