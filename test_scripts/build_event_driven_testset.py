"""
BetterTSE 事件驱动测试集生成器 (Event-Driven Testset Builder)
================================================================

核心理念：测试大模型时间序列编辑器能力的最好方式，
不是让它听懂"模糊的技术指令"，而是让它理解"现实世界的新闻/事件"，
并由它自己去推理物理影响。

范式转变：
- 旧范式：将"技术语言"变模糊
- 新范式：根据"技术变化"逆向编造"新闻快讯或业务通知"

三级事件驱动视角：
- Level 1: 直接业务指令（调度员视角）- 低模糊度
- Level 2: 宏观新闻播报（新闻主播视角）- 中度模糊
- Level 3: 无关联线索（社交媒体路人视角）- 高度模糊/极度间接

作者: BetterTSE Team
"""

import os
import sys
import re
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
try:
    from scipy.ndimage import gaussian_filter1d
except Exception:
    def gaussian_filter1d(array, sigma):
        sigma = float(sigma)
        values = np.asarray(array, dtype=np.float64)
        if sigma <= 0.0:
            return values
        radius = max(1, int(round(4.0 * sigma)))
        x = np.arange(-radius, radius + 1, dtype=np.float64)
        kernel = np.exp(-(x ** 2) / (2.0 * sigma * sigma))
        kernel /= np.sum(kernel)
        full = np.convolve(values, kernel, mode="full")
        start = (len(kernel) - 1) // 2
        end = start + len(values)
        return full[start:end]

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("EventDrivenTestSet")


try:
    from dotenv import load_dotenv
    _env = Path(__file__).resolve().parent.parent / ".env"
    if _env.exists():
        load_dotenv(_env)
except ImportError:
    pass

DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("OPENAI_API_KEY", "")


@dataclass
class EventDrivenPrompt:
    """事件驱动提示词"""
    prompt: str
    level: int
    level_name: str
    perspective: str


@dataclass
class EventDrivenSample:
    """事件驱动测试样本"""
    sample_id: str
    dataset_name: str
    target_feature: str
    feature_description: str
    task_type: str
    legacy_task_type: str
    injection_operator: str
    edit_intent_gt: Dict[str, Any]
    gt_start: int
    gt_end: int
    event_prompts: List[Dict]
    technical_ground_truth: str
    base_ts: List[float]
    target_ts: List[float]
    mask_gt: List[int]
    injection_config: Dict[str, Any]
    causal_scenario: str
    seq_len: int
    timestamp: str = ""


class CSVDataLoader:
    """真实CSV数据加载器"""
    
    DATASET_INFO = {
        "ETTh1": {
            "freq": "hourly",
            "desc": "Electricity Transformer Temperature (hourly)",
            "domain": "电力系统",
            "features_desc": {
                "HUFL": "High Useful Load (高压侧有效负荷)",
                "HULL": "High Useless Load (高压侧无效负荷)",
                "MUFL": "Middle Useful Load (中压侧有效负荷)",
                "MULL": "Middle Useless Load (中压侧无效负荷)",
                "LUFL": "Low Useful Load (低压侧有效负荷)",
                "LULL": "Low Useless Load (低压侧无效负荷)",
                "OT": "Oil Temperature (变压器油温)",
            }
        },
        "ETTm1": {"freq": "15min", "desc": "Electricity Transformer Temperature (15-min)", "domain": "电力系统"},
        "Traffic": {"freq": "hourly", "desc": "California Traffic Flow", "domain": "交通系统"},
    }
    
    def __init__(self, csv_path: str, dataset_name: str = "ETTh1"):
        self.csv_path = Path(csv_path)
        self.dataset_name = dataset_name
        self.data = None
        self.features = []
        self.feature_to_idx = {}
        self.timestamps = []
        self._series: Dict[str, np.ndarray] = {}
        self.domain = self.DATASET_INFO.get(dataset_name, {}).get("domain", "工业系统")
        self.feature_descriptions = self.DATASET_INFO.get(dataset_name, {}).get("features_desc", {})
        
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV文件不存在: {csv_path}")
        self._load_data()
        
    def _load_data(self):
        logger.info(f"正在加载数据集: {self.csv_path}")
        try:
            frame = pd.read_csv(self.csv_path)
        except Exception:
            raw = np.genfromtxt(self.csv_path, delimiter=",", names=True, dtype=None, encoding="utf-8")
            column_names = list(raw.dtype.names or [])
            if not column_names:
                raise
            self.features = [column for column in column_names if column != "date"]
            self._series = {}
            for column in column_names:
                values = np.asarray(raw[column])
                if column == "date":
                    self.timestamps = values.tolist()
                else:
                    self._series[column] = values.astype(np.float64)
            self.feature_to_idx = {feat: idx for idx, feat in enumerate(self.features)}
            self.data = np.column_stack([self._series[feat] for feat in self.features])
            logger.info(f"数据形状: {self.data.shape}")
            logger.info(f"特征列表 ({len(self.features)}): {self.features}")
            return
        else:
            if 'date' in frame.columns:
                self.timestamps = frame['date'].tolist()
                frame = frame.drop(columns=['date'])
            self.features = frame.columns.tolist()
            self.feature_to_idx = {feat: idx for idx, feat in enumerate(self.features)}
            self._series = {
                feat: np.asarray(frame[feat], dtype=np.float64)
                for feat in self.features
            }
            self.data = np.column_stack([self._series[feat] for feat in self.features])
        logger.info(f"数据形状: {self.data.shape}")
        logger.info(f"特征列表 ({len(self.features)}): {self.features}")
            
    def get_sequence(self, start_idx: int, seq_len: int, feature: str) -> Tuple[np.ndarray, List[str]]:
        end_idx = start_idx + seq_len
        total_length = int(self.data.shape[0]) if hasattr(self.data, "shape") else len(self.data)
        if end_idx > total_length:
            end_idx = total_length
            start_idx = max(0, end_idx - seq_len)
        
        sequence = np.asarray(self._series[feature][start_idx:end_idx], dtype=np.float64)
        
        timestamps = []
        if self.timestamps:
            timestamps = self.timestamps[start_idx:end_idx]
            
        return sequence, timestamps


class PhysicalInjector(ABC):
    """物理变化注入器基类"""
    
    TASK_TYPES = {
        "multiplier": "sustained_gain",
        "hard_zero": "shutdown_flatline",
        "noise_injection": "signal_corruption",
        "trend_injection": "transient_hump",
        "step_change": "regime_switch",
        "seasonality_injection": "seasonality_shift",
    }

    LEGACY_TASK_TYPES = {
        "multiplier": "heatwave_overload",
        "hard_zero": "facility_shutdown",
        "noise_injection": "sensor_offline",
        "trend_injection": "market_trend",
        "step_change": "device_switch",
        "seasonality_injection": "seasonality_shift",
    }
    
    CAUSAL_SCENARIOS = {
        "generic": {
            "multiplier": [
                "突发外部需求在一段时间内持续偏高",
                "关键活动开始后系统负载维持在明显偏高状态",
                "高峰时段到来后相关指标长时间停留在高位",
            ],
            "hard_zero": [
                "设备检修导致系统停机",
                "紧急故障导致设施关闭",
                "计划性维护导致服务中断",
            ],
            "noise_injection": [
                "传感器故障导致读数异常紊乱",
                "通信中断导致信号持续失真",
                "外部干扰导致输出出现杂乱跳变",
            ],
            "trend_injection": [
                "短时冲高后逐步恢复常态的需求波动",
                "某类突发因素带来阶段性抬升后缓慢回落",
                "事件刺激使相关指标先偏离后恢复稳定",
            ],
            "step_change": [
                "设备切换导致运行模式改变",
                "系统重置导致状态跳变",
                "控制策略调整导致基准状态突然切换",
            ],
            "seasonality_injection": [
                "运行节律突然变得更强，峰谷起伏被明显放大",
                "周期性调度让系统呈现更清晰的重复节拍",
                "外部约束改变后，原本的周期起伏被明显压平",
            ],
        },
        "power": {
            "multiplier": [
                "极端热浪使用电负荷在一段时间内持续偏高",
                "寒潮来袭后供暖需求长时间维持高位",
                "工业生产高峰期让负荷持续停留在偏高状态",
                "大型活动期间使整体用电量维持在高位",
            ],
            "hard_zero": [
                "设备检修导致系统停机",
                "主站保护动作导致区域停运",
                "计划性维护导致供电服务中断",
            ],
            "noise_injection": [
                "传感器故障导致监测信号异常紊乱",
                "通信中断导致遥测信号丢失",
                "电磁干扰导致读数持续跳变",
            ],
            "trend_injection": [
                "极端高温天气导致短时用电负荷冲高后回落",
                "工业园区开工带来阶段性负载抬升后逐步平稳",
                "气温骤变导致制冷或供暖需求短暂剧烈波动",
            ],
            "step_change": [
                "备用电源切换导致运行状态改变",
                "负载转移导致瞬时工作点切换",
                "控制策略调整导致基准负荷突然变化",
            ],
            "seasonality_injection": [
                "分时调度信号加强后，峰谷负荷节律变得更明显",
                "居民作息与气温共振让日内峰谷起伏更清晰",
                "错峰干预生效后，原本明显的日内峰谷被压平了一些",
            ],
        },
        "traffic": {
            "multiplier": [
                "节假日集中出行使车流量在一段时间内持续偏高",
                "大型赛事散场让道路通行压力维持在高位",
                "突发天气前集中出行使路网拥堵持续加剧一段时间",
            ],
            "hard_zero": [
                "主干道封控导致通行几乎中断",
                "重大事故处置导致道路临时关闭",
                "夜间施工封路导致流量接近停滞",
            ],
            "noise_injection": [
                "路侧传感器故障导致监测读数杂乱跳变",
                "通信链路不稳导致交通监测信号失真",
                "探测设备受干扰后输出出现无规律波动",
            ],
            "trend_injection": [
                "节假日集中出行导致流量短时激增随后平稳",
                "大型活动散场带来阶段性拥堵后逐步缓解",
                "天气骤变导致道路通行压力先升后降",
            ],
            "step_change": [
                "匝道开闭状态切换导致通行水平突然变化",
                "临时交通管制导致道路运行模式瞬时改变",
                "信号配时调整导致路段基准通行状态跳变",
            ],
            "seasonality_injection": [
                "通勤节律突然更集中，让早晚峰谷模式更明显",
                "阶段性分流后，原本规律的车流峰谷被削弱了一些",
                "节假日出行节奏变化让重复性的通行波动更突出",
            ],
        },
    }

    DIRECTIONAL_SCENARIOS = {
        "generic": {
            "trend_injection": {
                "up": [
                    "突发利好消息带来一阵明显提振，但热度随后逐步退去",
                    "短期刺激因素使系统一度被推到更高位，随后慢慢恢复常态",
                ],
                "down": [
                    "突发利空因素让系统一度承压走低，随后缓慢恢复",
                    "阶段性不利消息使相关状态先落到偏低水平，之后逐步回稳",
                ],
            },
            "step_change": {
                "up": [
                    "资源重新分配后系统切换到更高负载档位",
                    "策略调整使运行状态突然切到更高位并维持一段时间",
                ],
                "down": [
                    "限制措施生效后系统切换到更受限的运行状态",
                    "控制策略收紧使基准状态突然压到更低位并维持一段时间",
                ],
            },
        },
        "power": {
            "trend_injection": {
                "up": [
                    "供应紧张预期使短时负荷压力被推高，随后逐步回落",
                    "突发需求刺激让相关运行压力一度偏高，随后慢慢恢复",
                ],
                "down": [
                    "阶段性负荷转移让相关运行水平一度走低，随后逐步恢复",
                    "短时压降因素使监测对象先落到偏低状态，之后慢慢回稳",
                ],
            },
            "step_change": {
                "up": [
                    "负载重新切换后系统突然进入更高负荷档位",
                    "并网策略调整使运行状态瞬时切到更高位并维持一段时间",
                ],
                "down": [
                    "限载措施启动后系统突然切到更低负荷档位",
                    "保护策略介入使基准运行水平瞬时压到更低位并维持一段时间",
                ],
            },
        },
        "traffic": {
            "trend_injection": {
                "up": [
                    "突发消息刺激集中出行，相关路段压力一度被推高，随后才慢慢缓解",
                    "短时出行需求集中释放，让通行压力先明显偏高，之后逐步恢复",
                ],
                "down": [
                    "分流措施见效后，相关路段通行水平一度偏低，随后逐步回稳",
                    "短时绕行引导让该区域先落到较低通行状态，随后慢慢恢复常态",
                ],
            },
            "step_change": {
                "up": [
                    "匝道开放后相关路段突然切到更高通行状态并维持一段时间",
                    "管制解除使道路运行水平瞬时回到更高位并持续一阵",
                ],
                "down": [
                    "临时交通管制启动后相关匝道突然切到更受限的通行状态",
                    "匝道收紧控制后道路运行水平瞬时压到更低位并维持一段时间",
                ],
            },
        },
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

    def get_legacy_task_type(self) -> str:
        return self.LEGACY_TASK_TYPES.get(self.get_name(), "unknown")

    def get_injection_operator(self) -> str:
        return self.get_name()

    def get_edit_intent(self, injection_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        config = injection_config or {}
        injection_type = self.get_name()

        if injection_type == "multiplier":
            return {
                "effect_family": "multiplier",
                "shape": "scaled_surge",
                "direction": "up",
                "duration": "medium",
                "recovery": "gradual",
                "onset": "abrupt",
            }

        if injection_type == "hard_zero":
            return {
                "effect_family": "hard_zero",
                "shape": "flatline",
                "direction": "down",
                "duration": "medium",
                "recovery": "gradual",
                "onset": "gradual",
            }

        if injection_type == "noise_injection":
            return {
                "effect_family": "noise_injection",
                "shape": "irregular_noise",
                "direction": "neutral",
                "duration": "medium",
                "recovery": "none",
                "onset": "abrupt",
            }

        if injection_type == "trend_injection":
            direction = config.get("direction", "upward")
            return {
                "effect_family": "trend",
                "shape": "hump",
                "direction": "up" if direction == "upward" else "down",
                "duration": "medium",
                "recovery": "gradual",
                "onset": "gradual",
            }

        if injection_type == "step_change":
            direction = config.get("direction", "up")
            return {
                "effect_family": "step_change",
                "shape": "step",
                "direction": direction,
                "duration": "medium",
                "recovery": "gradual",
                "onset": "abrupt",
            }

        if injection_type == "seasonality_injection":
            return {
                "effect_family": "seasonality",
                "shape": "periodic",
                "direction": "neutral",
                "duration": "medium",
                "recovery": "smooth",
                "onset": "smooth",
            }

        return {
            "effect_family": "unknown",
            "shape": "unknown",
            "direction": "neutral",
            "duration": "medium",
            "recovery": "none",
            "onset": "unknown",
        }
    
    def get_causal_scenario(self, domain_key: str = "generic", injection_config: Optional[Dict[str, Any]] = None) -> str:
        direction = self._get_direction_key(injection_config or {})
        directional_bank = self.DIRECTIONAL_SCENARIOS.get(domain_key, {}).get(self.get_name(), {})
        if direction in directional_bank:
            scenarios = directional_bank[direction]
        else:
            domain_scenarios = self.CAUSAL_SCENARIOS.get(domain_key, self.CAUSAL_SCENARIOS["generic"])
            scenarios = domain_scenarios.get(self.get_name(), self.CAUSAL_SCENARIOS["generic"].get(self.get_name(), ["未知场景"]))
        return self.rng.choice(scenarios)

    def _get_direction_key(self, injection_config: Dict[str, Any]) -> str:
        name = self.get_name()
        if name == "multiplier":
            return "up"
        if name == "hard_zero":
            return "down"
        if name == "noise_injection":
            return "neutral"
        if name == "trend_injection":
            return "up" if injection_config.get("direction", "upward") == "upward" else "down"
        if name == "step_change":
            return injection_config.get("direction", "up")
        if name == "seasonality_injection":
            return "neutral"
        return "neutral"


class MultiplierInjector(PhysicalInjector):
    """
    乘法放大注入器
    功能：在指定时间段内将原始序列按随机倍数放大，模拟“极端热浪导致用电负荷激增”等场景
    """

    def __init__(self, rng: np.random.RandomState, multiplier_range: Tuple[float, float] = (2.0, 4.0)):
        """
        初始化
        :param rng: 随机数生成器，保证实验可复现
        :param multiplier_range: 放大倍数区间，默认 2~4 倍
        """
        super().__init__(rng)
        self.multiplier_range = multiplier_range

    def inject(self, base_ts: np.ndarray, start_step: int, duration: int) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        执行注入
        :param base_ts: 原始时间序列
        :param start_step: 注入起始步
        :param duration: 注入持续步长
        :return: (注入后序列,  Ground-Truth 掩码, 注入配置字典)

        物理建模：
        - 起点硬跳（热浪来袭时用电激增近似瞬时）
        - 末端线性软退出（事件消退过程是渐进的，非瞬时）
        - scale 数组在末尾从 multiplier 线性降回 1.0
        """
        target_ts = base_ts.copy()
        mask_gt = np.zeros(len(base_ts))
        end_step = min(start_step + duration, len(base_ts))
        region_len = end_step - start_step

        multiplier = self.rng.uniform(*self.multiplier_range)

        # 构造逐点缩放系数：主体为常数 multiplier，末端 ramp_out 步线性退回 1.0
        ramp_out = min(max(3, region_len // 5), 10)
        scale = np.ones(region_len) * multiplier
        scale[-ramp_out:] = np.linspace(multiplier, 1.0, ramp_out)

        target_ts[start_step:end_step] = base_ts[start_step:end_step] * scale
        mask_gt[start_step:end_step] = 1

        config = {
            "injection_type": "multiplier",
            "multiplier": round(float(multiplier), 2),
            "ramp_out": ramp_out,
            "start_step": start_step,
            "end_step": end_step - 1,
            "duration": duration,
        }
        return target_ts, mask_gt, config

    def get_name(self) -> str:
        """返回注入器名称，用于日志与配置识别"""
        return "multiplier"


class HardZeroInjector(PhysicalInjector):
    """
    硬归零注入器
    功能：将指定时间段内的序列值强制置零，模拟“设备检修导致系统停机”等场景
    """

    def inject(self, base_ts: np.ndarray, start_step: int, duration: int) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        执行注入
        :param base_ts: 原始时间序列
        :param start_step: 注入起始步
        :param duration: 注入持续步长
        :return: (注入后序列, Ground-Truth 掩码, 注入配置字典)

        物理建模：
        - 底值取数据低 3% 分位数，而非绝对 0（温度类特征不可能降到绝对零度，
          负荷类特征在停机后也会保持小量自耗电或传感器底噪）
        - 入口：start_step 前几步线性下滑至底值（设备切断需要短暂的放电过程）
        - 出口：end_step 前几步线性回升至恢复后的真实值（重启需要预热/升载）
        """
        target_ts = base_ts.copy()
        mask_gt = np.zeros(len(base_ts))
        end_step = min(start_step + duration, len(base_ts))
        region_len = end_step - start_step

        # 数据自适应物理底值：低3%分位数（远优于硬编码的 0.0）
        floor_value = float(np.percentile(base_ts, 3))

        # 过渡段长度：区间的 1/6，最少 1 步、最多 5 步
        ramp = min(5, max(1, region_len // 6))

        # 1. 主体区间填充底值
        target_ts[start_step:end_step] = floor_value

        # 2. 入口软降：从真实起始值渐变到底值
        entry_val = float(base_ts[start_step])
        target_ts[start_step:start_step + ramp] = np.linspace(entry_val, floor_value, ramp)

        # 3. 出口软升：从底值渐变到恢复后的真实值
        exit_val = float(base_ts[end_step]) if end_step < len(base_ts) else float(base_ts[-1])
        target_ts[end_step - ramp:end_step] = np.linspace(floor_value, exit_val, ramp)

        mask_gt[start_step:end_step] = 1

        config = {
            "injection_type": "hard_zero",
            "floor_value": round(floor_value, 4),
            "ramp": ramp,
            "start_step": start_step,
            "end_step": end_step - 1,
            "duration": duration,
        }
        return target_ts, mask_gt, config

    def get_name(self) -> str:
        """返回注入器名称，用于日志与配置识别"""
        return "hard_zero"


class NoiseInjector(PhysicalInjector):
    """
    传感器离线/干扰噪声注入器

    物理建模：传感器通信中断或受到强电磁干扰时，输出不再反映真实物理量，
    而是呈现为接近底部基线的高幅随机杂波（底噪 + 大方差）。

    关键修正（相比旧版）：
    - 旧版：基线硬编码 [3.0, 8.0]，与数据量纲完全无关；noise_level=0.1 几乎无噪声。
    - 新版：基线 = 数据最低值 + 微小偏移（数据自适应）；
            噪声幅度 = 数据标准差的 1.5~3 倍（远大于正常信号波动，清晰可辨）。
    """

    def __init__(self, rng: np.random.RandomState,
                 noise_multiplier: Tuple[float, float] = (1.5, 3.0)):
        super().__init__(rng)
        self.noise_multiplier = noise_multiplier

    def inject(self, base_ts: np.ndarray, start_step: int, duration: int) -> Tuple[np.ndarray, np.ndarray, Dict]:
        target_ts = base_ts.copy()
        mask_gt = np.zeros(len(base_ts))
        end_step = min(start_step + duration, len(base_ts))

        data_std   = float(np.std(base_ts))
        data_min   = float(np.min(base_ts))
        data_range = float(np.max(base_ts)) - data_min

        # 传感器底噪基线：数据最低值附近（2%~8% 分位偏移），而非绝对 0
        baseline = data_min + data_range * self.rng.uniform(0.02, 0.08)
        # 噪声标准差：数据标准差的随机倍数，使杂波幅度远超正常信号波动
        noise_std = data_std * self.rng.uniform(*self.noise_multiplier)

        noise = self.rng.normal(loc=baseline, scale=noise_std, size=end_step - start_step)
        target_ts[start_step:end_step] = noise
        mask_gt[start_step:end_step] = 1

        config = {
            "injection_type": "noise_injection",
            "baseline":   round(float(baseline), 4),
            "noise_std":  round(float(noise_std), 4),
            "data_std":   round(data_std, 4),
            "start_step": start_step,
            "end_step":   end_step - 1,
            "duration":   duration,
        }
        return target_ts, mask_gt, config

    def get_name(self) -> str:
        return "noise_injection"


class TrendInjector(PhysicalInjector):
    """
    临时负荷波动注入器（升余弦钟形脉冲）

    物理建模：极端天气/大型活动等短期事件导致负荷先升后降（或先降后升），
    整体呈钟形曲线，起止两端完全光滑，无硬边界跳变。

    关键修正（相比旧版）：
    - 旧版：线性单调趋势，终点处产生"悬崖式"硬跳变（从峰值偏移突然归零）。
    - 新版：升余弦窗 (1 - cos(t)) / 2，从 0 平滑上升到峰值再平滑回落，
            两端连续可微，无任何人工边界伪影。
    - 振幅取数据范围的 15%~40%，足够显眼且不产生物理上的荒谬值。
    """

    def inject(self, base_ts: np.ndarray, start_step: int, duration: int) -> Tuple[np.ndarray, np.ndarray, Dict]:
        target_ts = base_ts.copy()
        mask_gt = np.zeros(len(base_ts))
        end_step = min(start_step + duration, len(base_ts))
        region_len = end_step - start_step

        direction  = self.rng.choice(["upward", "downward"])
        data_range = float(np.max(base_ts)) - float(np.min(base_ts))

        # 振幅：数据范围的 15%~40%
        amplitude = data_range * self.rng.uniform(0.15, 0.40)
        if direction == "downward":
            amplitude = -abs(amplitude)

        # 升余弦窗：t ∈ [0, 2π]  =>  hump(t) = (1 - cos(t)) / 2
        # t=0  : hump=0（起点光滑）
        # t=π  : hump=1（中点峰值）
        # t=2π : hump=0（终点光滑）
        t    = np.linspace(0, 2.0 * np.pi, region_len)
        hump = (1.0 - np.cos(t)) / 2.0
        target_ts[start_step:end_step] += hump * amplitude
        mask_gt[start_step:end_step] = 1

        config = {
            "injection_type": "trend_injection",
            "direction":  direction,
            "amplitude":  round(float(amplitude), 4),
            "start_step": start_step,
            "end_step":   end_step - 1,
            "duration":   duration,
        }
        return target_ts, mask_gt, config

    def get_name(self) -> str:
        return "trend_injection"


class SeasonalityInjector(PhysicalInjector):
    """周期性注入器。"""

    def __init__(
        self,
        rng: np.random.RandomState,
        amplitude_ratio_range: Tuple[float, float] = (0.08, 0.28),
    ):
        super().__init__(rng)
        self.amplitude_ratio_range = amplitude_ratio_range

    def inject(self, base_ts: np.ndarray, start_step: int, duration: int) -> Tuple[np.ndarray, np.ndarray, Dict]:
        target_ts = base_ts.copy()
        mask_gt = np.zeros(len(base_ts))
        end_step = min(start_step + duration, len(base_ts))
        region_len = end_step - start_step
        region = np.asarray(base_ts[start_step:end_step], dtype=np.float64)
        data_range = max(float(np.max(base_ts)) - float(np.min(base_ts)), 1e-6)

        cycles = int(self.rng.choice([1, 2, 4]))
        amplitude_ratio = float(self.rng.uniform(*self.amplitude_ratio_range))
        amplitude = data_range * amplitude_ratio
        phase = float(self.rng.uniform(0.0, 2.0 * np.pi))
        u = np.linspace(0.0, 1.0, region_len)
        envelope = np.sin(np.pi * u)
        seasonal_wave = envelope * np.sin(2.0 * np.pi * cycles * u + phase)
        target_ts[start_step:end_step] = region + amplitude * seasonal_wave
        config = {
            "injection_type": "seasonality_injection",
            "cycles": cycles,
            "phase": round(float(phase), 4),
            "seasonal_amplitude": round(float(amplitude), 4),
            "seasonal_amplitude_ratio": round(float(amplitude_ratio), 4),
            "start_step": start_step,
            "end_step": end_step - 1,
            "duration": duration,
        }

        mask_gt[start_step:end_step] = 1
        return target_ts, mask_gt, config

    def get_name(self) -> str:
        return "seasonality_injection"


class StepChangeInjector(PhysicalInjector):
    def __init__(self, rng: np.random.RandomState, step_range: Tuple[float, float] = (0.3, 0.7)):
        super().__init__(rng)
        self.step_range = step_range
        
    def inject(self, base_ts: np.ndarray, start_step: int, duration: int) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        物理建模：
        - 起点：硬跳变（设备瞬时切换到新工作点，如备用机组并网）
        - 主体：维持在新工作点运行
        - 末端：渐进恢复（另一设备逐步接管负载，持续约 1/3 区间长度）

        关键修正（相比旧版）：
        - 旧版：起终两端均为硬跳变，实质是矩形脉冲，不符合"阶跃切换"语义。
        - 新版：保留起点硬跳（瞬时切换），末端线性渐降（渐进交接），
                物理上对应"故障切换后逐步恢复正常调度"。
        """
        target_ts = base_ts.copy()
        mask_gt = np.zeros(len(base_ts))
        end_step = min(start_step + duration, len(base_ts))
        region_len = end_step - start_step

        data_range     = float(np.max(base_ts)) - float(np.min(base_ts))
        step_magnitude = data_range * self.rng.uniform(*self.step_range)
        direction      = self.rng.choice(["up", "down"])
        if direction == "down":
            step_magnitude = -step_magnitude

        # 构造逐点偏移量：主体保持 step_magnitude，末端 1/3 线性退回 0
        ramp_out = min(max(3, region_len // 3), 20)
        magnitude_array = np.ones(region_len) * step_magnitude
        magnitude_array[-ramp_out:] = np.linspace(float(step_magnitude), 0.0, ramp_out)

        target_ts[start_step:end_step] += magnitude_array
        mask_gt[start_step:end_step] = 1

        config = {
            "injection_type": "step_change",
            "direction":  direction,
            "magnitude":  round(float(step_magnitude), 4),
            "ramp_out":   ramp_out,
            "start_step": start_step,
            "end_step":   end_step - 1,
            "duration":   duration,
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
        "seasonality_injection": SeasonalityInjector,
        "step_change": StepChangeInjector,
    }
    
    FEATURE_DESCRIPTIONS = {
        "HUFL": "高压侧有效负荷",
        "HULL": "高压侧无效负荷",
        "MUFL": "中压侧有效负荷",
        "MULL": "中压侧无效负荷",
        "LUFL": "低压侧有效负荷",
        "LULL": "低压侧无效负荷",
        "OT": "变压器油温",
    }
    
    def __init__(self, random_seed: Optional[int] = None):
        self.rng = np.random.RandomState(random_seed)
        
    def create_injector(self, injection_type: Optional[str] = None) -> PhysicalInjector:
        if injection_type is None:
            injection_type = self.rng.choice(list(self.INJECTOR_REGISTRY.keys()))
        injector_class = self.INJECTOR_REGISTRY[injection_type]
        return injector_class(self.rng)
    
    def get_feature_description(self, feature: str) -> str:
        return self.FEATURE_DESCRIPTIONS.get(feature, feature)

    @staticmethod
    def get_domain_key(dataset_name: str) -> str:
        dataset_name = (dataset_name or "").lower()
        if dataset_name.startswith("ett"):
            return "power"
        if dataset_name.startswith("traffic"):
            return "traffic"
        return "generic"
    
    def generate_random_config(self, seq_len: int) -> Tuple[int, int]:
        duration = self.rng.randint(max(5, seq_len // 20), max(10, seq_len // 8))
        start_step = self.rng.randint(seq_len // 8, seq_len - duration - seq_len // 8)
        return start_step, duration


class EventDrivenPromptGenerator:
    """
    事件驱动提示词生成器 (Few-Shot Learning 增强版)
    
    核心理念：将后台系统的【数学数据变化】转化为【真实世界的新闻事件或业务通报】
    
    三级事件驱动视角：
    - Level 1: 直接业务指令（调度员视角）- 低模糊度
    - Level 2: 宏观新闻播报（新闻主播视角）- 中度模糊
    - Level 3: 无关联线索（社交媒体路人视角）- 高度模糊/极度间接
    
    Few-Shot Learning 示例参考 Context-is-Key 基准测试（特别是 MontrealFire 任务）：
    用中立、客观的陈述语气，将复杂的因果交织和物理背景转化为自然语言
    """
    
    LEVEL_CONFIG = {
        1: {
            "name": "直接业务指令",
            "perspective": "调度员",
            "description": "以业务下发指令的口吻，指出具体的对象和定性的变化方向"
        },
        2: {
            "name": "宏观新闻播报",
            "perspective": "新闻主播",
            "description": "播报现实宏观事件，让专业人员推断数据变化"
        },
        3: {
            "name": "无关联线索",
            "perspective": "社交媒体路人",
            "description": "只描述生活场景/社会现象，完全依靠常识推理"
        }
    }
    
    FEW_SHOT_EXAMPLES = {
        1: """
【示例 1】
输入：领域=某系统关键指标, 线索=短时冲高后逐步回落, 时间=在今日运行中段
输出：请注意，在今日运行中段，相关运行压力会先明显抬高，随后再逐步恢复常态。

【示例 2】
输入：领域=某系统关键指标, 线索=状态突然切换并维持一段时间, 时间=在早班交接后
输出：请注意，在早班交接后，系统将切换到新的运行状态，并在一段时间内维持该水平。

【示例 3】
输入：领域=某系统关键指标, 线索=持续抬升, 时间=在夜间低谷期前
输出：请注意，在夜间低谷期前，相关指标会持续走高，请提前关注运行压力。

【示例 4】
输入：领域=某系统关键指标, 线索=读数杂乱跳变, 时间=在今日运行中段
输出：请注意，在今日运行中段，相关读数会出现杂乱跳变，建议尽快核查现场情况。

【示例 5】
输入：领域=某系统关键指标, 线索=周期起伏突然更明显, 时间=在夜间低谷期前
输出：请注意，在夜间低谷期前，相关运行节律会变得更明显，峰谷起伏将持续一阵。""",

        2: """
【示例 1】
输入：领域=某公共系统, 线索=短时冲高后回落, 时间=从今天中午开始
输出：最新通报，从今天中午开始，相关系统运行压力短时冲高，随后正逐步回到常态水平。

【示例 2】
输入：领域=某公共系统, 线索=持续抬升, 时间=预计在今晚深夜
输出：据最新消息，预计在今晚深夜，相关系统将持续承压，运行水平较平时明显偏高。

【示例 3】
输入：领域=某公共系统, 线索=降至极低水平并维持, 时间=从今天中午开始
输出：突发通报，从今天中午开始，相关系统服务能力明显下降，并将在一段时间内维持在极低水平。

【示例 4】
输入：领域=某公共系统, 线索=读数杂乱跳变, 时间=从今天中午开始
输出：监测部门通报，从今天中午开始，相关监测信号持续异常，输出表现出明显的杂乱跳变。

【示例 5】
输入：领域=某公共系统, 线索=周期节律被压平, 时间=预计在今晚深夜
输出：最新通报，预计在今晚深夜，相关系统原本清晰的峰谷节律会被压平，重复起伏不再像平时那样明显。""",

        3: """
【示例 1】
输入：领域=某公共系统, 线索=短时冲高后回落, 时间=刚才
输出：刚才那阵子一下子忙成一团，不过现在看着又慢慢缓下来了。

【示例 2】
输入：领域=某公共系统, 线索=降到很低并持续一阵, 时间=就快到半夜的时候
输出：就快到半夜的时候那边突然就没什么动静了，而且这种状态持续了好一阵。

【示例 3】
输入：领域=某公共系统, 线索=读数杂乱跳变, 时间=大清早的时候
输出：大清早的时候那边就开始一会儿高一会儿低，感觉完全不在正常状态里。

【示例 4】
输入：领域=某公共系统, 线索=突然切换到另一种状态并维持, 时间=刚才
输出：刚才那边像是突然换了个运行方式，后面一段时间都没再回到原来的样子。

【示例 5】
输入：领域=某公共系统, 线索=周期起伏更明显, 时间=就快到半夜的时候
输出：就快到半夜的时候那边一阵一阵的规律感突然特别明显，像是峰谷都被拉开了。"""
    }
    
    ROLE_DESCRIPTIONS = {
        1: "你是一线调度员。请下达明确的定性业务指令。必须包含时间提示，并保留一个可推断的变化线索，例如持续抬升、短时冲高后回落、突然切换、杂乱跳变、周期起伏更明显、跌至极低水平。禁止阿拉伯数字、倍数或百分比。",
        2: "你是客观的新闻发言人。必须包含正式的新闻时间副词。只客观播报现实事件和运行状态，不要直接说'数据上升下降'，但必须保留一个可推断的动态锚点，例如短时冲高后恢复、持续承压、信号杂乱跳变、周期节律增强或被压平、服务能力降到极低水平。",
        3: "你是社交媒体上的普通市民或现场路人。使用口语化时间词。可以非常生活化，但不能只说'异常'或'不对劲'；至少要留下一个弱语义线索，让人能推断是持续抬升、突然切换、杂乱跳变、周期起伏变强/变弱、短时冲高或明显停摆。"
    }

    DOMAIN_RULES = {
        "power": {
            "domain_label": "电力系统",
            "allowed_keywords": ["负荷", "油温", "供电", "变电站", "监测终端", "制冷设备", "电网调度"],
            "forbidden_keywords": ["车流", "道路", "高速", "匝道", "警戒线", "封路"],
        },
        "traffic": {
            "domain_label": "交通系统",
            "allowed_keywords": ["车流", "路网", "道路", "主干道", "匝道", "通行", "拥堵", "封控"],
            "forbidden_keywords": ["电网", "负荷", "变压器", "变电站", "制冷设备", "供暖需求", "工厂班次"],
        },
        "generic": {
            "domain_label": "工业系统",
            "allowed_keywords": [],
            "forbidden_keywords": [],
        },
    }

    DIRECTIONAL_IMPACT_HINTS = {
        "generic": {
            "multiplier": "让相关运行压力在一段时间内明显偏高",
            "hard_zero": "让相关服务能力被压到极低水平并持续一阵",
            "noise_injection": "让相关监测信号持续失真并无规律波动",
            "trend_injection_up": "先把相关状态推到偏高位置，随后再慢慢恢复",
            "trend_injection_down": "先把相关状态压到偏低位置，随后再慢慢恢复",
            "seasonality_injection": "让相关节律性起伏更明显，峰谷重复模式更清晰",
            "step_change_up": "让系统切到更高位的运行状态并维持一段时间",
            "step_change_down": "让系统切到更低位或更受限的运行状态并维持一段时间",
        },
        "power": {
            "multiplier": "让相关负荷压力在一段时间内明显偏高",
            "hard_zero": "让相关供给或服务能力被压到极低水平并持续一阵",
            "noise_injection": "让监测信号持续失真并出现无规律跳变",
            "trend_injection_up": "先把负荷压力推到偏高状态，随后再慢慢恢复",
            "trend_injection_down": "先把运行水平压到偏低状态，随后再慢慢回稳",
            "seasonality_injection": "让负荷峰谷节律更清晰、周期起伏更明显",
            "step_change_up": "让系统切到更高负荷档位并维持一段时间",
            "step_change_down": "让系统切到更低负荷档位并维持一段时间",
        },
        "traffic": {
            "multiplier": "让相关路段在一段时间内明显承压、通行水平偏高",
            "hard_zero": "让通行能力被压到极低水平并持续一阵",
            "noise_injection": "让监测信号持续失真并忽高忽低",
            "trend_injection_up": "先把通行压力推高，随后再慢慢缓解",
            "trend_injection_down": "先把通行水平压低，随后再慢慢恢复",
            "seasonality_injection": "让通勤型峰谷更明显，重复起伏更强",
            "step_change_up": "让路段切到更高通行状态并维持一段时间",
            "step_change_down": "让路段切到更受限或更低位的通行状态并维持一段时间",
        },
    }

    DIRECTIONAL_ANCHOR_KEYWORDS = {
        "up": ["偏高", "承压", "冲高", "推高", "更高位", "更高负荷", "更高通行", "抬高", "高位", "持续高位", "维持高位"],
        "down": ["极低", "低位", "受限", "压低", "更低位", "更低负荷", "更低通行", "停摆", "中断"],
        "neutral": ["跳变", "紊乱", "失真", "无规律", "忽高忽低", "周期", "节律", "峰谷", "起伏"],
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
            logger.info(f"事件驱动LLM客户端初始化成功 (FSL增强版): {self.base_url}")
        except ImportError:
            raise ImportError("请安装openai库: pip install openai httpx")
    
    def generate_event_prompts(
        self,
        dataset_name: str,
        domain_name: str,
        feature_name: str,
        feature_desc: str,
        injection_type: str,
        start_step: int,
        end_step: int,
        seq_len: int,
        causal_scenario: str,
        injection_config: Dict,
        num_levels: int = 3,
    ) -> List[EventDrivenPrompt]:
        """生成多级事件驱动提示词"""
        
        prompts = []
        for level in range(1, num_levels + 1):
            prompt_text = self._generate_single_prompt(
                level=level,
                dataset_name=dataset_name,
                domain_name=domain_name,
                feature_name=feature_name,
                feature_desc=feature_desc,
                injection_type=injection_type,
                start_step=start_step,
                end_step=end_step,
                seq_len=seq_len,
                causal_scenario=causal_scenario,
                injection_config=injection_config
            )
            
            level_config = self.LEVEL_CONFIG[level]
            prompts.append(EventDrivenPrompt(
                prompt=prompt_text,
                level=level,
                level_name=level_config["name"],
                perspective=level_config["perspective"]
            ))
            
        return prompts
    
    def _generate_single_prompt(
        self,
        level: int,
        dataset_name: str,
        domain_name: str,
        feature_name: str,
        feature_desc: str,
        injection_type: str,
        start_step: int,
        end_step: int,
        seq_len: int,
        causal_scenario: str,
        injection_config: Dict,
        max_retries: int = 3
    ) -> str:
        """生成基于新闻/事件驱动的模糊指令，带有Few-Shot Learning和自校验重试机制"""
        
        change_desc = self._get_change_description(injection_type, injection_config)
        few_shot_examples = self.FEW_SHOT_EXAMPLES.get(level, self.FEW_SHOT_EXAMPLES[2])
        role_desc = self.ROLE_DESCRIPTIONS.get(level, self.ROLE_DESCRIPTIONS[2])
        time_hint = self._get_time_hint(start_step, end_step, seq_len, level)
        domain_key = InjectorFactory.get_domain_key(dataset_name)
        domain_rules = self.DOMAIN_RULES.get(domain_key, self.DOMAIN_RULES["generic"])
        weak_signal = self._get_weak_signal_hint(injection_type, injection_config)
        directional_impact = self._get_directional_impact_hint(injection_type, injection_config, domain_key)
        
        system_prompt = f"""你是一个高级语料生成器，负责将后台的数值变化翻译为特定语气的文本。

【你的角色设定】：{role_desc}

【学习以下高质量示例的风格（Few-Shot Learning）】：
{few_shot_examples}

【当前领域约束】：
- 数据集：{dataset_name}
- 领域：{domain_name}
- 当前对象：{feature_desc}（{feature_name}）
- 允许贴近的领域词：{", ".join(domain_rules["allowed_keywords"]) if domain_rules["allowed_keywords"] else "无特别要求"}
- 绝对禁止跨领域词：{", ".join(domain_rules["forbidden_keywords"]) if domain_rules["forbidden_keywords"] else "无"}

【生成原则】：
- 你要学习的是“模糊、间接、自然”的表达风格，而不是示例中的具体行业设定。
- 你必须保留至少一个可推断的弱语义锚点，例如：
  持续抬升、短时冲高后回落、突然切换后维持、杂乱跳变、降到极低水平。
- 如果任务本身带有方向性，你必须保留一个“事件影响”的弱方向暗示，
  让人能推断是偏高、偏低、更受限、更宽松还是逐步缓解，而不是只看出“有异常”。
- 方向暗示应优先通过事件后果来表达，例如“使相关系统持续承压”“让通行能力被压到更低位”，
  而不要生硬地说“数据上升”或“数据下降”。
- 绝对不能只说“异常”“不对劲”“有波动”，否则无法区分任务类型。
- 严格保持在{domain_name}语境内，不要借用其他行业叙事模板。

【强制要求】：
仔细模仿上方示例的语气和结构。只输出最终的一句话（50字以内），禁止包含任何解释、分析或阿拉伯数字！"""

        user_prompt = f"""
请基于以下真实输入，生成一句话回复：
输入：数据集={dataset_name}, 领域={domain_name}, 对象={feature_desc} ({feature_name}), 变化={change_desc}, 弱线索={weak_signal}, 方向影响={directional_impact}, 诱因={causal_scenario}, 时间={time_hint}
输出："""

        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.7 + (attempt * 0.15),
                    max_tokens=100
                )
                result = response.choices[0].message.content.strip().strip('"\'')
                result = result.replace('**', '')
                
                is_valid, error_msg = self._validate_event_driven_fuzziness(
                    result,
                    level,
                    domain_key,
                    injection_type,
                    injection_config,
                )
                
                if is_valid:
                    return result
                else:
                    logger.warning(f"L{level}文本违规 (Attempt {attempt+1}): {error_msg} -> '{result}'")
                    user_prompt += f"\n\n注意！你刚才生成的'{result}'不合格！原因：{error_msg}。请完全重写，摒弃一切数学或数据视角的词汇，只描述现实事件！"
                    
            except Exception as e:
                logger.error(f"LLM调用异常: {e}")
                time.sleep(1)
                
        return self._get_fallback_prompt(level, feature_desc, causal_scenario, weak_signal, directional_impact, time_hint)
    
    def _validate_event_driven_fuzziness(
        self,
        text: str,
        level: int,
        domain_key: str,
        injection_type: str,
        injection_config: Dict,
    ) -> Tuple[bool, str]:
        """专门针对事件驱动指令的校验器"""
        
        if re.search(r'\d+(?:\.\d+)?(?:倍|%|个百分点)', text):
            return False, "绝对禁止出现具体的数值比例或倍数！"
                
        math_keywords = ['乘以', '除以', '加上', '减去', '方差', '均值', '基线', '斜率', '噪声']
        for kw in math_keywords:
            if kw in text:
                return False, f"你的回复太像数学题了，禁止使用技术词汇：{kw}"

        if level >= 2:
            data_words = ['数据', '曲线', '图表', '时间序列', '序列', '变压器读数', '监测值']
            for kw in data_words:
                if kw in text:
                    return False, f"你在直接描述数据！请描述现实世界中发生的事件，不要提'{kw}'。"
                    
        if level >= 3:
            direct_verbs = ['上升', '下降', '增加', '减少', '激增', '暴跌', '攀升', '升高', '降低']
            for kw in direct_verbs:
                if kw in text:
                    return False, f"太直接了！不要用'{kw}'来暗示变化，请只描述导致这个变化的现实场景。"

        weak_signal_words = ['持续', '回落', '恢复', '维持', '切换', '跳变', '停摆', '低位', '承压', '冲高', '紊乱', '周期', '节律', '峰谷', '起伏']
        if not any(word in text for word in weak_signal_words):
            return False, "表达过于空泛，必须保留至少一个可推断的弱语义锚点。"

        expected_direction = self._infer_expected_direction(injection_type, injection_config)
        if expected_direction != "neutral":
            direction_words = self.DIRECTIONAL_ANCHOR_KEYWORDS.get(expected_direction, [])
            if not any(word in text for word in direction_words):
                return False, "缺少弱方向线索，无法判断事件最终是推高、压低还是切到更受限/更高位状态。"

        domain_rules = self.DOMAIN_RULES.get(domain_key, self.DOMAIN_RULES["generic"])
        for kw in domain_rules["forbidden_keywords"]:
            if kw in text:
                return False, f"出现了跨领域词汇：{kw}"

        return True, ""

    def _get_weak_signal_hint(self, injection_type: str, config: Dict) -> str:
        weak_signals = {
            "multiplier": "明显偏高并维持一段时间，之后才逐步回落",
            "hard_zero": "快速降到极低水平，并维持一段时间后才恢复",
            "noise_injection": "读数变得杂乱无章，呈现持续跳变",
            "trend_injection": "先偏离常态后再回到稳定状态",
            "step_change": "突然切换到新的状态，并维持一段时间",
            "seasonality_injection": "周期起伏突然更明显，或者原有峰谷被压平一阵",
        }
        return weak_signals.get(injection_type, "出现可辨认但间接的运行变化")

    def _infer_expected_direction(self, injection_type: str, config: Dict) -> str:
        if injection_type == "multiplier":
            return "up"
        if injection_type == "hard_zero":
            return "down"
        if injection_type == "noise_injection":
            return "neutral"
        if injection_type == "trend_injection":
            return "up" if config.get("direction", "upward") == "upward" else "down"
        if injection_type == "step_change":
            return config.get("direction", "up")
        if injection_type == "seasonality_injection":
            return "neutral"
        return "neutral"

    def _get_directional_impact_hint(self, injection_type: str, config: Dict, domain_key: str) -> str:
        direction = self._infer_expected_direction(injection_type, config)
        if injection_type == "trend_injection":
            key = f"{injection_type}_{direction}"
        elif injection_type == "step_change":
            key = f"{injection_type}_{direction}"
        else:
            key = injection_type
        bank = self.DIRECTIONAL_IMPACT_HINTS.get(domain_key, self.DIRECTIONAL_IMPACT_HINTS["generic"])
        return bank.get(key, self.DIRECTIONAL_IMPACT_HINTS["generic"].get(key, "让相关系统表现出可推断的方向影响"))
    
    def _get_change_description(self, injection_type: str, config: Dict) -> str:
        """
        将注入配置翻译为供 LLM 反推 Prompt 的中文物理描述。
        描述应准确反映注入后信号的形态，而非注入参数本身，
        以便 LLM 能以此为锚点编造合理的事件性语言。
        """
        direction_zh = {"upward": "上升", "downward": "下降",
                        "up": "上跳", "down": "下跳"}
        descriptions = {
            "multiplier": (
                f"数值被抬升到更高水平并维持一段时间，末段再逐步回落"
            ),
            "hard_zero": (
                f"数值急剧下跌并长时间维持在底部低位（约 {config.get('floor_value', 0):.1f}），"
                f"随后缓慢恢复"
            ),
            "noise_injection": (
                f"数值丧失规律性，变为幅度约 ±{config.get('noise_std', 1):.1f} 的随机杂波"
                f"（基线约 {config.get('baseline', 0):.1f}）"
            ),
            "trend_injection": (
                f"数值先{'上升' if config.get('direction', 'upward') == 'upward' else '下降'}"
                f"后平稳回落，呈钟形波动，"
                f"峰值偏移约 {abs(config.get('amplitude', 0)):.2f}"
            ),
            "step_change": (
                f"数值瞬间{'抬升' if config.get('direction', 'up') == 'up' else '跌落'}"
                f" {abs(config.get('magnitude', 0)):.2f} 单位并持续运行，末期渐缓恢复"
            ),
            "seasonality_injection": (
                f"局部窗口中出现更明显的规律性周期起伏，"
                f"循环数约 {config.get('cycles', 0)} 个"
            ),
        }
        return descriptions.get(injection_type, "未知变化")
    
    def _get_time_hint(self, start_step: int, end_step: int, seq_len: int, level: int = 1) -> str:
        """将序列索引伪装成现实生活中的时间描述"""
        mid_point = (start_step + end_step) / 2
        ratio = mid_point / seq_len
        
        # 对于 L1 (调度员)，使用精确的业务时间
        if level == 1:
            if ratio < 0.33:
                return "在早班交接后"
            elif ratio < 0.66:
                return "在今日运行中段"
            else:
                return "在夜间低谷期前"
                
        # 对于 L2 (新闻主播)，使用正式的新闻时间副词
        elif level == 2:
            if ratio < 0.33:
                return "自今日清晨起"
            elif ratio < 0.66:
                return "从今天中午开始"
            else:
                return "预计在今晚深夜"
                
        # 对于 L3 (社媒路人)，使用极其口语化的时间
        else:
            if ratio < 0.33:
                return "大清早的时候"
            elif ratio < 0.66:
                return "刚才"
            else:
                return "就快到半夜的时候"
    
    def _get_fallback_prompt(
        self,
        level: int,
        feature_desc: str,
        causal_scenario: str,
        weak_signal: str,
        directional_impact: str,
        time_hint: str = "今天中午",
    ) -> str:
        """兜底的事件生成"""
        if level == 1:
            return f"请注意，{time_hint}，受{causal_scenario.split('导致')[0]}影响，{feature_desc}将{weak_signal}，并会{directional_impact}。"
        elif level == 2:
            event = causal_scenario.split('导致')[0] if '导致' in causal_scenario else causal_scenario
            return f"突发通报：{time_hint}，已确认{event}，相关系统将{weak_signal}，并会{directional_impact}。"
        else:
            event = causal_scenario.split('导致')[0] if '导致' in causal_scenario else causal_scenario
            return f"{time_hint}，{event}了，后面一阵子一直{weak_signal}，感觉像是{directional_impact}。"


class EventDrivenTestSetBuilder:
    """事件驱动测试集构建器"""
    
    def __init__(
        self,
        csv_path: str,
        dataset_name: str = "ETTh1",
        output_dir: str = "results/testsets/event_driven/generated",
        random_seed: Optional[int] = None,
        seq_len: int = 192,
        api_key: Optional[str] = None,
    ):
        self.data_loader = CSVDataLoader(csv_path, dataset_name)
        self.injector_factory = InjectorFactory(random_seed)
        self.seq_len = seq_len
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.random_seed = random_seed
        if random_seed is not None:
            np.random.seed(random_seed)
            
        self.samples: List[EventDrivenSample] = []
        
        self.api_key = api_key or DEEPSEEK_API_KEY
        self.prompt_generator = None
        if self.api_key:
            try:
                self.prompt_generator = EventDrivenPromptGenerator(self.api_key)
                logger.info("事件驱动提示词生成器初始化成功")
            except Exception as e:
                logger.warning(f"事件驱动生成器初始化失败: {e}")
    
    def build_single_sample(
        self,
        sample_id: str,
        start_idx: int,
        feature: Optional[str] = None,
        injection_type: Optional[str] = None,
    ) -> EventDrivenSample:
        """构建单个事件驱动测试样本"""
        
        if feature is None:
            feature = self.injector_factory.rng.choice(self.data_loader.features)
            
        base_ts, timestamps = self.data_loader.get_sequence(start_idx, self.seq_len, feature)
        
        injector = self.injector_factory.create_injector(injection_type)
        start_step, duration = self.injector_factory.generate_random_config(self.seq_len)
        
        target_ts, mask_gt, injection_config = injector.inject(base_ts, start_step, duration)
        
        domain_key = self.injector_factory.get_domain_key(self.data_loader.dataset_name)
        causal_scenario = injector.get_causal_scenario(domain_key=domain_key, injection_config=injection_config)
        feature_desc = self.data_loader.feature_descriptions.get(feature, self.injector_factory.get_feature_description(feature))
        
        event_prompts = []
        if self.prompt_generator:
            try:
                prompts = self.prompt_generator.generate_event_prompts(
                    dataset_name=self.data_loader.dataset_name,
                    domain_name=self.data_loader.domain,
                    feature_name=feature,
                    feature_desc=feature_desc,
                    injection_type=injector.get_name(),
                    start_step=injection_config['start_step'],
                    end_step=injection_config['end_step'],
                    seq_len=self.seq_len,
                    causal_scenario=causal_scenario,
                    injection_config=injection_config
                )
                event_prompts = [
                    {
                        "level": p.level,
                        "level_name": p.level_name,
                        "perspective": p.perspective,
                        "prompt": p.prompt
                    }
                    for p in prompts
                ]
            except Exception as e:
                logger.warning(f"事件驱动提示词生成失败: {e}")
        
        edit_intent_gt = injector.get_edit_intent(injection_config)
        task_type = injector.get_task_type()
        legacy_task_type = injector.get_legacy_task_type()
        injection_operator = injector.get_injection_operator()

        technical_gt = (
            f"[Technical Ground Truth]\n"
            f"Feature: {feature} ({feature_desc})\n"
            f"Injection Operator: {injection_operator}\n"
            f"Task Type: {task_type}\n"
            f"Legacy Task Type: {legacy_task_type}\n"
            f"Edit Intent GT: {json.dumps(edit_intent_gt, ensure_ascii=False)}\n"
            f"Region: [{injection_config['start_step']}, {injection_config['end_step']}]\n"
            f"Causal: {causal_scenario}"
        )
        
        sample = EventDrivenSample(
            sample_id=sample_id,
            dataset_name=self.data_loader.dataset_name,
            target_feature=feature,
            feature_description=feature_desc,
            task_type=task_type,
            legacy_task_type=legacy_task_type,
            injection_operator=injection_operator,
            edit_intent_gt=edit_intent_gt,
            gt_start=injection_config['start_step'],
            gt_end=injection_config['end_step'],
            event_prompts=event_prompts,
            technical_ground_truth=technical_gt,
            base_ts=base_ts.tolist(),
            target_ts=target_ts.tolist(),
            mask_gt=mask_gt.astype(int).tolist(),
            injection_config=injection_config,
            causal_scenario=causal_scenario,
            seq_len=self.seq_len,
            timestamp=datetime.now().isoformat(),
        )
        
        self.samples.append(sample)
        logger.info(f"样本 {sample_id} 构建: 特征={feature}, 类型={injector.get_name()}, 因果={causal_scenario}")
        
        return sample
    
    def build_batch(self, num_samples: int = 100) -> List[EventDrivenSample]:
        """批量构建测试样本"""
        
        logger.info(f"\n{'='*60}")
        logger.info(f"开始构建事件驱动测试集，共 {num_samples} 个样本")
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
    
    def save_json(self, filename: Optional[str] = None) -> str:
        """保存为JSON格式"""
        
        if filename is None:
            filename = f"event_driven_testset_{self.data_loader.dataset_name}_{len(self.samples)}.json"
            
        output_file = self.output_dir / filename
        
        testset_data = {
            "metadata": {
                "type": "event_driven",
                "description": "事件驱动测试集 - 测试大模型对现实世界新闻/事件的推理能力",
                "schema_version": "2.0",
                "dataset_name": self.data_loader.dataset_name,
                "total_samples": len(self.samples),
                "seq_len": self.seq_len,
                "features": self.data_loader.features,
                "created_at": datetime.now().isoformat(),
                "random_seed": self.random_seed,
                "level_definitions": {
                    "1": "直接业务指令（调度员视角）- 低模糊度",
                    "2": "宏观新闻播报（新闻主播视角）- 中度模糊",
                    "3": "无关联线索（社交媒体路人视角）- 高度模糊"
                }
            },
            "samples": [asdict(s) for s in self.samples],
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(testset_data, f, ensure_ascii=False, indent=2)
            
        logger.info(f"JSON文件已保存: {output_file}")
        
        return str(output_file)
    
    def save_csv(self, filename: Optional[str] = None) -> str:
        """保存为CSV格式"""
        
        if filename is None:
            filename = f"event_driven_testset_{self.data_loader.dataset_name}_{len(self.samples)}.csv"
            
        output_file = self.output_dir / filename
        
        csv_data = []
        for s in self.samples:
            row = {
                "sample_id": s.sample_id,
                "dataset_name": s.dataset_name,
                "target_feature": s.target_feature,
                "feature_description": s.feature_description,
                "task_type": s.task_type,
                "legacy_task_type": s.legacy_task_type,
                "injection_operator": s.injection_operator,
                "edit_intent_gt": json.dumps(s.edit_intent_gt, ensure_ascii=False),
                "gt_start": s.gt_start,
                "gt_end": s.gt_end,
                "causal_scenario": s.causal_scenario,
                "seq_len": s.seq_len,
                "timestamp": s.timestamp,
            }
            
            for ep in s.event_prompts:
                row[f"prompt_L{ep['level']}"] = ep['prompt']
                row[f"level_name_L{ep['level']}"] = ep['level_name']
                row[f"perspective_L{ep['level']}"] = ep['perspective']
            
            row["base_ts"] = json.dumps(s.base_ts)
            row["target_ts"] = json.dumps(s.target_ts)
            row["mask_gt"] = json.dumps(s.mask_gt)
            row["injection_config"] = json.dumps(s.injection_config)
            row["technical_ground_truth"] = s.technical_ground_truth
            
            csv_data.append(row)
        
        df = pd.DataFrame(csv_data)
        df.to_csv(output_file, index=False, encoding='utf-8')
        
        logger.info(f"CSV文件已保存: {output_file}")
        
        return str(output_file)
    
    def _describe_ts_change(self, sample: "EventDrivenSample") -> str:
        """
        将注入配置 + 实际时序数据转换为数学物理描述字符串。

        描述内容：
        - 编辑区间位置与长度
        - 区间内原始信号的均值、标准差、范围
        - 各注入类型对应的具体数值变化（幅度、倍数、均值漂移、波动性变化等）
        - 边界过渡特性
        """
        base   = np.array(sample.base_ts)
        target = np.array(sample.target_ts)
        cfg    = sample.injection_config

        s = sample.gt_start
        e = sample.gt_end + 1          # 切片右端（开区间）
        duration = e - s
        pct = duration / sample.seq_len * 100

        br = base[s:e]                 # 原始区间
        tr = target[s:e]               # 注入后区间
        b_mean, b_std = float(np.mean(br)), float(np.std(br))
        t_mean, t_std = float(np.mean(tr)), float(np.std(tr))
        delta_mean = t_mean - b_mean

        inj = cfg['injection_type']
        lines = []
        lines.append(
            f"  区间位置 : [{s}, {e-1}]，共 {duration} 步"
            f"（占序列 {pct:.1f}%，序列总长 {sample.seq_len}）"
        )
        lines.append(
            f"  原始统计 : 均值={b_mean:.4f}  标准差={b_std:.4f}"
            f"  范围=[{float(br.min()):.4f}, {float(br.max()):.4f}]"
        )

        if inj == 'trend_injection':
            amp = cfg.get('amplitude', 0.0)
            direction = '上升' if amp >= 0 else '下降'
            rel_pct = abs(amp) / max(abs(b_mean), 1e-8) * 100
            lines.append(f"  变化类型 : 升余弦钟形脉冲（{direction}）")
            lines.append(
                f"  峰值偏移 : {amp:+.4f}"
                f"（原始均值的 {rel_pct:.1f}%）"
            )
            lines.append(
                f"  注入后   : 均值={t_mean:.4f}（Δ={delta_mean:+.4f}）"
                f"  标准差={t_std:.4f}（原始 {b_std:.4f}）"
            )
            lines.append(f"  边界特性 : 两端完全光滑（升余弦窗），无硬跳变")

        elif inj == 'multiplier':
            mult     = cfg.get('multiplier', 1.0)
            ramp_out = cfg.get('ramp_out', 0)
            lines.append(f"  变化类型 : 乘法放大 ×{mult}")
            lines.append(
                f"  均值变化 : {b_mean:.4f} → {t_mean:.4f}"
                f"（放大 {t_mean/max(abs(b_mean),1e-8):.2f}×）"
            )
            lines.append(
                f"  标准差   : {b_std:.4f} → {t_std:.4f}"
                f"（波动性同步放大 {t_std/max(b_std,1e-8):.2f}×）"
            )
            lines.append(
                f"  边界特性 : 起点硬跳变（瞬时激增），"
                f"末端 {ramp_out} 步线性退回原值"
            )

        elif inj == 'hard_zero':
            floor = cfg.get('floor_value', 0.0)
            ramp  = cfg.get('ramp', 0)
            drop  = floor - b_mean
            drop_pct = drop / max(abs(b_mean), 1e-8) * 100
            lines.append(f"  变化类型 : 跌至数据自适应底部基线")
            lines.append(
                f"  底部基线 : {floor:.4f}"
                f"（原始均值 {b_mean:.4f}，"
                f"降幅 {drop:+.4f} / {drop_pct:+.1f}%）"
            )
            lines.append(
                f"  注入后   : 均值={t_mean:.4f}  标准差={t_std:.4f}"
                f"（主体趋近平线，std 由 {b_std:.4f} 降至 {t_std:.4f}）"
            )
            lines.append(
                f"  边界特性 : 入口/出口各 {ramp} 步线性软过渡，"
                f"中段严格维持底值"
            )

        elif inj == 'noise_injection':
            baseline  = cfg.get('baseline', 0.0)
            noise_std = cfg.get('noise_std', 0.0)
            data_std  = cfg.get('data_std', 1.0)
            snr_ratio = noise_std / max(data_std, 1e-8)
            mean_shift = baseline - b_mean
            lines.append(f"  变化类型 : 高斯杂波替换（传感器离线/强干扰）")
            lines.append(
                f"  杂波参数 : 均值={baseline:.4f}  标准差={noise_std:.4f}"
                f"（原始信号 std 的 {snr_ratio:.2f}×）"
            )
            lines.append(
                f"  均值漂移 : {b_mean:.4f} → {baseline:.4f}"
                f"（偏移 {mean_shift:+.4f}）"
            )
            lines.append(
                f"  波动性   : std {b_std:.4f} → {t_std:.4f}"
                f"（放大 {t_std/max(b_std,1e-8):.2f}×，区间已失去原信号规律）"
            )

        elif inj == 'seasonality_injection':
            cycles = int(cfg.get('cycles', 1))
            phase = float(cfg.get('phase', 0.0))
            amplitude = float(cfg.get('seasonal_amplitude', 0.0))
            amplitude_ratio = float(cfg.get('seasonal_amplitude_ratio', 0.0))
            lines.append("  变化类型 : 局部周期残差注入")
            lines.append(
                f"  周期参数 : 区间内固定 {cycles} 个循环  初相位={phase:.2f}"
            )
            lines.append(
                f"  周期幅度 : {amplitude:+.4f}"
                f"（约为全序列范围的 {amplitude_ratio * 100:.1f}%）"
            )
            lines.append(
                f"  注入后   : 均值={t_mean:.4f}（Δ={delta_mean:+.4f}）"
                f"  标准差={t_std:.4f}（原始 {b_std:.4f}）"
            )
            lines.append("  边界特性 : 首尾由平滑包络压到接近 0，避免编辑区边界跳变")

        elif inj == 'step_change':
            mag      = cfg.get('magnitude', 0.0)
            ramp_out = cfg.get('ramp_out', 0)
            data_range = float(np.max(base)) - float(np.min(base))
            mag_pct  = abs(mag) / max(data_range, 1e-8) * 100
            direction = '上跳' if mag >= 0 else '下跳'
            lines.append(f"  变化类型 : 阶跃{direction}")
            lines.append(
                f"  跳变幅度 : {mag:+.4f}"
                f"（全序列数据范围 {data_range:.4f} 的 {mag_pct:.1f}%）"
            )
            lines.append(
                f"  注入后   : 均值={t_mean:.4f}（Δ={delta_mean:+.4f}）"
                f"  标准差={t_std:.4f}（原始 {b_std:.4f}）"
            )
            lines.append(
                f"  边界特性 : 起点硬跳变（瞬时切换），"
                f"末端 {ramp_out} 步渐进恢复至原值"
            )

        return "\n".join(lines)

    def generate_report(self):
        """生成完整样本报告，每条样本包含精确的时序物理变化数学描述"""
        report_file = self.output_dir / "event_driven_report.txt"

        task_type_dist: Dict[str, int] = {}
        legacy_task_type_dist: Dict[str, int] = {}
        feature_dist:   Dict[str, int] = {}
        inj_type_dist:  Dict[str, int] = {}
        effect_family_dist: Dict[str, int] = {}

        for s in self.samples:
            task_type_dist[s.task_type] = task_type_dist.get(s.task_type, 0) + 1
            legacy_task_type_dist[s.legacy_task_type] = legacy_task_type_dist.get(s.legacy_task_type, 0) + 1
            feature_dist[s.target_feature] = feature_dist.get(s.target_feature, 0) + 1
            inj_type_dist[s.injection_operator] = inj_type_dist.get(s.injection_operator, 0) + 1
            effect_family = s.edit_intent_gt.get("effect_family", "unknown")
            effect_family_dist[effect_family] = effect_family_dist.get(effect_family, 0) + 1

        INJ_NAME_ZH = {
            'multiplier':      '乘法放大  (MultiplierInjector)',
            'hard_zero':       '底值归零  (HardZeroInjector)',
            'noise_injection':  '杂波替换  (NoiseInjector)',
            'trend_injection':  '钟形脉冲  (TrendInjector)',
            'seasonality_injection': '周期调制  (SeasonalityInjector)',
            'step_change':     '阶跃切换  (StepChangeInjector)',
        }

        SEP  = "=" * 70 + "\n"
        SEP2 = "-" * 70 + "\n"

        with open(report_file, 'w', encoding='utf-8') as f:
            # ── 头部信息 ──────────────────────────────────────────────
            f.write(SEP)
            f.write("BetterTSE 事件驱动测试集报告\n")
            f.write(SEP + "\n")

            f.write("【核心理念】\n")
            f.write("不是让模型听懂【模糊的技术指令】，而是让它理解【现实世界的新闻/事件】，\n")
            f.write("由模型自己推理出物理影响，再完成时序编辑。\n\n")

            f.write(f"数据集    : {self.data_loader.dataset_name}\n")
            f.write(f"样本数量  : {len(self.samples)}\n")
            f.write(f"序列长度  : {self.seq_len}\n")
            f.write("标签体系  : 通用形态语义（兼容保留 legacy task label）\n")
            f.write(f"生成时间  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write(SEP2)
            f.write("任务分布（通用 task_type）\n")
            f.write(SEP2)
            for tt, count in sorted(task_type_dist.items()):
                f.write(f"  {tt:20s} : {count}\n")
            f.write("\n")

            f.write(SEP2)
            f.write("编辑目标分布（effect_family）\n")
            f.write(SEP2)
            for family, count in sorted(effect_family_dist.items()):
                f.write(f"  {family:20s} : {count}\n")
            f.write("\n")

            f.write(SEP2)
            f.write("兼容旧标签分布（legacy_task_type）\n")
            f.write(SEP2)
            for tt, count in sorted(legacy_task_type_dist.items()):
                f.write(f"  {tt:20s} : {count}\n")
            f.write("\n")

            # ── 三级视角说明 ──────────────────────────────────────────
            f.write(SEP2)
            f.write("三级事件驱动视角\n")
            f.write(SEP2)
            f.write("  L1 直接业务指令（调度员视角）  — 低模糊度，含明确变化方向\n")
            f.write("  L2 宏观新闻播报（新闻主播视角）— 中度模糊，只描述宏观事件\n")
            f.write("  L3 无关联线索（社交媒体路人）  — 高度模糊，仅描述生活现象\n\n")

            # ── 注入类型说明 ──────────────────────────────────────────
            f.write(SEP2)
            f.write("注入类型说明（6 种物理变化）\n")
            f.write(SEP2)
            f.write("  multiplier     乘法放大  : 区间内信号整体乘以随机倍数（2~4×），\n")
            f.write("                             起点硬跳，末端渐退\n")
            f.write("  hard_zero      底值归零  : 区间内信号跌至数据低3%分位数，\n")
            f.write("                             入出口各有软过渡\n")
            f.write("  noise_injection 杂波替换 : 区间内信号被高斯杂波取代，\n")
            f.write("                             均值近数据底部，标准差为原std的1.5~3×\n")
            f.write("  trend_injection 钟形脉冲 : 区间内叠加升余弦钟形偏移（上升或下降），\n")
            f.write("                             两端完全光滑，无任何硬边界\n")
            f.write("  seasonality_injection 周期调制 : 区间内增强或削弱重复起伏，\n")
            f.write("                             保持局部周期语义，支持峰谷放大与压平\n")
            f.write("  step_change    阶跃切换  : 区间起点硬跳变，主体持续偏移，\n")
            f.write("                             末端1/3区间渐进恢复\n\n")

            # ── 分布统计 ──────────────────────────────────────────────
            f.write(SEP2)
            f.write("注入类型分布\n")
            f.write(SEP2)
            for k, v in sorted(inj_type_dist.items(), key=lambda x: -x[1]):
                bar = "█" * v
                f.write(f"  {k:<20} {v:>3} 条  {bar}\n")
            f.write("\n")

            f.write(SEP2)
            f.write("特征分布\n")
            f.write(SEP2)
            for k, v in sorted(feature_dist.items(), key=lambda x: -x[1]):
                bar = "█" * v
                f.write(f"  {k:<6} {v:>3} 条  {bar}\n")
            f.write("\n")

            # ── 全量样本详情 ──────────────────────────────────────────
            f.write(SEP2)
            f.write(f"全量样本详情（共 {len(self.samples)} 条）\n")
            f.write(SEP2)
            f.write("  每条格式：\n")
            f.write("    [编号] 特征 | 注入类型 | 因果场景\n")
            f.write("    时序物理变化（数学层面）\n")
            f.write("    L1/L2/L3 事件驱动提示词\n\n")

            for sample in self.samples:
                inj_type = sample.injection_config['injection_type']
                inj_zh   = INJ_NAME_ZH.get(inj_type, inj_type)
                f.write(f"{'─' * 70}\n")
                f.write(
                    f"[{sample.sample_id}] "
                    f"{sample.target_feature}（{sample.feature_description}）"
                    f" | {inj_zh}\n"
                )
                f.write(f"  因果场景 : {sample.causal_scenario}\n")

                # 数学物理描述
                f.write(self._describe_ts_change(sample) + "\n")

                # 三级提示词
                if sample.event_prompts:
                    f.write("  事件提示词:\n")
                    for ep in sample.event_prompts:
                        f.write(f"    [L{ep['level']} {ep['perspective']}] {ep['prompt']}\n")
                f.write("\n")

        logger.info(f"报告已保存: {report_file}")


def main():
    parser = argparse.ArgumentParser(description='BetterTSE 事件驱动测试集生成器')
    
    parser.add_argument('--csv-path', type=str, 
                        default=r'C:\Users\ghb\Desktop\李老师科研\BetterTSE-main\data\ETTh1.csv',
                        help='CSV数据文件路径')
    parser.add_argument('--dataset-name', type=str, default='ETTh1',
                        help='数据集名称')
    parser.add_argument('--output-dir', type=str, default='results/testsets/event_driven/generated',
                        help='输出目录')
    parser.add_argument('--num-samples', type=int, default=100,
                        help='构建样本数量')
    parser.add_argument('--seq-len', type=int, default=192,
                        help='序列长度')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    parser.add_argument('--api-key', type=str, default=None,
                        help='DeepSeek API密钥')
    
    args = parser.parse_args()
    
    logger.info("=" * 70)
    logger.info("BetterTSE 事件驱动测试集生成器")
    logger.info("=" * 70)
    logger.info(f"CSV文件: {args.csv_path}")
    logger.info(f"数据集: {args.dataset_name}")
    logger.info(f"样本数量: {args.num_samples}")
    logger.info(f"序列长度: {args.seq_len}")
    logger.info(f"输出目录: {args.output_dir}")
    
    try:
        builder = EventDrivenTestSetBuilder(
            csv_path=args.csv_path,
            dataset_name=args.dataset_name,
            output_dir=args.output_dir,
            random_seed=args.seed,
            seq_len=args.seq_len,
            api_key=args.api_key,
        )
        
        builder.build_batch(num_samples=args.num_samples)
        builder.save_json()
        builder.save_csv()
        builder.generate_report()
        
        logger.info("\n" + "=" * 70)
        logger.info("事件驱动测试集构建完成!")
        logger.info(f"输出目录: {builder.output_dir}")
        logger.info("=" * 70)
        
    except Exception as e:
        logger.error(f"构建失败: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
