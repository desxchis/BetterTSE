"""
BetterTSE 测试集构建工具 v2.0
============================

核心功能：基于真实公开数据集，构建时间序列编辑测试集

流程：
1. 加载真实CSV时间序列数据（ETTh1/ETTm1/Traffic等）
2. 注入确定的物理变化（Base TS → Target TS）
3. 用LLM反向生成多条不同模糊程度的宏观描述（3-5条Vague Prompts）
4. 输出四元组：{Base_TS, Vague_Prompts, Target_TS, Mask_GT}

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


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("BuildTestSet")


DEEPSEEK_API_KEY = "sk-5dffb5b7887e474b869647790c534f29"


@dataclass
class VaguePromptItem:
    """单条模糊指令"""
    prompt: str
    vagueness_level: int  # 1=轻微模糊, 2=中度模糊, 3=高度模糊, 4=极度模糊
    vagueness_label: str  # 模糊程度标签


@dataclass
class TestSetSample:
    """测试集样本 - 四元组格式"""
    sample_id: str
    dataset_name: str
    base_ts: List[float]
    target_ts: List[float]
    mask_gt: List[int]
    vague_prompts: List[Dict]
    technical_prompt: str
    injection_config: Dict[str, Any]
    feature_name: str
    seq_len: int
    timestamp: str = ""


class CSVDataLoader:
    """真实CSV数据加载器"""
    
    DATASET_INFO = {
        "ETTh1": {"freq": "hourly", "desc": "Electricity Transformer Temperature (hourly)", "features_desc": {
            "HUFL": "High Useful Load (高压侧有效负荷)",
            "HULL": "High Useless Load (高压侧无效负荷)",
            "MUFL": "Middle Useful Load (中压侧有效负荷)",
            "MULL": "Middle Useless Load (中压侧无效负荷)",
            "LUFL": "Low Useful Load (低压侧有效负荷)",
            "LULL": "Low Useless Load (低压侧无效负荷)",
            "OT": "Oil Temperature (变压器油温)"
        }},
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
        try:
            self.data = pd.read_csv(self.csv_path)
            logger.info(f"数据形状: {self.data.shape}")
            
            if 'date' in self.data.columns:
                self.timestamps = self.data['date'].tolist()
                self.data = self.data.drop(columns=['date'])
            
            self.features = self.data.columns.tolist()
            self.feature_to_idx = {feat: idx for idx, feat in enumerate(self.features)}
            logger.info(f"特征列表 ({len(self.features)}): {self.features}")
            
        except Exception as e:
            logger.error(f"加载数据失败: {e}")
            raise
            
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
    
    def get_feature_description(self, feature: str) -> str:
        return self.feature_descriptions.get(feature, feature)


class PhysicalInjector(ABC):
    """物理变化注入器基类"""
    
    def __init__(self, rng: np.random.RandomState):
        self.rng = rng
        
    @abstractmethod
    def inject(self, base_ts: np.ndarray, start_step: int, duration: int) -> Tuple[np.ndarray, np.ndarray, Dict]:
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        pass
    
    @abstractmethod
    def get_causal_description(self) -> str:
        pass


class MultiplierInjector(PhysicalInjector):
    """乘法倍数注入器 - 数值放大/缩小"""
    
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
    
    def get_causal_description(self) -> str:
        return "数值放大（如热浪导致负荷激增）"


class HardZeroInjector(PhysicalInjector):
    """强制归零注入器 - 数值置零"""
    
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
    
    def get_causal_description(self) -> str:
        return "强制归零（如设施关闭导致用量归零）"


class NoiseInjector(PhysicalInjector):
    """底噪替换注入器 - 替换为噪声"""
    
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
    
    def get_causal_description(self) -> str:
        return "底噪替换（如传感器离线）"


class TrendInjector(PhysicalInjector):
    """趋势注入器 - 添加上升/下降趋势"""
    
    def __init__(self, rng: np.random.RandomState):
        super().__init__(rng)
        
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
    
    def get_causal_description(self) -> str:
        return "趋势注入（如市场变化导致趋势反转）"


class StepChangeInjector(PhysicalInjector):
    """阶跃变化注入器 - 突然跳变"""
    
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
    
    def get_causal_description(self) -> str:
        return "阶跃变化（如设备切换导致数值跳变）"


class InjectorFactory:
    """注入器工厂"""
    
    INJECTOR_REGISTRY = {
        "multiplier": MultiplierInjector,
        "hard_zero": HardZeroInjector,
        "noise_injection": NoiseInjector,
        "trend_injection": TrendInjector,
        "step_change": StepChangeInjector,
    }
    
    CAUSAL_SCENARIOS = {
        "multiplier": [
            "热浪导致空调用电量激增",
            "促销活动导致销量翻倍",
            "节假日导致流量大幅上升",
            "极端天气导致负荷飙升",
        ],
        "hard_zero": [
            "设施关闭导致用电量归零",
            "系统维护导致流量归零",
            "设备故障导致读数为零",
            "计划停机导致数据归零",
        ],
        "noise_injection": [
            "传感器离线导致读数为底噪",
            "设备维护期间数据异常",
            "信号中断导致随机读数",
            "通信故障导致数据漂移",
        ],
        "trend_injection": [
            "市场变化导致上升趋势",
            "设备老化导致下降趋势",
            "季节变化导致趋势转变",
            "政策调整导致趋势反转",
        ],
        "step_change": [
            "设备切换导致数值跳变",
            "配置更改导致阶跃变化",
            "模式切换导致突然变化",
            "系统重启导致数值跳变",
        ],
    }
    
    def __init__(self, random_seed: Optional[int] = None):
        self.rng = np.random.RandomState(random_seed)
        
    def create_injector(self, injection_type: Optional[str] = None) -> PhysicalInjector:
        if injection_type is None:
            injection_type = self.rng.choice(list(self.INJECTOR_REGISTRY.keys()))
        injector_class = self.INJECTOR_REGISTRY[injection_type]
        return injector_class(self.rng)
    
    def get_random_scenario(self, injection_type: str) -> str:
        scenarios = self.CAUSAL_SCENARIOS.get(injection_type, ["未知场景"])
        return self.rng.choice(scenarios)
    
    def generate_random_config(self, seq_len: int) -> Tuple[int, int]:
        duration = self.rng.randint(max(5, seq_len // 20), max(10, seq_len // 8))
        start_step = self.rng.randint(seq_len // 8, seq_len - duration - seq_len // 8)
        return start_step, duration


class MultiLevelVaguePromptGenerator:
    """多模糊程度提示词生成器 - 生成3-5条不同模糊程度的指令"""
    
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
            self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
            logger.info(f"LLM客户端初始化成功: {self.base_url}")
        except ImportError:
            raise ImportError("请安装openai库: pip install openai")
            
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
        num_prompts: int = 4,
    ) -> List[VaguePromptItem]:
        """生成多条不同模糊程度的指令"""
        
        prompts = []
        vagueness_levels = list(range(1, min(num_prompts + 1, 5)))
        
        for level in vagueness_levels:
            prompt = self._generate_single_prompt(
                level=level,
                feature_name=feature_name,
                feature_desc=feature_desc,
                injection_type=injection_type,
                start_step=start_step,
                end_step=end_step,
                seq_len=seq_len,
                causal_scenario=causal_scenario,
                injection_config=injection_config,
            )
            prompts.append(VaguePromptItem(
                prompt=prompt,
                vagueness_level=level,
                vagueness_label=self.VAGUENESS_LEVELS[level]["label"]
            ))
            
        return prompts
        
    def _generate_single_prompt(
        self,
        level: int,
        feature_name: str,
        feature_desc: str,
        injection_type: str,
        start_step: int,
        end_step: int,
        seq_len: int,
        causal_scenario: str,
        injection_config: Dict,
    ) -> str:
        """生成单条指定模糊程度的指令"""
        
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
        
        system_prompt = f"""你是一个业务分析师，擅长用自然语言描述数据变化。
你的任务是将技术性的数据变化描述转换为口语化的自然语言指令。

{level_instructions.get(level, level_instructions[2])}

【严格要求】：
1. 只返回一句指令，不要任何多余的解释
2. 语言风格要自然，像是给AI助手下达任务
3. 必须符合指定的模糊程度"""

        user_prompt = f"""请将以下技术描述转换为模糊程度{level}的自然语言指令：

目标特征: {feature_name} ({feature_desc})
变化类型: {change_description}
发生位置: {relative_position} (约占总长度的{(start_step + end_step) / 2 / seq_len * 100:.0f}%处)
因果场景: {causal_scenario}

请生成一句模糊程度{level}的指令："""

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.8,
                max_tokens=150
            )
            result = response.choices[0].message.content.strip().strip('"\'')
            return result
        except Exception as e:
            logger.error(f"LLM调用失败: {e}")
            return self._get_fallback_prompt(level, feature_name, relative_position, change_description)
            
    def _get_fallback_prompt(self, level: int, feature_name: str, relative_position: str, change_desc: str) -> str:
        """LLM调用失败时的备用提示词"""
        fallbacks = {
            1: f"请调整{feature_name}在{relative_position}的数据，大约有{change_desc}的变化。",
            2: f"请调整{feature_name}在{relative_position}的数据，出现了明显变化。",
            3: f"{feature_name}在某个时间段出现了异常，请调整一下。",
            4: "数据中有一段出现了业务相关的异常情况，请帮忙修正。",
        }
        return fallbacks.get(level, fallbacks[2])
            
    def _get_relative_position(self, start_step: int, end_step: int, seq_len: int) -> str:
        mid_point = (start_step + end_step) / 2
        ratio = mid_point / seq_len
        
        if ratio < 0.25:
            return "观测窗口的前段"
        elif ratio < 0.5:
            return "观测窗口的前中段"
        elif ratio < 0.75:
            return "观测窗口的后中段"
        else:
            return "观测窗口的后段"
            
    def _get_change_description(self, injection_type: str, config: Dict) -> str:
        descriptions = {
            "multiplier": f"数值放大到约{config.get('multiplier', 2):.1f}倍",
            "hard_zero": "数值强制归零",
            "noise_injection": "数值变为底噪水平",
            "trend_injection": f"出现{config.get('direction', 'upward')}趋势",
            "step_change": f"发生{config.get('direction', 'up')}向阶跃跳变",
        }
        return descriptions.get(injection_type, "数值发生变化")


class TechnicalPromptBuilder:
    """技术性提示词构建器"""
    
    @staticmethod
    def build(
        feature_name: str,
        feature_desc: str,
        injection_type: str,
        start_step: int,
        end_step: int,
        seq_len: int,
        causal_scenario: str,
        injection_config: Dict,
    ) -> str:
        """构建详细的技术性描述"""
        
        change_desc = {
            "multiplier": f"数值乘以{injection_config.get('multiplier', 2):.1f}倍",
            "hard_zero": "数值强制设为0",
            "noise_injection": f"数值替换为底噪（基准值约{injection_config.get('baseline', 5):.1f}）",
            "trend_injection": f"添加{injection_config.get('direction', 'upward')}趋势（斜率{injection_config.get('slope', 0):.4f}）",
            "step_change": f"数值{injection_config.get('direction', 'up')}跳变{abs(injection_config.get('magnitude', 0)):.2f}",
        }
        
        prompt = f"""[Technical Description]
Feature: {feature_name} ({feature_desc})
Sequence Length: {seq_len} steps
Edit Region: steps {start_step} to {end_step} (duration: {end_step - start_step + 1} steps)
Change Type: {injection_type}
Change Detail: {change_desc.get(injection_type, 'N/A')}
Causal Scenario: {causal_scenario}
"""
        return prompt


class TestSetBuilder:
    """测试集构建器"""
    
    def __init__(
        self,
        csv_path: str,
        dataset_name: str = "ETTh1",
        api_key: Optional[str] = None,
        output_dir: str = "test_sets",
        random_seed: Optional[int] = None,
        seq_len: int = 192,
        num_prompts_per_sample: int = 4,
    ):
        self.data_loader = CSVDataLoader(csv_path, dataset_name)
        self.injector_factory = InjectorFactory(random_seed)
        self.seq_len = seq_len
        self.num_prompts_per_sample = num_prompts_per_sample
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.llm_generator = None
        if api_key:
            self.llm_generator = MultiLevelVaguePromptGenerator(api_key)
            
        self.random_seed = random_seed
        if random_seed is not None:
            np.random.seed(random_seed)
            
        self.samples: List[TestSetSample] = []
        
    def build_single_sample(
        self,
        sample_id: str,
        start_idx: int,
        feature: Optional[str] = None,
        injection_type: Optional[str] = None,
    ) -> TestSetSample:
        """构建单个测试样本"""
        
        if feature is None:
            feature = self.injector_factory.rng.choice(self.data_loader.features)
            
        base_ts, timestamps = self.data_loader.get_sequence(start_idx, self.seq_len, feature)
        
        injector = self.injector_factory.create_injector(injection_type)
        start_step, duration = self.injector_factory.generate_random_config(self.seq_len)
        
        target_ts, mask_gt, injection_config = injector.inject(base_ts, start_step, duration)
        
        causal_scenario = self.injector_factory.get_random_scenario(injector.get_name())
        feature_desc = self.data_loader.get_feature_description(feature)
        
        technical_prompt = TechnicalPromptBuilder.build(
            feature_name=feature,
            feature_desc=feature_desc,
            injection_type=injector.get_name(),
            start_step=injection_config['start_step'],
            end_step=injection_config['end_step'],
            seq_len=self.seq_len,
            causal_scenario=causal_scenario,
            injection_config=injection_config,
        )
        
        if self.llm_generator:
            logger.info(f"调用LLM生成{self.num_prompts_per_sample}条不同模糊程度的提示词...")
            vague_prompt_items = self.llm_generator.generate_multiple_prompts(
                feature_name=feature,
                feature_desc=feature_desc,
                injection_type=injector.get_name(),
                start_step=injection_config['start_step'],
                end_step=injection_config['end_step'],
                seq_len=self.seq_len,
                causal_scenario=causal_scenario,
                injection_config=injection_config,
                num_prompts=self.num_prompts_per_sample,
            )
            time.sleep(0.3)
        else:
            relative_pos = self._get_relative_position(start_step, injection_config['end_step'], self.seq_len)
            vague_prompt_items = [
                VaguePromptItem(
                    prompt=f"请调整{feature}在{relative_pos}的数据。",
                    vagueness_level=i,
                    vagueness_label=f"模糊等级{i}"
                )
                for i in range(1, self.num_prompts_per_sample + 1)
            ]
            
        vague_prompts_dict = [
            {"prompt": item.prompt, "vagueness_level": item.vagueness_level, "vagueness_label": item.vagueness_label}
            for item in vague_prompt_items
        ]
            
        sample = TestSetSample(
            sample_id=sample_id,
            dataset_name=self.data_loader.dataset_name,
            base_ts=base_ts.tolist(),
            target_ts=target_ts.tolist(),
            mask_gt=mask_gt.astype(int).tolist(),
            vague_prompts=vague_prompts_dict,
            technical_prompt=technical_prompt,
            injection_config=injection_config,
            feature_name=feature,
            seq_len=self.seq_len,
            timestamp=datetime.now().isoformat(),
        )
        
        self.samples.append(sample)
        
        logger.info(f"样本 {sample_id} 构建完成:")
        logger.info(f"  特征: {feature} ({feature_desc})")
        logger.info(f"  注入类型: {injector.get_name()}")
        logger.info(f"  编辑区间: [{injection_config['start_step']}, {injection_config['end_step']}]")
        logger.info(f"  因果场景: {causal_scenario}")
        logger.info(f"  生成指令数: {len(vague_prompt_items)}")
        for i, item in enumerate(vague_prompt_items):
            logger.info(f"    [{item.vagueness_label}] {item.prompt[:50]}...")
        
        return sample
        
    def build_batch(
        self,
        num_samples: int = 10,
        features: Optional[List[str]] = None,
        injection_types: Optional[List[str]] = None,
    ) -> List[TestSetSample]:
        """批量构建测试样本"""
        
        logger.info(f"\n{'='*70}")
        logger.info(f"开始构建测试集，共 {num_samples} 个样本")
        logger.info(f"数据集: {self.data_loader.dataset_name}")
        logger.info(f"序列长度: {self.seq_len}")
        logger.info(f"每样本指令数: {self.num_prompts_per_sample}")
        logger.info(f"{'='*70}\n")
        
        max_start = len(self.data_loader.data) - self.seq_len
        step = max(1, max_start // num_samples)
        
        for i in range(num_samples):
            start_idx = (i * step) % max_start
            
            feature = None
            if features:
                feature = features[i % len(features)]
                
            injection_type = None
            if injection_types:
                injection_type = injection_types[i % len(injection_types)]
                
            sample_id = f"{self.data_loader.dataset_name}_{i+1:04d}"
            
            try:
                self.build_single_sample(
                    sample_id=sample_id,
                    start_idx=start_idx,
                    feature=feature,
                    injection_type=injection_type,
                )
            except Exception as e:
                logger.error(f"样本 {sample_id} 构建失败: {e}")
                continue
                
        return self.samples
        
    def save_testset(self, filename: Optional[str] = None) -> str:
        """保存测试集"""
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"testset_{self.data_loader.dataset_name}_{timestamp}.json"
            
        output_file = self.output_dir / filename
        
        testset_data = {
            "metadata": {
                "dataset_name": self.data_loader.dataset_name,
                "total_samples": len(self.samples),
                "seq_len": self.seq_len,
                "features": self.data_loader.features,
                "feature_descriptions": self.data_loader.feature_descriptions,
                "num_prompts_per_sample": self.num_prompts_per_sample,
                "created_at": datetime.now().isoformat(),
                "random_seed": self.random_seed,
            },
            "statistics": self._compute_statistics(),
            "samples": [asdict(s) for s in self.samples],
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(testset_data, f, ensure_ascii=False, indent=2)
            
        logger.info(f"\n测试集已保存至: {output_file}")
        
        self._save_summary_report()
        
        return str(output_file)
        
    def _compute_statistics(self) -> Dict:
        """计算测试集统计信息"""
        if not self.samples:
            return {}
            
        injection_type_dist = {}
        feature_dist = {}
        duration_list = []
        vagueness_dist = {}
        
        for s in self.samples:
            it = s.injection_config.get('injection_type', 'unknown')
            injection_type_dist[it] = injection_type_dist.get(it, 0) + 1
            
            fn = s.feature_name
            feature_dist[fn] = feature_dist.get(fn, 0) + 1
            
            duration_list.append(s.injection_config.get('duration', 0))
            
            for vp in s.vague_prompts:
                level = vp.get('vagueness_level', 0)
                vagueness_dist[level] = vagueness_dist.get(level, 0) + 1
            
        return {
            "injection_type_distribution": injection_type_dist,
            "feature_distribution": feature_dist,
            "vagueness_level_distribution": vagueness_dist,
            "avg_duration": float(np.mean(duration_list)) if duration_list else 0,
            "min_duration": int(min(duration_list)) if duration_list else 0,
            "max_duration": int(max(duration_list)) if duration_list else 0,
        }
        
    def _save_summary_report(self):
        """保存摘要报告"""
        report_file = self.output_dir / f"testset_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        stats = self._compute_statistics()
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write("BetterTSE 测试集构建报告\n")
            f.write("=" * 70 + "\n\n")
            
            f.write(f"数据集: {self.data_loader.dataset_name}\n")
            f.write(f"样本数量: {len(self.samples)}\n")
            f.write(f"序列长度: {self.seq_len}\n")
            f.write(f"每样本指令数: {self.num_prompts_per_sample}\n")
            f.write(f"构建时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("-" * 70 + "\n")
            f.write("注入类型分布:\n")
            f.write("-" * 70 + "\n")
            for it, count in stats.get('injection_type_distribution', {}).items():
                f.write(f"  {it}: {count}\n")
            f.write("\n")
            
            f.write("-" * 70 + "\n")
            f.write("特征分布:\n")
            f.write("-" * 70 + "\n")
            for fn, count in stats.get('feature_distribution', {}).items():
                f.write(f"  {fn}: {count}\n")
            f.write("\n")
            
            f.write("-" * 70 + "\n")
            f.write("模糊程度分布:\n")
            f.write("-" * 70 + "\n")
            for level, count in sorted(stats.get('vagueness_level_distribution', {}).items()):
                label = MultiLevelVaguePromptGenerator.VAGUENESS_LEVELS.get(level, {}).get('label', f'等级{level}')
                f.write(f"  等级{level} ({label}): {count}条\n")
            f.write("\n")
            
            f.write("-" * 70 + "\n")
            f.write("样本示例 (前3个):\n")
            f.write("-" * 70 + "\n")
            for i, sample in enumerate(self.samples[:3]):
                f.write(f"\n样本 {i+1}: {sample.sample_id}\n")
                f.write(f"  特征: {sample.feature_name}\n")
                f.write(f"  注入类型: {sample.injection_config.get('injection_type', 'N/A')}\n")
                f.write(f"  编辑区间: [{sample.injection_config.get('start_step', 0)}, {sample.injection_config.get('end_step', 0)}]\n")
                f.write(f"  模糊指令 ({len(sample.vague_prompts)}条):\n")
                for vp in sample.vague_prompts:
                    f.write(f"    [{vp.get('vagueness_label', 'N/A')}] {vp.get('prompt', 'N/A')}\n")
                f.write(f"  技术提示词:\n{sample.technical_prompt}\n")
                
        logger.info(f"摘要报告已保存至: {report_file}")
        
    def _get_relative_position(self, start_step: int, end_step: int, seq_len: int) -> str:
        mid_point = (start_step + end_step) / 2
        ratio = mid_point / seq_len
        
        if ratio < 0.25:
            return "观测窗口的前段"
        elif ratio < 0.5:
            return "观测窗口的前中段"
        elif ratio < 0.75:
            return "观测窗口的后中段"
        else:
            return "观测窗口的后段"


def main():
    parser = argparse.ArgumentParser(description='BetterTSE 测试集构建工具 v2.0')
    
    parser.add_argument('--csv-path', type=str, 
                        default=r'C:\Users\ghb\Desktop\李老师科研\BetterTSE-main\data\ETTh1.csv',
                        help='CSV数据文件路径')
    parser.add_argument('--dataset-name', type=str, default='ETTh1',
                        help='数据集名称')
    parser.add_argument('--api-key', type=str, default=DEEPSEEK_API_KEY,
                        help='DeepSeek API密钥')
    parser.add_argument('--output-dir', type=str, default='test_sets',
                        help='输出目录')
    parser.add_argument('--num-samples', type=int, default=10,
                        help='构建样本数量')
    parser.add_argument('--seq-len', type=int, default=192,
                        help='序列长度')
    parser.add_argument('--num-prompts', type=int, default=4,
                        help='每样本生成的模糊指令数量（3-5）')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    parser.add_argument('--features', type=str, nargs='+', default=None,
                        help='指定特征列表')
    parser.add_argument('--injection-types', type=str, nargs='+', default=None,
                        choices=['multiplier', 'hard_zero', 'noise_injection', 'trend_injection', 'step_change'],
                        help='指定注入类型')
    
    args = parser.parse_args()
    
    num_prompts = max(3, min(args.num_prompts, 5))
    
    logger.info("=" * 70)
    logger.info("BetterTSE 测试集构建工具 v2.0")
    logger.info("=" * 70)
    logger.info(f"CSV文件: {args.csv_path}")
    logger.info(f"数据集: {args.dataset_name}")
    logger.info(f"样本数量: {args.num_samples}")
    logger.info(f"序列长度: {args.seq_len}")
    logger.info(f"每样本指令数: {num_prompts}")
    logger.info(f"输出目录: {args.output_dir}")
    
    try:
        builder = TestSetBuilder(
            csv_path=args.csv_path,
            dataset_name=args.dataset_name,
            api_key=args.api_key,
            output_dir=args.output_dir,
            random_seed=args.seed,
            seq_len=args.seq_len,
            num_prompts_per_sample=num_prompts,
        )
        
        builder.build_batch(
            num_samples=args.num_samples,
            features=args.features,
            injection_types=args.injection_types,
        )
        
        output_file = builder.save_testset()
        
        logger.info("\n" + "=" * 70)
        logger.info("测试集构建完成!")
        logger.info(f"输出文件: {output_file}")
        logger.info("=" * 70)
        
    except Exception as e:
        logger.error(f"构建失败: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
