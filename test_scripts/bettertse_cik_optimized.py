"""
BetterTSE 时间序列编辑测试框架 - CiK风格优化版
===============================================

针对ETTh1数据集特点优化：
1. 时间尺度适配：ETTh1是小时级数据，duration缩短到3-6小时
2. 单一特征指向：Prompt只针对单一特征，不涉及多特征交互
3. 时间步适配：start_step使用14-17代表下午时段

作者: BetterTSE Team
"""

import os
import sys
import json
import logging
import argparse
import time
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict, field
import numpy as np
import pandas as pd


def mean_squared_error(y_true, y_pred):
    """计算均方误差"""
    return np.mean((np.array(y_true) - np.array(y_pred)) ** 2)


def mean_absolute_error(y_true, y_pred):
    """计算平均绝对误差"""
    return np.mean(np.abs(np.array(y_true) - np.array(y_pred)))


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(name)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger("BetterTSE_CiK_Optimized")


@dataclass
class EvaluationMetrics:
    """评估指标"""
    t_iou: float = 0.0
    feature_accuracy: float = 0.0
    mse_edit_region: float = 0.0
    mae_edit_region: float = 0.0
    mse_preserve_region: float = 0.0
    mae_preserve_region: float = 0.0
    editability_score: float = 0.0
    preservability_score: float = 0.0


@dataclass
class TestSample:
    """测试样本"""
    sample_id: str
    base_ts: List[float]
    target_ts: List[float]
    generated_ts: Optional[List[float]] = None
    gt_mask: Optional[List[int]] = None
    pred_mask: Optional[List[int]] = None
    vague_prompt: str = ""
    cik_prompt: str = ""
    gt_config: Dict = field(default_factory=dict)
    llm_prediction: Dict = field(default_factory=dict)
    metrics: Dict = field(default_factory=dict)


DEEPSEEK_API_KEY = "sk-5dffb5b7887e474b869647790c534f29"


class CSVDataLoader:
    """真实CSV数据加载器"""
    
    def __init__(self, csv_path: str):
        self.csv_path = Path(csv_path)
        self.data = None
        self.features = []
        self.feature_to_idx = {}
        
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV文件不存在: {csv_path}")
            
        self._load_data()
        
    def _load_data(self):
        """加载CSV数据"""
        logger.info(f"正在加载CSV文件: {self.csv_path}")
        
        try:
            self.data = pd.read_csv(self.csv_path)
            logger.info(f"成功加载数据，形状: {self.data.shape}")
            
            numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
            self.features = numeric_cols
            self.feature_to_idx = {feat: idx for idx, feat in enumerate(self.features)}
            
            logger.info(f"检测到 {len(self.features)} 个数值特征: {self.features}")
            
        except Exception as e:
            logger.error(f"加载CSV文件失败: {e}")
            raise
            
    def get_sequence(self, start_idx: int = 0, seq_len: int = 100, features: Optional[List[str]] = None) -> np.ndarray:
        if features is None:
            features = self.features
        end_idx = start_idx + seq_len
        
        if end_idx > len(self.data):
            end_idx = len(self.data)
            start_idx = max(0, end_idx - seq_len)
            
        sequence = self.data[features].iloc[start_idx:end_idx].values
        return sequence.astype(np.float64)
    
    def get_feature_data(self, feature: str, start_idx: int = 0, seq_len: int = 100) -> np.ndarray:
        return self.get_sequence(start_idx, seq_len, [feature]).flatten()
    
    def get_statistics(self) -> Dict:
        stats = {
            "total_rows": len(self.data),
            "features": self.features,
            "feature_stats": {}
        }
        for feat in self.features:
            col_data = self.data[feat]
            stats["feature_stats"][feat] = {
                "mean": float(col_data.mean()),
                "std": float(col_data.std()),
                "min": float(col_data.min()),
                "max": float(col_data.max())
            }
        return stats


class CiKStyleDataSynthesizerOptimized:
    """
    CiK风格数据合成器 - 针对ETTh1优化版
    
    优化点:
    1. duration: 3-6小时 (适配小时级数据)
    2. 单一特征指向的Prompt
    3. 时间步: 14-17代表下午时段
    """
    
    FEATURE_CONTEXTS = {
        "HUFL": {
            "name": "High Useful Load (高压侧有效负荷)",
            "unit": "MW",
            "scenarios": ["heatwave", "industrial_demand", "peak_hours"]
        },
        "HULL": {
            "name": "High Useless Load (高压侧无效负荷)",
            "unit": "MW",
            "scenarios": ["reactive_power", "capacitor_switching"]
        },
        "MUFL": {
            "name": "Middle Useful Load (中压侧有效负荷)",
            "unit": "MW",
            "scenarios": ["distribution_load", "commercial_area"]
        },
        "MULL": {
            "name": "Middle Useless Load (中压侧无效负荷)",
            "unit": "MW",
            "scenarios": ["power_factor_correction"]
        },
        "LUFL": {
            "name": "Low Useful Load (低压侧有效负荷)",
            "unit": "MW",
            "scenarios": ["residential_area", "lighting_load"]
        },
        "LULL": {
            "name": "Low Useless Load (低压侧无效负荷)",
            "unit": "MW",
            "scenarios": ["small_motors", "household_appliances"]
        },
        "OT": {
            "name": "Oil Temperature (变压器油温)",
            "unit": "°C",
            "scenarios": ["cooling_system", "ambient_temperature", "load_correlation"]
        }
    }
    
    def __init__(self, seq_len: int, features: List[str], random_seed: Optional[int] = None):
        self.seq_len = seq_len
        self.features = features
        self.feature_to_idx = {feat: idx for idx, feat in enumerate(self.features)}
        self.rng = np.random.RandomState(random_seed)
        
    def _generate_base_ts(self, df_source: Optional[pd.DataFrame] = None, start_idx: int = 0) -> np.ndarray:
        if df_source is None:
            return np.cumsum(self.rng.randn(self.seq_len, len(self.features)) * 0.1, axis=0) + 20
        return df_source[self.features].iloc[start_idx:start_idx + self.seq_len].values.copy()
    
    def task_heatwave_overload(
        self,
        df_source: Optional[pd.DataFrame] = None,
        start_idx: int = 0,
        start_step: Optional[int] = None,
        duration: Optional[int] = None,
        multiplier: Optional[float] = None,
        include_distractor: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str, Dict]:
        """
        场景1: 热浪导致高负载 (优化版)
        
        物理逻辑：突发热浪导致特定时间段内电力负荷激增
        时间尺度适配: duration=3-6小时 (下午时段)
        """
        if start_step is None:
            start_step = self.rng.randint(14, 17)  # 14:00-17:00 (下午时段)
        if duration is None:
            duration = self.rng.randint(3, 6)  # 缩短到3-6小时
            
        base_ts = self._generate_base_ts(df_source, start_idx)
        target_ts = base_ts.copy()
        mask_gt = np.zeros_like(base_ts)
        
        target_feature = 'HUFL' if 'HUFL' in self.features else self.features[0]
        feat_idx = self.feature_to_idx[target_feature]
        end_step = min(start_step + duration - 1, self.seq_len - 1)
        
        if multiplier is None:
            multiplier = self.rng.uniform(1.5, 2.5)
        
        target_ts[start_step:end_step + 1, feat_idx] *= multiplier
        mask_gt[start_step:end_step + 1, feat_idx] = 1.0
        
        feature_ctx = self.FEATURE_CONTEXTS.get(target_feature, {})
        feature_name = feature_ctx.get('name', target_feature)
        unit = feature_ctx.get('unit', 'units')
        
        # 优化后的Prompt: 单一特征指向
        prompt = (
            f"[Context] You are a grid monitoring expert analyzing electrical transformer data. "
            f"According to the meteorological log, a severe heatwave struck the city during the afternoon (around step {start_step}). "
            f"During this period, the {feature_name} experienced "
            f"an extreme surge, peaking at approximately {multiplier:.1f} times its normal capacity due to excessive AC usage. "
            f"Please correct the expected sequence to reflect this overload event."
        )
        
        if include_distractor:
            distractor_events = [
                f"A routine inspection was also performed at step {self.rng.randint(0, start_step - 5)}, with no issues found.",
                f"Neighboring sensors reported normal readings throughout the period.",
                f"There was also a minor voltage fluctuation at step {self.rng.randint(0, start_step - 5)}, but was automatically corrected."
            ]
            prompt += f" Note: {self.rng.choice(distractor_events)}"
            
        gt_params = {
            "task_type": "Heatwave_Overload",
            "target_feature": target_feature,
            "start_step": start_step,
            "end_step": end_step,
            "multiplier": float(multiplier),
            "causal_context": "heatwave -> AC usage surge -> load increase",
            "has_distractor": include_distractor
        }
        
        return base_ts, target_ts, mask_gt, prompt, gt_params
    
    def task_sensor_maintenance(
        self,
        df_source: Optional[pd.DataFrame] = None,
        start_idx: int = 0,
        start_step: Optional[int] = None,
        duration: Optional[int] = None,
        include_distractor: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str, Dict]:
        """
        场景2: 传感器维护掉线 (优化版)
        
        物理逻辑：设备计划内维护，导致指标变为底噪
        时间尺度适配: duration=3-6小时
        """
        if start_step is None:
            start_step = self.rng.randint(14, 17)
        if duration is None:
            duration = self.rng.randint(3, 6)
            
        base_ts = self._generate_base_ts(df_source, start_idx)
        target_ts = base_ts.copy()
        mask_gt = np.zeros_like(base_ts)
        
        target_feature = 'OT' if 'OT' in self.features else self.features[-1]
        feat_idx = self.feature_to_idx[target_feature]
        end_step = min(start_step + duration - 1, self.seq_len - 1)
        
        ambient_baseline = self.rng.uniform(3.0, 8.0)
        ambient_noise = self.rng.normal(loc=ambient_baseline, scale=0.5, size=(duration,))
        target_ts[start_step:end_step + 1, feat_idx] = ambient_noise
        mask_gt[start_step:end_step + 1, feat_idx] = 1.0
        
        feature_ctx = self.FEATURE_CONTEXTS.get(target_feature, {})
        feature_name = feature_ctx.get('name', target_feature)
        unit = feature_ctx.get('unit', 'units')
        
        # 优化后的Prompt: 单一特征指向
        prompt = (
            f"[Context] The system logs indicate that a scheduled maintenance for the {feature_name} "
            f"was initiated at the afternoon (around step {start_step}) and completed by the afternoon (around step {end_step}). "
            f"During this period, the transformer was temporarily taken offline, causing the reading to drop "
            f"and flatline around the ambient baseline level (approx {ambient_baseline:.1f} {unit}). "
            f"Modify the sequence to incorporate this planned downtime."
        )
        
        if include_distractor:
            distractor_events = [
                f"A routine inspection was also performed at step {self.rng.randint(0, start_step - 5)}, with no issues found.",
                f"Neighboring sensors reported normal readings throughout the period.",
                f"The maintenance was originally scheduled for step {start_step - 10} but was postponed."
            ]
            prompt += f" Note: {self.rng.choice(distractor_events)}"
            
        gt_params = {
            "task_type": "Sensor_Maintenance",
            "target_feature": target_feature,
            "start_step": start_step,
            "end_step": end_step,
            "ambient_baseline": float(ambient_baseline),
            "causal_context": "scheduled maintenance -> system offline -> baseline reading",
            "has_distractor": include_distractor
        }
        
        return base_ts, target_ts, mask_gt, prompt, gt_params
    
    def task_scheduled_shutdown(
        self,
        df_source: Optional[pd.DataFrame] = None,
        start_idx: int = 0,
        start_step: Optional[int] = None,
        duration: Optional[int] = None,
        include_distractor: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str, Dict]:
        """
        场景3: 计划停机 (优化版)
        
        物理逻辑：计划内的系统停机，指标逐渐下降
        时间尺度适配: duration=3-6小时
        """
        if start_step is None:
            start_step = self.rng.randint(14, 17)
        if duration is None:
            duration = self.rng.randint(3, 6)
            
        base_ts = self._generate_base_ts(df_source, start_idx)
        target_ts = base_ts.copy()
        mask_gt = np.zeros_like(base_ts)
        
        target_feature = 'HUFL' if 'HUFL' in self.features else self.features[0]
        feat_idx = self.feature_to_idx[target_feature]
        end_step = min(start_step + duration - 1, self.seq_len - 1)
        
        shutdown_curve = np.linspace(1.0, 0.05, duration)
        for i, ratio in enumerate(shutdown_curve):
            if start_step + i < self.seq_len:
                target_ts[start_step + i, feat_idx] *= ratio
                
        mask_gt[start_step:end_step + 1, feat_idx] = 1.0
        
        feature_ctx = self.FEATURE_CONTEXTS.get(target_feature, {})
        feature_name = feature_ctx.get('name', target_feature)
        unit = feature_ctx.get('unit', 'units')
        
        # 优化后的Prompt: 单一特征指向
        prompt = (
            f"[Context] A planned system shutdown was executed starting at the afternoon (around step {start_step}). "
            f"The {feature_name} gradually decreased over {duration} steps as the system was powered down safely. "
            f"By step {end_step}, the reading reached near-zero levels. "
            f"Modify the sequence to show this controlled shutdown procedure."
        )
        
        if include_distractor:
            prompt += f" Note: The shutdown was originally planned for step {start_step + 10} but was moved forward."
            
        gt_params = {
            "task_type": "Scheduled_Shutdown",
            "target_feature": target_feature,
            "start_step": start_step,
            "end_step": end_step,
            "shutdown_type": "gradual",
            "causal_context": "planned shutdown -> gradual power down -> near-zero reading",
            "has_distractor": include_distractor
        }
        
        return base_ts, target_ts, mask_gt, prompt, gt_params
    
    def generate_random_task(
        self,
        df_source: Optional[pd.DataFrame] = None,
        start_idx: int = 0,
        task_type: Optional[str] = None,
        include_distractor: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str, Dict]:
        """
        随机生成一个CiK风格的任务
        """
        if task_type is None:
            task_type = self.rng.choice(["Heatwave_Overload", "Sensor_Maintenance", "Scheduled_Shutdown"])
            
        task_methods = {
            "Heatwave_Overload": self.task_heatwave_overload,
            "Sensor_Maintenance": self.task_sensor_maintenance,
            "Scheduled_Shutdown": self.task_scheduled_shutdown
        }
        
        method = task_methods.get(task_type, self.task_heatwave_overload)
        
        return method(
            df_source=df_source,
            start_idx=start_idx,
            include_distractor=include_distractor
        )


class RealLLMClient:
    """
    真实LLM API客户端
    支持DeepSeek和OpenAI API
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.deepseek.com",
        model_name: str = "deepseek-chat"
    ):
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("OPENAI_API_KEY")
        self.base_url = base_url
        self.model_name = model_name
        self.client = None
        
        if not self.api_key:
            logger.warning(
                "未找到API密钥！将使用默认预测模式运行。"
            )
            return
            
        self._init_client()
        
    def _init_client(self):
        try:
            from openai import OpenAI
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )
            logger.info(f"LLM客户端初始化成功: {self.base_url}, 模型: {self.model_name}")
        except ImportError:
            raise ImportError("请安装openai库: pip install openai")
            
    def generate_vague_prompt(self, cik_prompt: str, temperature: float = 0.7) -> str:
        if not self.client:
            return "请调整中间那段时间的数据。"
            
        system_prompt = """你是一个缺乏技术背景，但对业务场景非常熟悉的业务员。
我会提供给你一个详细的技术描述，你需要将其转换为口语化、模糊、宏观的自然语言指令。

严格要求：
1. 绝对不能出现具体的时间步索引（如 step 60）、具体的数值或比例。
2. 只能用时间段（如"下午"、"半夜"、"中段"）和定性描述（如"骤降"、"稍微平缓一点"、"拔高一个台阶"）来表达。
3. 语言风格要自然，像是给AI助手下达任务。
4. 只返回一句指令，不要任何多余的解释或前缀。
5. 保留事件的核心因果关系（如"热浪"、"维护"、"故障"等）。"""

        user_prompt = f"技术描述：\n{cik_prompt}\n\n请生成一句模糊的宏观指令："

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temperature,
                max_tokens=150
            )
            
            result = response.choices[0].message.content.strip().strip('"\'')
            logger.info(f"LLM生成模糊提示词成功: {result[:50]}...")
            return result
            
        except Exception as e:
            logger.error(f"LLM API调用失败: {e}")
            return "请调整中间那段时间的数据。"
            
    def predict_edit_region(self, vague_prompt: str, seq_len: int, features: List[str]) -> Dict:
        if not self.client:
            return {
                "target_feature": features[0] if features else "OT",
                "start_step": seq_len // 4,
                "end_step": seq_len // 2
            }
            
        system_prompt = f"""你是一个时间序列分析专家。根据用户提供的模糊描述，你需要预测编辑的区域。

可用特征: {features}
序列总长度: {seq_len}

请返回一个JSON格式的预测结果，包含：
- target_feature: 目标特征名称
- start_step: 预测的起始步骤（整数）
- end_step: 预测的结束步骤（整数）

只返回JSON，不要其他内容。"""

        user_prompt = f"用户指令：{vague_prompt}\n\n请预测编辑区域："

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=100
            )
            
            content = response.choices[0].message.content.strip()
            
            json_start = content.find('{')
            json_end = content.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                json_str = content[json_start:json_end]
                result = json.loads(json_str)
                
                result['start_step'] = max(0, min(result.get('start_step', 0), seq_len - 1))
                result['end_step'] = max(result['start_step'] + 1, min(result.get('end_step', seq_len), seq_len))
                
                logger.info(f"LLM预测编辑区域: {result}")
                return result
            else:
                raise ValueError("无法从响应中解析JSON")
                
        except json.JSONDecodeError as e:
            logger.error(f"JSON解析失败: {e}")
            return {
                "target_feature": features[0] if features else "OT",
                "start_step": seq_len // 4,
                "end_step": seq_len // 2
            }
        except Exception as e:
            logger.error(f"LLM预测失败: {e}")
            return {
                "target_feature": features[0] if features else "OT",
                "start_step": seq_len // 4,
                "end_step": seq_len // 2
            }


class TSEditEvaluator:
    """时间序列编辑评估器"""
    
    def compute_t_iou(self, gt_start: int, gt_end: int, pred_start: int, pred_end: int) -> float:
        gt_set = set(range(gt_start, gt_end + 1))
        pred_set = set(range(pred_start, pred_end + 1))
        
        intersection = len(gt_set.intersection(pred_set))
        union = len(gt_set.union(pred_set))
        
        return intersection / union if union > 0 else 0.0
    
    def compute_partitioned_metrics(
        self,
        base_ts: np.ndarray,
        target_ts: np.ndarray,
        generated_ts: np.ndarray,
        mask: np.ndarray
    ) -> Dict[str, float]:
        mask_bool = mask.astype(bool)
        inverse_mask_bool = ~mask_bool
        
        metrics = {}
        
        if np.any(mask_bool):
            target_edit = target_ts[mask_bool]
            generated_edit = generated_ts[mask_bool]
            metrics['mse_edit_region'] = float(mean_squared_error(target_edit, generated_edit))
            metrics['mae_edit_region'] = float(mean_absolute_error(target_edit, generated_edit))
        else:
            metrics['mse_edit_region'] = 0.0
            metrics['mae_edit_region'] = 0.0
            
        if np.any(inverse_mask_bool):
            base_preserve = base_ts[inverse_mask_bool]
            generated_preserve = generated_ts[inverse_mask_bool]
            metrics['mse_preserve_region'] = float(mean_squared_error(base_preserve, generated_preserve))
            metrics['mae_preserve_region'] = float(mean_absolute_error(base_preserve, generated_preserve))
        else:
            metrics['mse_preserve_region'] = 0.0
            metrics['mae_preserve_region'] = 0.0
            
        data_range = np.max(base_ts) - np.min(base_ts)
        if data_range > 0:
            metrics['editability_score'] = 1.0 - (metrics['mae_edit_region'] / data_range)
            metrics['preservability_score'] = 1.0 - (metrics['mae_preserve_region'] / data_range)
        else:
            metrics['editability_score'] = 0.0
            metrics['preservability_score'] = 1.0
            
        return metrics
    
    def evaluate(
        self,
        base_ts: np.ndarray,
        target_ts: np.ndarray,
        generated_ts: np.ndarray,
        gt_mask: np.ndarray,
        gt_config: Dict,
        llm_prediction: Dict
    ) -> EvaluationMetrics:
        t_iou = self.compute_t_iou(
            gt_config['start_step'],
            gt_config['end_step'],
            llm_prediction.get('start_step', 0),
            llm_prediction.get('end_step', 0)
        )
        
        feature_accuracy = 1.0 if llm_prediction.get('target_feature') == gt_config.get('target_feature') else 0.0
        
        partitioned_metrics = self.compute_partitioned_metrics(
            base_ts, target_ts, generated_ts, gt_mask
        )
        
        metrics = EvaluationMetrics(
            t_iou=t_iou,
            feature_accuracy=feature_accuracy,
            mse_edit_region=partitioned_metrics['mse_edit_region'],
            mae_edit_region=partitioned_metrics['mae_edit_region'],
            mse_preserve_region=partitioned_metrics['mse_preserve_region'],
            mae_preserve_region=partitioned_metrics['mae_preserve_region'],
            editability_score=partitioned_metrics['editability_score'],
            preservability_score=partitioned_metrics['preservability_score']
        )
        
        return metrics


class BetterTSETestPipelineOptimized:
    """BetterTSE完整测试流程 - 针对ETTh1优化版"""
    
    def __init__(
        self,
        csv_path: str,
        api_key: Optional[str] = None,
        output_dir: str = "test_results",
        random_seed: Optional[int] = None
    ):
        self.data_loader = CSVDataLoader(csv_path)
        self.cik_synthesizer = CiKStyleDataSynthesizerOptimized(
            seq_len=100,
            features=self.data_loader.features,
            random_seed=random_seed
        )
        self.llm_client = RealLLMClient(api_key=api_key) if api_key or os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("OPENAI_API_KEY") else None
        self.evaluator = TSEditEvaluator()
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.random_seed = random_seed
        if random_seed is not None:
            np.random.seed(random_seed)
            
        self.samples: List[TestSample] = []
        
    def run_single_test(
        self,
        sample_id: str,
        start_idx: int,
        seq_len: int,
        task_type: Optional[str] = None,
        include_distractor: bool = False
    ) -> TestSample:
        logger.info(f"\n{'='*60}")
        logger.info(f"开始测试样本: {sample_id}")
        logger.info(f"{'='*60}")
        
        self.cik_synthesizer.seq_len = seq_len
        
        base_ts_multi, target_ts_multi, gt_mask_multi, cik_prompt, gt_params = \
            self.cik_synthesizer.generate_random_task(
                df_source=self.data_loader.data,
                start_idx=start_idx,
                task_type=task_type,
                include_distractor=include_distractor
            )
        
        target_feature = gt_params.get('target_feature', self.data_loader.features[0])
        feat_idx = self.data_loader.feature_to_idx.get(target_feature, 0)
        
        base_ts = base_ts_multi[:, feat_idx]
        target_ts = target_ts_multi[:, feat_idx]
        gt_mask = gt_mask_multi[:, feat_idx]
        
        logger.info(f"任务类型: {gt_params.get('task_type', 'unknown')}")
        logger.info(f"目标特征: {target_feature}")
        logger.info(f"编辑区间: [{gt_params['start_step']}, {gt_params['end_step']}]")
        logger.info(f"因果上下文: {gt_params.get('causal_context', 'N/A')}")
        
        if self.llm_client:
            logger.info("调用LLM生成模糊提示词...")
            vague_prompt = self.llm_client.generate_vague_prompt(cik_prompt)
            
            time.sleep(0.5)
            
            logger.info("调用LLM预测编辑区域...")
            llm_prediction = self.llm_client.predict_edit_region(
                vague_prompt=vague_prompt,
                seq_len=seq_len,
                features=self.data_loader.features
            )
            
            time.sleep(0.5)
        else:
            logger.warning("未配置LLM客户端，使用默认值")
            vague_prompt = "请调整中间那段时间的数据。"
            llm_prediction = {
                "target_feature": target_feature,
                "start_step": gt_params['start_step'],
                "end_step": gt_params['end_step']
            }
            
        pred_mask = np.zeros(seq_len, dtype=np.float64)
        pred_start = llm_prediction.get('start_step', gt_params['start_step'])
        pred_end = llm_prediction.get('end_step', gt_params['end_step'])
        pred_mask[pred_start:pred_end] = 1.0
        
        generated_ts = pred_mask * target_ts + (1 - pred_mask) * base_ts
        
        metrics = self.evaluator.evaluate(
            base_ts=base_ts,
            target_ts=target_ts,
            generated_ts=generated_ts,
            gt_mask=gt_mask,
            gt_config=gt_params,
            llm_prediction=llm_prediction
        )
        
        sample = TestSample(
            sample_id=sample_id,
            base_ts=base_ts.tolist(),
            target_ts=target_ts.tolist(),
            generated_ts=generated_ts.tolist(),
            gt_mask=gt_mask.tolist(),
            pred_mask=pred_mask.tolist(),
            vague_prompt=vague_prompt,
            cik_prompt=cik_prompt,
            gt_config=gt_params,
            llm_prediction=llm_prediction,
            metrics=asdict(metrics)
        )
        
        self.samples.append(sample)
        
        logger.info(f"样本 {sample_id} 测试完成:")
        logger.info(f"  - 任务类型: {gt_params.get('task_type', 'N/A')}")
        logger.info(f"  - t-IoU: {metrics.t_iou:.4f}")
        logger.info(f"  - 特征准确率: {metrics.feature_accuracy:.1%}")
        logger.info(f"  - 编辑区MSE: {metrics.mse_edit_region:.4f}")
        logger.info(f"  - 保留区MSE: {metrics.mse_preserve_region:.4f}")
        logger.info(f"  - Editability: {metrics.editability_score:.4f}")
        logger.info(f"  - Preservability: {metrics.preservability_score:.4f}")
        logger.info(f"  - CiK Prompt: {cik_prompt[:100]}...")
        logger.info(f"  - 模糊提示词: {vague_prompt}")
        
        return sample
    
    def run_batch_tests(
        self,
        num_samples: int = 10,
        seq_len: int = 100,
        task_types: Optional[List[str]] = None,
        include_distractor_ratio: float = 0.3
    ) -> List[TestSample]:
        logger.info(f"\n开始批量测试，共 {num_samples} 个样本")
        logger.info(f"干扰项比例: {include_distractor_ratio:.0%}")
        
        max_start = len(self.data_loader.data) - seq_len
        step = max(1, max_start // num_samples)
        
        for i in range(num_samples):
            start_idx = (i * step) % max_start
            
            task_type = None
            if task_types:
                task_type = task_types[i % len(task_types)]
                
            include_distractor = np.random.random() < include_distractor_ratio
                
            sample_id = f"TSE_CiK_Opt_{i+1:03d}"
            
            try:
                self.run_single_test(
                    sample_id=sample_id,
                    start_idx=start_idx,
                    seq_len=seq_len,
                    task_type=task_type,
                    include_distractor=include_distractor
                )
            except Exception as e:
                logger.error(f"样本 {sample_id} 测试失败: {e}")
                continue
                
        return self.samples
    
    def save_results(self, filename: str = "cik_optimized_test_results.json"):
        output_file = self.output_dir / filename
        
        results = {
            "metadata": {
                "csv_path": str(self.data_loader.csv_path),
                "total_samples": len(self.samples),
                "test_time": datetime.now().isoformat(),
                "random_seed": self.random_seed,
                "framework": "BetterTSE-CiK Optimized"
            },
            "data_statistics": self.data_loader.get_statistics(),
            "summary": self._compute_summary(),
            "samples": [asdict(s) for s in self.samples]
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
            
        logger.info(f"\n结果已保存至: {output_file}")
        
        self._save_report()
        
    def _compute_summary(self) -> Dict:
        if not self.samples:
            return {}
            
        t_ious = [s.metrics['t_iou'] for s in self.samples]
        feature_accs = [s.metrics['feature_accuracy'] for s in self.samples]
        editability_scores = [s.metrics['editability_score'] for s in self.samples]
        preservability_scores = [s.metrics['preservability_score'] for s in self.samples]
        
        distractor_samples = [s for s in self.samples if s.gt_config.get('has_distractor', False)]
        
        return {
            "avg_t_iou": float(np.mean(t_ious)),
            "std_t_iou": float(np.std(t_ious)),
            "avg_feature_accuracy": float(np.mean(feature_accs)),
            "avg_editability": float(np.mean(editability_scores)),
            "avg_preservability": float(np.mean(preservability_scores)),
            "task_type_distribution": self._count_task_types(),
            "distractor_samples": len(distractor_samples),
            "distractor_ratio": len(distractor_samples) / len(self.samples) if self.samples else 0
        }
    
    def _count_task_types(self) -> Dict[str, int]:
        counts = {}
        for s in self.samples:
            tt = s.gt_config.get('task_type', 'unknown')
            counts[tt] = counts.get(tt, 0) + 1
        return counts
    
    def _save_report(self):
        report_file = self.output_dir / "cik_optimized_evaluation_report.txt"
        
        summary = self._compute_summary()
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write("BetterTSE 时间序列编辑测试评估报告 (CiK风格优化版)\n")
            f.write("=" * 70 + "\n\n")
            
            f.write(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"数据文件: {self.data_loader.csv_path}\n")
            f.write(f"样本数量: {len(self.samples)}\n")
            f.write(f"框架版本: BetterTSE-CiK Optimized\n\n")
            
            f.write("-" * 70 + "\n")
            f.write("一、整体评估指标\n")
            f.write("-" * 70 + "\n")
            f.write(f"平均 t-IoU:           {summary.get('avg_t_iou', 0):.4f}\n")
            f.write(f"t-IoU 标准差:         {summary.get('std_t_iou', 0):.4f}\n")
            f.write(f"平均特征准确率:       {summary.get('avg_feature_accuracy', 0):.1%}\n")
            f.write(f"平均 Editability:     {summary.get('avg_editability', 0):.4f}\n")
            f.write(f"平均 Preservability:  {summary.get('avg_preservability', 0):.4f}\n\n")
            
            f.write("-" * 70 + "\n")
            f.write("二、CiK风格任务分布\n")
            f.write("-" * 70 + "\n")
            for tt, count in summary.get('task_type_distribution', {}).items():
                f.write(f"  {tt}: {count}\n")
            f.write(f"\n干扰项样本数: {summary.get('distractor_samples', 0)} ")
            f.write(f"({summary.get('distractor_ratio', 0):.0%})\n\n")
            
            f.write("-" * 70 + "\n")
            f.write("三、样本详情（前5个）\n")
            f.write("-" * 70 + "\n")
            for i, sample in enumerate(self.samples[:5]):
                f.write(f"\n样本 {i+1}: {sample.sample_id}\n")
                f.write(f"  任务类型: {sample.gt_config.get('task_type', 'N/A')}\n")
                f.write(f"  目标特征: {sample.gt_config.get('target_feature', 'N/A')}\n")
                f.write(f"  编辑区间: [{sample.gt_config.get('start_step', 0)}, {sample.gt_config.get('end_step', 0)}]\n")
                f.write(f"  因果上下文: {sample.gt_config.get('causal_context', 'N/A')}\n")
                f.write(f"  包含干扰项: {'是' if sample.gt_config.get('has_distractor', False) else '否'}\n")
                f.write(f"  CiK Prompt: {sample.cik_prompt[:80]}...\n")
                f.write(f"  模糊提示词: {sample.vague_prompt}\n")
                f.write(f"  t-IoU: {sample.metrics['t_iou']:.4f}\n")
                f.write(f"  Edit区MSE: {sample.metrics['mse_edit_region']:.4f}\n")
                f.write(f"  保留区MSE: {sample.metrics['mse_preserve_region']:.4f}\n")
                f.write(f"  Editability: {sample.metrics['editability_score']:.4f}\n")
                f.write(f"  Preservability: {sample.metrics['preservability_score']:.4f}\n")
                
        logger.info(f"评估报告已保存至: {report_file}")


def main():
    parser = argparse.ArgumentParser(
        description='BetterTSE 时间序列编辑测试脚本 (CiK风格优化版)',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--csv-path',
        type=str,
        default=r'C:\Users\ghb\Desktop\李老师科研\BetterTSE-main\data\ETTh1.csv',
        help='CSV数据文件路径'
    )
    
    parser.add_argument(
        '--api-key',
        type=str,
        default=DEEPSEEK_API_KEY,
        help='LLM API密钥（默认使用内置DeepSeek密钥）'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='test_results',
        help='输出目录'
    )
    
    parser.add_argument(
        '--num-samples',
        type=int,
        default=10,
        help='测试样本数量'
    )
    
    parser.add_argument(
        '--seq-len',
        type=int,
        default=100,
        help='序列长度'
    )
    
    parser.add_argument(
        '--distractor-ratio',
        type=float,
        default=0.3,
        help='干扰项样本比例'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='随机种子'
    )
    
    args = parser.parse_args()
    
    log_file = Path(args.output_dir) / f"cik_optimized_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - [%(levelname)s] - %(name)s - %(message)s'))
    logging.getLogger().addHandler(file_handler)
    
    logger.info("=" * 70)
    logger.info("BetterTSE 时间序列编辑测试框架 (CiK风格优化版)")
    logger.info("=" * 70)
    logger.info(f"CSV文件: {args.csv_path}")
    logger.info(f"输出目录: {args.output_dir}")
    logger.info(f"样本数量: {args.num_samples}")
    logger.info(f"序列长度: {args.seq_len}")
    logger.info(f"干扰项比例: {args.distractor_ratio:.0%}")
    
    try:
        pipeline = BetterTSETestPipelineOptimized(
            csv_path=args.csv_path,
            api_key=args.api_key,
            output_dir=args.output_dir,
            random_seed=args.seed
        )
        
        pipeline.run_batch_tests(
            num_samples=args.num_samples,
            seq_len=args.seq_len,
            include_distractor_ratio=args.distractor_ratio
        )
        
        pipeline.save_results()
        
        logger.info("\n" + "=" * 70)
        logger.info("测试完成!")
        logger.info("=" * 70)
        
    except Exception as e:
        logger.error(f"测试失败: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
