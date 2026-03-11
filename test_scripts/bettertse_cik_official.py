"""
BetterTSE 时间序列编辑测试框架 - CiK官方范式版
==============================================

完全遵循 Context is Key (CiK) 官方开源代码库逻辑：
1. 面向对象(OOP)的任务类设计
2. 物理变化注入算子：乘法倍数激增、强制归零/底噪替换
3. 结构化Prompt模板：Background + Scenario + Constraint
4. 严格的时间序列切分：历史(past) + 未来/预测(future)

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


def mean_squared_error(y_true, y_pred):
    return np.mean((np.array(y_true) - np.array(y_pred)) ** 2)


def mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(np.array(y_true) - np.array(y_pred)))


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(name)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("BetterTSE_CiK_Official")


@dataclass
class EvaluationMetrics:
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
    def __init__(self, csv_path: str):
        self.csv_path = Path(csv_path)
        self.data = None
        self.features = []
        self.feature_to_idx = {}
        
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV文件不存在: {csv_path}")
        self._load_data()
        
    def _load_data(self):
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
            
    def get_sequence(self, start_idx: int = 0, seq_len: int = 192, features: Optional[List[str]] = None) -> np.ndarray:
        if features is None:
            features = self.features
        end_idx = start_idx + seq_len
        
        if end_idx > len(self.data):
            end_idx = len(self.data)
            start_idx = max(0, end_idx - seq_len)
            
        sequence = self.data[features].iloc[start_idx:end_idx].values
        return sequence.astype(np.float64)


class BaseCiKTask(ABC):
    """
    CiK基础任务类 - 遵循CiK官方OOP设计
    
    所有任务类继承此基类，实现具体的物理变化注入逻辑
    """
    
    def __init__(self, seq_len: int, features: List[str], feature_to_idx: Dict[str, int], rng: np.random.RandomState):
        self.seq_len = seq_len
        self.features = features
        self.feature_to_idx = feature_to_idx
        self.rng = rng
        
    @abstractmethod
    def inject(self, base_ts: np.ndarray, start_step: int, duration: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str, Dict]:
        """
        注入物理变化
        
        Returns:
            Tuple[base_ts, target_ts, mask_gt, cik_prompt, gt_params]
        """
        pass
    
    def _build_cik_prompt(self, background: str, scenario: str, constraint: str) -> str:
        """
        构建CiK风格的结构化Prompt
        
        CiK的Prompt分为三部分：
        - Background: 背景知识
        - Scenario: 突发事件场景
        - Constraint: 约束条件
        """
        return f"[Background] {background}\n[Scenario] {scenario}\n[Constraint] {constraint}"


class ElectricityIncreaseTask(BaseCiKTask):
    """
    借鉴CiK: ElectricityIncreaseInPredictionTask
    
    物理逻辑：热浪导致电力负荷(HUFL)激增
    注入算子：乘法倍数激增 (Multiplier)
    """
    
    def inject(self, base_ts: np.ndarray, start_step: int, duration: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str, Dict]:
        target_ts = base_ts.copy()
        mask_gt = np.zeros_like(base_ts)
        
        target_feature = 'HUFL' if 'HUFL' in self.features else self.features[0]
        feat_idx = self.feature_to_idx[target_feature]
        end_step = min(start_step + duration, self.seq_len)
        
        multiplier = self.rng.choice([2.0, 2.5, 3.0, 3.5, 4.0])
        target_ts[start_step:end_step, feat_idx] *= multiplier
        mask_gt[start_step:end_step, feat_idx] = 1.0
        
        background = "You are an expert forecasting system monitoring power transformer metrics (ETT data)."
        scenario = (f"According to meteorological alerts, a severe heatwave occurred from time step {start_step} "
                    f"to {end_step - 1}, driving massive air conditioning usage.")
        constraint = (f"During this exact period, the High Useful Load (HUFL) surged to approximately "
                      f"{multiplier:.1f} times its normal volume. It returned to normal historical patterns immediately after.")
        
        cik_prompt = self._build_cik_prompt(background, scenario, constraint)
        
        gt_params = {
            "task_type": "ElectricityIncrease",
            "target_feature": target_feature,
            "start_step": start_step,
            "end_step": end_step - 1,
            "multiplier": float(multiplier),
            "injection_type": "multiplier",
            "causal_context": "heatwave -> AC usage surge -> load increase"
        }
        
        return base_ts, target_ts, mask_gt, cik_prompt, gt_params


class SensorMaintenanceTask(BaseCiKTask):
    """
    借鉴CiK: SensorMaintenanceInPredictionTask
    
    物理逻辑：计划内维护导致油温(OT)失去监控并掉至底噪
    注入算子：强制归零/底噪替换 (Hard Overwrite)
    """
    
    def inject(self, base_ts: np.ndarray, start_step: int, duration: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str, Dict]:
        target_ts = base_ts.copy()
        mask_gt = np.zeros_like(base_ts)
        
        target_feature = 'OT' if 'OT' in self.features else self.features[-1]
        feat_idx = self.feature_to_idx[target_feature]
        end_step = min(start_step + duration, self.seq_len)
        
        ambient_baseline = self.rng.uniform(3.0, 8.0)
        ambient_noise = self.rng.normal(loc=ambient_baseline, scale=0.5, size=(end_step - start_step,))
        target_ts[start_step:end_step, feat_idx] = ambient_noise
        mask_gt[start_step:end_step, feat_idx] = 1.0
        
        background = "You are analyzing the Oil Temperature (OT) sensor logs of a power transformer."
        scenario = (f"System logs show a scheduled maintenance outage starting at step {start_step} "
                    f"and lasting for {end_step - start_step} steps.")
        constraint = ("During this outage window, the OT sensor went offline and its readings plummeted to "
                      f"the ambient baseline noise level (around {ambient_baseline:.1f}). Please correct the numerical sequence to reflect this outage.")
        
        cik_prompt = self._build_cik_prompt(background, scenario, constraint)
        
        gt_params = {
            "task_type": "SensorMaintenance",
            "target_feature": target_feature,
            "start_step": start_step,
            "end_step": end_step - 1,
            "ambient_baseline": float(ambient_baseline),
            "injection_type": "hard_overwrite_to_noise",
            "causal_context": "scheduled maintenance -> system offline -> baseline reading"
        }
        
        return base_ts, target_ts, mask_gt, cik_prompt, gt_params


class BuildingClosedTask(BaseCiKTask):
    """
    借鉴CiK: ATMBuildingClosedTask
    
    物理逻辑：大楼关闭导致用电量归零
    注入算子：强制归零 (Hard Zero)
    """
    
    def inject(self, base_ts: np.ndarray, start_step: int, duration: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str, Dict]:
        target_ts = base_ts.copy()
        mask_gt = np.zeros_like(base_ts)
        
        target_feature = 'HUFL' if 'HUFL' in self.features else self.features[0]
        feat_idx = self.feature_to_idx[target_feature]
        end_step = min(start_step + duration, self.seq_len)
        
        target_ts[start_step:end_step, feat_idx] = 0.0
        mask_gt[start_step:end_step, feat_idx] = 1.0
        
        background = "You are monitoring the electrical load patterns of a commercial building complex."
        scenario = (f"The building management system reports that the facility was completely closed from step {start_step} "
                    f"to {end_step - 1} for a scheduled holiday.")
        constraint = ("During this closure period, all electrical loads were shut down, resulting in zero consumption. "
                      "The load patterns should show a complete flatline at zero during this interval.")
        
        cik_prompt = self._build_cik_prompt(background, scenario, constraint)
        
        gt_params = {
            "task_type": "BuildingClosed",
            "target_feature": target_feature,
            "start_step": start_step,
            "end_step": end_step - 1,
            "injection_type": "hard_zero",
            "causal_context": "building closure -> all systems off -> zero load"
        }
        
        return base_ts, target_ts, mask_gt, cik_prompt, gt_params


class TrendReversalTask(BaseCiKTask):
    """
    趋势反转任务
    
    物理逻辑：市场/系统状态变化导致趋势反转
    注入算子：趋势注入 (Trend Injection)
    """
    
    def inject(self, base_ts: np.ndarray, start_step: int, duration: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str, Dict]:
        target_ts = base_ts.copy()
        mask_gt = np.zeros_like(base_ts)
        
        target_feature = self.rng.choice(self.features)
        feat_idx = self.feature_to_idx[target_feature]
        end_step = min(start_step + duration, self.seq_len)
        
        direction = self.rng.choice(["upward", "downward"])
        segment_range = np.max(target_ts[:, feat_idx]) - np.min(target_ts[:, feat_idx])
        slope = segment_range * self.rng.uniform(0.1, 0.3) / duration
        
        if direction == "downward":
            slope = -abs(slope)
        else:
            slope = abs(slope)
            
        trend = np.linspace(0, slope * duration, end_step - start_step)
        target_ts[start_step:end_step, feat_idx] += trend
        mask_gt[start_step:end_step, feat_idx] = 1.0
        
        background = f"You are analyzing the {target_feature} time series data from a monitoring system."
        scenario = (f"A significant market/system event occurred from step {start_step} to {end_step - 1}, "
                    f"causing a fundamental shift in the underlying dynamics.")
        constraint = (f"During this period, the {target_feature} exhibited a strong {direction} trend. "
                      f"Please adjust the sequence to reflect this directional change.")
        
        cik_prompt = self._build_cik_prompt(background, scenario, constraint)
        
        gt_params = {
            "task_type": "TrendReversal",
            "target_feature": target_feature,
            "start_step": start_step,
            "end_step": end_step - 1,
            "direction": direction,
            "slope": float(slope),
            "injection_type": "trend_injection",
            "causal_context": f"market event -> {direction} trend -> directional shift"
        }
        
        return base_ts, target_ts, mask_gt, cik_prompt, gt_params


class CiKTaskFactory:
    """
    CiK任务工厂 - 遵循CiK官方设计模式
    """
    
    TASK_REGISTRY = {
        "ElectricityIncrease": ElectricityIncreaseTask,
        "SensorMaintenance": SensorMaintenanceTask,
        "BuildingClosed": BuildingClosedTask,
        "TrendReversal": TrendReversalTask
    }
    
    def __init__(self, seq_len: int, features: List[str], feature_to_idx: Dict[str, int], random_seed: Optional[int] = None):
        self.seq_len = seq_len
        self.features = features
        self.feature_to_idx = feature_to_idx
        self.rng = np.random.RandomState(random_seed)
        
    def create_task(self, task_type: Optional[str] = None) -> BaseCiKTask:
        """创建任务实例"""
        if task_type is None:
            task_type = self.rng.choice(list(self.TASK_REGISTRY.keys()))
            
        task_class = self.TASK_REGISTRY.get(task_type, ElectricityIncreaseTask)
        return task_class(self.seq_len, self.features, self.feature_to_idx, self.rng)
    
    def generate_random_config(self) -> Tuple[int, int]:
        """生成随机的start_step和duration"""
        duration = self.rng.randint(5, 15)
        start_step = self.rng.randint(int(self.seq_len * 0.1), int(self.seq_len * 0.6))
        return start_step, duration


class RealLLMClient:
    def __init__(self, api_key: Optional[str] = None, base_url: str = "https://api.deepseek.com", model_name: str = "deepseek-chat"):
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("OPENAI_API_KEY")
        self.base_url = base_url
        self.model_name = model_name
        self.client = None
        
        if not self.api_key:
            logger.warning("未找到API密钥！将使用默认预测模式运行。")
            return
        self._init_client()
        
    def _init_client(self):
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
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
2. 只能用时间段（如"下午"、"半夜"、"观测初期"、"观测后期"）和定性描述（如"骤降"、"稍微平缓一点"、"拔高一个台阶"）来表达。
3. 语言风格要自然，像是给AI助手下达任务。
4. 只返回一句指令，不要任何多余的解释或前缀。
5. 保留事件的核心因果关系（如"热浪"、"维护"、"关闭"等）。

【关键约束 - 必须遵守】：
6. 必须明确提及目标特征名称（如"高压侧有效负荷HUFL"、"变压器油温OT"），不能省略或替换成其他名称。
7. 必须用相对时间方位来暗示事件发生的时间段，例如：
   - "在观测窗口的前半段" 或 "在序列初期"
   - "在观测窗口的中段" 或 "在序列中间"
   - "在观测窗口的后半段" 或 "在序列后期"
   - "大约在观测开始后的三分之一处"
8. 描述中要同时包含：特征名 + 相对时间方位 + 变化类型（如骤降、飙升、归零等）。"""

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
            logger.info(f"LLM生成模糊提示词成功: {result[:60]}...")
            return result
        except Exception as e:
            logger.error(f"LLM API调用失败: {e}")
            return "请调整中间那段时间的数据。"
            
    def predict_edit_region(self, vague_prompt: str, seq_len: int, features: List[str]) -> Dict:
        if not self.client:
            return {"target_feature": features[0] if features else "OT", "start_step": seq_len // 4, "end_step": seq_len // 2}
            
        system_prompt = f"""你是一个时间序列分析专家。根据用户提供的模糊描述，你需要预测编辑的区域。

可用特征: {features}
序列总长度: {seq_len} 步

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
        except Exception as e:
            logger.error(f"LLM预测失败: {e}")
            return {"target_feature": features[0] if features else "OT", "start_step": seq_len // 4, "end_step": seq_len // 2}


class TSEditEvaluator:
    def compute_t_iou(self, gt_start: int, gt_end: int, pred_start: int, pred_end: int) -> float:
        gt_set = set(range(gt_start, gt_end + 1))
        pred_set = set(range(pred_start, pred_end + 1))
        intersection = len(gt_set.intersection(pred_set))
        union = len(gt_set.union(pred_set))
        return intersection / union if union > 0 else 0.0
    
    def compute_partitioned_metrics(self, base_ts: np.ndarray, target_ts: np.ndarray, generated_ts: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
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
    
    def evaluate(self, base_ts: np.ndarray, target_ts: np.ndarray, generated_ts: np.ndarray, gt_mask: np.ndarray, gt_config: Dict, llm_prediction: Dict) -> EvaluationMetrics:
        t_iou = self.compute_t_iou(
            gt_config['start_step'],
            gt_config['end_step'],
            llm_prediction.get('start_step', 0),
            llm_prediction.get('end_step', 0)
        )
        feature_accuracy = 1.0 if llm_prediction.get('target_feature') == gt_config.get('target_feature') else 0.0
        partitioned_metrics = self.compute_partitioned_metrics(base_ts, target_ts, generated_ts, gt_mask)
        
        return EvaluationMetrics(
            t_iou=t_iou,
            feature_accuracy=feature_accuracy,
            mse_edit_region=partitioned_metrics['mse_edit_region'],
            mae_edit_region=partitioned_metrics['mae_edit_region'],
            mse_preserve_region=partitioned_metrics['mse_preserve_region'],
            mae_preserve_region=partitioned_metrics['mae_preserve_region'],
            editability_score=partitioned_metrics['editability_score'],
            preservability_score=partitioned_metrics['preservability_score']
        )


class BetterTSETestPipelineCiKOfficial:
    """BetterTSE完整测试流程 - CiK官方范式版"""
    
    def __init__(self, csv_path: str, api_key: Optional[str] = None, output_dir: str = "test_results", random_seed: Optional[int] = None, seq_len: int = 192):
        self.data_loader = CSVDataLoader(csv_path)
        self.seq_len = seq_len
        self.task_factory = CiKTaskFactory(
            seq_len=seq_len,
            features=self.data_loader.features,
            feature_to_idx=self.data_loader.feature_to_idx,
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
        
    def run_single_test(self, sample_id: str, start_idx: int, task_type: Optional[str] = None) -> TestSample:
        logger.info(f"\n{'='*60}")
        logger.info(f"开始测试样本: {sample_id}")
        logger.info(f"{'='*60}")
        
        base_ts_multi = self.data_loader.get_sequence(start_idx, self.seq_len)
        
        task = self.task_factory.create_task(task_type)
        start_step, duration = self.task_factory.generate_random_config()
        
        base_ts, target_ts, gt_mask, cik_prompt, gt_params = task.inject(base_ts_multi, start_step, duration)
        
        target_feature = gt_params.get('target_feature', self.data_loader.features[0])
        feat_idx = self.data_loader.feature_to_idx.get(target_feature, 0)
        
        base_ts_1d = base_ts[:, feat_idx]
        target_ts_1d = target_ts[:, feat_idx]
        gt_mask_1d = gt_mask[:, feat_idx]
        
        logger.info(f"任务类型: {gt_params.get('task_type', 'unknown')}")
        logger.info(f"目标特征: {target_feature}")
        logger.info(f"编辑区间: [{gt_params['start_step']}, {gt_params['end_step']}]")
        logger.info(f"注入类型: {gt_params.get('injection_type', 'N/A')}")
        logger.info(f"因果上下文: {gt_params.get('causal_context', 'N/A')}")
        
        if self.llm_client:
            logger.info("调用LLM生成模糊提示词...")
            vague_prompt = self.llm_client.generate_vague_prompt(cik_prompt)
            time.sleep(0.5)
            
            logger.info("调用LLM预测编辑区域...")
            llm_prediction = self.llm_client.predict_edit_region(
                vague_prompt=vague_prompt,
                seq_len=self.seq_len,
                features=self.data_loader.features
            )
            time.sleep(0.5)
        else:
            logger.warning("未配置LLM客户端，使用默认值")
            vague_prompt = "请调整中间那段时间的数据。"
            llm_prediction = {"target_feature": target_feature, "start_step": gt_params['start_step'], "end_step": gt_params['end_step']}
            
        pred_mask = np.zeros(self.seq_len, dtype=np.float64)
        pred_start = llm_prediction.get('start_step', gt_params['start_step'])
        pred_end = llm_prediction.get('end_step', gt_params['end_step'])
        pred_mask[pred_start:pred_end] = 1.0
        
        generated_ts = pred_mask * target_ts_1d + (1 - pred_mask) * base_ts_1d
        
        metrics = self.evaluator.evaluate(
            base_ts=base_ts_1d,
            target_ts=target_ts_1d,
            generated_ts=generated_ts,
            gt_mask=gt_mask_1d,
            gt_config=gt_params,
            llm_prediction=llm_prediction
        )
        
        sample = TestSample(
            sample_id=sample_id,
            base_ts=base_ts_1d.tolist(),
            target_ts=target_ts_1d.tolist(),
            generated_ts=generated_ts.tolist(),
            gt_mask=gt_mask_1d.tolist(),
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
        logger.info(f"  - CiK Prompt: {cik_prompt[:80]}...")
        logger.info(f"  - 模糊提示词: {vague_prompt}")
        
        return sample
    
    def run_batch_tests(self, num_samples: int = 10, task_types: Optional[List[str]] = None) -> List[TestSample]:
        logger.info(f"\n开始批量测试，共 {num_samples} 个样本")
        logger.info(f"序列长度: {self.seq_len}")
        
        max_start = len(self.data_loader.data) - self.seq_len
        step = max(1, max_start // num_samples)
        
        for i in range(num_samples):
            start_idx = (i * step) % max_start
            task_type = None
            if task_types:
                task_type = task_types[i % len(task_types)]
            sample_id = f"TSE_CiK_Official_{i+1:03d}"
            
            try:
                self.run_single_test(sample_id=sample_id, start_idx=start_idx, task_type=task_type)
            except Exception as e:
                logger.error(f"样本 {sample_id} 测试失败: {e}")
                continue
        return self.samples
    
    def save_results(self, filename: str = "cik_official_test_results.json"):
        output_file = self.output_dir / filename
        
        results = {
            "metadata": {
                "csv_path": str(self.data_loader.csv_path),
                "total_samples": len(self.samples),
                "test_time": datetime.now().isoformat(),
                "random_seed": self.random_seed,
                "seq_len": self.seq_len,
                "framework": "BetterTSE-CiK Official Paradigm"
            },
            "data_statistics": {"total_rows": len(self.data_loader.data), "features": self.data_loader.features},
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
        
        positive_iou_samples = sum(1 for iou in t_ious if iou > 0)
        
        task_type_dist = {}
        injection_type_dist = {}
        for s in self.samples:
            tt = s.gt_config.get('task_type', 'unknown')
            task_type_dist[tt] = task_type_dist.get(tt, 0) + 1
            it = s.gt_config.get('injection_type', 'unknown')
            injection_type_dist[it] = injection_type_dist.get(it, 0) + 1
        
        return {
            "avg_t_iou": float(np.mean(t_ious)),
            "std_t_iou": float(np.std(t_ious)),
            "max_t_iou": float(np.max(t_ious)),
            "positive_iou_count": positive_iou_samples,
            "positive_iou_ratio": positive_iou_samples / len(self.samples) if self.samples else 0,
            "avg_feature_accuracy": float(np.mean(feature_accs)),
            "avg_editability": float(np.mean(editability_scores)),
            "avg_preservability": float(np.mean(preservability_scores)),
            "task_type_distribution": task_type_dist,
            "injection_type_distribution": injection_type_dist
        }
    
    def _save_report(self):
        report_file = self.output_dir / "cik_official_evaluation_report.txt"
        summary = self._compute_summary()
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write("BetterTSE 时间序列编辑测试评估报告 (CiK官方范式版)\n")
            f.write("=" * 70 + "\n\n")
            
            f.write(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"数据文件: {self.data_loader.csv_path}\n")
            f.write(f"样本数量: {len(self.samples)}\n")
            f.write(f"序列长度: {self.seq_len}\n")
            f.write(f"框架版本: BetterTSE-CiK Official Paradigm\n\n")
            
            f.write("-" * 70 + "\n")
            f.write("一、整体评估指标\n")
            f.write("-" * 70 + "\n")
            f.write(f"平均 t-IoU:           {summary.get('avg_t_iou', 0):.4f}\n")
            f.write(f"t-IoU 标准差:         {summary.get('std_t_iou', 0):.4f}\n")
            f.write(f"最大 t-IoU:           {summary.get('max_t_iou', 0):.4f}\n")
            f.write(f"t-IoU > 0 样本数:     {summary.get('positive_iou_count', 0)} ({summary.get('positive_iou_ratio', 0):.1%})\n")
            f.write(f"平均特征准确率:       {summary.get('avg_feature_accuracy', 0):.1%}\n")
            f.write(f"平均 Editability:     {summary.get('avg_editability', 0):.4f}\n")
            f.write(f"平均 Preservability:  {summary.get('avg_preservability', 0):.4f}\n\n")
            
            f.write("-" * 70 + "\n")
            f.write("二、任务类型分布\n")
            f.write("-" * 70 + "\n")
            for tt, count in summary.get('task_type_distribution', {}).items():
                f.write(f"  {tt}: {count}\n")
            f.write("\n")
            
            f.write("-" * 70 + "\n")
            f.write("三、注入算子分布\n")
            f.write("-" * 70 + "\n")
            for it, count in summary.get('injection_type_distribution', {}).items():
                f.write(f"  {it}: {count}\n")
            f.write("\n")
            
            f.write("-" * 70 + "\n")
            f.write("四、样本详情（前5个）\n")
            f.write("-" * 70 + "\n")
            for i, sample in enumerate(self.samples[:5]):
                f.write(f"\n样本 {i+1}: {sample.sample_id}\n")
                f.write(f"  任务类型: {sample.gt_config.get('task_type', 'N/A')}\n")
                f.write(f"  目标特征: {sample.gt_config.get('target_feature', 'N/A')}\n")
                f.write(f"  编辑区间: [{sample.gt_config.get('start_step', 0)}, {sample.gt_config.get('end_step', 0)}]\n")
                f.write(f"  注入类型: {sample.gt_config.get('injection_type', 'N/A')}\n")
                f.write(f"  因果上下文: {sample.gt_config.get('causal_context', 'N/A')}\n")
                f.write(f"  CiK Prompt: {sample.cik_prompt[:100]}...\n")
                f.write(f"  模糊提示词: {sample.vague_prompt}\n")
                f.write(f"  t-IoU: {sample.metrics['t_iou']:.4f}\n")
                f.write(f"  特征准确率: {sample.metrics['feature_accuracy']:.1%}\n")
                f.write(f"  Edit区MSE: {sample.metrics['mse_edit_region']:.4f}\n")
                f.write(f"  保留区MSE: {sample.metrics['mse_preserve_region']:.4f}\n")
                f.write(f"  Editability: {sample.metrics['editability_score']:.4f}\n")
                f.write(f"  Preservability: {sample.metrics['preservability_score']:.4f}\n")
                
        logger.info(f"评估报告已保存至: {report_file}")


def main():
    parser = argparse.ArgumentParser(description='BetterTSE 时间序列编辑测试脚本 (CiK官方范式版)')
    
    parser.add_argument('--csv-path', type=str, default=r'C:\Users\ghb\Desktop\李老师科研\BetterTSE-main\data\ETTh1.csv')
    parser.add_argument('--api-key', type=str, default=DEEPSEEK_API_KEY)
    parser.add_argument('--output-dir', type=str, default='test_results')
    parser.add_argument('--num-samples', type=int, default=10)
    parser.add_argument('--seq-len', type=int, default=192)
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    log_file = Path(args.output_dir) / f"cik_official_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - [%(levelname)s] - %(name)s - %(message)s'))
    logging.getLogger().addHandler(file_handler)
    
    logger.info("=" * 70)
    logger.info("BetterTSE 时间序列编辑测试框架 (CiK官方范式版)")
    logger.info("=" * 70)
    logger.info(f"CSV文件: {args.csv_path}")
    logger.info(f"输出目录: {args.output_dir}")
    logger.info(f"样本数量: {args.num_samples}")
    logger.info(f"序列长度: {args.seq_len}")
    
    try:
        pipeline = BetterTSETestPipelineCiKOfficial(
            csv_path=args.csv_path,
            api_key=args.api_key,
            output_dir=args.output_dir,
            random_seed=args.seed,
            seq_len=args.seq_len
        )
        
        pipeline.run_batch_tests(num_samples=args.num_samples)
        pipeline.save_results()
        
        logger.info("\n" + "=" * 70)
        logger.info("测试完成!")
        logger.info("=" * 70)
        
    except Exception as e:
        logger.error(f"测试失败: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
