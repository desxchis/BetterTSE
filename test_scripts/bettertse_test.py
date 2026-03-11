"""
BetterTSE 时间序列编辑测试框架
===============================

完整实现以下功能：
1. 真实CSV数据读取（ETTh1等数据集）
2. 确定性物理变化注入机制
3. 真实LLM API调用生成模糊描述
4. 完整评估指标：t-IoU、分区MSE/MAE（Editability/Preservability）

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
logger = logging.getLogger("BetterTSE_Evaluator")


@dataclass
class ChangeConfig:
    """物理变化配置"""
    change_type: str
    start_idx: int
    end_idx: int
    target_feature: str
    intensity: float = 1.0
    direction: str = "up"
    description: str = ""


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
    gt_config: Dict = field(default_factory=dict)
    llm_prediction: Dict = field(default_factory=dict)
    metrics: Dict = field(default_factory=dict)


class CSVDataLoader:
    """
    真实CSV数据加载器
    直接读取本地CSV文件，不使用任何模拟数据
    """
    
    def __init__(self, csv_path: str):
        """
        初始化数据加载器
        
        Args:
            csv_path: CSV文件路径
        """
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
            
    def get_sequence(
        self, 
        start_idx: int = 0, 
        seq_len: int = 100,
        features: Optional[List[str]] = None
    ) -> np.ndarray:
        """
        获取指定长度的序列
        
        Args:
            start_idx: 起始索引
            seq_len: 序列长度
            features: 要提取的特征列表
            
        Returns:
            np.ndarray: 时间序列数据
        """
        if features is None:
            features = self.features
            
        end_idx = start_idx + seq_len
        
        if end_idx > len(self.data):
            logger.warning(f"请求的序列超出数据范围，将调整起始位置")
            end_idx = len(self.data)
            start_idx = max(0, end_idx - seq_len)
            
        sequence = self.data[features].iloc[start_idx:end_idx].values
        
        logger.info(f"提取序列: 起始={start_idx}, 长度={len(sequence)}, 特征数={len(features)}")
        
        return sequence.astype(np.float64)
    
    def get_feature_data(self, feature: str, start_idx: int = 0, seq_len: int = 100) -> np.ndarray:
        """获取单个特征的数据"""
        return self.get_sequence(start_idx, seq_len, [feature]).flatten()
    
    def get_statistics(self) -> Dict:
        """获取数据统计信息"""
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


class PhysicalChangeInjector:
    """
    确定性物理变化注入器
    确保变化过程可复现且参数可配置
    """
    
    CHANGE_TYPES = [
        "amplification",      # 放大
        "attenuation",        # 衰减
        "baseline_shift",     # 基线平移
        "trend_injection",    # 趋势注入
        "event_drop",         # 事件骤降
        "anomaly_spike",      # 异常波动
        "smoothing"           # 平滑
    ]
    
    def __init__(self, random_seed: Optional[int] = None):
        """初始化注入器"""
        self.rng = np.random.RandomState(random_seed)
        self.history = []
        
    def inject_change(
        self,
        base_ts: np.ndarray,
        config: ChangeConfig
    ) -> Tuple[np.ndarray, np.ndarray, str]:
        """
        注入物理变化
        
        Args:
            base_ts: 基础时间序列
            config: 变化配置
            
        Returns:
            Tuple[target_ts, mask, description]
        """
        target_ts = base_ts.copy()
        seq_len = len(base_ts)
        
        mask = np.zeros(seq_len, dtype=np.float64)
        start_idx = max(0, config.start_idx)
        end_idx = min(config.end_idx, seq_len)
        
        mask[start_idx:end_idx] = 1.0
        
        segment = target_ts[start_idx:end_idx]
        intensity = config.intensity
        
        if config.change_type == "amplification":
            mean_val = np.mean(segment)
            target_ts[start_idx:end_idx] = (segment - mean_val) * intensity + mean_val
            desc = f"在索引[{start_idx}:{end_idx}]区间，数值被放大{intensity:.1f}倍"
            
        elif config.change_type == "attenuation":
            mean_val = np.mean(segment)
            target_ts[start_idx:end_idx] = (segment - mean_val) / intensity + mean_val
            desc = f"在索引[{start_idx}:{end_idx}]区间，数值被衰减至{1/intensity:.1%}"
            
        elif config.change_type == "baseline_shift":
            shift = np.mean(base_ts) * intensity * 0.5
            if config.direction == "down":
                shift = -shift
            target_ts[start_idx:end_idx] = segment + shift
            desc = f"在索引[{start_idx}:{end_idx}]区间，基线{'上移' if config.direction == 'up' else '下移'}{abs(shift):.2f}"
            
        elif config.change_type == "trend_injection":
            slope = (np.max(base_ts) - np.min(base_ts)) * intensity * 0.1 / (end_idx - start_idx)
            if config.direction == "down":
                slope = -slope
            trend = np.linspace(0, slope * (end_idx - start_idx), end_idx - start_idx)
            target_ts[start_idx:end_idx] = segment + trend
            desc = f"在索引[{start_idx}:{end_idx}]区间，注入{'上升' if config.direction == 'up' else '下降'}趋势"
            
        elif config.change_type == "event_drop":
            ratio = 1.0 - intensity * 0.8
            target_ts[start_idx:end_idx] = segment * ratio
            desc = f"在索引[{start_idx}:{end_idx}]区间，数值骤降至{ratio:.0%}"
            
        elif config.change_type == "anomaly_spike":
            noise_std = np.std(base_ts) * intensity
            noise = self.rng.normal(0, noise_std, end_idx - start_idx)
            target_ts[start_idx:end_idx] = segment + noise
            desc = f"在索引[{start_idx}:{end_idx}]区间，注入异常波动(噪声标准差={noise_std:.2f})"
            
        elif config.change_type == "smoothing":
            window = min(5, end_idx - start_idx)
            if window > 1:
                smoothed = pd.Series(segment).rolling(window, min_periods=1, center=True).mean().values
                target_ts[start_idx:end_idx] = smoothed
            desc = f"在索引[{start_idx}:{end_idx}]区间，应用平滑处理(窗口={window})"
            
        else:
            desc = f"未知的变化类型: {config.change_type}"
            
        self.history.append({
            "config": asdict(config),
            "description": desc
        })
        
        return target_ts, mask, desc
    
    def generate_random_config(
        self,
        seq_len: int,
        target_feature: str = "OT",
        change_type: Optional[str] = None
    ) -> ChangeConfig:
        """生成随机变化配置"""
        if change_type is None:
            change_type = self.rng.choice(self.CHANGE_TYPES)
            
        window_size = self.rng.randint(int(seq_len * 0.15), int(seq_len * 0.35))
        start_idx = self.rng.randint(int(seq_len * 0.1), int(seq_len * 0.6))
        end_idx = min(start_idx + window_size, seq_len - 1)
        
        intensity = self.rng.uniform(0.3, 0.8)
        direction = self.rng.choice(["up", "down"])
        
        return ChangeConfig(
            change_type=change_type,
            start_idx=start_idx,
            end_idx=end_idx,
            target_feature=target_feature,
            intensity=intensity,
            direction=direction
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
        """
        初始化LLM客户端
        
        Args:
            api_key: API密钥
            base_url: API基础URL
            model_name: 模型名称
        """
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("OPENAI_API_KEY")
        self.base_url = base_url
        self.model_name = model_name
        self.client = None
        
        if not self.api_key:
            raise ValueError(
                "未找到API密钥！请通过以下方式之一提供：\n"
                "1. 传入api_key参数\n"
                "2. 设置环境变量DEEPSEEK_API_KEY\n"
                "3. 设置环境变量OPENAI_API_KEY"
            )
            
        self._init_client()
        
    def _init_client(self):
        """初始化OpenAI客户端"""
        try:
            from openai import OpenAI
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )
            logger.info(f"LLM客户端初始化成功: {self.base_url}, 模型: {self.model_name}")
        except ImportError:
            raise ImportError("请安装openai库: pip install openai")
            
    def generate_vague_prompt(
        self,
        scenario: str,
        change_type: str,
        physical_desc: str,
        position_desc: str,
        temperature: float = 0.7
    ) -> str:
        """
        生成模糊提示词
        
        Args:
            scenario: 业务场景
            change_type: 变化类型
            physical_desc: 物理变化描述
            position_desc: 位置描述
            temperature: 温度参数
            
        Returns:
            str: 生成的模糊提示词
        """
        system_prompt = """你是一个缺乏技术背景，但对业务场景非常熟悉的业务员。
我会提供给你一条时间序列的【底层物理变化】，你需要根据指定的【业务场景】，将其包装成一句口语化、模糊、宏观的自然语言指令。

严格要求：
1. 绝对不能出现具体的索引值（如 index 40 到 60）、具体的修改数值或比例。
2. 只能用时间段（如"下午"、"半夜"、"中段"）和定性描述（如"骤降"、"稍微平缓一点"、"拔高一个台阶"）来表达。
3. 语言风格要自然，像是给AI助手下达任务。
4. 只返回一句指令，不要任何多余的解释或前缀。
5. 描述要具有业务场景的真实感。"""

        user_prompt = f"""
【业务场景】：{scenario}
【意图类别】：{change_type}
【底层物理变化】：{physical_desc}
【位置参考】：变化发生在序列的{position_desc}部分

请生成1句符合要求的模糊指令："""

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
            logger.info(f"LLM生成提示词成功: {result[:50]}...")
            return result
            
        except Exception as e:
            logger.error(f"LLM API调用失败: {e}")
            raise
            
    def predict_edit_region(
        self,
        vague_prompt: str,
        seq_len: int,
        features: List[str]
    ) -> Dict:
        """
        让LLM根据模糊提示词预测编辑区域
        
        Args:
            vague_prompt: 模糊提示词
            seq_len: 序列长度
            features: 特征列表
            
        Returns:
            Dict: 预测结果 {target_feature, start_step, end_step}
        """
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
            raise


class TSEditEvaluator:
    """
    时间序列编辑评估器
    实现完整的评估指标：t-IoU、分区MSE/MAE
    """
    
    def __init__(self):
        pass
        
    def compute_t_iou(
        self,
        gt_start: int,
        gt_end: int,
        pred_start: int,
        pred_end: int
    ) -> float:
        """
        计算时间交并比 (t-IoU)
        
        Args:
            gt_start: 真实起始位置
            gt_end: 真实结束位置
            pred_start: 预测起始位置
            pred_end: 预测结束位置
            
        Returns:
            float: t-IoU值
        """
        gt_set = set(range(gt_start, gt_end + 1))
        pred_set = set(range(pred_start, pred_end + 1))
        
        intersection = len(gt_set.intersection(pred_set))
        union = len(gt_set.union(pred_set))
        
        t_iou = intersection / union if union > 0 else 0.0
        
        return t_iou
    
    def compute_partitioned_metrics(
        self,
        base_ts: np.ndarray,
        target_ts: np.ndarray,
        generated_ts: np.ndarray,
        mask: np.ndarray
    ) -> Dict[str, float]:
        """
        计算分区评估指标
        
        Args:
            base_ts: 基础时间序列
            target_ts: 目标时间序列
            generated_ts: 生成的时间序列
            mask: 编辑区域掩码
            
        Returns:
            Dict: 评估指标
        """
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
        """
        完整评估
        
        Args:
            base_ts: 基础时间序列
            target_ts: 目标时间序列
            generated_ts: 生成的时间序列
            gt_mask: 真实掩码
            gt_config: 真实配置
            llm_prediction: LLM预测结果
            
        Returns:
            EvaluationMetrics: 评估指标
        """
        t_iou = self.compute_t_iou(
            gt_config['start_idx'],
            gt_config['end_idx'],
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


class BetterTSETestPipeline:
    """
    BetterTSE完整测试流程
    """
    
    SCENARIOS = {
        "OT": "电力变压器油温监控",
        "HUFL": "高压负载监控",
        "HULL": "高压低负载监控",
        "MUFL": "中压负载监控",
        "MULL": "中压低负载监控",
        "LUFL": "低压负载监控",
        "LULL": "低压低负载监控"
    }
    
    def __init__(
        self,
        csv_path: str,
        api_key: Optional[str] = None,
        output_dir: str = "test_results",
        random_seed: Optional[int] = None
    ):
        """
        初始化测试流程
        
        Args:
            csv_path: CSV数据文件路径
            api_key: LLM API密钥
            output_dir: 输出目录
            random_seed: 随机种子
        """
        self.data_loader = CSVDataLoader(csv_path)
        self.change_injector = PhysicalChangeInjector(random_seed)
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
        target_feature: str = "OT",
        change_type: Optional[str] = None
    ) -> TestSample:
        """
        运行单个测试
        
        Args:
            sample_id: 样本ID
            start_idx: 数据起始索引
            seq_len: 序列长度
            target_feature: 目标特征
            change_type: 变化类型
            
        Returns:
            TestSample: 测试样本
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"开始测试样本: {sample_id}")
        logger.info(f"{'='*60}")
        
        base_ts = self.data_loader.get_feature_data(target_feature, start_idx, seq_len)
        
        config = self.change_injector.generate_random_config(
            seq_len=seq_len,
            target_feature=target_feature,
            change_type=change_type
        )
        
        target_ts, gt_mask, physical_desc = self.change_injector.inject_change(base_ts, config)
        
        scenario = self.SCENARIOS.get(target_feature, "时间序列监控")
        position_desc = self._get_position_description(config.start_idx, config.end_idx, seq_len)
        
        if self.llm_client:
            logger.info("调用LLM生成模糊提示词...")
            vague_prompt = self.llm_client.generate_vague_prompt(
                scenario=scenario,
                change_type=config.change_type,
                physical_desc=physical_desc,
                position_desc=position_desc
            )
            
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
            vague_prompt = f"请调整{position_desc}部分的数据"
            llm_prediction = {
                "target_feature": target_feature,
                "start_step": config.start_idx,
                "end_step": config.end_idx
            }
            
        pred_mask = np.zeros(seq_len, dtype=np.float64)
        pred_start = llm_prediction.get('start_step', config.start_idx)
        pred_end = llm_prediction.get('end_step', config.end_idx)
        pred_mask[pred_start:pred_end] = 1.0
        
        generated_ts = pred_mask * target_ts + (1 - pred_mask) * base_ts
        
        gt_config = {
            "change_type": config.change_type,
            "start_idx": config.start_idx,
            "end_idx": config.end_idx,
            "target_feature": config.target_feature,
            "intensity": config.intensity,
            "direction": config.direction
        }
        
        metrics = self.evaluator.evaluate(
            base_ts=base_ts,
            target_ts=target_ts,
            generated_ts=generated_ts,
            gt_mask=gt_mask,
            gt_config=gt_config,
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
            gt_config=gt_config,
            llm_prediction=llm_prediction,
            metrics=asdict(metrics)
        )
        
        self.samples.append(sample)
        
        logger.info(f"样本 {sample_id} 测试完成:")
        logger.info(f"  - t-IoU: {metrics.t_iou:.4f}")
        logger.info(f"  - 特征准确率: {metrics.feature_accuracy:.1%}")
        logger.info(f"  - 编辑区MSE: {metrics.mse_edit_region:.4f}")
        logger.info(f"  - 保留区MSE: {metrics.mse_preserve_region:.4f}")
        logger.info(f"  - Editability: {metrics.editability_score:.4f}")
        logger.info(f"  - Preservability: {metrics.preservability_score:.4f}")
        
        return sample
    
    def run_batch_tests(
        self,
        num_samples: int = 10,
        seq_len: int = 100,
        target_feature: str = "OT",
        change_types: Optional[List[str]] = None
    ) -> List[TestSample]:
        """
        批量运行测试
        
        Args:
            num_samples: 样本数量
            seq_len: 序列长度
            target_feature: 目标特征
            change_types: 变化类型列表
            
        Returns:
            List[TestSample]: 测试样本列表
        """
        logger.info(f"\n开始批量测试，共 {num_samples} 个样本")
        
        max_start = len(self.data_loader.data) - seq_len
        step = max(1, max_start // num_samples)
        
        for i in range(num_samples):
            start_idx = (i * step) % max_start
            change_type = None
            if change_types:
                change_type = change_types[i % len(change_types)]
                
            sample_id = f"TSE_{target_feature}_{i+1:03d}"
            
            try:
                self.run_single_test(
                    sample_id=sample_id,
                    start_idx=start_idx,
                    seq_len=seq_len,
                    target_feature=target_feature,
                    change_type=change_type
                )
            except Exception as e:
                logger.error(f"样本 {sample_id} 测试失败: {e}")
                continue
                
        return self.samples
    
    def _get_position_description(self, start_idx: int, end_idx: int, seq_len: int) -> str:
        """获取位置描述"""
        mid_ratio = (start_idx + end_idx) / 2 / seq_len
        
        if mid_ratio < 0.25:
            return "前段"
        elif mid_ratio < 0.5:
            return "中前段"
        elif mid_ratio < 0.75:
            return "中后段"
        else:
            return "后段"
    
    def save_results(self, filename: str = "test_results.json"):
        """保存测试结果"""
        output_file = self.output_dir / filename
        
        results = {
            "metadata": {
                "csv_path": str(self.data_loader.csv_path),
                "total_samples": len(self.samples),
                "test_time": datetime.now().isoformat(),
                "random_seed": self.random_seed
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
        """计算汇总统计"""
        if not self.samples:
            return {}
            
        t_ious = [s.metrics['t_iou'] for s in self.samples]
        feature_accs = [s.metrics['feature_accuracy'] for s in self.samples]
        editability_scores = [s.metrics['editability_score'] for s in self.samples]
        preservability_scores = [s.metrics['preservability_score'] for s in self.samples]
        
        return {
            "avg_t_iou": float(np.mean(t_ious)),
            "std_t_iou": float(np.std(t_ious)),
            "avg_feature_accuracy": float(np.mean(feature_accs)),
            "avg_editability": float(np.mean(editability_scores)),
            "avg_preservability": float(np.mean(preservability_scores)),
            "change_type_distribution": self._count_change_types()
        }
    
    def _count_change_types(self) -> Dict[str, int]:
        """统计变化类型分布"""
        counts = {}
        for s in self.samples:
            ct = s.gt_config.get('change_type', 'unknown')
            counts[ct] = counts.get(ct, 0) + 1
        return counts
    
    def _save_report(self):
        """保存文本报告"""
        report_file = self.output_dir / "evaluation_report.txt"
        
        summary = self._compute_summary()
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write("BetterTSE 时间序列编辑测试评估报告\n")
            f.write("=" * 70 + "\n\n")
            
            f.write(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"数据文件: {self.data_loader.csv_path}\n")
            f.write(f"样本数量: {len(self.samples)}\n\n")
            
            f.write("-" * 70 + "\n")
            f.write("一、整体评估指标\n")
            f.write("-" * 70 + "\n")
            f.write(f"平均 t-IoU:           {summary.get('avg_t_iou', 0):.4f}\n")
            f.write(f"t-IoU 标准差:         {summary.get('std_t_iou', 0):.4f}\n")
            f.write(f"平均特征准确率:       {summary.get('avg_feature_accuracy', 0):.1%}\n")
            f.write(f"平均 Editability:     {summary.get('avg_editability', 0):.4f}\n")
            f.write(f"平均 Preservability:  {summary.get('avg_preservability', 0):.4f}\n\n")
            
            f.write("-" * 70 + "\n")
            f.write("二、变化类型分布\n")
            f.write("-" * 70 + "\n")
            for ct, count in summary.get('change_type_distribution', {}).items():
                f.write(f"  {ct}: {count}\n")
            f.write("\n")
            
            f.write("-" * 70 + "\n")
            f.write("三、样本详情（前5个）\n")
            f.write("-" * 70 + "\n")
            for i, sample in enumerate(self.samples[:5]):
                f.write(f"\n样本 {i+1}: {sample.sample_id}\n")
                f.write(f"  变化类型: {sample.gt_config.get('change_type', 'N/A')}\n")
                f.write(f"  编辑区间: [{sample.gt_config.get('start_idx', 0)}, {sample.gt_config.get('end_idx', 0)}]\n")
                f.write(f"  模糊提示词: {sample.vague_prompt}\n")
                f.write(f"  t-IoU: {sample.metrics['t_iou']:.4f}\n")
                f.write(f"  Editability: {sample.metrics['editability_score']:.4f}\n")
                f.write(f"  Preservability: {sample.metrics['preservability_score']:.4f}\n")
                
        logger.info(f"评估报告已保存至: {report_file}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='BetterTSE 时间序列编辑测试脚本',
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
        default=None,
        help='LLM API密钥（也可通过环境变量设置）'
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
        '--target-feature',
        type=str,
        default='OT',
        help='目标特征'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='随机种子'
    )
    
    args = parser.parse_args()
    
    log_file = Path(args.output_dir) / f"test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - [%(levelname)s] - %(name)s - %(message)s'))
    logging.getLogger().addHandler(file_handler)
    
    logger.info("=" * 70)
    logger.info("BetterTSE 时间序列编辑测试框架")
    logger.info("=" * 70)
    logger.info(f"CSV文件: {args.csv_path}")
    logger.info(f"输出目录: {args.output_dir}")
    logger.info(f"样本数量: {args.num_samples}")
    logger.info(f"序列长度: {args.seq_len}")
    logger.info(f"目标特征: {args.target_feature}")
    
    try:
        pipeline = BetterTSETestPipeline(
            csv_path=args.csv_path,
            api_key=args.api_key,
            output_dir=args.output_dir,
            random_seed=args.seed
        )
        
        pipeline.run_batch_tests(
            num_samples=args.num_samples,
            seq_len=args.seq_len,
            target_feature=args.target_feature
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
