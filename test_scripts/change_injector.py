import logging
from typing import Dict, Tuple, Optional, List, Any
from dataclasses import dataclass
import numpy as np
import pandas as pd

from config import ChangeType, CHANGE_DESCRIPTIONS


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ChangeParameters:
    """物理变化参数"""
    change_type: ChangeType
    start_idx: int
    end_idx: int
    intensity: float
    direction: str = "upward"
    ratio: float = 1.0
    shift_value: float = 0.0
    noise_std: float = 0.0
    smoothing_window: int = 5
    slope: float = 0.0
    

@dataclass
class ChangeResult:
    """物理变化结果"""
    base_ts: np.ndarray
    target_ts: np.ndarray
    parameters: ChangeParameters
    physical_description: str
    change_mask: np.ndarray
    metrics: Dict[str, float]


class PhysicalChangeInjector:
    """
    确定性物理变化注入器
    实现从Base TS到Target TS的可复现、可配置的变化注入
    """
    
    def __init__(self, random_seed: Optional[int] = None):
        self.rng = np.random.RandomState(random_seed)
        self._change_history: List[ChangeResult] = []
        
    def inject_change(
        self,
        base_ts: np.ndarray,
        change_type: ChangeType,
        start_idx: Optional[int] = None,
        end_idx: Optional[int] = None,
        intensity: Optional[float] = None,
        **kwargs
    ) -> ChangeResult:
        """
        注入物理变化
        
        Args:
            base_ts: 基础时间序列
            change_type: 变化类型
            start_idx: 变化起始索引
            end_idx: 变化结束索引
            intensity: 变化强度 (0-1)
            **kwargs: 其他参数
            
        Returns:
            ChangeResult: 变化结果对象
        """
        base_ts = np.asarray(base_ts, dtype=np.float64).copy()
        seq_len = len(base_ts)
        
        if start_idx is None or end_idx is None:
            start_idx, end_idx = self._generate_window(seq_len)
            
        start_idx = max(0, min(start_idx, seq_len - 1))
        end_idx = max(start_idx + 1, min(end_idx, seq_len))
        
        if intensity is None:
            intensity = self.rng.uniform(0.3, 0.7)
            
        params = self._create_parameters(
            change_type, start_idx, end_idx, intensity, **kwargs
        )
        
        target_ts, physical_desc = self._apply_change(base_ts, params)
        
        change_mask = np.zeros(seq_len, dtype=bool)
        change_mask[start_idx:end_idx] = True
        
        metrics = self._compute_metrics(base_ts, target_ts, change_mask)
        
        result = ChangeResult(
            base_ts=base_ts,
            target_ts=target_ts,
            parameters=params,
            physical_description=physical_desc,
            change_mask=change_mask,
            metrics=metrics
        )
        
        self._change_history.append(result)
        
        return result
    
    def inject_random_change(
        self,
        base_ts: np.ndarray,
        change_types: Optional[List[ChangeType]] = None,
        **kwargs
    ) -> ChangeResult:
        """
        注入随机类型的物理变化
        
        Args:
            base_ts: 基础时间序列
            change_types: 可选的变化类型列表
            **kwargs: 其他参数
            
        Returns:
            ChangeResult: 变化结果对象
        """
        if change_types is None:
            change_types = list(ChangeType)
            
        change_type = self.rng.choice(change_types)
        return self.inject_change(base_ts, change_type, **kwargs)
    
    def _generate_window(self, seq_len: int) -> Tuple[int, int]:
        """生成随机窗口"""
        window_ratio = self.rng.uniform(0.2, 0.4)
        window_size = max(5, int(seq_len * window_ratio))
        
        position_ratio = self.rng.uniform(0.1, 0.5)
        start_idx = int(seq_len * position_ratio)
        end_idx = min(start_idx + window_size, seq_len - 1)
        
        return start_idx, end_idx
    
    def _create_parameters(
        self,
        change_type: ChangeType,
        start_idx: int,
        end_idx: int,
        intensity: float,
        **kwargs
    ) -> ChangeParameters:
        """创建变化参数"""
        params = ChangeParameters(
            change_type=change_type,
            start_idx=start_idx,
            end_idx=end_idx,
            intensity=intensity
        )
        
        if change_type == ChangeType.EVENT_DROP:
            params.ratio = kwargs.get('ratio', 0.2 + intensity * 0.3)
            
        elif change_type == ChangeType.BASELINE_SHIFT:
            params.shift_value = kwargs.get('shift_value', None)
            
        elif change_type == ChangeType.ANOMALY_SPIKE:
            params.noise_std = kwargs.get('noise_std', None)
            
        elif change_type == ChangeType.TREND_SMOOTHING:
            params.smoothing_window = kwargs.get('window', 5)
            
        elif change_type == ChangeType.AMPLIFICATION:
            params.ratio = kwargs.get('ratio', 1.3 + intensity * 0.7)
            
        elif change_type == ChangeType.ATTENUATION:
            params.ratio = kwargs.get('ratio', 0.3 + intensity * 0.3)
            
        elif change_type == ChangeType.TREND_INJECTION:
            params.direction = kwargs.get('direction', self.rng.choice(["上升", "下降"]))
            params.slope = kwargs.get('slope', None)
            
        return params
    
    def _apply_change(
        self,
        base_ts: np.ndarray,
        params: ChangeParameters
    ) -> Tuple[np.ndarray, str]:
        """应用物理变化"""
        target_ts = base_ts.copy()
        start_idx = params.start_idx
        end_idx = params.end_idx
        
        if params.change_type == ChangeType.EVENT_DROP:
            target_ts, desc = self._apply_event_drop(target_ts, params)
            
        elif params.change_type == ChangeType.BASELINE_SHIFT:
            target_ts, desc = self._apply_baseline_shift(target_ts, params)
            
        elif params.change_type == ChangeType.ANOMALY_SPIKE:
            target_ts, desc = self._apply_anomaly_spike(target_ts, params)
            
        elif params.change_type == ChangeType.TREND_SMOOTHING:
            target_ts, desc = self._apply_trend_smoothing(target_ts, params)
            
        elif params.change_type == ChangeType.AMPLIFICATION:
            target_ts, desc = self._apply_amplification(target_ts, params)
            
        elif params.change_type == ChangeType.ATTENUATION:
            target_ts, desc = self._apply_attenuation(target_ts, params)
            
        elif params.change_type == ChangeType.TREND_INJECTION:
            target_ts, desc = self._apply_trend_injection(target_ts, params)
            
        else:
            desc = "未发生变化"
            
        return target_ts, desc
    
    def _apply_event_drop(
        self,
        ts: np.ndarray,
        params: ChangeParameters
    ) -> Tuple[np.ndarray, str]:
        """应用事件导致的骤降"""
        start_idx, end_idx = params.start_idx, params.end_idx
        ratio = params.ratio
        
        original_mean = np.mean(ts[start_idx:end_idx])
        ts[start_idx:end_idx] = ts[start_idx:end_idx] * ratio
        
        desc = CHANGE_DESCRIPTIONS[ChangeType.EVENT_DROP]["zh"].format(ratio=ratio)
        return ts, desc
    
    def _apply_baseline_shift(
        self,
        ts: np.ndarray,
        params: ChangeParameters
    ) -> Tuple[np.ndarray, str]:
        """应用基线平移"""
        start_idx, end_idx = params.start_idx, params.end_idx
        
        if params.shift_value is None:
            shift_value = np.mean(ts) * (0.3 + params.intensity * 0.5)
        else:
            shift_value = params.shift_value
            
        ts[start_idx:end_idx] = ts[start_idx:end_idx] + shift_value
        
        desc = CHANGE_DESCRIPTIONS[ChangeType.BASELINE_SHIFT]["zh"].format(shift=shift_value)
        return ts, desc
    
    def _apply_anomaly_spike(
        self,
        ts: np.ndarray,
        params: ChangeParameters
    ) -> Tuple[np.ndarray, str]:
        """应用异常波动"""
        start_idx, end_idx = params.start_idx, params.end_idx
        
        if params.noise_std is None:
            noise_std = np.std(ts) * (1.5 + params.intensity * 1.5)
        else:
            noise_std = params.noise_std
            
        noise = self.rng.normal(0, noise_std, end_idx - start_idx)
        ts[start_idx:end_idx] = ts[start_idx:end_idx] + noise
        
        desc = CHANGE_DESCRIPTIONS[ChangeType.ANOMALY_SPIKE]["zh"]
        return ts, desc
    
    def _apply_trend_smoothing(
        self,
        ts: np.ndarray,
        params: ChangeParameters
    ) -> Tuple[np.ndarray, str]:
        """应用趋势平滑"""
        start_idx, end_idx = params.start_idx, params.end_idx
        window = params.smoothing_window
        
        window = min(window, end_idx - start_idx)
        
        if window > 1:
            segment = ts[start_idx:end_idx]
            smoothed = pd.Series(segment).rolling(
                window, min_periods=1, center=True
            ).mean().values
            ts[start_idx:end_idx] = smoothed
            
        desc = CHANGE_DESCRIPTIONS[ChangeType.TREND_SMOOTHING]["zh"]
        return ts, desc
    
    def _apply_amplification(
        self,
        ts: np.ndarray,
        params: ChangeParameters
    ) -> Tuple[np.ndarray, str]:
        """应用放大"""
        start_idx, end_idx = params.start_idx, params.end_idx
        ratio = params.ratio
        
        segment_mean = np.mean(ts[start_idx:end_idx])
        ts[start_idx:end_idx] = (ts[start_idx:end_idx] - segment_mean) * ratio + segment_mean
        
        desc = CHANGE_DESCRIPTIONS[ChangeType.AMPLIFICATION]["zh"].format(ratio=ratio)
        return ts, desc
    
    def _apply_attenuation(
        self,
        ts: np.ndarray,
        params: ChangeParameters
    ) -> Tuple[np.ndarray, str]:
        """应用衰减"""
        start_idx, end_idx = params.start_idx, params.end_idx
        ratio = params.ratio
        
        segment_mean = np.mean(ts[start_idx:end_idx])
        ts[start_idx:end_idx] = (ts[start_idx:end_idx] - segment_mean) * ratio + segment_mean
        
        desc = CHANGE_DESCRIPTIONS[ChangeType.ATTENUATION]["zh"].format(ratio=ratio)
        return ts, desc
    
    def _apply_trend_injection(
        self,
        ts: np.ndarray,
        params: ChangeParameters
    ) -> Tuple[np.ndarray, str]:
        """注入趋势"""
        start_idx, end_idx = params.start_idx, params.end_idx
        direction = params.direction
        
        if params.slope is None:
            segment_range = np.max(ts) - np.min(ts)
            slope = segment_range * (0.1 + params.intensity * 0.2) / (end_idx - start_idx)
        else:
            slope = params.slope
            
        if direction == "下降":
            slope = -abs(slope)
        else:
            slope = abs(slope)
            
        trend = np.linspace(0, slope * (end_idx - start_idx), end_idx - start_idx)
        ts[start_idx:end_idx] = ts[start_idx:end_idx] + trend
        
        desc = CHANGE_DESCRIPTIONS[ChangeType.TREND_INJECTION]["zh"].format(direction=direction)
        return ts, desc
    
    def _compute_metrics(
        self,
        base_ts: np.ndarray,
        target_ts: np.ndarray,
        change_mask: np.ndarray
    ) -> Dict[str, float]:
        """计算变化指标"""
        changed_region = change_mask
        
        if np.sum(changed_region) == 0:
            return {"mse": 0.0, "mae": 0.0, "change_magnitude": 0.0}
        
        base_changed = base_ts[changed_region]
        target_changed = target_ts[changed_region]
        
        mse = float(np.mean((base_changed - target_changed) ** 2))
        mae = float(np.mean(np.abs(base_changed - target_changed)))
        
        change_magnitude = float(np.abs(np.mean(target_changed) - np.mean(base_changed)))
        
        return {
            "mse": mse,
            "mae": mae,
            "change_magnitude": change_magnitude,
            "relative_change": float(change_magnitude / (np.mean(base_changed) + 1e-8))
        }
    
    def get_change_history(self) -> List[ChangeResult]:
        """获取变化历史"""
        return self._change_history.copy()
    
    def clear_history(self):
        """清除历史记录"""
        self._change_history.clear()
        
    def reproduce_change(
        self,
        base_ts: np.ndarray,
        params: ChangeParameters
    ) -> ChangeResult:
        """
        根据参数复现变化
        
        Args:
            base_ts: 基础时间序列
            params: 变化参数
            
        Returns:
            ChangeResult: 变化结果
        """
        return self.inject_change(
            base_ts,
            params.change_type,
            params.start_idx,
            params.end_idx,
            params.intensity,
            ratio=params.ratio,
            shift_value=params.shift_value,
            noise_std=params.noise_std,
            window=params.smoothing_window,
            direction=params.direction,
            slope=params.slope
        )


def validate_change_injection(
    base_ts: np.ndarray,
    target_ts: np.ndarray,
    change_mask: np.ndarray,
    tolerance: float = 1e-6
) -> Dict[str, Any]:
    """
    验证变化注入的正确性
    
    Args:
        base_ts: 基础时间序列
        target_ts: 目标时间序列
        change_mask: 变化掩码
        tolerance: 容差
        
    Returns:
        Dict: 验证结果
    """
    unchanged_region = ~change_mask
    
    unchanged_diff = np.abs(base_ts[unchanged_region] - target_ts[unchanged_region])
    unchanged_valid = np.all(unchanged_diff < tolerance)
    
    changed_region = change_mask
    changed_diff = np.abs(base_ts[changed_region] - target_ts[changed_region])
    changed_valid = np.any(changed_diff > tolerance)
    
    return {
        "is_valid": unchanged_valid and changed_valid,
        "unchanged_region_correct": unchanged_valid,
        "changed_region_correct": changed_valid,
        "max_unchanged_diff": float(np.max(unchanged_diff)) if len(unchanged_diff) > 0 else 0.0,
        "max_changed_diff": float(np.max(changed_diff)) if len(changed_diff) > 0 else 0.0,
        "change_ratio": float(np.sum(change_mask) / len(base_ts))
    }
