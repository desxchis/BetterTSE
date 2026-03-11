import os
import logging
import urllib.request
import zipfile
import gzip
import shutil
from typing import Optional, Tuple, List, Dict
from pathlib import Path

import numpy as np
import pandas as pd

from config import DatasetConfig, DATASET_CONFIGS, ScenarioType


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatasetLoader:
    """
    时间序列数据集加载器
    支持从本地加载或自动下载公开数据集
    """
    
    def __init__(self, data_dir: str = "data", auto_download: bool = True):
        self.data_dir = Path(data_dir)
        self.auto_download = auto_download
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._cache: Dict[str, np.ndarray] = {}
        
    def load_dataset(
        self, 
        dataset_name: str, 
        column: Optional[str] = None,
        seq_len: Optional[int] = None
    ) -> Tuple[np.ndarray, Dict]:
        """
        加载指定数据集
        
        Args:
            dataset_name: 数据集名称 (ETTh1, ETTm1, Traffic, Weather, Exchange)
            column: 要加载的列名，如果为None则使用配置中的默认列
            seq_len: 序列长度，如果为None则返回完整数据
            
        Returns:
            Tuple[np.ndarray, Dict]: (时间序列数据, 元数据信息)
        """
        if dataset_name not in DATASET_CONFIGS:
            raise ValueError(f"未知数据集: {dataset_name}. 可用数据集: {list(DATASET_CONFIGS.keys())}")
        
        config = DATASET_CONFIGS[dataset_name]
        
        if column is None:
            column = config.column_name
            
        cache_key = f"{dataset_name}_{column}"
        if cache_key in self._cache:
            data = self._cache[cache_key]
        else:
            file_path = self._get_data_file(config)
            data = self._load_csv_data(file_path, column)
            self._cache[cache_key] = data
            
        metadata = {
            "dataset_name": dataset_name,
            "column": column,
            "total_length": len(data),
            "description": config.description,
            "mean": float(np.mean(data)),
            "std": float(np.std(data)),
            "min": float(np.min(data)),
            "max": float(np.max(data))
        }
        
        if seq_len is not None:
            data = self._extract_sequence(data, seq_len)
            
        return data, metadata
    
    def load_multiple_sequences(
        self,
        dataset_name: str,
        num_sequences: int,
        seq_len: int,
        column: Optional[str] = None,
        random_seed: Optional[int] = None
    ) -> Tuple[List[np.ndarray], List[Dict]]:
        """
        从数据集中加载多条不重叠的序列
        
        Args:
            dataset_name: 数据集名称
            num_sequences: 要提取的序列数量
            seq_len: 每条序列的长度
            column: 列名
            random_seed: 随机种子
            
        Returns:
            Tuple[List[np.ndarray], List[Dict]]: (序列列表, 元数据列表)
        """
        full_data, base_metadata = self.load_dataset(dataset_name, column)
        
        if len(full_data) < num_sequences * seq_len:
            logger.warning(
                f"数据长度({len(full_data)})不足以提取{num_sequences}条不重叠的{seq_len}长度序列"
                f"，将使用有重叠的采样策略"
            )
            return self._sample_with_overlap(full_data, num_sequences, seq_len, random_seed, base_metadata)
        
        sequences = []
        metadata_list = []
        
        if random_seed is not None:
            np.random.seed(random_seed)
            
        max_start = len(full_data) - seq_len
        step = max(1, max_start // num_sequences)
        
        for i in range(num_sequences):
            start_idx = min(i * step, max_start)
            seq = full_data[start_idx:start_idx + seq_len].copy()
            sequences.append(seq)
            
            meta = base_metadata.copy()
            meta.update({
                "sequence_index": i,
                "start_idx": int(start_idx),
                "seq_len": seq_len
            })
            metadata_list.append(meta)
            
        return sequences, metadata_list
    
    def _sample_with_overlap(
        self,
        data: np.ndarray,
        num_sequences: int,
        seq_len: int,
        random_seed: Optional[int],
        base_metadata: Dict
    ) -> Tuple[List[np.ndarray], List[Dict]]:
        """允许重叠的采样策略"""
        if random_seed is not None:
            np.random.seed(random_seed)
            
        sequences = []
        metadata_list = []
        max_start = len(data) - seq_len
        
        start_indices = np.random.choice(max_start + 1, num_sequences, replace=True)
        
        for i, start_idx in enumerate(start_indices):
            seq = data[start_idx:start_idx + seq_len].copy()
            sequences.append(seq)
            
            meta = base_metadata.copy()
            meta.update({
                "sequence_index": i,
                "start_idx": int(start_idx),
                "seq_len": seq_len
            })
            metadata_list.append(meta)
            
        return sequences, metadata_list
    
    def _get_data_file(self, config: DatasetConfig) -> Path:
        """获取数据文件路径，如果不存在则自动下载"""
        file_path = self.data_dir / config.file_pattern
        
        if file_path.exists():
            logger.info(f"找到本地数据文件: {file_path}")
            return file_path
            
        if not self.auto_download:
            raise FileNotFoundError(
                f"数据文件不存在: {file_path}，且 auto_download=False"
            )
            
        logger.info(f"正在下载数据集 {config.name}...")
        return self._download_dataset(config)
    
    def _download_dataset(self, config: DatasetConfig) -> Path:
        """下载数据集"""
        file_path = self.data_dir / config.file_pattern
        
        try:
            if config.name == "Traffic":
                return self._download_traffic(config)
            elif config.name == "Exchange":
                return self._download_exchange(config)
            else:
                return self._download_csv(config)
        except Exception as e:
            logger.error(f"下载失败: {e}")
            logger.info("将使用合成数据代替...")
            return self._generate_synthetic_data(config)
    
    def _download_csv(self, config: DatasetConfig) -> Path:
        """下载CSV格式数据"""
        file_path = self.data_dir / config.file_pattern
        
        try:
            urllib.request.urlretrieve(config.url, file_path)
            logger.info(f"下载完成: {file_path}")
            return file_path
        except Exception as e:
            raise RuntimeError(f"下载 {config.name} 失败: {e}")
    
    def _download_traffic(self, config: DatasetConfig) -> Path:
        """下载Traffic数据集（txt格式）"""
        txt_path = self.data_dir / "traffic.txt"
        csv_path = self.data_dir / config.file_pattern
        
        try:
            urllib.request.urlretrieve(config.url, txt_path)
            logger.info("Traffic数据下载完成，正在转换为CSV格式...")
            
            df = pd.read_csv(txt_path, sep=',', header=None)
            df.to_csv(csv_path, index=False)
            txt_path.unlink()
            
            logger.info(f"转换完成: {csv_path}")
            return csv_path
        except Exception as e:
            raise RuntimeError(f"处理Traffic数据失败: {e}")
    
    def _download_exchange(self, config: DatasetConfig) -> Path:
        """下载Exchange数据集"""
        txt_path = self.data_dir / "exchange_rate.txt"
        csv_path = self.data_dir / config.file_pattern
        
        try:
            urllib.request.urlretrieve(config.url, txt_path)
            logger.info("Exchange数据下载完成，正在转换为CSV格式...")
            
            df = pd.read_csv(txt_path, sep=',', header=None)
            df.to_csv(csv_path, index=False)
            txt_path.unlink()
            
            logger.info(f"转换完成: {csv_path}")
            return csv_path
        except Exception as e:
            raise RuntimeError(f"处理Exchange数据失败: {e}")
    
    def _generate_synthetic_data(self, config: DatasetConfig, num_points: int = 10000) -> Path:
        """生成合成数据作为后备"""
        logger.warning(f"生成合成数据模拟 {config.name}...")
        
        np.random.seed(42)
        
        if config.name in ["ETTh1", "ETTm1"]:
            t = np.linspace(0, 100, num_points)
            data = (
                10 * np.sin(2 * np.pi * t / 24) +  
                5 * np.sin(2 * np.pi * t / 168) +  
                30 +  
                np.random.normal(0, 2, num_points)
            )
        elif config.name == "Traffic":
            t = np.linspace(0, 100, num_points)
            data = (
                0.5 * np.sin(2 * np.pi * t / 24) +
                0.3 * np.sin(2 * np.pi * t / 168) +
                0.5 +
                np.random.uniform(0, 0.2, num_points)
            )
        elif config.name == "Weather":
            t = np.linspace(0, 100, num_points)
            data = (
                10 * np.sin(2 * np.pi * t / 365) +
                20 +
                np.random.normal(0, 3, num_points)
            )
        else:
            t = np.linspace(0, 100, num_points)
            data = np.cumsum(np.random.normal(0, 0.01, num_points)) + 1
            
        df = pd.DataFrame({config.column_name: data})
        file_path = self.data_dir / config.file_pattern
        df.to_csv(file_path, index=False)
        
        logger.info(f"合成数据已保存: {file_path}")
        return file_path
    
    def _load_csv_data(self, file_path: Path, column: str) -> np.ndarray:
        """从CSV文件加载数据"""
        try:
            df = pd.read_csv(file_path)
            
            if column in df.columns:
                data = df[column].values
            elif column.isdigit() or column in [str(i) for i in range(len(df.columns))]:
                col_idx = int(column)
                data = df.iloc[:, col_idx].values
            else:
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    logger.warning(f"列 '{column}' 不存在，使用第一个数值列 '{numeric_cols[0]}'")
                    data = df[numeric_cols[0]].values
                else:
                    raise ValueError(f"文件中没有找到数值列: {file_path}")
                    
            data = data.astype(np.float64)
            data = data[~np.isnan(data)]
            
            logger.info(f"成功加载数据，长度: {len(data)}")
            return data
            
        except Exception as e:
            raise RuntimeError(f"加载CSV数据失败 {file_path}: {e}")
    
    def _extract_sequence(self, data: np.ndarray, seq_len: int) -> np.ndarray:
        """从数据中提取指定长度的序列"""
        if len(data) <= seq_len:
            return np.pad(data, (0, seq_len - len(data)), mode='edge')
        
        start_idx = np.random.randint(0, len(data) - seq_len)
        return data[start_idx:start_idx + seq_len].copy()
    
    def get_scenario_for_dataset(self, dataset_name: str) -> ScenarioType:
        """根据数据集名称返回对应的场景类型"""
        mapping = {
            "ETTh1": ScenarioType.ELECTRICITY,
            "ETTm1": ScenarioType.ELECTRICITY,
            "Traffic": ScenarioType.TRAFFIC,
            "Weather": ScenarioType.WEATHER,
            "Exchange": ScenarioType.EXCHANGE,
        }
        return mapping.get(dataset_name, ScenarioType.ELECTRICITY)
    
    def clear_cache(self):
        """清除缓存"""
        self._cache.clear()
        logger.info("缓存已清除")


def create_synthetic_sequence(
    seq_len: int = 100,
    pattern: str = "sine",
    noise_level: float = 0.1,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    创建合成时间序列数据
    
    Args:
        seq_len: 序列长度
        pattern: 模式类型 (sine, trend, step, combined)
        noise_level: 噪声水平
        seed: 随机种子
        
    Returns:
        np.ndarray: 合成的时间序列
    """
    if seed is not None:
        np.random.seed(seed)
        
    t = np.linspace(0, 4 * np.pi, seq_len)
    
    if pattern == "sine":
        data = np.sin(t) * 10 + 20
    elif pattern == "trend":
        data = np.linspace(0, 30, seq_len)
    elif pattern == "step":
        data = np.ones(seq_len) * 10
        data[seq_len//3:2*seq_len//3] = 20
    elif pattern == "combined":
        data = (
            np.sin(t) * 5 +
            np.linspace(0, 10, seq_len) +
            15
        )
    else:
        data = np.sin(t) * 10 + 20
        
    if noise_level > 0:
        noise = np.random.normal(0, noise_level * np.std(data), seq_len)
        data = data + noise
        
    return data
