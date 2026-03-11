import os
import json
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np

from config import ChangeType, ScenarioType, TestConfig, get_config
from data_loader import DatasetLoader, create_synthetic_sequence
from change_injector import PhysicalChangeInjector, ChangeResult, validate_change_injection
from llm_interface import VaguePromptGenerator, LLMResponse, evaluate_prompt_quality


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TestSample:
    """测试样本"""
    sample_id: str
    dataset_name: str
    scenario: str
    base_ts: List[float]
    target_ts: List[float]
    vague_prompt: str
    gt_change_type: str
    gt_start_idx: int
    gt_end_idx: int
    gt_physical_desc: str
    gt_parameters: Dict[str, Any]
    llm_response: Dict[str, Any]
    validation_result: Dict[str, Any]
    prompt_quality: Dict[str, Any]
    metadata: Dict[str, Any]


@dataclass
class TestResult:
    """测试结果"""
    total_samples: int
    successful_samples: int
    failed_samples: int
    avg_prompt_quality: float
    avg_change_magnitude: float
    change_type_distribution: Dict[str, int]
    scenario_distribution: Dict[str, int]
    samples: List[TestSample]


class TestPipeline:
    """
    完整的测试流程管道
    """
    
    def __init__(
        self,
        config: Optional[TestConfig] = None,
        data_dir: str = "data",
        output_dir: Optional[str] = None,
        use_mock_llm: bool = False,
        random_seed: Optional[int] = None
    ):
        self.config = config or get_config()
        self.output_dir = Path(output_dir or self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.data_loader = DatasetLoader(data_dir=data_dir)
        self.change_injector = PhysicalChangeInjector(random_seed=random_seed)
        self.prompt_generator = VaguePromptGenerator(use_mock=use_mock_llm)
        
        self.random_seed = random_seed
        if random_seed is not None:
            np.random.seed(random_seed)
            
        self._test_samples: List[TestSample] = []
        
    def run_full_test(
        self,
        dataset_names: Optional[List[str]] = None,
        num_samples_per_dataset: int = 10,
        seq_len: int = 100,
        change_types: Optional[List[ChangeType]] = None,
        save_results: bool = True
    ) -> TestResult:
        """
        运行完整测试流程
        
        Args:
            dataset_names: 要测试的数据集名称列表
            num_samples_per_dataset: 每个数据集的样本数量
            seq_len: 序列长度
            change_types: 要测试的变化类型列表
            save_results: 是否保存结果
            
        Returns:
            TestResult: 测试结果
        """
        logger.info("=" * 60)
        logger.info("开始运行完整测试流程")
        logger.info("=" * 60)
        
        if dataset_names is None:
            dataset_names = ["ETTh1", "ETTm1", "Traffic"]
            
        if change_types is None:
            change_types = list(ChangeType)
            
        total_samples = len(dataset_names) * num_samples_per_dataset
        logger.info(f"计划生成 {total_samples} 个测试样本")
        
        self._test_samples = []
        sample_idx = 0
        
        for dataset_name in dataset_names:
            logger.info(f"\n处理数据集: {dataset_name}")
            
            try:
                sequences, metadatas = self.data_loader.load_multiple_sequences(
                    dataset_name=dataset_name,
                    num_sequences=num_samples_per_dataset,
                    seq_len=seq_len,
                    random_seed=self.random_seed
                )
            except Exception as e:
                logger.warning(f"加载 {dataset_name} 失败: {e}，使用合成数据")
                sequences = [
                    create_synthetic_sequence(seq_len, seed=self.random_seed + i)
                    for i in range(num_samples_per_dataset)
                ]
                metadatas = [{"dataset_name": dataset_name, "seq_len": seq_len}] * num_samples_per_dataset
                
            scenario = self.data_loader.get_scenario_for_dataset(dataset_name)
            
            for i, (base_ts, metadata) in enumerate(zip(sequences, metadatas)):
                sample_idx += 1
                logger.info(f"  处理样本 {sample_idx}/{total_samples}")
                
                sample = self._process_single_sample(
                    base_ts=base_ts,
                    sample_id=f"TSE_{dataset_name}_{i+1:03d}",
                    dataset_name=dataset_name,
                    scenario=scenario,
                    change_types=change_types,
                    metadata=metadata
                )
                
                if sample is not None:
                    self._test_samples.append(sample)
                    
        result = self._compile_results()
        
        if save_results:
            self._save_results(result)
            
        logger.info("\n" + "=" * 60)
        logger.info("测试完成!")
        logger.info(f"成功样本: {result.successful_samples}/{result.total_samples}")
        logger.info(f"平均提示词质量: {result.avg_prompt_quality:.3f}")
        logger.info("=" * 60)
        
        return result
    
    def _process_single_sample(
        self,
        base_ts: np.ndarray,
        sample_id: str,
        dataset_name: str,
        scenario: ScenarioType,
        change_types: List[ChangeType],
        metadata: Dict
    ) -> Optional[TestSample]:
        """处理单个样本"""
        try:
            change_result = self.change_injector.inject_random_change(
                base_ts=base_ts,
                change_types=change_types
            )
            
            validation = validate_change_injection(
                base_ts=change_result.base_ts,
                target_ts=change_result.target_ts,
                change_mask=change_result.change_mask
            )
            
            if not validation['is_valid']:
                logger.warning(f"变化注入验证失败: {validation}")
                
            llm_response = self.prompt_generator.generate_prompt(
                scenario=scenario,
                change_type=change_result.parameters.change_type,
                physical_desc=change_result.physical_description,
                total_len=len(base_ts),
                start_idx=change_result.parameters.start_idx,
                end_idx=change_result.parameters.end_idx
            )
            
            prompt_quality = evaluate_prompt_quality(
                prompt=llm_response.response,
                change_type=change_result.parameters.change_type
            )
            
            sample = TestSample(
                sample_id=sample_id,
                dataset_name=dataset_name,
                scenario=scenario.value,
                base_ts=change_result.base_ts.tolist(),
                target_ts=change_result.target_ts.tolist(),
                vague_prompt=llm_response.response,
                gt_change_type=change_result.parameters.change_type.value,
                gt_start_idx=change_result.parameters.start_idx,
                gt_end_idx=change_result.parameters.end_idx,
                gt_physical_desc=change_result.physical_description,
                gt_parameters={
                    'intensity': change_result.parameters.intensity,
                    'ratio': change_result.parameters.ratio,
                    'shift_value': change_result.parameters.shift_value,
                    'direction': change_result.parameters.direction
                },
                llm_response={
                    'success': llm_response.success,
                    'latency': llm_response.latency,
                    'error_message': llm_response.error_message
                },
                validation_result=validation,
                prompt_quality=prompt_quality,
                metadata=metadata
            )
            
            return sample
            
        except Exception as e:
            logger.error(f"处理样本 {sample_id} 时出错: {e}")
            return None
    
    def run_pilot_test(
        self,
        num_samples: int = 5,
        seq_len: int = 100
    ) -> TestResult:
        """
        运行小规模预测试
        
        Args:
            num_samples: 样本数量
            seq_len: 序列长度
            
        Returns:
            TestResult: 测试结果
        """
        logger.info("运行预测试 (Pilot Test)...")
        
        return self.run_full_test(
            dataset_names=["ETTh1"],
            num_samples_per_dataset=num_samples,
            seq_len=seq_len,
            save_results=True
        )
    
    def run_change_type_test(
        self,
        change_type: ChangeType,
        num_samples: int = 10,
        seq_len: int = 100
    ) -> TestResult:
        """
        测试特定变化类型
        
        Args:
            change_type: 变化类型
            num_samples: 样本数量
            seq_len: 序列长度
            
        Returns:
            TestResult: 测试结果
        """
        logger.info(f"测试变化类型: {change_type.value}")
        
        return self.run_full_test(
            dataset_names=["ETTh1"],
            num_samples_per_dataset=num_samples,
            seq_len=seq_len,
            change_types=[change_type],
            save_results=True
        )
    
    def run_scenario_test(
        self,
        scenario: ScenarioType,
        num_samples: int = 10,
        seq_len: int = 100
    ) -> TestResult:
        """
        测试特定场景
        
        Args:
            scenario: 场景类型
            num_samples: 样本数量
            seq_len: 序列长度
            
        Returns:
            TestResult: 测试结果
        """
        logger.info(f"测试场景: {scenario.value}")
        
        dataset_mapping = {
            ScenarioType.TRAFFIC: "Traffic",
            ScenarioType.ELECTRICITY: "ETTh1",
            ScenarioType.WEATHER: "Weather",
            ScenarioType.EXCHANGE: "Exchange",
            ScenarioType.STOCK: "Exchange"
        }
        
        dataset_name = dataset_mapping.get(scenario, "ETTh1")
        
        return self.run_full_test(
            dataset_names=[dataset_name],
            num_samples_per_dataset=num_samples,
            seq_len=seq_len,
            save_results=True
        )
    
    def _compile_results(self) -> TestResult:
        """编译测试结果"""
        total = len(self._test_samples)
        successful = sum(1 for s in self._test_samples if s.validation_result.get('is_valid', False))
        failed = total - successful
        
        quality_scores = [s.prompt_quality.get('total_score', 0) for s in self._test_samples]
        avg_quality = np.mean(quality_scores) if quality_scores else 0.0
        
        magnitudes = [
            np.mean(np.abs(np.array(s.base_ts) - np.array(s.target_ts)))
            for s in self._test_samples
        ]
        avg_magnitude = np.mean(magnitudes) if magnitudes else 0.0
        
        change_type_dist = {}
        for s in self._test_samples:
            ct = s.gt_change_type
            change_type_dist[ct] = change_type_dist.get(ct, 0) + 1
            
        scenario_dist = {}
        for s in self._test_samples:
            sc = s.scenario
            scenario_dist[sc] = scenario_dist.get(sc, 0) + 1
            
        return TestResult(
            total_samples=total,
            successful_samples=successful,
            failed_samples=failed,
            avg_prompt_quality=avg_quality,
            avg_change_magnitude=avg_magnitude,
            change_type_distribution=change_type_dist,
            scenario_distribution=scenario_dist,
            samples=self._test_samples
        )
    
    def _convert_to_serializable(self, obj):
        """将numpy类型转换为Python原生类型"""
        if isinstance(obj, dict):
            return {k: self._convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        else:
            return obj
    
    def _save_results(self, result: TestResult):
        """保存测试结果"""
        output_file = self.output_dir / "test_results.json"
        
        result_dict = {
            'summary': {
                'total_samples': result.total_samples,
                'successful_samples': result.successful_samples,
                'failed_samples': result.failed_samples,
                'avg_prompt_quality': result.avg_prompt_quality,
                'avg_change_magnitude': result.avg_change_magnitude,
                'change_type_distribution': result.change_type_distribution,
                'scenario_distribution': result.scenario_distribution
            },
            'samples': [asdict(s) for s in result.samples]
        }
        
        result_dict = self._convert_to_serializable(result_dict)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result_dict, f, ensure_ascii=False, indent=2)
            
        logger.info(f"结果已保存至: {output_file}")
        
        self._save_summary_report(result)
        
    def _save_summary_report(self, result: TestResult):
        """保存摘要报告"""
        report_file = self.output_dir / "summary_report.txt"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("BetterTSE 测试报告\n")
            f.write("=" * 60 + "\n\n")
            
            f.write("1. 测试概览\n")
            f.write("-" * 40 + "\n")
            f.write(f"总样本数: {result.total_samples}\n")
            f.write(f"成功样本: {result.successful_samples}\n")
            f.write(f"失败样本: {result.failed_samples}\n")
            f.write(f"成功率: {result.successful_samples/result.total_samples*100:.1f}%\n\n")
            
            f.write("2. 提示词质量评估\n")
            f.write("-" * 40 + "\n")
            f.write(f"平均质量分数: {result.avg_prompt_quality:.3f}\n\n")
            
            f.write("3. 变化类型分布\n")
            f.write("-" * 40 + "\n")
            for ct, count in result.change_type_distribution.items():
                f.write(f"  {ct}: {count}\n")
            f.write("\n")
            
            f.write("4. 场景分布\n")
            f.write("-" * 40 + "\n")
            for sc, count in result.scenario_distribution.items():
                f.write(f"  {sc}: {count}\n")
            f.write("\n")
            
            f.write("5. 样本示例\n")
            f.write("-" * 40 + "\n")
            for i, sample in enumerate(result.samples[:3]):
                f.write(f"\n样本 {i+1}: {sample.sample_id}\n")
                f.write(f"  变化类型: {sample.gt_change_type}\n")
                f.write(f"  变化区间: [{sample.gt_start_idx}, {sample.gt_end_idx}]\n")
                f.write(f"  物理描述: {sample.gt_physical_desc}\n")
                f.write(f"  生成的提示词: {sample.vague_prompt}\n")
                f.write(f"  提示词质量: {sample.prompt_quality.get('total_score', 0):.3f}\n")
                
        logger.info(f"摘要报告已保存至: {report_file}")


def run_quick_test():
    """运行快速测试"""
    pipeline = TestPipeline(
        use_mock_llm=True,
        random_seed=42
    )
    
    result = pipeline.run_pilot_test(num_samples=5)
    
    print("\n" + "=" * 60)
    print("快速测试完成!")
    print(f"生成样本数: {result.total_samples}")
    print(f"平均提示词质量: {result.avg_prompt_quality:.3f}")
    print("=" * 60)
    
    return result


if __name__ == "__main__":
    run_quick_test()
