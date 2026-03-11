import os
import json
import logging
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from datetime import datetime
import numpy as np

from config import ChangeType, ScenarioType


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ResultEvaluator:
    """
    结果评估器
    提供多维度的评估和分析功能
    """
    
    def __init__(self, output_dir: str = "test_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def evaluate_test_results(self, results_file: str) -> Dict[str, Any]:
        """
        评估测试结果文件
        
        Args:
            results_file: 结果文件路径
            
        Returns:
            Dict: 评估报告
        """
        with open(results_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        samples = data.get('samples', [])
        
        report = {
            'evaluation_time': datetime.now().isoformat(),
            'total_samples': len(samples),
            'dimensions': self._evaluate_all_dimensions(samples),
            'statistics': self._compute_statistics(samples),
            'recommendations': self._generate_recommendations(samples)
        }
        
        report_file = self.output_dir / "evaluation_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
            
        logger.info(f"评估报告已保存至: {report_file}")
        
        return report
    
    def _evaluate_all_dimensions(self, samples: List[Dict]) -> Dict[str, Any]:
        """评估所有维度"""
        dimensions = {}
        
        dimensions['prompt_quality'] = self._evaluate_prompt_quality(samples)
        dimensions['change_injection'] = self._evaluate_change_injection(samples)
        dimensions['llm_performance'] = self._evaluate_llm_performance(samples)
        dimensions['data_distribution'] = self._evaluate_distribution(samples)
        
        return dimensions
    
    def _evaluate_prompt_quality(self, samples: List[Dict]) -> Dict[str, Any]:
        """评估提示词质量"""
        quality_scores = []
        dimension_scores = {
            'no_specific_numbers': [],
            'vague_language': [],
            'time_reference': [],
            'natural_language': [],
            'semantic_match': []
        }
        
        for sample in samples:
            pq = sample.get('prompt_quality', {})
            quality_scores.append(pq.get('total_score', 0))
            
            ds = pq.get('dimension_scores', {})
            for key in dimension_scores:
                dimension_scores[key].append(ds.get(key, 0))
                
        return {
            'mean_score': float(np.mean(quality_scores)) if quality_scores else 0,
            'std_score': float(np.std(quality_scores)) if quality_scores else 0,
            'min_score': float(np.min(quality_scores)) if quality_scores else 0,
            'max_score': float(np.max(quality_scores)) if quality_scores else 0,
            'acceptable_ratio': sum(1 for s in quality_scores if s >= 0.6) / len(quality_scores) if quality_scores else 0,
            'dimension_analysis': {
                k: {'mean': float(np.mean(v)), 'std': float(np.std(v))}
                for k, v in dimension_scores.items()
            }
        }
    
    def _evaluate_change_injection(self, samples: List[Dict]) -> Dict[str, Any]:
        """评估变化注入"""
        validation_results = []
        change_magnitudes = []
        relative_changes = []
        
        for sample in samples:
            vr = sample.get('validation_result', {})
            validation_results.append(vr.get('is_valid', False))
            
            base_ts = np.array(sample.get('base_ts', []))
            target_ts = np.array(sample.get('target_ts', []))
            
            if len(base_ts) > 0 and len(target_ts) > 0:
                magnitude = float(np.mean(np.abs(base_ts - target_ts)))
                change_magnitudes.append(magnitude)
                
                mean_base = np.mean(base_ts)
                if mean_base != 0:
                    relative_changes.append(magnitude / abs(mean_base))
                    
        return {
            'validation_success_rate': sum(validation_results) / len(validation_results) if validation_results else 0,
            'mean_change_magnitude': float(np.mean(change_magnitudes)) if change_magnitudes else 0,
            'std_change_magnitude': float(np.std(change_magnitudes)) if change_magnitudes else 0,
            'mean_relative_change': float(np.mean(relative_changes)) if relative_changes else 0
        }
    
    def _evaluate_llm_performance(self, samples: List[Dict]) -> Dict[str, Any]:
        """评估LLM性能"""
        success_count = 0
        latencies = []
        error_messages = []
        
        for sample in samples:
            llm_resp = sample.get('llm_response', {})
            if llm_resp.get('success', False):
                success_count += 1
            else:
                error_msg = llm_resp.get('error_message', '')
                if error_msg:
                    error_messages.append(error_msg)
                    
            latencies.append(llm_resp.get('latency', 0))
            
        return {
            'success_rate': success_count / len(samples) if samples else 0,
            'mean_latency': float(np.mean(latencies)) if latencies else 0,
            'std_latency': float(np.std(latencies)) if latencies else 0,
            'error_types': self._categorize_errors(error_messages)
        }
    
    def _categorize_errors(self, error_messages: List[str]) -> Dict[str, int]:
        """分类错误类型"""
        categories = {
            'api_key': 0,
            'rate_limit': 0,
            'timeout': 0,
            'network': 0,
            'other': 0
        }
        
        for msg in error_messages:
            msg_lower = msg.lower()
            if 'api' in msg_lower and 'key' in msg_lower:
                categories['api_key'] += 1
            elif 'rate' in msg_lower or 'limit' in msg_lower:
                categories['rate_limit'] += 1
            elif 'timeout' in msg_lower:
                categories['timeout'] += 1
            elif 'network' in msg_lower or 'connection' in msg_lower:
                categories['network'] += 1
            else:
                categories['other'] += 1
                
        return categories
    
    def _evaluate_distribution(self, samples: List[Dict]) -> Dict[str, Any]:
        """评估数据分布"""
        change_types = {}
        scenarios = {}
        datasets = {}
        
        for sample in samples:
            ct = sample.get('gt_change_type', 'unknown')
            change_types[ct] = change_types.get(ct, 0) + 1
            
            sc = sample.get('scenario', 'unknown')
            scenarios[sc] = scenarios.get(sc, 0) + 1
            
            ds = sample.get('dataset_name', 'unknown')
            datasets[ds] = datasets.get(ds, 0) + 1
            
        return {
            'change_types': change_types,
            'scenarios': scenarios,
            'datasets': datasets,
            'balance_score': self._compute_balance_score(change_types)
        }
    
    def _compute_balance_score(self, distribution: Dict[str, int]) -> float:
        """计算分布平衡分数"""
        if not distribution:
            return 0.0
            
        values = list(distribution.values())
        total = sum(values)
        
        if total == 0:
            return 0.0
            
        expected_ratio = 1.0 / len(values)
        actual_ratios = [v / total for v in values]
        
        deviations = [abs(r - expected_ratio) for r in actual_ratios]
        mean_deviation = np.mean(deviations)
        
        return float(1.0 - mean_deviation * 2)
    
    def _compute_statistics(self, samples: List[Dict]) -> Dict[str, Any]:
        """计算统计信息"""
        if not samples:
            return {}
            
        base_lengths = [len(s.get('base_ts', [])) for s in samples]
        window_sizes = [
            s.get('gt_end_idx', 0) - s.get('gt_start_idx', 0)
            for s in samples
        ]
        position_ratios = [
            s.get('gt_start_idx', 0) / max(s.get('base_ts', []).__len__(), 1)
            for s in samples
        ]
        
        return {
            'sequence_length': {
                'mean': float(np.mean(base_lengths)),
                'std': float(np.std(base_lengths)),
                'min': int(np.min(base_lengths)),
                'max': int(np.max(base_lengths))
            },
            'window_size': {
                'mean': float(np.mean(window_sizes)),
                'std': float(np.std(window_sizes)),
                'min': int(np.min(window_sizes)),
                'max': int(np.max(window_sizes))
            },
            'position_distribution': {
                'mean_start_ratio': float(np.mean(position_ratios)),
                'std_start_ratio': float(np.std(position_ratios))
            }
        }
    
    def _generate_recommendations(self, samples: List[Dict]) -> List[str]:
        """生成改进建议"""
        recommendations = []
        
        quality_scores = [
            s.get('prompt_quality', {}).get('total_score', 0)
            for s in samples
        ]
        avg_quality = np.mean(quality_scores) if quality_scores else 0
        
        if avg_quality < 0.5:
            recommendations.append(
                "提示词质量较低，建议优化LLM的system prompt，"
                "增加更多关于模糊描述的示例。"
            )
        elif avg_quality < 0.7:
            recommendations.append(
                "提示词质量中等，可以尝试调整temperature参数，"
                "或增加更多业务场景的上下文信息。"
            )
            
        no_number_scores = [
            s.get('prompt_quality', {}).get('dimension_scores', {}).get('no_specific_numbers', 0)
            for s in samples
        ]
        if np.mean(no_number_scores) < 0.8:
            recommendations.append(
                "部分提示词包含具体数值，建议在prompt中更明确地禁止使用数字。"
            )
            
        validation_rate = sum(
            1 for s in samples 
            if s.get('validation_result', {}).get('is_valid', False)
        ) / len(samples) if samples else 0
        
        if validation_rate < 0.95:
            recommendations.append(
                "变化注入验证存在失败案例，建议检查变化注入逻辑的边界条件处理。"
            )
            
        llm_success_rate = sum(
            1 for s in samples
            if s.get('llm_response', {}).get('success', False)
        ) / len(samples) if samples else 0
        
        if llm_success_rate < 0.9:
            recommendations.append(
                "LLM调用存在失败情况，建议检查API配置或增加重试机制。"
            )
            
        if not recommendations:
            recommendations.append("测试结果良好，建议扩大测试规模进行更全面的评估。")
            
        return recommendations


class ResultVisualizer:
    """
    结果可视化器
    """
    
    def __init__(self, output_dir: str = "test_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def visualize_sample(
        self,
        sample: Dict,
        save_path: Optional[str] = None
    ) -> Optional[str]:
        """
        可视化单个样本
        
        Args:
            sample: 样本数据
            save_path: 保存路径
            
        Returns:
            str: 保存的文件路径
        """
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib未安装，跳过可视化")
            return None
            
        base_ts = np.array(sample.get('base_ts', []))
        target_ts = np.array(sample.get('target_ts', []))
        
        if len(base_ts) == 0:
            return None
            
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        ax1 = axes[0, 0]
        x = np.arange(len(base_ts))
        ax1.plot(x, base_ts, 'b-', label='Base TS', alpha=0.7)
        ax1.plot(x, target_ts, 'r-', label='Target TS', alpha=0.7)
        
        start_idx = sample.get('gt_start_idx', 0)
        end_idx = sample.get('gt_end_idx', len(base_ts))
        ax1.axvspan(start_idx, end_idx, alpha=0.2, color='yellow', label='Change Region')
        ax1.set_xlabel('Time Step')
        ax1.set_ylabel('Value')
        ax1.set_title('Time Series Comparison')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2 = axes[0, 1]
        diff = target_ts - base_ts
        ax2.bar(x, diff, alpha=0.7, color='green')
        ax2.axvspan(start_idx, end_idx, alpha=0.2, color='yellow')
        ax2.set_xlabel('Time Step')
        ax2.set_ylabel('Difference')
        ax2.set_title('Change Magnitude')
        ax2.grid(True, alpha=0.3)
        
        ax3 = axes[1, 0]
        pq = sample.get('prompt_quality', {})
        scores = pq.get('dimension_scores', {})
        if scores:
            labels = list(scores.keys())
            values = list(scores.values())
            bars = ax3.barh(labels, values, color='steelblue')
            ax3.set_xlim(0, 1)
            ax3.set_xlabel('Score')
            ax3.set_title('Prompt Quality Dimensions')
            for bar, val in zip(bars, values):
                ax3.text(val + 0.02, bar.get_y() + bar.get_height()/2, 
                        f'{val:.2f}', va='center')
        
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        info_text = f"""
Sample ID: {sample.get('sample_id', 'N/A')}
Dataset: {sample.get('dataset_name', 'N/A')}
Scenario: {sample.get('scenario', 'N/A')}

Change Type: {sample.get('gt_change_type', 'N/A')}
Change Region: [{start_idx}, {end_idx}]
Physical Description: {sample.get('gt_physical_desc', 'N/A')}

Generated Prompt:
"{sample.get('vague_prompt', 'N/A')}"

Prompt Quality Score: {pq.get('total_score', 0):.3f}
Validation: {'✓ Passed' if sample.get('validation_result', {}).get('is_valid', False) else '✗ Failed'}
"""
        ax4.text(0.1, 0.9, info_text, transform=ax4.transAxes, 
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.output_dir / f"sample_{sample.get('sample_id', 'unknown')}.png"
        else:
            save_path = Path(save_path)
            
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return str(save_path)
    
    def visualize_summary(
        self,
        results_file: str,
        save_path: Optional[str] = None
    ) -> Optional[str]:
        """
        可视化测试结果摘要
        
        Args:
            results_file: 结果文件路径
            save_path: 保存路径
            
        Returns:
            str: 保存的文件路径
        """
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib未安装，跳过可视化")
            return None
            
        with open(results_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        samples = data.get('samples', [])
        summary = data.get('summary', {})
        
        if not samples:
            return None
            
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        
        ax1 = axes[0, 0]
        quality_scores = [
            s.get('prompt_quality', {}).get('total_score', 0)
            for s in samples
        ]
        ax1.hist(quality_scores, bins=20, color='steelblue', edgecolor='white', alpha=0.7)
        ax1.axvline(np.mean(quality_scores), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(quality_scores):.3f}')
        ax1.set_xlabel('Quality Score')
        ax1.set_ylabel('Count')
        ax1.set_title('Prompt Quality Distribution')
        ax1.legend()
        
        ax2 = axes[0, 1]
        change_types = summary.get('change_type_distribution', {})
        if change_types:
            ax2.pie(change_types.values(), labels=change_types.keys(), autopct='%1.1f%%',
                   colors=plt.cm.Set3.colors[:len(change_types)])
            ax2.set_title('Change Type Distribution')
            
        ax3 = axes[0, 2]
        scenarios = summary.get('scenario_distribution', {})
        if scenarios:
            ax3.pie(scenarios.values(), labels=scenarios.keys(), autopct='%1.1f%%',
                   colors=plt.cm.Pastel1.colors[:len(scenarios)])
            ax3.set_title('Scenario Distribution')
            
        ax4 = axes[1, 0]
        latencies = [
            s.get('llm_response', {}).get('latency', 0)
            for s in samples
        ]
        ax4.hist(latencies, bins=20, color='coral', edgecolor='white', alpha=0.7)
        ax4.set_xlabel('Latency (s)')
        ax4.set_ylabel('Count')
        ax4.set_title('LLM Response Latency')
        
        ax5 = axes[1, 1]
        change_magnitudes = [
            np.mean(np.abs(np.array(s.get('base_ts', [])) - np.array(s.get('target_ts', []))))
            for s in samples
        ]
        ax5.hist(change_magnitudes, bins=20, color='seagreen', edgecolor='white', alpha=0.7)
        ax5.set_xlabel('Change Magnitude')
        ax5.set_ylabel('Count')
        ax5.set_title('Change Magnitude Distribution')
        
        ax6 = axes[1, 2]
        ax6.axis('off')
        
        summary_text = f"""
Test Summary
{'='*30}

Total Samples: {summary.get('total_samples', 0)}
Successful: {summary.get('successful_samples', 0)}
Failed: {summary.get('failed_samples', 0)}

Average Prompt Quality: {summary.get('avg_prompt_quality', 0):.3f}
Average Change Magnitude: {summary.get('avg_change_magnitude', 0):.3f}

Change Types: {len(change_types)}
Scenarios: {len(scenarios)}
"""
        ax6.text(0.1, 0.9, summary_text, transform=ax6.transAxes,
                fontsize=11, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.output_dir / "summary_visualization.png"
        else:
            save_path = Path(save_path)
            
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return str(save_path)
    
    def generate_all_visualizations(self, results_file: str) -> List[str]:
        """
        生成所有可视化
        
        Args:
            results_file: 结果文件路径
            
        Returns:
            List[str]: 生成的文件路径列表
        """
        with open(results_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        samples = data.get('samples', [])
        saved_files = []
        
        for sample in samples[:10]:
            path = self.visualize_sample(sample)
            if path:
                saved_files.append(path)
                
        summary_path = self.visualize_summary(results_file)
        if summary_path:
            saved_files.append(summary_path)
            
        logger.info(f"生成了 {len(saved_files)} 个可视化文件")
        
        return saved_files


def generate_dataset_for_bettertse(
    output_file: str = "tse_pilot_dataset.json",
    num_samples: int = 50,
    seq_len: int = 100,
    use_real_data: bool = True,
    use_mock_llm: bool = True
) -> str:
    """
    生成BetterTSE格式的数据集
    
    Args:
        output_file: 输出文件路径
        num_samples: 样本数量
        seq_len: 序列长度
        use_real_data: 是否使用真实数据
        use_mock_llm: 是否使用模拟LLM
        
    Returns:
        str: 输出文件路径
    """
    from test_pipeline import TestPipeline
    
    pipeline = TestPipeline(
        use_mock_llm=use_mock_llm,
        random_seed=42
    )
    
    result = pipeline.run_full_test(
        dataset_names=["ETTh1", "ETTm1", "Traffic"] if use_real_data else None,
        num_samples_per_dataset=num_samples // 3,
        seq_len=seq_len,
        save_results=False
    )
    
    dataset = []
    for sample in result.samples:
        dataset.append({
            "sample_id": sample.sample_id,
            "scenario": sample.scenario,
            "vague_prompt": sample.vague_prompt,
            "gt_action_type": sample.gt_change_type,
            "gt_start_idx": sample.gt_start_idx,
            "gt_end_idx": sample.gt_end_idx,
            "base_ts": sample.base_ts,
            "target_ts": sample.target_ts,
            "gt_physical_desc": sample.gt_physical_desc,
            "gt_parameters": sample.gt_parameters
        })
        
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
        
    logger.info(f"数据集已生成: {output_file}")
    logger.info(f"总样本数: {len(dataset)}")
    
    return output_file


if __name__ == "__main__":
    evaluator = ResultEvaluator()
    
    results_file = "test_results/test_results.json"
    if os.path.exists(results_file):
        report = evaluator.evaluate_test_results(results_file)
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print("请先运行测试生成结果文件")
