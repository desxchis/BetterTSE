import os
import sys
import json
import logging
import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional, List

from config import ChangeType, ScenarioType, TestConfig, get_config
from data_loader import DatasetLoader, create_synthetic_sequence
from change_injector import PhysicalChangeInjector, validate_change_injection
from llm_interface import VaguePromptGenerator, evaluate_prompt_quality
from test_pipeline import TestPipeline, run_quick_test
from result_evaluator import ResultEvaluator, ResultVisualizer, generate_dataset_for_bettertse


def setup_logging(log_dir: str, verbose: bool = True):
    """配置日志系统"""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"test_{timestamp}.log"
    
    level = logging.DEBUG if verbose else logging.INFO
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    return str(log_file)


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='BetterTSE 测试脚本 - 时间序列编辑数据生成与评估',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 运行快速预测试
  python main.py --mode pilot --samples 5
  
  # 运行完整测试
  python main.py --mode full --samples 50 --datasets ETTh1 ETTm1 Traffic
  
  # 生成BetterTSE格式数据集
  python main.py --mode generate --output tse_dataset.json --samples 100
  
  # 评估已有结果
  python main.py --mode evaluate --input test_results/test_results.json
        """
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['pilot', 'full', 'generate', 'evaluate', 'visualize'],
        default='pilot',
        help='运行模式: pilot(预测试), full(完整测试), generate(生成数据集), evaluate(评估), visualize(可视化)'
    )
    
    parser.add_argument(
        '--samples',
        type=int,
        default=10,
        help='样本数量 (默认: 10)'
    )
    
    parser.add_argument(
        '--seq-len',
        type=int,
        default=100,
        help='序列长度 (默认: 100)'
    )
    
    parser.add_argument(
        '--datasets',
        nargs='+',
        default=['ETTh1'],
        help='数据集列表 (默认: ETTh1)'
    )
    
    parser.add_argument(
        '--change-types',
        nargs='+',
        default=None,
        choices=[ct.value for ct in ChangeType],
        help='变化类型列表'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='test_results',
        help='输出目录或文件路径 (默认: test_results)'
    )
    
    parser.add_argument(
        '--input',
        type=str,
        default=None,
        help='输入文件路径 (用于evaluate模式)'
    )
    
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data',
        help='数据目录 (默认: data)'
    )
    
    parser.add_argument(
        '--log-dir',
        type=str,
        default='logs',
        help='日志目录 (默认: logs)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='随机种子 (默认: 42)'
    )
    
    parser.add_argument(
        '--use-real-data',
        action='store_true',
        help='使用真实数据集'
    )
    
    parser.add_argument(
        '--use-mock-llm',
        action='store_true',
        default=True,
        help='使用模拟LLM (默认: True)'
    )
    
    parser.add_argument(
        '--api-key',
        type=str,
        default=None,
        help='LLM API密钥 (也可通过环境变量DEEPSEEK_API_KEY设置)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        default=True,
        help='详细输出 (默认: True)'
    )
    
    return parser.parse_args()


def run_pilot_test(args):
    """运行预测试"""
    logging.info("=" * 60)
    logging.info("运行预测试模式 (Pilot Test)")
    logging.info("=" * 60)
    
    pipeline = TestPipeline(
        data_dir=args.data_dir,
        output_dir=args.output,
        use_mock_llm=args.use_mock_llm,
        random_seed=args.seed
    )
    
    result = pipeline.run_pilot_test(
        num_samples=args.samples,
        seq_len=args.seq_len
    )
    
    return result


def run_full_test(args):
    """运行完整测试"""
    logging.info("=" * 60)
    logging.info("运行完整测试模式 (Full Test)")
    logging.info("=" * 60)
    
    change_types = None
    if args.change_types:
        change_types = [ChangeType(ct) for ct in args.change_types]
    
    pipeline = TestPipeline(
        data_dir=args.data_dir,
        output_dir=args.output,
        use_mock_llm=args.use_mock_llm,
        random_seed=args.seed
    )
    
    result = pipeline.run_full_test(
        dataset_names=args.datasets,
        num_samples_per_dataset=max(1, args.samples // len(args.datasets)),
        seq_len=args.seq_len,
        change_types=change_types,
        save_results=True
    )
    
    return result


def run_generate_dataset(args):
    """生成数据集"""
    logging.info("=" * 60)
    logging.info("生成BetterTSE格式数据集")
    logging.info("=" * 60)
    
    output_file = args.output
    if not output_file.endswith('.json'):
        output_file = os.path.join(args.output, 'tse_dataset.json')
    
    output_path = generate_dataset_for_bettertse(
        output_file=output_file,
        num_samples=args.samples,
        seq_len=args.seq_len,
        use_real_data=args.use_real_data,
        use_mock_llm=args.use_mock_llm
    )
    
    logging.info(f"数据集已生成: {output_path}")
    
    return output_path


def run_evaluate(args):
    """评估结果"""
    logging.info("=" * 60)
    logging.info("评估测试结果")
    logging.info("=" * 60)
    
    if args.input is None:
        args.input = os.path.join(args.output, 'test_results.json')
    
    if not os.path.exists(args.input):
        logging.error(f"输入文件不存在: {args.input}")
        return None
    
    evaluator = ResultEvaluator(output_dir=args.output)
    report = evaluator.evaluate_test_results(args.input)
    
    print("\n" + "=" * 60)
    print("评估报告摘要")
    print("=" * 60)
    print(f"总样本数: {report['total_samples']}")
    print(f"提示词质量: {report['dimensions']['prompt_quality']['mean_score']:.3f}")
    print(f"验证成功率: {report['dimensions']['change_injection']['validation_success_rate']:.1%}")
    print(f"LLM成功率: {report['dimensions']['llm_performance']['success_rate']:.1%}")
    print("\n建议:")
    for rec in report['recommendations']:
        print(f"  - {rec}")
    
    return report


def run_visualize(args):
    """可视化结果"""
    logging.info("=" * 60)
    logging.info("生成可视化")
    logging.info("=" * 60)
    
    if args.input is None:
        args.input = os.path.join(args.output, 'test_results.json')
    
    if not os.path.exists(args.input):
        logging.error(f"输入文件不存在: {args.input}")
        return None
    
    visualizer = ResultVisualizer(output_dir=args.output)
    saved_files = visualizer.generate_all_visualizations(args.input)
    
    logging.info(f"生成了 {len(saved_files)} 个可视化文件")
    
    return saved_files


def main():
    """主函数"""
    args = parse_arguments()
    
    log_file = setup_logging(args.log_dir, args.verbose)
    logging.info(f"日志文件: {log_file}")
    
    if args.api_key:
        os.environ['DEEPSEEK_API_KEY'] = args.api_key
    
    try:
        if args.mode == 'pilot':
            result = run_pilot_test(args)
        elif args.mode == 'full':
            result = run_full_test(args)
        elif args.mode == 'generate':
            result = run_generate_dataset(args)
        elif args.mode == 'evaluate':
            result = run_evaluate(args)
        elif args.mode == 'visualize':
            result = run_visualize(args)
        else:
            logging.error(f"未知模式: {args.mode}")
            sys.exit(1)
            
        logging.info("\n" + "=" * 60)
        logging.info("执行完成!")
        logging.info("=" * 60)
        
    except KeyboardInterrupt:
        logging.warning("\n用户中断执行")
        sys.exit(0)
    except Exception as e:
        logging.error(f"执行出错: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
