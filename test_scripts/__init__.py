"""
BetterTSE 测试脚本包

该包提供完整的测试脚本系统，用于：
1. 加载真实公开的CSV格式时间序列数据集
2. 实现确定性物理变化注入机制
3. 集成LLM接口生成模糊宏观描述
4. 评估和可视化测试结果

主要模块:
- config: 配置管理
- data_loader: 数据集加载器
- change_injector: 物理变化注入器
- llm_interface: LLM接口
- test_pipeline: 测试流程管道
- result_evaluator: 结果评估器
"""

from config import (
    ChangeType,
    ScenarioType,
    DatasetConfig,
    ChangeInjectionConfig,
    LLMConfig,
    TestConfig,
    DATASET_CONFIGS,
    get_config,
    get_llm_config
)

from data_loader import (
    DatasetLoader,
    create_synthetic_sequence
)

from change_injector import (
    PhysicalChangeInjector,
    ChangeParameters,
    ChangeResult,
    validate_change_injection
)

from llm_interface import (
    BaseLLMClient,
    OpenAICompatibleClient,
    MockLLMClient,
    VaguePromptGenerator,
    LLMResponse,
    evaluate_prompt_quality
)

from test_pipeline import (
    TestPipeline,
    TestSample,
    TestResult,
    run_quick_test
)

from result_evaluator import (
    ResultEvaluator,
    ResultVisualizer,
    generate_dataset_for_bettertse
)

__version__ = "1.0.0"
__author__ = "BetterTSE Team"

__all__ = [
    # Config
    'ChangeType',
    'ScenarioType',
    'DatasetConfig',
    'ChangeInjectionConfig',
    'LLMConfig',
    'TestConfig',
    'DATASET_CONFIGS',
    'get_config',
    'get_llm_config',
    
    # Data Loader
    'DatasetLoader',
    'create_synthetic_sequence',
    
    # Change Injector
    'PhysicalChangeInjector',
    'ChangeParameters',
    'ChangeResult',
    'validate_change_injection',
    
    # LLM Interface
    'BaseLLMClient',
    'OpenAICompatibleClient',
    'MockLLMClient',
    'VaguePromptGenerator',
    'LLMResponse',
    'evaluate_prompt_quality',
    
    # Test Pipeline
    'TestPipeline',
    'TestSample',
    'TestResult',
    'run_quick_test',
    
    # Result Evaluator
    'ResultEvaluator',
    'ResultVisualizer',
    'generate_dataset_for_bettertse'
]
