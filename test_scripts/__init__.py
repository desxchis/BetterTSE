"""
BetterTSE 测试脚本包

该包提供完整的测试脚本系统，用于：
1. 加载真实公开的CSV格式时间序列数据集
2. 实现确定性物理变化注入机制
3. 集成LLM接口生成模糊宏观描述
4. 评估和可视化测试结果

主要模块（请直接按需从子模块导入，而非通过此包顶层）:
  from test_scripts.bettertse_cik_official import BetterTSETestPipelineCiKOfficial, TSEditEvaluator
  from test_scripts.build_mini_benchmark import ...
  from test_scripts.result_evaluator import ResultEvaluator
  ...

注意：test_scripts/config.py 与项目根目录 config.py 同名，
      请勿在顶层 __init__.py 中做通配导入，以避免 sys.modules 冲突。
"""

__version__ = "1.0.0"
__author__ = "BetterTSE Team"
