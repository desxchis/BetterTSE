import os
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from enum import Enum


class ChangeType(Enum):
    EVENT_DROP = "event_drop"
    BASELINE_SHIFT = "baseline_shift"
    ANOMALY_SPIKE = "anomaly_spike"
    TREND_SMOOTHING = "trend_smoothing"
    AMPLIFICATION = "amplification"
    ATTENUATION = "attenuation"
    TREND_INJECTION = "trend_injection"


class ScenarioType(Enum):
    TRAFFIC = "交通流量 (Traffic)"
    ELECTRICITY = "电力负荷 (Electricity)"
    WEATHER = "气温监控 (Weather)"
    EXCHANGE = "汇率变化 (Exchange)"
    STOCK = "股票走势 (Stock)"


@dataclass
class DatasetConfig:
    name: str
    url: str
    file_pattern: str
    column_name: str
    description: str
    seq_len: int = 100
    train_ratio: float = 0.7
    val_ratio: float = 0.15


@dataclass
class ChangeInjectionConfig:
    change_type: ChangeType
    window_ratio_min: float = 0.2
    window_ratio_max: float = 0.4
    position_min: float = 0.1
    position_max: float = 0.5
    intensity_range: tuple = (0.2, 0.8)
    noise_level: float = 0.1


@dataclass
class LLMConfig:
    api_key: str = ""
    base_url: str = "https://api.deepseek.com"
    model_name: str = "deepseek-chat"
    temperature: float = 0.7
    max_tokens: int = 150
    retry_times: int = 3
    retry_delay: float = 1.0


@dataclass
class TestConfig:
    num_samples: int = 50
    seq_len: int = 100
    output_dir: str = "test_results"
    log_dir: str = "logs"
    random_seed: int = 42
    save_visualization: bool = True
    verbose: bool = True


DATASET_CONFIGS: Dict[str, DatasetConfig] = {
    "ETTh1": DatasetConfig(
        name="ETTh1",
        url="https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv",
        file_pattern="ETTh1.csv",
        column_name="OT",
        description="电力变压器温度数据(小时级)"
    ),
    "ETTm1": DatasetConfig(
        name="ETTm1",
        url="https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTm1.csv",
        file_pattern="ETTm1.csv",
        column_name="OT",
        description="电力变压器温度数据(分钟级)"
    ),
    "Traffic": DatasetConfig(
        name="Traffic",
        url="https://raw.githubusercontent.com/laiguokun/multivariate-time-series-data/master/traffic/traffic.txt",
        file_pattern="traffic.csv",
        column_name="0",
        description="加州交通流量数据"
    ),
    "Weather": DatasetConfig(
        name="Weather",
        url="https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/Weather.csv",
        file_pattern="Weather.csv",
        column_name="OT",
        description="天气预报数据"
    ),
    "Exchange": DatasetConfig(
        name="Exchange",
        url="https://raw.githubusercontent.com/laiguokun/multivariate-time-series-data/master/exchange_rate/exchange_rate.txt",
        file_pattern="exchange.csv",
        column_name="0",
        description="汇率变化数据"
    ),
}

CHANGE_DESCRIPTIONS: Dict[ChangeType, Dict[str, str]] = {
    ChangeType.EVENT_DROP: {
        "zh": "在指定区间，数值急剧萎缩至原来的 {ratio:.0%}",
        "en": "Values dropped sharply to {ratio:.0%} of original in the specified range"
    },
    ChangeType.BASELINE_SHIFT: {
        "zh": "在指定区间，整体数值基线被大幅抬高了 {shift:.2f} 个单位",
        "en": "Baseline shifted up by {shift:.2f} units in the specified range"
    },
    ChangeType.ANOMALY_SPIKE: {
        "zh": "在指定区间，出现了剧烈且无规律的异常波动",
        "en": "Severe irregular fluctuations appeared in the specified range"
    },
    ChangeType.TREND_SMOOTHING: {
        "zh": "在指定区间，原本的尖锐波动被抹平，变得非常平缓",
        "en": "Sharp fluctuations smoothed out in the specified range"
    },
    ChangeType.AMPLIFICATION: {
        "zh": "在指定区间，数值被放大了 {ratio:.1f} 倍",
        "en": "Values amplified by {ratio:.1f}x in the specified range"
    },
    ChangeType.ATTENUATION: {
        "zh": "在指定区间，数值被压缩至原来的 {ratio:.0%}",
        "en": "Values attenuated to {ratio:.0%} of original in the specified range"
    },
    ChangeType.TREND_INJECTION: {
        "zh": "在指定区间，注入了{direction}的线性趋势",
        "en": "A {direction} linear trend was injected in the specified range"
    },
}

SCENARIO_CONTEXTS: Dict[ScenarioType, Dict[str, List[str]]] = {
    ScenarioType.TRAFFIC: {
        "time_periods": ["早高峰", "晚高峰", "深夜", "下午", "凌晨", "中午"],
        "events": ["交通事故", "道路施工", "节假日", "恶劣天气", "大型活动"],
        "descriptions": ["交通流量", "车流密度", "道路拥堵程度"]
    },
    ScenarioType.ELECTRICITY: {
        "time_periods": ["用电高峰期", "深夜低谷", "工作日白天", "周末", "节假日"],
        "events": ["极端天气", "设备故障", "大型活动", "工业生产调整"],
        "descriptions": ["用电负荷", "电力消耗", "电网负载"]
    },
    ScenarioType.WEATHER: {
        "time_periods": ["清晨", "午后", "傍晚", "夜间", "凌晨"],
        "events": ["冷空气来袭", "暖流经过", "暴风雨", "晴朗天气"],
        "descriptions": ["温度变化", "气温波动", "温度趋势"]
    },
    ScenarioType.EXCHANGE: {
        "time_periods": ["开盘时段", "午间", "收盘时段", "夜间交易"],
        "events": ["政策发布", "经济数据公布", "国际事件", "市场波动"],
        "descriptions": ["汇率变化", "货币走势", "交易波动"]
    },
    ScenarioType.STOCK: {
        "time_periods": ["开盘", "午盘", "尾盘", "盘后"],
        "events": ["财报发布", "重大公告", "市场情绪变化", "行业动态"],
        "descriptions": ["股价走势", "交易量", "价格波动"]
    },
}

LLM_SYSTEM_PROMPT = """你是一个缺乏技术背景，但对业务场景非常熟悉的业务员。
我现在会提供给你一条时间序列的【底层物理变化】，你需要根据指定的【业务场景】，将其包装成一句口语化、模糊、宏观的自然语言指令。

严格要求：
1. 绝对不能出现具体的索引值（如 index 40 到 60）、具体的修改数值或比例。
2. 只能用时间段（如"下午"、"半夜"、"中段"）和定性描述（如"骤降"、"稍微平缓一点"、"拔高一个台阶"）来表达。
3. 语言风格要自然，像是给 AI 助手下达任务。
4. 只返回一句指令，不要任何多余的解释或前缀。
5. 描述要具有业务场景的真实感，让人感觉这是实际工作中会遇到的情况。"""

LLM_USER_PROMPT_TEMPLATE = """
【业务场景】：{scenario} (总长度 {total_len} 步的时间序列)
【意图类别】：{change_type}
【底层物理变化】：{physical_desc}
【时间参考】：变化发生在序列的{position_desc}部分

请生成 1 句符合要求的模糊指令："""


def get_config():
    config = TestConfig()
    config.output_dir = os.environ.get("TSE_OUTPUT_DIR", config.output_dir)
    config.log_dir = os.environ.get("TSE_LOG_DIR", config.log_dir)
    return config


def get_llm_config():
    config = LLMConfig()
    config.api_key = os.environ.get("DEEPSEEK_API_KEY", os.environ.get("OPENAI_API_KEY", ""))
    config.base_url = os.environ.get("LLM_BASE_URL", config.base_url)
    config.model_name = os.environ.get("LLM_MODEL_NAME", config.model_name)
    return config
