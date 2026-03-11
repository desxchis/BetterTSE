import os
import logging
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod

from config import (
    LLMConfig, get_llm_config, ChangeType, ScenarioType,
    LLM_SYSTEM_PROMPT, LLM_USER_PROMPT_TEMPLATE, SCENARIO_CONTEXTS
)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """LLM响应结果"""
    prompt: str
    response: str
    success: bool
    error_message: str = ""
    latency: float = 0.0
    tokens_used: int = 0


class BaseLLMClient(ABC):
    """LLM客户端基类"""
    
    @abstractmethod
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> LLMResponse:
        """生成响应"""
        pass
    
    @abstractmethod
    def batch_generate(
        self, 
        prompts: List[str], 
        system_prompt: Optional[str] = None
    ) -> List[LLMResponse]:
        """批量生成响应"""
        pass


class OpenAICompatibleClient(BaseLLMClient):
    """OpenAI兼容API客户端"""
    
    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or get_llm_config()
        self._client = None
        
    def _get_client(self):
        """延迟初始化客户端"""
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(
                    api_key=self.config.api_key,
                    base_url=self.config.base_url
                )
            except ImportError:
                raise ImportError(
                    "请安装openai库: pip install openai"
                )
        return self._client
    
    def generate(
        self, 
        prompt: str, 
        system_prompt: Optional[str] = None
    ) -> LLMResponse:
        """生成单个响应"""
        if not self.config.api_key:
            return self._fallback_response(prompt, "API Key未配置")
        
        client = self._get_client()
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        for attempt in range(self.config.retry_times):
            try:
                start_time = time.time()
                
                response = client.chat.completions.create(
                    model=self.config.model_name,
                    messages=messages,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens
                )
                
                latency = time.time() - start_time
                
                return LLMResponse(
                    prompt=prompt,
                    response=response.choices[0].message.content.strip().strip('"\''),
                    success=True,
                    latency=latency,
                    tokens_used=response.usage.total_tokens if hasattr(response, 'usage') else 0
                )
                
            except Exception as e:
                error_msg = str(e)
                logger.warning(f"API调用失败 (尝试 {attempt + 1}/{self.config.retry_times}): {error_msg}")
                
                if attempt < self.config.retry_times - 1:
                    time.sleep(self.config.retry_delay * (attempt + 1))
                else:
                    return self._fallback_response(prompt, error_msg)
    
    def batch_generate(
        self, 
        prompts: List[str], 
        system_prompt: Optional[str] = None
    ) -> List[LLMResponse]:
        """批量生成响应"""
        results = []
        for i, prompt in enumerate(prompts):
            logger.info(f"正在处理第 {i+1}/{len(prompts)} 个提示词...")
            result = self.generate(prompt, system_prompt)
            results.append(result)
            
            if i < len(prompts) - 1:
                time.sleep(0.5)
                
        return results
    
    def _fallback_response(self, prompt: str, error_message: str) -> LLMResponse:
        """生成后备响应"""
        fallback_responses = [
            "请帮我调整一下中间那段时间的数据。",
            "能不能把那部分数据变得平缓一些？",
            "需要修改一下那个区间的变化趋势。",
            "帮忙处理一下那段时间的异常情况。",
            "请调整一下那个时段的数据特征。"
        ]
        
        import random
        fallback = random.choice(fallback_responses)
        
        return LLMResponse(
            prompt=prompt,
            response=fallback,
            success=False,
            error_message=error_message
        )


class MockLLMClient(BaseLLMClient):
    """模拟LLM客户端（用于测试）"""
    
    def __init__(self):
        self.templates = {
            ChangeType.EVENT_DROP: [
                "那段时间出了点状况，数据应该会明显下降，帮我模拟一下这个情况。",
                "中间那个时段发生了突发事件，导致数值骤降，请帮我调整。",
                "有段时间出现了异常，数据应该大幅减少才对。"
            ],
            ChangeType.BASELINE_SHIFT: [
                "那段时间整体水平应该比现在高一个台阶，帮我调整一下。",
                "那个时段的数据基线需要往上抬一抬。",
                "这期间的数据整体偏低了，应该提高一些。"
            ],
            ChangeType.ANOMALY_SPIKE: [
                "那段时间数据出现了很多异常波动，帮我模拟这个情况。",
                "中间有些脏数据，波动很剧烈，请保留这个特征。",
                "那个时段传感器可能有问题，数据乱跳。"
            ],
            ChangeType.TREND_SMOOTHING: [
                "那段时间的变化太剧烈了，能不能让它平滑一些？",
                "中间那段波动太尖锐，帮我抹平一点。",
                "那个时段的数据起伏太大，需要变得更平缓。"
            ],
            ChangeType.AMPLIFICATION: [
                "那段时间的变化幅度需要更明显一些。",
                "中间那个区间的波动应该更剧烈才对。",
                "那个时段的数据起伏太小了，放大一些。"
            ],
            ChangeType.ATTENUATION: [
                "那段时间的波动太大了，能不能压缩一下？",
                "中间那个区间的变化幅度需要减小。",
                "那个时段的数据起伏太剧烈，收敛一点。"
            ],
            ChangeType.TREND_INJECTION: [
                "那段时间应该有一个明显的上升趋势。",
                "中间那个区间需要加入一个下降的趋势。",
                "那个时段的数据走势需要改变方向。"
            ]
        }
        
    def generate(
        self, 
        prompt: str, 
        system_prompt: Optional[str] = None
    ) -> LLMResponse:
        """生成模拟响应"""
        import random
        time.sleep(0.1)
        
        for change_type, templates in self.templates.items():
            if change_type.value in prompt:
                return LLMResponse(
                    prompt=prompt,
                    response=random.choice(templates),
                    success=True,
                    latency=0.1
                )
        
        return LLMResponse(
            prompt=prompt,
            response="请帮我调整一下中间那段数据。",
            success=True,
            latency=0.1
        )
    
    def batch_generate(
        self, 
        prompts: List[str], 
        system_prompt: Optional[str] = None
    ) -> List[LLMResponse]:
        """批量生成模拟响应"""
        return [self.generate(p, system_prompt) for p in prompts]


class VaguePromptGenerator:
    """模糊提示词生成器"""
    
    def __init__(
        self, 
        llm_client: Optional[BaseLLMClient] = None,
        use_mock: bool = False
    ):
        if llm_client is not None:
            self.client = llm_client
        elif use_mock:
            self.client = MockLLMClient()
        else:
            self.client = OpenAICompatibleClient()
            
    def generate_prompt(
        self,
        scenario: ScenarioType,
        change_type: ChangeType,
        physical_desc: str,
        total_len: int,
        start_idx: int,
        end_idx: int
    ) -> LLMResponse:
        """
        生成模糊提示词
        
        Args:
            scenario: 业务场景
            change_type: 变化类型
            physical_desc: 物理变化描述
            total_len: 序列总长度
            start_idx: 变化起始位置
            end_idx: 变化结束位置
            
        Returns:
            LLMResponse: LLM响应结果
        """
        position_desc = self._get_position_description(start_idx, end_idx, total_len)
        
        user_prompt = LLM_USER_PROMPT_TEMPLATE.format(
            scenario=scenario.value,
            change_type=self._get_change_type_display(change_type),
            physical_desc=physical_desc,
            total_len=total_len,
            position_desc=position_desc
        )
        
        return self.client.generate(user_prompt, LLM_SYSTEM_PROMPT)
    
    def batch_generate_prompts(
        self,
        requests: List[Dict]
    ) -> List[LLMResponse]:
        """
        批量生成模糊提示词
        
        Args:
            requests: 请求列表，每个请求包含scenario, change_type, physical_desc等字段
            
        Returns:
            List[LLMResponse]: LLM响应结果列表
        """
        prompts = []
        
        for req in requests:
            position_desc = self._get_position_description(
                req['start_idx'], req['end_idx'], req['total_len']
            )
            
            prompt = LLM_USER_PROMPT_TEMPLATE.format(
                scenario=req['scenario'].value,
                change_type=self._get_change_type_display(req['change_type']),
                physical_desc=req['physical_desc'],
                total_len=req['total_len'],
                position_desc=position_desc
            )
            prompts.append(prompt)
            
        return self.client.batch_generate(prompts, LLM_SYSTEM_PROMPT)
    
    def _get_position_description(self, start_idx: int, end_idx: int, total_len: int) -> str:
        """获取位置描述"""
        start_ratio = start_idx / total_len
        end_ratio = end_idx / total_len
        mid_ratio = (start_ratio + end_ratio) / 2
        
        if mid_ratio < 0.25:
            return "前段"
        elif mid_ratio < 0.5:
            return "中前段"
        elif mid_ratio < 0.75:
            return "中后段"
        else:
            return "后段"
    
    def _get_change_type_display(self, change_type: ChangeType) -> str:
        """获取变化类型的显示名称"""
        display_names = {
            ChangeType.EVENT_DROP: "事件骤降",
            ChangeType.BASELINE_SHIFT: "基线平移",
            ChangeType.ANOMALY_SPIKE: "异常波动",
            ChangeType.TREND_SMOOTHING: "趋势平滑",
            ChangeType.AMPLIFICATION: "特征放大",
            ChangeType.ATTENUATION: "特征衰减",
            ChangeType.TREND_INJECTION: "趋势注入"
        }
        return display_names.get(change_type, change_type.value)


def evaluate_prompt_quality(prompt: str, change_type: ChangeType) -> Dict:
    """
    评估生成的提示词质量
    
    Args:
        prompt: 生成的提示词
        change_type: 变化类型
        
    Returns:
        Dict: 评估结果
    """
    issues = []
    scores = {}
    
    import re
    number_patterns = [
        r'\d+\.?\d*%',
        r'index\s*\d+',
        r'第\s*\d+\s*[步个]',
        r'\d+\s*到\s*\d+',
        r'倍',
        r'\d+\s*个单位'
    ]
    
    has_numbers = False
    for pattern in number_patterns:
        if re.search(pattern, prompt):
            has_numbers = True
            issues.append(f"包含具体数值信息: {pattern}")
            
    scores['no_specific_numbers'] = 0 if has_numbers else 1
    
    vague_words = ['稍微', '一点', '比较', '大概', '有些', '明显', '大幅', '小幅', '平缓', '剧烈']
    vague_count = sum(1 for word in vague_words if word in prompt)
    scores['vague_language'] = min(vague_count / 2, 1)
    
    time_words = ['早', '晚', '中午', '下午', '凌晨', '深夜', '前', '后', '中', '开始', '结束']
    has_time_ref = any(word in prompt for word in time_words)
    scores['time_reference'] = 1 if has_time_ref else 0.5
    
    natural_patterns = ['帮我', '能不能', '请', '需要', '应该', '能不能']
    is_natural = any(pattern in prompt for pattern in natural_patterns)
    scores['natural_language'] = 1 if is_natural else 0.5
    
    type_keywords = {
        ChangeType.EVENT_DROP: ['下降', '骤降', '减少', '跌', '低'],
        ChangeType.BASELINE_SHIFT: ['抬高', '提高', '上升', '台阶', '整体'],
        ChangeType.ANOMALY_SPIKE: ['异常', '波动', '脏', '乱', '跳'],
        ChangeType.TREND_SMOOTHING: ['平滑', '平缓', '抹平', '缓和'],
        ChangeType.AMPLIFICATION: ['放大', '剧烈', '明显', '增强'],
        ChangeType.ATTENUATION: ['压缩', '减小', '收敛', '减弱'],
        ChangeType.TREND_INJECTION: ['趋势', '上升', '下降', '走势']
    }
    
    keywords = type_keywords.get(change_type, [])
    keyword_match = sum(1 for kw in keywords if kw in prompt)
    scores['semantic_match'] = min(keyword_match / 2, 1)
    
    total_score = sum(scores.values()) / len(scores)
    
    return {
        'total_score': total_score,
        'dimension_scores': scores,
        'issues': issues,
        'is_acceptable': total_score >= 0.6 and len(issues) == 0
    }
