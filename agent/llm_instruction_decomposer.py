"""LLM-based instruction decomposer for intelligent time series editing.

This module uses LLM to understand natural language editing instructions
and decompose them into structured editing operations.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import numpy as np
from openai import OpenAI


class LLMInstructionDecomposer:
    """Use LLM to decompose user editing instructions.
    
    This class leverages LLM's natural language understanding to:
    1. Parse complex editing intents
    2. Identify target regions
    3. Generate appropriate editing operations
    """
    
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        """Initialize the LLM-based decomposer.
        
        Args:
            api_key: API key for LLM service (defaults to OPENAI_API_KEY env var)
            base_url: Base URL for LLM API (defaults to OPENAI_BASE_URL env var)
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL")
        
        if not self.api_key:
            raise ValueError("API key not provided. Set OPENAI_API_KEY environment variable.")
        
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        self.model = "deepseek-chat"
        
    def decompose(
        self,
        user_instruction: str,
        ts_length: int,
        ts_values: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Decompose user instruction using LLM.
        
        Args:
            user_instruction: User's editing request in natural language
            ts_length: Length of the time series
            ts_values: Optional time series values for context
            
        Returns:
            Dictionary containing structured editing plan
        """
        # Prepare context about the time series
        ts_context = self._prepare_ts_context(ts_length, ts_values)
        
        # Build prompt for LLM
        prompt = self._build_decomposition_prompt(user_instruction, ts_context)
        
        # Call LLM
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert time series editing assistant. Analyze user instructions and output structured editing plans in JSON format."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            result["original_instruction"] = user_instruction
            return result
            
        except Exception as e:
            print(f"LLM decomposition failed: {e}")
            # Fallback to default behavior
            return {
                "intent": "general",
                "region_selection": {
                    "method": "manual",
                    "start_idx": 0,
                    "end_idx": ts_length,
                    "reasoning": "LLM failed, defaulting to entire series"
                },
                "editing_operations": [
                    {
                        "name": "tedit_edit",
                        "parameters": {"n_samples": 1, "sampler": "ddim"},
                        "reasoning": "Default editing operation"
                    }
                ],
                "is_two_stage": False,
                "original_instruction": user_instruction
            }
    
    def _prepare_ts_context(
        self,
        ts_length: int,
        ts_values: Optional[np.ndarray]
    ) -> str:
        """Prepare time series context for LLM."""
        context = f"Time Series Length: {ts_length}\n"
        
        if ts_values is not None:
            context += f"Mean: {np.mean(ts_values):.2f}\n"
            context += f"Std: {np.std(ts_values):.2f}\n"
            context += f"Min: {np.min(ts_values):.2f}\n"
            context += f"Max: {np.max(ts_values):.2f}\n"
            
            # Sample some values for context
            sample_indices = np.linspace(0, len(ts_values)-1, min(10, len(ts_values)), dtype=int)
            sample_values = ts_values[sample_indices]
            context += f"Sample values at indices {sample_indices.tolist()}: {sample_values.tolist()}\n"
        
        return context
    
    def _build_decomposition_prompt(
        self,
        instruction: str,
        ts_context: str
    ) -> str:
        """Build prompt for LLM decomposition."""
        prompt = f"""Analyze the following time series editing instruction and output a structured editing plan.

## Time Series Information
{ts_context}

## User Instruction
"{instruction}"

## Task
1. Identify the editing intent (trend, seasonality, volatility, anomaly, smoothing, etc.)
2. Determine if this requires region selection (specific part of the series) or the entire series
3. Specify the editing operations needed

## Output Format
Return a JSON object with this structure:
{{
    "intent": "detected_intent",
    "region_selection": {{
        "method": "manual" or "semantic",
        "start_idx": start_index (0-based, inclusive),
        "end_idx": end_index (exclusive),
        "reasoning": "why this region was selected"
    }},
    "editing_operations": [
        {{
            "name": "operation_name",
            "parameters": {{parameter_dict}},
            "reasoning": "why this operation"
        }}
    ],
    "is_two_stage": true/false,
    "natural_language_summary": "brief summary of the editing plan"
}}

## Available Operations
- tedit_edit: General editing with TEdit
- tedit_change_trend: Modify trend (parameters: trend_direction: "increase"/"decrease")
- tedit_change_seasonality: Modify seasonality
- tedit_change_volatility: Modify volatility (parameters: volatility_level: "high"/"low")
- tedit_edit_region: Edit specific region

## Examples

Example 1:
Instruction: "Make the first half grow faster"
Output: {{
    "intent": "trend",
    "region_selection": {{
        "method": "manual",
        "start_idx": 0,
        "end_idx": 25,
        "reasoning": "First half of the 50-point series"
    }},
    "editing_operations": [
        {{
            "name": "tedit_change_trend",
            "parameters": {{"trend_direction": "increase"}},
            "reasoning": "User wants to increase growth rate"
        }}
    ],
    "is_two_stage": true,
    "natural_language_summary": "Increase trend in the first half of the series"
}}

Example 2:
Instruction: "Reduce the noise in the middle section"
Output: {{
    "intent": "volatility",
    "region_selection": {{
        "method": "manual",
        "start_idx": 15,
        "end_idx": 35,
        "reasoning": "Middle section of the series"
    }},
    "editing_operations": [
        {{
            "name": "tedit_change_volatility",
            "parameters": {{"volatility_level": "low"}},
            "reasoning": "Reduce noise means decreasing volatility"
        }}
    ],
    "is_two_stage": true,
    "natural_language_summary": "Decrease volatility in the middle section"
}}

Now analyze the instruction and output the JSON:"""
        
        return prompt
    
    def select_region_with_llm(
        self,
        instruction: str,
        ts_values: np.ndarray,
        intent: str
    ) -> Dict[str, Any]:
        """Use LLM to select the best region for editing.
        
        Args:
            instruction: User instruction
            ts_values: Time series values
            intent: Detected editing intent
            
        Returns:
            Region selection result
        """
        # Prepare time series data
        ts_data = ts_values.tolist()
        
        prompt = f"""Given a time series editing task, select the most appropriate region to edit.

## Time Series Data (length: {len(ts_values)})
{ts_data}

## Editing Intent
{intent}

## User Instruction
"{instruction}"

## Task
Analyze the time series and select the region that best matches the editing intent.
For example:
- If intent is "volatility", select the most volatile region
- If intent is "trend", select where trend change is most needed
- If intent is "anomaly", select where anomalies exist

## Output Format
Return a JSON object:
{{
    "start_idx": start_index,
    "end_idx": end_index,
    "confidence": 0.0-1.0,
    "reasoning": "explanation of why this region was selected",
    "method": "llm_semantic"
}}

Output only the JSON:"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert at analyzing time series data."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            # Ensure indices are valid
            result["start_idx"] = max(0, min(result.get("start_idx", 0), len(ts_values) - 1))
            result["end_idx"] = max(result["start_idx"] + 1, min(result.get("end_idx", len(ts_values)), len(ts_values)))
            return result
            
        except Exception as e:
            print(f"LLM region selection failed: {e}")
            # Fallback: select middle region
            mid = len(ts_values) // 2
            return {
                "start_idx": mid - 5,
                "end_idx": mid + 5,
                "confidence": 0.5,
                "reasoning": "LLM failed, using default middle region",
                "method": "fallback"
            }


_llm_decomposer_instance: Optional[LLMInstructionDecomposer] = None


def get_llm_decomposer(
    api_key: Optional[str] = None,
    base_url: Optional[str] = None
) -> LLMInstructionDecomposer:
    """Get or create a singleton LLM decomposer instance.
    
    Args:
        api_key: API key for LLM service
        base_url: Base URL for LLM API
        
    Returns:
        LLMInstructionDecomposer instance
    """
    global _llm_decomposer_instance
    if _llm_decomposer_instance is None:
        _llm_decomposer_instance = LLMInstructionDecomposer(api_key, base_url)
    return _llm_decomposer_instance
