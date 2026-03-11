"""Instruction decomposer for two-stage time series editing.

This module implements the instruction decomposition mechanism that breaks down
complex editing requests into a two-stage process:
1. Region selection (like bounding box)
2. Specific editing operations on the selected region
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class InstructionDecomposer:
    """Decompose user editing instructions into region selection and editing steps.

    This class analyzes user's editing intent and generates a structured
    two-stage editing plan:
    - Stage 1: Select the region to edit (like bounding box)
    - Stage 2: Apply specific editing operations to the selected region

    Attributes:
        intent_keywords: Dictionary mapping keywords to editing intents
        region_patterns: Regex patterns for region specifications
    """

    def __init__(self):
        """Initialize the instruction decomposer."""
        self.intent_keywords = {
            "trend": [
                "increase trend", "decrease trend", "steeper trend",
                "flatter trend", "make it grow faster", "slow down growth",
                "change slope", "adjust trend", "trend adjustment"
            ],
            "volatility": [
                "increase volatility", "decrease volatility", "more fluctuating",
                "less fluctuating", "reduce noise", "add noise", "smooth out",
                "make it more variable", "stabilize", "reduce volatility",
                "decrease volatility", "increase fluctuation", "decrease fluctuation"
            ],
            "seasonality": [
                "change seasonality", "modify seasonal pattern", "add seasonality",
                "remove seasonality", "change period", "change amplitude"
            ],
            "anomaly": [
                "remove outliers", "fix anomalies", "smooth spikes",
                "handle extreme values", "clean data"
            ],
            "smoothing": [
                "smooth", "smooth out", "make smoother", "reduce jaggedness"
            ],
            "interpolation": [
                "fill gaps", "interpolate", "fill missing", "impute"
            ]
        }

        self.region_patterns = {
            "indices": r"(?:from\s+)?(\d+)\s*(?:to|until|-)\s*(\d+)",
            "relative": r"(?:first|last|beginning|end|middle|second\s+half)",
            "percentage": r"(?:top|bottom)\s+(\d+)%"
        }

    def decompose(
        self,
        user_instruction: str,
        ts_length: int,
        ts_values: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Decompose user instruction into two-stage editing plan.

        Args:
            user_instruction: User's editing request in natural language
            ts_length: Length of the time series
            ts_values: Optional time series values for context-aware decomposition

        Returns:
            Dictionary containing:
            - intent: Detected editing intent
            - region_selection: Parameters for region selection
            - editing_operations: List of editing operations to apply
            - is_two_stage: Whether this is a two-stage editing task
        """
        instruction_lower = user_instruction.lower()

        intent = self._detect_intent(instruction_lower)

        region_selection = self._extract_region_spec(
            instruction_lower, ts_length, ts_values, intent
        )

        editing_operations = self._generate_editing_operations(
            instruction_lower, intent, region_selection
        )

        is_two_stage = region_selection.get("needs_selection", False)

        return {
            "intent": intent,
            "region_selection": region_selection,
            "editing_operations": editing_operations,
            "is_two_stage": is_two_stage,
            "original_instruction": user_instruction
        }

    def _detect_intent(self, instruction: str) -> str:
        """Detect the primary editing intent from user instruction.

        Args:
            instruction: Lowercase user instruction

        Returns:
            Detected intent string
        """
        for intent, keywords in self.intent_keywords.items():
            for keyword in keywords:
                if keyword in instruction:
                    return intent

        return "general"

    def _extract_region_spec(
        self,
        instruction: str,
        ts_length: int,
        ts_values: Optional[np.ndarray],
        intent: str,
    ) -> Dict[str, Any]:
        """Extract region specification from instruction.

        Args:
            instruction: Lowercase user instruction
            ts_length: Length of time series
            ts_values: Optional time series values
            intent: Detected editing intent

        Returns:
            Dictionary with region selection parameters
        """
        region_spec = {
            "method": "auto",
            "needs_selection": False,
            "start_idx": None,
            "end_idx": None,
            "reasoning": ""
        }

        indices_match = re.search(self.region_patterns["indices"], instruction)
        if indices_match:
            start = int(indices_match.group(1))
            end = int(indices_match.group(2))
            region_spec.update({
                "method": "manual",
                "start_idx": min(start, ts_length - 1),
                "end_idx": min(end, ts_length),
                "reasoning": "User specified explicit indices"
            })
            return region_spec

        if "entire" in instruction or "whole" in instruction or "all" in instruction:
            region_spec.update({
                "method": "manual",
                "start_idx": 0,
                "end_idx": ts_length,
                "reasoning": "User specified entire series"
            })
            return region_spec

        if ts_values is not None:
            region_spec.update({
                "method": "semantic",
                "needs_selection": True,
                "intent": intent,
                "reasoning": f"Will select region based on {intent} intent"
            })
        else:
            region_spec.update({
                "method": "manual",
                "start_idx": 0,
                "end_idx": ts_length,
                "reasoning": "No time series values provided, defaulting to entire series"
            })

        return region_spec

    def _generate_editing_operations(
        self,
        instruction: str,
        intent: str,
        region_spec: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Generate editing operations based on intent.

        Args:
            instruction: Lowercase user instruction
            intent: Detected editing intent
            region_spec: Region selection specification

        Returns:
            List of editing operation dictionaries
        """
        operations = []

        if intent == "trend":
            operations.append(self._create_trend_operation(instruction))
        elif intent == "volatility":
            operations.append(self._create_volatility_operation(instruction))
        elif intent == "seasonality":
            operations.append(self._create_seasonality_operation(instruction))
        elif intent == "anomaly":
            operations.append(self._create_anomaly_operation(instruction))
        elif intent == "smoothing":
            operations.append(self._create_smoothing_operation(instruction))
        elif intent == "interpolation":
            operations.append(self._create_interpolation_operation(instruction))
        else:
            operations.append(self._create_general_operation(instruction))

        for op in operations:
            if region_spec["start_idx"] is not None:
                op["start_idx"] = region_spec["start_idx"]
            if region_spec["end_idx"] is not None:
                op["end_idx"] = region_spec["end_idx"]

        return operations

    def _create_trend_operation(self, instruction: str) -> Dict[str, Any]:
        """Create trend adjustment operation."""
        if any(kw in instruction for kw in ["increase", "steeper", "faster"]):
            return {
                "name": "increase_trend",
                "parameters": {"factor": 1.5},
                "reasoning": "User wants to increase trend magnitude"
            }
        elif any(kw in instruction for kw in ["decrease", "flatter", "slower"]):
            return {
                "name": "decrease_trend",
                "parameters": {"factor": 0.5},
                "reasoning": "User wants to decrease trend magnitude"
            }
        else:
            return {
                "name": "apply_trend_in_region",
                "parameters": {"slope": 0.1},
                "reasoning": "User wants to apply a trend"
            }

    def _create_volatility_operation(self, instruction: str) -> Dict[str, Any]:
        """Create volatility adjustment operation."""
        if any(kw in instruction for kw in ["increase", "more", "add"]):
            return {
                "name": "increase_volatility",
                "parameters": {"factor": 1.5},
                "reasoning": "User wants to increase volatility"
            }
        elif any(kw in instruction for kw in ["decrease", "less", "reduce", "smooth"]):
            return {
                "name": "decrease_volatility",
                "parameters": {"factor": 0.5},
                "reasoning": "User wants to decrease volatility"
            }
        else:
            return {
                "name": "smooth_region",
                "parameters": {"window": 3},
                "reasoning": "User wants to smooth the series"
            }

    def _create_seasonality_operation(self, instruction: str) -> Dict[str, Any]:
        """Create seasonality modification operation."""
        return {
            "name": "tedit_change_seasonality",
            "parameters": {"n_samples": 1, "sampler": "ddim"},
            "reasoning": "User wants to change seasonality pattern"
        }

    def _create_anomaly_operation(self, instruction: str) -> Dict[str, Any]:
        """Create anomaly removal operation."""
        return {
            "name": "remove_anomalies_in_region",
            "parameters": {
                "method": "zscore",
                "threshold": 3.0,
                "fill_method": "interpolate"
            },
            "reasoning": "User wants to remove anomalies"
        }

    def _create_smoothing_operation(self, instruction: str) -> Dict[str, Any]:
        """Create smoothing operation."""
        return {
            "name": "smooth_region",
            "parameters": {"window": 3},
            "reasoning": "User wants to smooth the series"
        }

    def _create_interpolation_operation(self, instruction: str) -> Dict[str, Any]:
        """Create interpolation operation."""
        return {
            "name": "interpolate_region",
            "parameters": {"method": "linear"},
            "reasoning": "User wants to interpolate values"
        }

    def _create_general_operation(self, instruction: str) -> Dict[str, Any]:
        """Create general editing operation."""
        return {
            "name": "tedit_edit",
            "parameters": {"n_samples": 1, "sampler": "ddim"},
            "reasoning": "General editing request, using TEdit"
        }

    def format_for_llm(self, decomposition: Dict[str, Any]) -> str:
        """Format decomposition result for LLM consumption.

        Args:
            decomposition: Decomposition result from decompose()

        Returns:
            Formatted string for LLM
        """
        output = [
            "Instruction Decomposition Result:",
            f"Intent: {decomposition['intent']}",
            f"Is Two-Stage: {decomposition['is_two_stage']}",
            "",
            "Region Selection:",
            json.dumps(decomposition['region_selection'], indent=2),
            "",
            "Editing Operations:",
            json.dumps(decomposition['editing_operations'], indent=2),
            "",
            f"Original Instruction: {decomposition['original_instruction']}"
        ]

        return "\n".join(output)

    def parse_llm_response(
        self,
        llm_response: str
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """Parse LLM response containing editing instructions.

        Args:
            llm_response: LLM's response string

        Returns:
            Tuple of (parsed_edit_instruction, error_message)
        """
        try:
            edit_match = re.search(r"<edit>(.*?)</edit>", llm_response, re.DOTALL)
            if edit_match:
                edit_json = edit_match.group(1).strip()
                parsed = json.loads(edit_json)
                return parsed, None

            region_match = re.search(r"<region>(.*?)</region>", llm_response, re.DOTALL)
            if region_match:
                region_json = region_match.group(1).strip()
                parsed = json.loads(region_json)
                return {"region_selection": parsed}, None

            return None, "No valid <edit> or <region> tag found in response"

        except json.JSONDecodeError as e:
            return None, f"Failed to parse JSON: {e}"
        except Exception as e:
            return None, f"Unexpected error: {e}"


_decomposer_instance: Optional[InstructionDecomposer] = None


def get_decomposer() -> InstructionDecomposer:
    """Get or create a singleton decomposer instance.

    Returns:
        InstructionDecomposer instance
    """
    global _decomposer_instance
    if _decomposer_instance is None:
        _decomposer_instance = InstructionDecomposer()
    return _decomposer_instance
