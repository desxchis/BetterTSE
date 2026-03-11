"""Simple region selector without scipy dependency for testing."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np


class SimpleRegionSelector:
    """Simple region selector without scipy dependency.

    This class provides basic region selection strategies for testing.
    """

    def __init__(
        self,
        min_length: int = 5,
        max_length: int = 50,
    ):
        """Initialize simple region selector.

        Args:
            min_length: Minimum region length (default: 5)
            max_length: Maximum region length (default: 50)
        """
        self.min_length = min_length
        self.max_length = max_length

    def select_region(
        self,
        ts: np.ndarray,
        intent: str,
        method: str = "manual",
        **kwargs
    ) -> Dict[str, Any]:
        """Select a region for editing.

        Args:
            ts: Input time series
            intent: Editing intent
            method: Selection method
            **kwargs: Additional parameters

        Returns:
            Dictionary containing selection result
        """
        if method == "manual":
            return self._select_manual(ts, **kwargs)
        elif method == "semantic":
            return self._select_semantic(ts, intent, **kwargs)
        elif method == "statistical":
            return self._select_semantic(ts, intent, **kwargs)
        else:
            return self._select_default_region(ts)

    def _select_manual(
        self,
        ts: np.ndarray,
        start_idx: int = 0,
        end_idx: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Select region using manual indices."""
        if end_idx is None:
            end_idx = len(ts)

        start_idx = max(0, start_idx)
        end_idx = min(len(ts), end_idx)

        if start_idx >= end_idx:
            raise ValueError(f"Invalid region: start_idx ({start_idx}) >= end_idx ({end_idx})")

        return {
            "start_idx": start_idx,
            "end_idx": end_idx,
            "method": "manual",
            "confidence": 1.0,
            "reasoning": "User-specified region"
        }

    def _select_semantic(
        self,
        ts: np.ndarray,
        intent: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Select region based on semantic intent."""
        if intent == "trend":
            return self._select_trend_region(ts)
        elif intent == "volatility":
            return self._select_volatility_region(ts)
        elif intent == "anomaly":
            return self._select_anomaly_region(ts)
        else:
            return self._select_default_region(ts)

    def _select_trend_region(
        self,
        ts: np.ndarray
    ) -> Dict[str, Any]:
        """Select region with strongest trend."""
        window_size = min(self.max_length, len(ts) // 4)
        window_size = max(window_size, self.min_length)

        best_start = 0
        best_end = len(ts)
        best_trend_strength = 0

        for start in range(0, len(ts) - window_size + 1, window_size // 2):
            end = min(start + window_size, len(ts))
            region = ts[start:end]

            if len(region) < 2:
                continue

            x = np.arange(len(region))
            coeffs = np.polyfit(x, region, 1)
            slope = coeffs[0]
            trend_strength = abs(slope)

            if trend_strength > best_trend_strength:
                best_trend_strength = trend_strength
                best_start = start
                best_end = end

        confidence = min(1.0, best_trend_strength * 10)

        return {
            "start_idx": best_start,
            "end_idx": best_end,
            "method": "semantic_trend",
            "confidence": confidence,
            "reasoning": f"Selected region with strongest trend (strength: {best_trend_strength:.3f})"
        }

    def _select_volatility_region(
        self,
        ts: np.ndarray
    ) -> Dict[str, Any]:
        """Select region with highest volatility."""
        window_size = min(self.max_length, len(ts) // 4)
        window_size = max(window_size, self.min_length)

        best_start = 0
        best_end = len(ts)
        best_volatility = 0

        for start in range(0, len(ts) - window_size + 1, window_size // 2):
            end = min(start + window_size, len(ts))
            region = ts[start:end]
            volatility = np.std(region)

            if volatility > best_volatility:
                best_volatility = volatility
                best_start = start
                best_end = end

        confidence = min(1.0, best_volatility / 2.0)

        return {
            "start_idx": best_start,
            "end_idx": best_end,
            "method": "semantic_volatility",
            "confidence": confidence,
            "reasoning": f"Selected region with highest volatility (std: {best_volatility:.3f})"
        }

    def _select_anomaly_region(
        self,
        ts: np.ndarray
    ) -> Dict[str, Any]:
        """Select region containing most anomalies."""
        mean = np.mean(ts)
        std = np.std(ts)
        threshold = 2.0

        anomalies = np.abs(ts - mean) > threshold * std

        if not np.any(anomalies):
            return self._select_default_region(ts)

        anomaly_indices = np.where(anomalies)[0]

        if len(anomaly_indices) == 1:
            idx = anomaly_indices[0]
            start_idx = max(0, idx - self.min_length // 2)
            end_idx = min(len(ts), idx + self.min_length // 2)
        else:
            start_idx = anomaly_indices[0]
            end_idx = anomaly_indices[-1] + 1

        start_idx = max(0, start_idx)
        end_idx = min(len(ts), end_idx)

        if start_idx >= end_idx:
            start_idx = max(0, start_idx - self.min_length)
            end_idx = start_idx + self.min_length
            if end_idx > len(ts):
                end_idx = len(ts)

        confidence = min(1.0, np.sum(anomalies) / (end_idx - start_idx))

        return {
            "start_idx": start_idx,
            "end_idx": end_idx,
            "method": "statistical_anomaly",
            "confidence": confidence,
            "reasoning": f"Selected region containing {np.sum(anomalies)} anomalies"
        }

    def _select_default_region(
        self,
        ts: np.ndarray
    ) -> Dict[str, Any]:
        """Select default region (middle of series)."""
        mid_point = len(ts) // 2
        half_length = min(self.max_length // 2, len(ts) // 4)
        half_length = max(half_length, self.min_length // 2)

        start_idx = max(0, mid_point - half_length)
        end_idx = min(len(ts), mid_point + half_length)

        return {
            "start_idx": start_idx,
            "end_idx": end_idx,
            "method": "default",
            "confidence": 0.5,
            "reasoning": "Selected middle region as default"
        }


_simple_selector_instance: Optional[SimpleRegionSelector] = None


def get_simple_selector(
    min_length: int = 5,
    max_length: int = 50,
    force_reload: bool = False
) -> SimpleRegionSelector:
    """Get or create a singleton simple selector instance.

    Args:
        min_length: Minimum region length
        max_length: Maximum region length
        force_reload: Force reload of instance

    Returns:
        SimpleRegionSelector instance
    """
    global _simple_selector_instance
    if _simple_selector_instance is None or force_reload:
        _simple_selector_instance = SimpleRegionSelector(min_length, max_length)
    return _simple_selector_instance
