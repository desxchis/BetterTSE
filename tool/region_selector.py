"""Region selector for two-stage time series editing.

This module implements region selection strategies that identify
appropriate regions for editing based on semantic intent or
statistical characteristics.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats
from scipy.signal import find_peaks


class RegionSelector:
    """Select regions of time series for editing.

    This class provides multiple strategies for selecting regions:
    - Semantic-based: Select based on editing intent
    - Statistical-based: Select based on anomalies or patterns
    - Manual: Use user-specified indices

    Attributes:
        min_length: Minimum region length to select
        max_length: Maximum region length to select
    """

    def __init__(
        self,
        min_length: int = 5,
        max_length: int = 50,
    ):
        """Initialize region selector.

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
        method: str = "semantic",
        threshold: float = 0.5,
        **kwargs
    ) -> Dict[str, Any]:
        """Select a region for editing.

        Args:
            ts: Input time series
            intent: Editing intent (e.g., "trend", "volatility", "anomaly")
            method: Selection method ("semantic", "statistical", "manual")
            threshold: Threshold for statistical methods
            **kwargs: Additional parameters

        Returns:
            Dictionary containing:
            - start_idx: Start index of selected region
            - end_idx: End index of selected region
            - method: Method used
            - confidence: Confidence score (0-1)
            - reasoning: Explanation of selection
        """
        if method == "manual":
            return self._select_manual(ts, **kwargs)
        elif method == "semantic":
            return self._select_semantic(ts, intent, threshold, **kwargs)
        elif method == "statistical":
            return self._select_statistical(ts, intent, threshold, **kwargs)
        else:
            raise ValueError(f"Unknown selection method: {method}")

    def _select_manual(
        self,
        ts: np.ndarray,
        start_idx: int = 0,
        end_idx: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Select region using manual indices.

        Args:
            ts: Input time series
            start_idx: Start index
            end_idx: End index (None means end of series)

        Returns:
            Selection result dictionary
        """
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
        threshold: float,
        **kwargs
    ) -> Dict[str, Any]:
        """Select region based on semantic intent.

        Args:
            ts: Input time series
            intent: Editing intent
            threshold: Confidence threshold
            **kwargs: Additional parameters

        Returns:
            Selection result dictionary
        """
        if intent == "trend":
            return self._select_trend_region(ts, threshold)
        elif intent == "volatility":
            return self._select_volatility_region(ts, threshold)
        elif intent == "anomaly":
            return self._select_anomaly_region(ts, threshold)
        elif intent == "smoothing":
            return self._select_smoothing_region(ts, threshold)
        else:
            return self._select_default_region(ts)

    def _select_statistical(
        self,
        ts: np.ndarray,
        intent: str,
        threshold: float,
        **kwargs
    ) -> Dict[str, Any]:
        """Select region based on statistical analysis.

        Args:
            ts: Input time series
            intent: Editing intent
            threshold: Statistical threshold
            **kwargs: Additional parameters

        Returns:
            Selection result dictionary
        """
        if intent == "anomaly":
            return self._select_anomaly_region(ts, threshold)
        elif intent == "volatility":
            return self._select_high_variance_region(ts, threshold)
        else:
            return self._select_default_region(ts)

    def _select_trend_region(
        self,
        ts: np.ndarray,
        threshold: float
    ) -> Dict[str, Any]:
        """Select region with strongest trend.

        Args:
            ts: Input time series
            threshold: Minimum trend strength threshold

        Returns:
            Selection result dictionary
        """
        window_size = min(self.max_length, len(ts) // 4)
        window_size = max(window_size, self.min_length)

        best_start = 0
        best_end = len(ts)
        best_trend_strength = 0

        for start in range(0, len(ts) - window_size + 1, window_size // 2):
            end = min(start + window_size, len(ts))
            region = ts[start:end]

            x = np.arange(len(region))
            slope, _, r_value, _, _ = stats.linregress(x, region)
            trend_strength = abs(slope) * abs(r_value)

            if trend_strength > best_trend_strength:
                best_trend_strength = trend_strength
                best_start = start
                best_end = end

        confidence = min(1.0, best_trend_strength / threshold)

        return {
            "start_idx": best_start,
            "end_idx": best_end,
            "method": "semantic_trend",
            "confidence": confidence,
            "reasoning": f"Selected region with strongest trend (strength: {best_trend_strength:.3f})"
        }

    def _select_volatility_region(
        self,
        ts: np.ndarray,
        threshold: float
    ) -> Dict[str, Any]:
        """Select region with highest volatility.

        Args:
            ts: Input time series
            threshold: Minimum volatility threshold

        Returns:
            Selection result dictionary
        """
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

        confidence = min(1.0, best_volatility / threshold)

        return {
            "start_idx": best_start,
            "end_idx": best_end,
            "method": "semantic_volatility",
            "confidence": confidence,
            "reasoning": f"Selected region with highest volatility (std: {best_volatility:.3f})"
        }

    def _select_anomaly_region(
        self,
        ts: np.ndarray,
        threshold: float
    ) -> Dict[str, Any]:
        """Select region containing most anomalies.

        Args:
            ts: Input time series
            threshold: Z-score threshold for anomaly detection

        Returns:
            Selection result dictionary
        """
        z_scores = np.abs(stats.zscore(ts))
        anomalies = z_scores > threshold

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

        confidence = min(1.0, np.sum(anomalies) / (end_idx - start_idx))

        return {
            "start_idx": start_idx,
            "end_idx": end_idx,
            "method": "statistical_anomaly",
            "confidence": confidence,
            "reasoning": f"Selected region containing {np.sum(anomalies)} anomalies"
        }

    def _select_high_variance_region(
        self,
        ts: np.ndarray,
        threshold: float
    ) -> Dict[str, Any]:
        """Select region with highest variance.

        Args:
            ts: Input time series
            threshold: Variance threshold

        Returns:
            Selection result dictionary
        """
        window_size = min(self.max_length, len(ts) // 4)
        window_size = max(window_size, self.min_length)

        best_start = 0
        best_end = len(ts)
        best_variance = 0

        for start in range(0, len(ts) - window_size + 1, window_size // 2):
            end = min(start + window_size, len(ts))
            region = ts[start:end]
            variance = np.var(region)

            if variance > best_variance:
                best_variance = variance
                best_start = start
                best_end = end

        confidence = min(1.0, best_variance / threshold)

        return {
            "start_idx": best_start,
            "end_idx": best_end,
            "method": "statistical_variance",
            "confidence": confidence,
            "reasoning": f"Selected region with highest variance (var: {best_variance:.3f})"
        }

    def _select_smoothing_region(
        self,
        ts: np.ndarray,
        threshold: float
    ) -> Dict[str, Any]:
        """Select region that would benefit most from smoothing.

        Args:
            ts: Input time series
            threshold: Roughness threshold

        Returns:
            Selection result dictionary
        """
        window_size = min(self.max_length, len(ts) // 4)
        window_size = max(window_size, self.min_length)

        best_start = 0
        best_end = len(ts)
        best_roughness = 0

        for start in range(0, len(ts) - window_size + 1, window_size // 2):
            end = min(start + window_size, len(ts))
            region = ts[start:end]

            if len(region) < 2:
                continue

            roughness = np.mean(np.abs(np.diff(region, 2)))

            if roughness > best_roughness:
                best_roughness = roughness
                best_start = start
                best_end = end

        confidence = min(1.0, best_roughness / threshold)

        return {
            "start_idx": best_start,
            "end_idx": best_end,
            "method": "semantic_smoothing",
            "confidence": confidence,
            "reasoning": f"Selected region with highest roughness (roughness: {best_roughness:.3f})"
        }

    def _select_default_region(
        self,
        ts: np.ndarray
    ) -> Dict[str, Any]:
        """Select default region (middle of series).

        Args:
            ts: Input time series

        Returns:
            Selection result dictionary
        """
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

    def select_multiple_regions(
        self,
        ts: np.ndarray,
        intent: str,
        method: str = "semantic",
        n_regions: int = 3,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Select multiple regions for editing.

        Args:
            ts: Input time series
            intent: Editing intent
            method: Selection method
            n_regions: Number of regions to select
            **kwargs: Additional parameters

        Returns:
            List of selection result dictionaries
        """
        regions = []

        if method == "anomaly":
            return self._select_anomaly_regions(ts, n_regions, **kwargs)
        elif method == "peak":
            return self._select_peak_regions(ts, n_regions, **kwargs)
        else:
            return self._select_diverse_regions(ts, intent, n_regions, **kwargs)

    def _select_anomaly_regions(
        self,
        ts: np.ndarray,
        n_regions: int,
        threshold: float = 3.0,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Select multiple regions containing anomalies.

        Args:
            ts: Input time series
            n_regions: Number of regions
            threshold: Z-score threshold

        Returns:
            List of selection results
        """
        z_scores = np.abs(stats.zscore(ts))
        anomalies = z_scores > threshold

        if not np.any(anomalies):
            return [self._select_default_region(ts)]

        anomaly_indices = np.where(anomalies)[0]

        regions = []
        for idx in anomaly_indices[:n_regions]:
            start_idx = max(0, idx - self.min_length // 2)
            end_idx = min(len(ts), idx + self.min_length // 2)
            regions.append({
                "start_idx": start_idx,
                "end_idx": end_idx,
                "method": "anomaly",
                "confidence": 0.8,
                "reasoning": f"Region containing anomaly at index {idx}"
            })

        return regions

    def _select_peak_regions(
        self,
        ts: np.ndarray,
        n_regions: int,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Select regions around peaks.

        Args:
            ts: Input time series
            n_regions: Number of regions

        Returns:
            List of selection results
        """
        peaks, _ = find_peaks(ts)

        if len(peaks) == 0:
            return [self._select_default_region(ts)]

        regions = []
        for peak_idx in peaks[:n_regions]:
            start_idx = max(0, peak_idx - self.min_length // 2)
            end_idx = min(len(ts), peak_idx + self.min_length // 2)
            regions.append({
                "start_idx": start_idx,
                "end_idx": end_idx,
                "method": "peak",
                "confidence": 0.7,
                "reasoning": f"Region around peak at index {peak_idx}"
            })

        return regions

    def _select_diverse_regions(
        self,
        ts: np.ndarray,
        intent: str,
        n_regions: int,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Select diverse regions across the time series.

        Args:
            ts: Input time series
            intent: Editing intent
            n_regions: Number of regions

        Returns:
            List of selection results
        """
        step = len(ts) // (n_regions + 1)
        regions = []

        for i in range(n_regions):
            start_idx = i * step
            end_idx = min(start_idx + self.max_length, (i + 1) * step)
            end_idx = min(end_idx, len(ts))

            regions.append({
                "start_idx": start_idx,
                "end_idx": end_idx,
                "method": "diverse",
                "confidence": 0.6,
                "reasoning": f"Region {i+1} of {n_regions} diverse regions"
            })

        return regions


_selector_instance: Optional[RegionSelector] = None


def get_selector(
    min_length: int = 5,
    max_length: int = 50,
    force_reload: bool = False
) -> RegionSelector:
    """Get or create a singleton selector instance.

    Args:
        min_length: Minimum region length
        max_length: Maximum region length
        force_reload: Force reload of instance

    Returns:
        RegionSelector instance
    """
    global _selector_instance
    if _selector_instance is None or force_reload:
        _selector_instance = RegionSelector(min_length, max_length)
    return _selector_instance
