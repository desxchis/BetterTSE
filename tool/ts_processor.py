"""ts_generators.py
===================

This module collects a suite of deterministic building blocks for synthesising
time series without relying on any random number generators. These functions
are intended for use by high-level agents that need reproducible behaviour
given the same inputs. Each function accepts NumPy arrays or array-like
sequences for the time index or signal values and returns a NumPy array of
matching shape. Scalar parameters are broadcast where sensible.

The functions span simple trends, periodic components, pulses, envelopes,
filters, resampling routines and multivariate couplings. See each
docstring for details on the expected inputs and semantics.

Example
-------

```python
import numpy as np
import ts_generators as tsg

# Create an hourly time vector for one week
t = np.arange(7 * 24, dtype=float)

# Build a synthetic series: linear trend + daily seasonality + weekend boost
trend = tsg.linear_trend(t, slope=0.05, intercept=10)
daily = tsg.sine_wave(t, period=24, amplitude=2.0)
weekend_mask = tsg.periodic_duty_cycle_mask(t, period=7 * 24, duty_cycle=2 * 24 / (7 * 24), phase_offset=5 * 24)
weekend_boost = 3.0 * weekend_mask

signal = trend + daily + weekend_boost
```
"""

from __future__ import annotations

import numpy as np
from typing import Callable, Iterable, Sequence, Tuple, Union, Optional

import pandas as pd

ArrayLike = Union[np.ndarray, Sequence[float], Sequence[int]]


def _to_numpy(arr: ArrayLike, dtype: Optional[np.dtype] = None) -> np.ndarray:
    """Convert input to a NumPy array without copying if already an array.

    Parameters
    ----------
    arr : array_like, required
        Input sequence.
    dtype : numpy dtype, optional
        Desired dtype of returned array. If None, infer from input.

    Returns
    -------
    ndarray
        NumPy array view or copy of the input.
    """
    if isinstance(arr, np.ndarray):
        if dtype is not None and arr.dtype != dtype:
            return arr.astype(dtype)
        return arr
    return np.asarray(arr, dtype=dtype)


# --- transforms ---

def log_transform(x: ArrayLike, epsilon: float = 1e-6) -> np.ndarray:
    """Element-wise natural logarithm with numerical guard.

    Applies ``y = log(x + epsilon)`` to each element, where a small positive
    ``epsilon`` prevents ``log(0)`` when inputs contain zeros.

    Parameters
    ----------
    x : array_like, required
        Input values. Can be any array-like of numeric types.
    epsilon : float, optional
        Small positive offset added before taking the logarithm. Default ``1e-6``.

    Returns
    -------
    ndarray
        Array of the same shape as ``x`` with transformed values.

    Notes
    -----
    If any ``x + epsilon <= 0``, the result for those positions will be
    ``-inf`` (for exactly 0) or ``nan`` (for negatives), following NumPy's
    ``log`` semantics.
    """
    x_arr = _to_numpy(x, dtype=float)
    return np.log(x_arr + epsilon)


def standardize(x: ArrayLike, method: str = "zscore") -> np.ndarray:
    """Remove location and scale from a series.

    Supports two deterministic scalers:

    - ``'zscore'``: ``(x - mean) / std`` using NaN-aware mean/std.
    - ``'robust'``: ``(x - median) / (Q3 - Q1)`` using NaN-aware quantiles.

    Parameters
    ----------
    x : array_like, required
        Input values to standardize.
    method : {'zscore', 'robust'}, optional
        Standardization method. Default is ``'zscore'``.

    Returns
    -------
    ndarray
        Standardized values with the same shape as ``x``.

    Notes
    -----
    If the scale term is zero (std for ``'zscore'`` or ``Q3 - Q1`` for
    ``'robust'``), a zero array of the same shape is returned.
    """
    x_arr = _to_numpy(x, dtype=float)
    if method == "zscore":
        mu = float(np.nanmean(x_arr))
        sigma = float(np.nanstd(x_arr, ddof=0))
        if sigma == 0:
            return np.zeros_like(x_arr, dtype=float)
        return (x_arr - mu) / sigma
    elif method == "robust":
        q1 = float(np.nanquantile(x_arr, 0.25))
        q3 = float(np.nanquantile(x_arr, 0.75))
        denom = q3 - q1
        med = float(np.nanmedian(x_arr))
        if denom == 0:
            return np.zeros_like(x_arr, dtype=float)
        return (x_arr - med) / denom
    else:
        raise ValueError("method must be 'zscore' or 'robust'")


# ---------------------------------------------------------------------------
# Missing-value handling (ffill / bfill / linear / median)
# ---------------------------------------------------------------------------

def fill_missing(x: ArrayLike, method: str = "ffill", limit: Optional[int] = None) -> np.ndarray:
    """Fill missing values (NaN) in a 1-D series.

    Parameters
    ----------
    x : array_like
        Input sequence.
    method : {'ffill','bfill','linear','median'}
        Forward-fill, backward-fill, linear interpolation, or global-median fill.
    limit : int | None, optional
        Maximum number of consecutive NaNs to fill (where applicable).

    Returns
    -------
    ndarray
        Array with imputed values (same length as input).
    """
    s = pd.Series(_to_numpy(x, dtype=float))
    if method == "ffill":
        out = s.ffill(limit=limit)
    elif method == "bfill":
        out = s.bfill(limit=limit)
    elif method == "linear":
        out = s.interpolate(method="linear", limit=limit)
    elif method == "median":
        out = s.fillna(s.median())
    else:
        raise ValueError(
            "method must be one of {'ffill','bfill','linear','median'}")
    return out.to_numpy(dtype=float)


# ---------------------------------------------------------------------------
# Outlier handling
# ---------------------------------------------------------------------------

def remove_outliers(x: ArrayLike, method: str = "zscore", threshold: float = 3.0) -> Tuple[np.ndarray, np.ndarray]:
    """Mask outliers via z-score or IQR rules and return cleaned values + indices.

    Parameters
    ----------
    x : array_like
        Input sequence.
    method : {'zscore','iqr'}
        * 'zscore'  -> mask points with |z| > threshold
        * 'iqr'     -> Tukey fences: [Q1 - k*IQR, Q3 + k*IQR], where k = threshold
    threshold : float
        Cutoff for the chosen method.

    Returns
    -------
    (cleaned, removed_idx) : (ndarray, ndarray)
        * cleaned     -> same shape as x, with outliers set to NaN
        * removed_idx -> zero-based integer indices of masked points
    """
    s = pd.Series(_to_numpy(x, dtype=float))
    if method == "zscore":
        mu = s.mean()
        sigma = s.std(ddof=0)
        if sigma == 0 or np.isnan(sigma):
            mask = pd.Series(False, index=s.index)
        else:
            z = (s - mu).abs() / sigma
            mask = z > threshold
    elif method == "iqr":
        q1, q3 = s.quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - threshold * iqr, q3 + threshold * iqr
        mask = (s < lower) | (s > upper)
    else:
        raise ValueError("method must be 'zscore' or 'iqr'")
    cleaned = s.mask(mask)
    removed = np.where(mask.to_numpy())[0].astype(int)
    return cleaned.to_numpy(dtype=float), removed


# ---------------------------------------------------------------------------
# Region-based editing functions
# ---------------------------------------------------------------------------

def smooth_region(x: ArrayLike, start_idx: int, end_idx: int, window: int = 3) -> np.ndarray:
    """Apply moving average smoothing to a specific region of the time series.

    Parameters
    ----------
    x : array_like
        Input time series.
    start_idx : int
        Start index of the region to smooth (inclusive).
    end_idx : int
        End index of the region to smooth (exclusive).
    window : int, optional
        Window size for moving average. Default is 3.

    Returns
    -------
    ndarray
        Time series with smoothed region.
    """
    x_arr = _to_numpy(x, dtype=float)
    if window < 1:
        raise ValueError("window must be >= 1")
    if start_idx < 0 or end_idx > len(x_arr) or start_idx >= end_idx:
        raise ValueError("Invalid region indices")
    
    result = x_arr.copy()
    region = result[start_idx:end_idx]
    if len(region) >= window:
        kernel = np.ones(window, dtype=float) / window
        smoothed = np.convolve(region, kernel, mode='same')
        result[start_idx:end_idx] = smoothed
    return result


def interpolate_region(x: ArrayLike, start_idx: int, end_idx: int, method: str = "linear") -> np.ndarray:
    """Interpolate values in a specific region using specified method.

    Parameters
    ----------
    x : array_like
        Input time series.
    start_idx : int
        Start index of the region to interpolate (inclusive).
    end_idx : int
        End index of the region to interpolate (exclusive).
    method : {'linear', 'nearest', 'zero'}, optional
        Interpolation method. Default is 'linear'.

    Returns
    -------
    ndarray
        Time series with interpolated region.
    """
    x_arr = _to_numpy(x, dtype=float)
    if start_idx < 0 or end_idx > len(x_arr) or start_idx >= end_idx:
        raise ValueError("Invalid region indices")
    
    result = x_arr.copy()
    
    if start_idx > 0 and end_idx < len(x_arr):
        left_val = result[start_idx - 1]
        right_val = result[end_idx]
        region_length = end_idx - start_idx
        
        if method == "linear":
            interpolated = np.linspace(left_val, right_val, region_length + 2)[1:-1]
        elif method == "nearest":
            mid_point = region_length // 2
            interpolated = np.where(np.arange(region_length) < mid_point, left_val, right_val)
        elif method == "zero":
            interpolated = np.full(region_length, left_val)
        else:
            raise ValueError("method must be 'linear', 'nearest', or 'zero'")
        
        result[start_idx:end_idx] = interpolated
    elif start_idx > 0:
        result[start_idx:end_idx] = result[start_idx - 1]
    elif end_idx < len(x_arr):
        result[start_idx:end_idx] = result[end_idx]
    
    return result


def remove_anomalies_in_region(
    x: ArrayLike,
    start_idx: int,
    end_idx: int,
    method: str = "zscore",
    threshold: float = 3.0,
    fill_method: str = "interpolate"
) -> np.ndarray:
    """Remove anomalies in a specific region and fill them.

    Parameters
    ----------
    x : array_like
        Input time series.
    start_idx : int
        Start index of the region (inclusive).
    end_idx : int
        End index of the region (exclusive).
    method : {'zscore', 'iqr'}, optional
        Method for detecting anomalies. Default is 'zscore'.
    threshold : float, optional
        Threshold for anomaly detection. Default is 3.0.
    fill_method : {'interpolate', 'mean', 'median'}, optional
        Method to fill removed anomalies. Default is 'interpolate'.

    Returns
    -------
    ndarray
        Time series with anomalies removed and filled.
    """
    x_arr = _to_numpy(x, dtype=float)
    if start_idx < 0 or end_idx > len(x_arr) or start_idx >= end_idx:
        raise ValueError("Invalid region indices")
    
    result = x_arr.copy()
    region = result[start_idx:end_idx]
    
    if method == "zscore":
        mu = np.nanmean(region)
        sigma = np.nanstd(region)
        if sigma == 0 or np.isnan(sigma):
            anomaly_mask = np.zeros_like(region, dtype=bool)
        else:
            z = np.abs((region - mu) / sigma)
            anomaly_mask = z > threshold
    elif method == "iqr":
        q1, q3 = np.nanquantile(region, [0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - threshold * iqr, q3 + threshold * iqr
        anomaly_mask = (region < lower) | (region > upper)
    else:
        raise ValueError("method must be 'zscore' or 'iqr'")
    
    if np.any(anomaly_mask):
        anomaly_indices = np.where(anomaly_mask)[0]
        global_indices = anomaly_indices + start_idx
        
        if fill_method == "interpolate":
            result = interpolate_region(result, global_indices[0], global_indices[-1] + 1, method="linear")
        elif fill_method == "mean":
            valid_values = region[~anomaly_mask]
            fill_value = np.nanmean(valid_values) if len(valid_values) > 0 else 0
            result[global_indices] = fill_value
        elif fill_method == "median":
            valid_values = region[~anomaly_mask]
            fill_value = np.nanmedian(valid_values) if len(valid_values) > 0 else 0
            result[global_indices] = fill_value
        else:
            raise ValueError("fill_method must be 'interpolate', 'mean', or 'median'")
    
    return result


def apply_trend_in_region(
    x: ArrayLike,
    start_idx: int,
    end_idx: int,
    slope: float,
    offset: float = 0.0
) -> np.ndarray:
    """Apply a linear trend to a specific region of the time series.

    Parameters
    ----------
    x : array_like
        Input time series.
    start_idx : int
        Start index of the region (inclusive).
    end_idx : int
        End index of the region (exclusive).
    slope : float
        Slope of the linear trend.
    offset : float, optional
        Vertical offset at the start of the region. Default is 0.0.

    Returns
    -------
    ndarray
        Time series with trend applied to the region.
    """
    x_arr = _to_numpy(x, dtype=float)
    if start_idx < 0 or end_idx > len(x_arr) or start_idx >= end_idx:
        raise ValueError("Invalid region indices")
    
    result = x_arr.copy()
    region_length = end_idx - start_idx
    trend_values = offset + slope * np.arange(region_length, dtype=float)
    result[start_idx:end_idx] = trend_values
    return result


def scale_region(
    x: ArrayLike,
    start_idx: int,
    end_idx: int,
    scale_factor: float,
    center: bool = False
) -> np.ndarray:
    """Scale values in a specific region of the time series.

    Parameters
    ----------
    x : array_like
        Input time series.
    start_idx : int
        Start index of the region (inclusive).
    end_idx : int
        End index of the region (exclusive).
    scale_factor : float
        Scaling factor to apply.
    center : bool, optional
        If True, scale around the region mean. Default is False.

    Returns
    -------
    ndarray
        Time series with scaled region.
    """
    x_arr = _to_numpy(x, dtype=float)
    if start_idx < 0 or end_idx > len(x_arr) or start_idx >= end_idx:
        raise ValueError("Invalid region indices")
    
    result = x_arr.copy()
    region = result[start_idx:end_idx].copy()
    
    if center:
        region_mean = np.nanmean(region)
        region = (region - region_mean) * scale_factor + region_mean
    else:
        region = region * scale_factor
    
    result[start_idx:end_idx] = region
    return result


def adjust_trend_in_region(
    x: ArrayLike,
    start_idx: int,
    end_idx: int,
    factor: float = 1.5
) -> np.ndarray:
    """Adjust the trend in a specific region of time series.

    Parameters
    ----------
    x : array_like
        Input time series.
    start_idx : int
        Start index of the region (inclusive).
    end_idx : int
        End index of the region (exclusive).
    factor : float, optional
        Factor to adjust the trend by. Values > 1 increase trend,
        values < 1 decrease trend. Default is 1.5.

    Returns
    -------
    ndarray
        Time series with adjusted trend in region.
    """
    x_arr = _to_numpy(x, dtype=float)
    if start_idx < 0 or end_idx > len(x_arr) or start_idx >= end_idx:
        raise ValueError("Invalid region indices")
    
    result = x_arr.copy()
    region = result[start_idx:end_idx].copy()
    
    x = np.arange(len(region), dtype=float)
    coeffs = np.polyfit(x, region, 1)
    current_trend = coeffs[0]
    new_trend = current_trend * factor
    
    trend_component = current_trend * x
    new_trend_component = new_trend * x
    
    detrended = region - trend_component
    new_region = detrended + new_trend_component
    
    result[start_idx:end_idx] = new_region
    return result


def increase_trend(
    x: ArrayLike,
    start_idx: int,
    end_idx: int,
    factor: float = 1.5
) -> np.ndarray:
    """Increase the trend in a specific region of time series.

    Parameters
    ----------
    x : array_like
        Input time series.
    start_idx : int
        Start index of the region (inclusive).
    end_idx : int
        End index of the region (exclusive).
    factor : float, optional
        Factor to increase the trend by (factor > 1). Default is 1.5.

    Returns
    -------
    ndarray
        Time series with increased trend in region.
    """
    return adjust_trend_in_region(x, start_idx, end_idx, factor)


def decrease_trend(
    x: ArrayLike,
    start_idx: int,
    end_idx: int,
    factor: float = 0.5
) -> np.ndarray:
    """Decrease the trend in a specific region of time series.

    Parameters
    ----------
    x : array_like
        Input time series.
    start_idx : int
        Start index of the region (inclusive).
    end_idx : int
        End index of the region (exclusive).
    factor : float, optional
        Factor to decrease the trend by (0 < factor < 1). Default is 0.5.

    Returns
    -------
    ndarray
        Time series with decreased trend in region.
    """
    return adjust_trend_in_region(x, start_idx, end_idx, factor)


def adjust_volatility_in_region(
    x: ArrayLike,
    start_idx: int,
    end_idx: int,
    factor: float = 1.5
) -> np.ndarray:
    """Adjust volatility in a specific region of time series.

    Parameters
    ----------
    x : array_like
        Input time series.
    start_idx : int
        Start index of the region (inclusive).
    end_idx : int
        End index of the region (exclusive).
    factor : float, optional
        Factor to adjust volatility by. Values > 1 increase volatility,
        values < 1 decrease volatility. Default is 1.5.

    Returns
    -------
    ndarray
        Time series with adjusted volatility in region.
    """
    x_arr = _to_numpy(x, dtype=float)
    if start_idx < 0 or end_idx > len(x_arr) or start_idx >= end_idx:
        raise ValueError("Invalid region indices")
    
    result = x_arr.copy()
    region = result[start_idx:end_idx].copy()
    
    region_mean = np.nanmean(region)
    deviations = region - region_mean
    
    new_region = region_mean + deviations * factor
    
    result[start_idx:end_idx] = new_region
    return result


def increase_volatility(
    x: ArrayLike,
    start_idx: int,
    end_idx: int,
    factor: float = 1.5
) -> np.ndarray:
    """Increase the volatility in a specific region of time series.

    Parameters
    ----------
    x : array_like
        Input time series.
    start_idx : int
        Start index of region (inclusive).
    end_idx : int
        End index of region (exclusive).
    factor : float, optional
        Factor to increase the volatility by (factor > 1). Default is 1.5.

    Returns
    -------
    ndarray
        Time series with increased volatility in region.
    """
    return adjust_volatility_in_region(x, start_idx, end_idx, factor)


def decrease_volatility(
    x: ArrayLike,
    start_idx: int,
    end_idx: int,
    factor: float = 0.5
) -> np.ndarray:
    """Decrease the volatility in a specific region of time series.

    Parameters
    ----------
    x : array_like
        Input time series.
    start_idx : int
        Start index of region (inclusive).
    end_idx : int
        End index of region (exclusive).
    factor : float, optional
        Factor to decrease the volatility by (0 < factor < 1). Default is 0.5.

    Returns
    -------
    ndarray
        Time series with decreased volatility in region.
    """
    return adjust_volatility_in_region(x, start_idx, end_idx, factor)
