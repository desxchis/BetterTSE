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
from typing import Callable, Iterable, Sequence, Union, Optional

from modules.utils import timestamps_to_numeric

ArrayLike = Union[np.ndarray, Sequence[float], Sequence[int]]


def _to_numpy(arr: ArrayLike, dtype: Optional[np.dtype] = None) -> np.ndarray:
    """Convert input to a 1-D NumPy array without copying if already an array.

    Parameters
    ----------
    arr : array_like, required
        Input sequence.
    dtype : numpy dtype, optional
        Desired dtype of returned array. If None, infer from input.

    Returns
    -------
    ndarray
        1-D NumPy array view or copy of the input.
    """
    if isinstance(arr, np.ndarray):
        out = arr if dtype is None or arr.dtype == dtype else arr.astype(dtype)
    else:
        out = np.asarray(arr, dtype=dtype)
    if out.ndim != 1:
        out = out.reshape(-1)
    return out


# --- simple deterministic forecasts ---

def repeat_last_value_forecast(x: ArrayLike, horizon: int) -> np.ndarray:
    """Naïve forecast using the last observed value.

    Repeats the last finite (non-NaN, non-inf) observation from ``x`` for the
    next ``horizon`` steps.

    Parameters
    ----------
    x : array_like, required
        Historical series.
    horizon : int, required
        Number of future steps to forecast. Must be non-negative.

    Returns
    -------
    ndarray
        Forecast of length ``horizon`` filled with the last finite value.

    Raises
    ------
    ValueError
        If ``x`` contains no finite values.
    """
    x_arr = _to_numpy(x, dtype=float)
    finite = x_arr[np.isfinite(x_arr)]
    if finite.size == 0:
        raise ValueError("x contains no finite values")
    return np.full(horizon, finite[-1], dtype=float)


def repeat_last_season_forecast(x: ArrayLike, horizon: int, seasonal_period: int) -> np.ndarray:
    """Seasonal naïve forecast by repeating the latest season.

    Extracts the last complete seasonal pattern of length ``seasonal_period``
    from the finite portion of ``x`` and repeats it to cover ``horizon`` steps.

    Parameters
    ----------
    x : array_like, required
        Historical series.
    horizon : int, required
        Number of future steps to forecast. Must be non-negative.
    seasonal_period : int, required
        Seasonal cycle length. Must not exceed the number of finite
        observations in ``x``.

    Returns
    -------
    ndarray
        Forecast of length ``horizon`` constructed by tiling the last season.

    Raises
    ------
    ValueError
        If ``seasonal_period`` is larger than the count of finite observations.
    """
    x_arr = _to_numpy(x, dtype=float)
    finite = x_arr[np.isfinite(x_arr)]
    n = finite.size
    if seasonal_period > n:
        raise ValueError(
            "seasonal_period larger than number of valid observations")
    pattern = finite[-seasonal_period:]
    reps = int(np.ceil(horizon / seasonal_period))
    return np.tile(pattern, reps)[:horizon].astype(float)


def linear_drift_forecast(x: ArrayLike, horizon: int) -> np.ndarray:
    """Drift forecast via linear extrapolation.

    Fits a line between the first and last finite observations in ``x`` and
    extrapolates that trend forward for ``horizon`` steps.

    Parameters
    ----------
    x : array_like, required
        Historical series.
    horizon : int, required
        Number of future steps to forecast. Must be non-negative.

    Returns
    -------
    ndarray
        Forecast of length ``horizon`` obtained by linear drift.

    Raises
    ------
    ValueError
        If fewer than two finite observations are available.
    """
    x_arr = _to_numpy(x, dtype=float)
    finite = x_arr[np.isfinite(x_arr)]
    n = finite.size
    if n < 2:
        raise ValueError("need at least two observations for drift")
    slope = (finite[-1] - finite[0]) / (n - 1)
    return finite[-1] + slope * np.arange(1, horizon + 1, dtype=float)


def trailing_mean_forecast(x: ArrayLike, horizon: int, window: int) -> np.ndarray:
    """Flat forecast from the recent moving average.

    Uses the mean of the last ``window`` finite observations from ``x`` as a
    constant forecast over the next ``horizon`` steps.

    Parameters
    ----------
    x : array_like, required
        Historical series.
    horizon : int, required
        Number of future steps to forecast. Must be non-negative.
    window : int, required
        Number of most recent finite observations to average. Must be >= 1
        and not exceed the number of finite observations in ``x``.

    Returns
    -------
    ndarray
        Forecast of length ``horizon`` filled with the recent mean.

    Raises
    ------
    ValueError
        If ``window`` is larger than the number of finite observations.
    """
    x_arr = _to_numpy(x, dtype=float)
    finite = x_arr[np.isfinite(x_arr)]
    if window > finite.size:
        raise ValueError("window larger than number of valid observations")
    mean_val = float(np.mean(finite[-window:]))
    return np.full(horizon, mean_val, dtype=float)


def simple_exp_smoothing_forecast(x: ArrayLike, horizon: int, alpha: Optional[float] = None) -> np.ndarray:
    """Simple Exponential Smoothing (SES) with fixed final level.

    Computes an SES level from the finite observations in ``x`` using smoothing
    factor ``alpha`` and returns a constant forecast equal to that level for
    ``horizon`` steps. If ``alpha`` is not provided, it defaults to
    ``2 / (n + 1)`` where ``n`` is the number of finite observations.

    Parameters
    ----------
    x : array_like, required
        Historical series.
    horizon : int, required
        Number of future steps to forecast. Must be non-negative.
    alpha : float, optional
        Smoothing factor in ``(0, 1]``. If ``None``, uses ``2/(n+1)``.

    Returns
    -------
    ndarray
        Constant forecast of length ``horizon`` equal to the final SES level.

    Raises
    ------
    ValueError
        If ``x`` contains no finite values.
    """
    x_arr = _to_numpy(x, dtype=float)
    finite = x_arr[np.isfinite(x_arr)]
    n = finite.size
    if n == 0:
        raise ValueError("x contains no finite values")
    if alpha is None:
        alpha = 2.0 / (n + 1)
    level = float(finite[0])
    for v in finite[1:]:
        level = alpha * float(v) + (1.0 - alpha) * level
    return np.full(horizon, level, dtype=float)


def constant_signal(t: ArrayLike, value: float) -> np.ndarray:
    """Generate a constant (flat) signal.

    Given an array of times ``t``, return an array of the same shape filled
    with the constant ``value``. Useful as a baseline or offset.

    Parameters
    ----------
    t : array_like, required
        1-D time points. Only the shape is used.
    value : float, required
        Constant value to fill.

    Returns
    -------
    ndarray
        Array of constant values with the same shape as ``t``.
    """
    t_arr = _to_numpy(t)
    return np.full_like(t_arr, fill_value=value, dtype=float)


def linear_trend(t: ArrayLike, slope: float, intercept: float) -> np.ndarray:
    """Generate a linear trend.

    Computes ``intercept + slope * t`` elementwise.

    Parameters
    ----------
    t : array_like, required
        1-D time points.
    slope : float, required
        Coefficient of the linear term.
    intercept : float, required
        Constant offset.

    Returns
    -------
    ndarray
        Linearly increasing or decreasing sequence.
    """
    t_arr = _to_numpy(t, dtype=float)
    return intercept + slope * t_arr


def changepoint_linear_trend(
    t: ArrayLike,
    knots: Sequence[float],
    deltas: Sequence[float],
    offset: float = 0.0,
) -> np.ndarray:
    """Piecewise linear trend with changepoints.

    Constructs a function with zero intercept that changes its slope by
    ``delta[i]`` at each ``knots[i]``. The knots must be in ascending order.
    Combine with ``linear_trend`` or ``constant_signal`` to add a baseline slope or offset
    when needed.

    Parameters
    ----------
    t : array_like, required
        1-D time points.
    knots : sequence of float, required
        Monotonically increasing changepoints.
    deltas : sequence of float, required
        Slope changes at each knot. Must have the same length as ``knots``.

    offset : float, optional
        Constant baseline added to the piecewise trend. Default 0.0.

    Returns
    -------
    ndarray
        Piecewise linear sequence shifted by ``offset``.
    """
    t_arr = _to_numpy(t, dtype=float)
    if len(knots) != len(deltas):
        raise ValueError("knots and deltas must have the same length")
    y = np.zeros_like(t_arr, dtype=float)
    cumulative_slopes = np.cumsum(deltas)
    for tau, cum_delta in zip(knots, cumulative_slopes):
        y += cum_delta * np.maximum(0.0, t_arr - tau)
    return y + offset


def sigmoid_transition(
    t: ArrayLike,
    center: float,
    width: float,
    low: float,
    high: float,
) -> np.ndarray:
    """Smooth transition between ``low`` and ``high`` using a logistic profile."""
    if width <= 0:
        raise ValueError("width must be positive")
    t_arr = _to_numpy(t, dtype=float)
    scale = (t_arr - float(center)) / float(width)
    return float(low) + (float(high) - float(low)) / (1.0 + np.exp(-scale))


def sine_wave(
    t: ArrayLike,
    period: float,
    amplitude: float,
    phase: float = 0.0,
    offset: float = 0.0,
) -> np.ndarray:
    """Simple sinusoidal (sine) wave.

    Produces a smooth periodic oscillation:

        y(t) = amplitude * sin(2π / period * t + phase)

    ``phase`` is in radians. To create a cosine wave, set ``phase = π/2``.

    Parameters
    ----------
    t : array_like, required
        1-D time points.
    period : float, required
        Length of one full cycle in the units of ``t``.
    amplitude : float, required
        Amplitude of the wave.
    phase : float, optional
        Phase offset in radians. Default 0.0.

    offset : float, optional
        Baseline added to the sinusoid. Default 0.0.

    Returns
    -------
    ndarray
        Sinusoid evaluated at ``t`` and shifted by ``offset``.
    """
    t_arr = _to_numpy(t, dtype=float)
    return offset + amplitude * np.sin(2.0 * np.pi * t_arr / period + phase)


def fourier_series_seasonality(
    t: ArrayLike,
    period: float,
    n_harmonics: int,
    cos_coeffs: Sequence[float],
    sin_coeffs: Sequence[float],
    offset: float = 0.0,
) -> np.ndarray:
    """Seasonal signal as a sum of sine and cosine harmonics.

        y(t) = sum_{h=1..n_harmonics} [ cos_coeffs[h-1]*cos(2π h t / period)
                                        + sin_coeffs[h-1]*sin(2π h t / period) ]

    Parameters
    ----------
    t : array_like, required
        1-D time points.
    period : float, required
        Fundamental period of the seasonality.
    n_harmonics : int, required
        Number of harmonics (>= 1).
    cos_coeffs : sequence of float, required
        Cosine coefficients of length ``n_harmonics``.
    sin_coeffs : sequence of float, required
        Sine coefficients of length ``n_harmonics``.

    offset : float, optional
        Constant baseline added after summing harmonics. Default 0.0.

    Returns
    -------
    ndarray
        Composite seasonal signal shifted by ``offset``.
    """
    if n_harmonics < 1:
        raise ValueError("n_harmonics must be at least 1")
    if len(cos_coeffs) != n_harmonics or len(sin_coeffs) != n_harmonics:
        raise ValueError(
            "cos_coeffs and sin_coeffs must have length n_harmonics")
    t_arr = _to_numpy(t, dtype=float)
    y = np.zeros_like(t_arr, dtype=float)
    for h in range(1, n_harmonics + 1):
        omega = 2.0 * np.pi * h / period
        y += cos_coeffs[h - 1] * np.cos(omega * t_arr) + \
            sin_coeffs[h - 1] * np.sin(omega * t_arr)
    return y + offset


def sum_of_sinusoids(
    t: ArrayLike,
    freqs: Sequence[float],
    amplitudes: Sequence[float],
    phases: Sequence[float],
    offset: float = 0.0,
) -> np.ndarray:
    """Combine arbitrary sinusoids defined by frequency, amplitude and phase."""
    if not (len(freqs) == len(amplitudes) == len(phases)):
        raise ValueError("freqs, amplitudes and phases must have the same length")
    t_arr = _to_numpy(t, dtype=float)
    y = np.full_like(t_arr, fill_value=float(offset), dtype=float)
    for freq, amp, phase in zip(freqs, amplitudes, phases):
        y += float(amp) * np.sin(2.0 * np.pi * float(freq) * t_arr + float(phase))
    return y


def periodic_pulse_train(
    t: ArrayLike,
    period: float,
    width: float,
    amplitude: float,
    phase_offset: float = 0.0,
    offset: float = 0.0,
) -> np.ndarray:
    """Generate a periodic rectangular pulse train with fixed amplitude."""
    if period <= 0:
        raise ValueError("period must be positive")
    if width < 0:
        raise ValueError("width must be non-negative")
    t_arr = _to_numpy(t, dtype=float)
    effective_width = min(float(width), float(period))
    phase = (t_arr + float(phase_offset)) % float(period)
    mask = phase < effective_width
    return float(offset) + float(amplitude) * mask.astype(float)


def delayed_step_signal(
    t: ArrayLike,
    t0: float,
    amplitude: float,
    offset: float = 0.0,
) -> np.ndarray:
    """Step function (Heaviside-like).

    Returns an array that is 0 for ``t < t0`` and ``amplitude`` for
    ``t >= t0``.

    Parameters
    ----------
    t : array_like, required
        1-D time points.
    t0 : float, required
        Step location.
    amplitude : float, required
        Magnitude of the step.

    offset : float, optional
        Baseline added to the step function. Default 0.0.

    Returns
    -------
    ndarray
        Step function values at ``t`` shifted by ``offset``.
    """
    t_arr = _to_numpy(t, dtype=float)
    return np.where(t_arr >= t0, amplitude, 0.0) + offset


def delayed_ramp_signal(
    t: ArrayLike,
    t0: float,
    slope: float,
    offset: float = 0.0,
) -> np.ndarray:
    """Ramp function.

    Starts at zero for ``t < t0`` and increases linearly with slope
    thereafter: ``slope * max(0, t - t0)``.

    Parameters
    ----------
    t : array_like, required
        1-D time points.
    t0 : float, required
        Onset time of the ramp.
    slope : float, required
        Rate of increase after ``t0``.

    offset : float, optional
        Baseline added to the ramp. Default 0.0.

    Returns
    -------
    ndarray
        Ramp values at ``t`` shifted by ``offset``.
    """
    t_arr = _to_numpy(t, dtype=float)
    return slope * np.maximum(0.0, t_arr - t0) + offset


def unit_boxcar_pulse(
    t: ArrayLike,
    center: float,
    width: float,
    offset: float = 0.0,
) -> np.ndarray:
    """Rectangular (boxcar) pulse.

    Generates a unit-height window centred at ``center`` and spanning ``width``
    units of time (symmetric).

    Parameters
    ----------
    t : array_like, required
        1-D time points.
    center : float, required
        Centre of the boxcar.
    width : float, required
        Full width of the pulse. Must be non-negative.

    offset : float, optional
        Baseline added to the boxcar. Default 0.0.

    Returns
    -------
    ndarray
        Boxcar values at ``t`` shifted by ``offset``.
    """
    if width < 0:
        raise ValueError("width must be non-negative")
    t_arr = _to_numpy(t, dtype=float)
    half = width / 2.0
    return np.where(np.abs(t_arr - center) <= half, 1.0, 0.0) + offset


def unit_gaussian_pulse(
    t: ArrayLike,
    mu: float,
    sigma: float,
    offset: float = 0.0,
) -> np.ndarray:
    """Gaussian bell curve centred at ``mu``.

        y(t) = exp(-0.5 * ((t - mu) / sigma)**2)

    Parameters
    ----------
    t : array_like, required
        1-D time points.
    mu : float, required
        Centre of the pulse.
    sigma : float, required
        Standard deviation. Must be > 0.

    offset : float, optional
        Baseline added to the Gaussian pulse. Default 0.0.

    Returns
    -------
    ndarray
        Gaussian pulse values shifted by ``offset``.
    """
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    t_arr = _to_numpy(t, dtype=float)
    z = (t_arr - mu) / sigma
    return np.exp(-0.5 * z * z) + offset


def unit_exponential_decay_pulse(
    t: ArrayLike,
    t0: float,
    tau: float,
    offset: float = 0.0,
) -> np.ndarray:
    """Exponential decay starting at ``t0``.

        y(t) = exp(-(t - t0) / tau) * 1_{t >= t0}

    Parameters
    ----------
    t : array_like, required
        1-D time points.
    t0 : float, required
        Onset time.
    tau : float, required
        Decay time constant (must be positive).

    offset : float, optional
        Baseline added to the exponential decay. Default 0.0.

    Returns
    -------
    ndarray
        Exponentially decaying signal shifted by ``offset``.
    """
    if tau <= 0:
        raise ValueError("tau must be positive")
    t_arr = _to_numpy(t, dtype=float)
    z = np.maximum(0.0, t_arr - t0)
    return np.exp(-z / tau) * (t_arr >= t0) + offset


def kernel_shot_noise_signal(
    t: ArrayLike,
    events: Sequence[float],
    kernel_fn: Callable[[np.ndarray], np.ndarray],
    offset: float = 0.0,
) -> np.ndarray:
    """Generate a shot noise signal from fixed event times.

    Each event at time ``e`` contributes a kernel centred at ``e``. The
    kernel function ``kernel_fn`` should accept an array of delays ``(t - e)``
    and return an array of the same shape.

    Parameters
    ----------
    t : array_like, required
        1-D time points.
    events : sequence of float, required
        Times at which pulses occur.
    kernel_fn : callable, required
        Function mapping delays (t - event) to pulse shape values.

    offset : float, optional
        Baseline added after summing all kernel contributions. Default 0.0.

    Returns
    -------
    ndarray
        Shot noise signal at ``t`` shifted by ``offset``.
    """
    t_arr = _to_numpy(t, dtype=float)
    y = np.zeros_like(t_arr, dtype=float)
    if not callable(kernel_fn):
        raise ValueError("kernel_fn must be callable")
    for e in events:
        delays = t_arr - float(e)
        contribution = kernel_fn(delays)
        if contribution.shape != y.shape:
            raise ValueError(
                "kernel_fn must return array of same shape as input")
        y += contribution
    return y + offset


def apply_amplitude_envelope(x: ArrayLike, envelope: ArrayLike) -> np.ndarray:
    """Apply an amplitude envelope to a base signal.

    Elementwise multiplication of ``x`` and ``envelope``. Both inputs
    must be the same length.

    Parameters
    ----------
    x : array_like, required
        Base signal.
    envelope : array_like, required
        Scaling factors per sample. Same shape as ``x``.

    Returns
    -------
    ndarray
        Modulated signal ``x * envelope``.
    """
    x_arr = _to_numpy(x, dtype=float)
    env_arr = _to_numpy(envelope, dtype=float)
    if x_arr.shape != env_arr.shape:
        raise ValueError("x and envelope must have the same shape")
    return x_arr * env_arr


def periodic_duty_cycle_mask(
    t: ArrayLike,
    period: float,
    duty_cycle: float,
    phase_offset: float = 0.0,
    offset: float = 0.0,
) -> np.ndarray:
    """Deterministic on/off mask with a duty cycle.

    Returns a 0/1 mask that is 1 for a fraction ``duty_cycle`` of each period
    and 0 otherwise. ``phase_offset`` shifts the start of the active window
    in the same time units as ``t``.

    Parameters
    ----------
    t : array_like, required
        1-D time points.
    period : float, required
        Length of one cycle.
    duty_cycle : float, required
        Fraction of the period during which the mask is 1. Must satisfy 0 < duty_cycle <= 1.
    phase_offset : float, optional
        Phase shift in time units (same units as ``t``). Default 0.0.

    offset : float, optional
        Baseline added to the mask values. Default 0.0.

    Returns
    -------
    ndarray
        Mask array shifted by ``offset``.
    """
    if not (0.0 < duty_cycle <= 1.0):
        raise ValueError("duty_cycle must be in the interval (0, 1]")
    t_arr = _to_numpy(t, dtype=float)
    frac = ((t_arr + phase_offset) % period) / period
    return (frac < duty_cycle).astype(float) + offset


def calendar_rule_mask(
    dates: ArrayLike,
    rule: str,
    holidays: Optional[Iterable[np.datetime64]] = None,
) -> np.ndarray:
    """Generate a deterministic mask based on calendar rules.

    Accepts numpy ``datetime64`` values and returns a 0/1 mask according to ``rule``:

    - 'weekend'    : 1 on Saturdays/Sundays, 0 otherwise
    - 'weekday'    : 1 on Monday–Friday, 0 on weekends
    - 'month_end'  : 1 on the last calendar day of each month
    - 'month_start': 1 on the first calendar day of each month
    - 'holiday'    : 1 on given ``holidays`` list/set

    Parameters
    ----------
    dates : array_like of datetime64, required
        Dates or times.
    rule : str, required
        One of 'weekend', 'weekday', 'month_end', 'month_start', 'holiday'.
    holidays : iterable of datetime64, optional
        Required if ``rule='holiday'``. Specifies which dates to mark as holidays.

    Returns
    -------
    ndarray
        0/1 mask array of the same shape as ``dates``.
    """
    d_arr = _to_numpy(dates)
    if d_arr.dtype.kind not in {'M'}:
        raise TypeError("dates must be a numpy datetime64 array")
    days = d_arr.astype('datetime64[D]')
    days_int = days.astype('int64')
    dow = (days_int + 4) % 7  # Monday=0, Sunday=6
    if rule == 'weekend':
        return ((dow >= 5).astype(float))
    elif rule == 'weekday':
        return ((dow < 5).astype(float))
    elif rule == 'month_end':
        next_day = (days + np.timedelta64(1, 'D')).astype('datetime64[M]')
        this_month = days.astype('datetime64[M]')
        return ((next_day != this_month).astype(float))
    elif rule == 'month_start':
        prev_day = (days - np.timedelta64(1, 'D')).astype('datetime64[M]')
        this_month = days.astype('datetime64[M]')
        return ((prev_day != this_month).astype(float))
    elif rule == 'holiday':
        if holidays is None:
            raise ValueError("holidays must be provided when rule='holiday'")
        holiday_set = {np.datetime64(h).astype(
            'datetime64[D]') for h in holidays}
        return np.array([1.0 if d in holiday_set else 0.0 for d in days.astype('datetime64[D]')], dtype=float)
    else:
        raise ValueError(f"Unsupported rule '{rule}'")


def moving_average_filter(x: ArrayLike, window: int) -> np.ndarray:
    """Compute a simple moving average (boxcar filter).

    Parameters
    ----------
    x : array_like, required
        1-D signal to smooth.
    window : int, required
        Width of the averaging window (>= 1).

    Returns
    -------
    ndarray
        Smoothed signal (same length as ``x``).
    """
    if window < 1:
        raise ValueError("window must be >= 1")
    x_arr = _to_numpy(x, dtype=float)
    if window == 1:
        return x_arr.copy()
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(x_arr, kernel, mode='same')


def exponential_smoothing_filter(x: ArrayLike, alpha: float) -> np.ndarray:
    """Exponential smoothing of a time series.

    Recurrence:
        y[0] = x[0]
        y[t] = alpha * x[t] + (1 - alpha) * y[t-1]

    Parameters
    ----------
    x : array_like, required
        Input signal.
    alpha : float, required
        Smoothing parameter, 0 < alpha <= 1.

    Returns
    -------
    ndarray
        Smoothed signal.
    """
    if not (0.0 < alpha <= 1.0):
        raise ValueError("alpha must be in (0, 1]")
    x_arr = _to_numpy(x, dtype=float)
    y = np.empty_like(x_arr, dtype=float)
    y[0] = x_arr[0]
    for i in range(1, len(x_arr)):
        y[i] = alpha * x_arr[i] + (1.0 - alpha) * y[i - 1]
    return y


def convolve_1d_signal(x: ArrayLike, kernel: ArrayLike, mode: str = 'same') -> np.ndarray:
    """Perform linear convolution with an arbitrary kernel.

    Wrapper around `np.convolve`.

    Parameters
    ----------
    x : array_like, required
        Input signal to be filtered.
    kernel : array_like, required
        Convolution kernel (impulse response).
    mode : {'full', 'valid', 'same'}, optional
        Convolution mode passed to `np.convolve`. Default 'same'.

    Returns
    -------
    ndarray
        Convolved signal.
    """
    x_arr = _to_numpy(x, dtype=float)
    k_arr = _to_numpy(kernel, dtype=float)
    return np.convolve(x_arr, k_arr, mode=mode)


def difference_signal(x: ArrayLike, order: int = 1) -> np.ndarray:
    """Compute discrete differences of a signal.

    Equivalent to repeated application of `np.diff`. Output length is
    ``len(x) - order``.

    Parameters
    ----------
    x : array_like, required
        Input signal.
    order : int, optional
        Number of times to apply differencing (>= 1 and < len(x)). Default 1.

    Returns
    -------
    ndarray
        Differenced signal.
    """
    if order < 1:
        raise ValueError("order must be >= 1")
    x_arr = _to_numpy(x, dtype=float)
    if order >= len(x_arr):
        raise ValueError("order must be less than length of x")
    y = x_arr.copy()
    for _ in range(order):
        y = np.diff(y)
    return y


def cumulative_sum_signal(x: ArrayLike) -> np.ndarray:
    """Cumulative sum (discrete integration).

    Parameters
    ----------
    x : array_like, required
        Input increments or values to integrate.

    Returns
    -------
    ndarray
        Cumulative sum of ``x``.
    """
    x_arr = _to_numpy(x, dtype=float)
    return np.cumsum(x_arr)


def blockwise_aggregate(
    x: ArrayLike,
    block_size: int,
    agg: str = 'mean',
) -> np.ndarray:
    """Aggregate a signal in non-overlapping blocks.

    Downsamples ``x`` by grouping consecutive samples into blocks of
    length ``block_size`` and applying an aggregation function:
    'mean', 'sum', 'max', or 'min'. If the last block is partial, it is included.

    Parameters
    ----------
    x : array_like, required
        Input signal.
    block_size : int, required
        Number of samples per block (>= 1).
    agg : {'mean','sum','max','min'}, optional
        Aggregation operation. Default 'mean'.

    Returns
    -------
    ndarray
        Downsampled signal.
    """
    if block_size < 1:
        raise ValueError("block_size must be >= 1")
    x_arr = _to_numpy(x, dtype=float)
    n = len(x_arr)
    num_blocks = (n + block_size - 1) // block_size
    result = np.empty(num_blocks, dtype=float)
    for i in range(num_blocks):
        start = i * block_size
        end = min(start + block_size, n)
        block = x_arr[start:end]
        if agg == 'mean':
            result[i] = block.mean()
        elif agg == 'sum':
            result[i] = block.sum()
        elif agg == 'max':
            result[i] = block.max()
        elif agg == 'min':
            result[i] = block.min()
        else:
            raise ValueError(f"Unsupported agg '{agg}'")
    return result


def resample_to_time_index(
    x: ArrayLike,
    t: ArrayLike,
    new_t: ArrayLike,
    method: str = 'linear',
) -> np.ndarray:
    """Resample a signal onto a new time grid by interpolation.

    Supported methods:
    - 'linear'  : linear interpolation (default)
    - 'nearest' : nearest neighbour
    - 'zero'    : zero-order hold (previous value)

    Values outside the original domain are extrapolated by endpoint hold.

    Parameters
    ----------
    x : array_like, required
        Values of the signal at times ``t``.
    t : array_like, required
        Times corresponding to ``x`` (sorted ascending).
    new_t : array_like, required
        Target times (sorted ascending).
    method : {'linear','nearest','zero'}, optional
        Interpolation method. Default 'linear'.

    Returns
    -------
    ndarray
        Interpolated values at ``new_t``.
    """
    x_arr = _to_numpy(x, dtype=float)
    t_arr = _to_numpy(t, dtype=float)
    new_t_arr = _to_numpy(new_t, dtype=float)
    if len(x_arr) != len(t_arr):
        raise ValueError("x and t must have the same length")
    if len(x_arr) == 0:
        return np.zeros_like(new_t_arr, dtype=float)
    if not np.all(np.diff(t_arr) >= 0):
        raise ValueError("t must be sorted ascending")
    if method == 'linear':
        return np.interp(new_t_arr, t_arr, x_arr)
    elif method == 'zero':
        indices = np.searchsorted(t_arr, new_t_arr, side='right') - 1
        indices = np.clip(indices, 0, len(x_arr) - 1)
        return x_arr[indices]
    elif method == 'nearest':
        idx = np.searchsorted(t_arr, new_t_arr)
        idx_left = np.clip(idx - 1, 0, len(t_arr) - 1)
        idx_right = np.clip(idx, 0, len(t_arr) - 1)
        left_diff = np.abs(new_t_arr - t_arr[idx_left])
        right_diff = np.abs(t_arr[idx_right] - new_t_arr)
        choose_right = right_diff < left_diff
        indices = np.where(choose_right, idx_right, idx_left)
        return x_arr[indices]
    else:
        raise ValueError(f"Unsupported method '{method}'")


def apply_time_warp(t: ArrayLike, warp_fn: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    """Apply a monotonic time reparameterisation.

    Given a time vector ``t`` and a strictly increasing mapping
    ``warp_fn: t -> τ``, return ``warp_fn(t)``.

    Parameters
    ----------
    t : array_like, required
        1-D time points to transform.
    warp_fn : callable, required
        Monotonic mapping from ``t`` to new time. Must accept and
        return NumPy arrays of the same shape.

    Returns
    -------
    ndarray
        Transformed time values ``warp_fn(t)``.
    """
    t_arr = _to_numpy(t, dtype=float)
    if not callable(warp_fn):
        raise ValueError("warp_fn must be callable")
    new_vals = warp_fn(t_arr)
    if not isinstance(new_vals, np.ndarray):
        new_vals = np.asarray(new_vals, dtype=float)
    if new_vals.shape != t_arr.shape:
        raise ValueError(
            "warp_fn must return an array of the same shape as input")
    return new_vals


def piecewise_linear_time_warp(
    t: ArrayLike,
    knots: Sequence[float],
    rates: Sequence[float],
) -> np.ndarray:
    """Apply a piecewise-linear time warp with segment-specific rates."""
    t_arr = _to_numpy(t, dtype=float)
    if t_arr.size == 0:
        return np.array([], dtype=float)
    if np.any(np.diff(t_arr) < 0):
        raise ValueError("t must be sorted in non-decreasing order")
    knots_arr = np.asarray(knots, dtype=float)
    rates_arr = np.asarray(rates, dtype=float)
    if knots_arr.size + 1 != rates_arr.size:
        raise ValueError("rates must have length len(knots) + 1")
    if knots_arr.size > 0 and np.any(np.diff(knots_arr) <= 0):
        raise ValueError("knots must be strictly increasing")
    if np.any(rates_arr <= 0):
        raise ValueError("rates must be positive to preserve order")
    t_min = float(t_arr[0])
    t_shifted = t_arr - t_min
    max_span = float(t_shifted[-1])
    if knots_arr.size > 0:
        if knots_arr[0] < t_min or knots_arr[-1] > t_arr[-1]:
            raise ValueError("knots must lie within the range of t")
    knots_shifted = knots_arr - t_min
    edges = np.concatenate(([0.0], knots_shifted, [max_span]))
    segment_lengths = np.diff(edges)
    segment_offsets = np.zeros(rates_arr.size, dtype=float)
    cumulative = 0.0
    for idx in range(rates_arr.size):
        segment_offsets[idx] = cumulative
        cumulative += rates_arr[idx] * segment_lengths[idx]
    segment_indices = np.searchsorted(knots_shifted, t_shifted, side='right')
    segment_starts = edges[segment_indices]
    warped = segment_offsets[segment_indices] + rates_arr[segment_indices] * (t_shifted - segment_starts)
    return warped


def apply_signal_mask(x: ArrayLike, mask: ArrayLike) -> np.ndarray:
    """Apply a 0/1 mask to a signal.

    Parameters
    ----------
    x : array_like, required
        Signal to be masked.
    mask : array_like, required
        0/1 mask of the same length as ``x``.

    Returns
    -------
    ndarray
        Elementwise product ``x * mask``.
    """
    x_arr = _to_numpy(x, dtype=float)
    m_arr = _to_numpy(mask, dtype=float)
    if x_arr.shape != m_arr.shape:
        raise ValueError("x and mask must have the same shape")
    return x_arr * m_arr


def apply_nan_mask(x: ArrayLike, mask: ArrayLike) -> np.ndarray:
    """Apply a mask that inserts NaNs where the mask is False."""
    x_arr = _to_numpy(x, dtype=float)
    m_arr = _to_numpy(mask, dtype=float)
    if x_arr.shape != m_arr.shape:
        raise ValueError("x and mask must have the same shape")
    out = x_arr.copy()
    out[m_arr <= 0.5] = np.nan
    return out


def winsorize_signal(x: ArrayLike, lower_q: float = 0.01, upper_q: float = 0.99) -> np.ndarray:
    """Clip signal tails to the specified quantiles (inclusive)."""
    if not (0.0 <= lower_q <= upper_q <= 1.0):
        raise ValueError("quantiles must satisfy 0 <= lower_q <= upper_q <= 1")
    x_arr = _to_numpy(x, dtype=float)
    if x_arr.size == 0:
        return x_arr.copy()
    valid = x_arr[np.isfinite(x_arr)]
    if valid.size == 0:
        return x_arr.copy()
    low = float(np.nanquantile(valid, lower_q))
    high = float(np.nanquantile(valid, upper_q))
    return np.clip(x_arr, low, high)


def lag_weighted_filter(x: ArrayLike, lags: Sequence[int], weights: Sequence[float]) -> np.ndarray:
    """Finite impulse response (FIR) filter with explicit lags.

        y[t] = sum_k weights[k] * x[t - lags[k]]

    Parameters
    ----------
    x : array_like, required
        Input signal.
    lags : sequence of int, required
        Non-negative lag offsets.
    weights : sequence of float, required
        Coefficients for each lag. Same length as ``lags``.

    Returns
    -------
    ndarray
        Filtered signal of the same length as ``x``.
    """
    if len(lags) != len(weights):
        raise ValueError("lags and weights must have the same length")
    if any(l < 0 for l in lags):
        raise ValueError("lags must be non-negative")
    x_arr = _to_numpy(x, dtype=float)
    n = len(x_arr)
    y = np.zeros_like(x_arr, dtype=float)
    for lag, w in zip(lags, weights):
        if lag == 0:
            y += w * x_arr
        else:
            y[lag:] += w * x_arr[:-lag]
    return y
