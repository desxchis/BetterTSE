"""Auto‑generated tool descriptions for functions in ``ts_generators.py``.

This module exposes a list named ``description`` containing metadata for each
public function defined in ``ts_generators.py``.  These descriptions are
intended to guide a language model agent when invoking the available
time‑series generators and utility routines.  Each entry in the list
corresponds to a single function and includes its human‑readable
description, the function name, a breakdown of required and optional
parameters (with defaults where applicable) and high‑level tags that
categorise the function.  Parameter types are expressed using simple
primitives (e.g. ``int``, ``float``, ``array_like``, ``callable``) so that
the agent can supply appropriate arguments.
"""

# Note: The structure of each dictionary matches the examples provided in
# ``example.py``: a ``description`` string, the ``name`` of the tool,
# ``optional_parameters`` and ``required_parameters`` (both lists of
# dictionaries with name/type/default/description) and a list of ``tag``
# strings to organise related tools.

description = [
    {
        "description": "Generate a naïve forecast by repeating the last finite value of a series for the specified number of future steps.",
        "name": "repeat_last_value_forecast",
        "optional_parameters": [],
        "required_parameters": [
            {"name": "x", "type": "array_like", "default": None, "description": "Historical series used to find the last finite observation."},
            {"name": "horizon", "type": "int", "default": None, "description": "Number of future time steps to forecast."},
        ],
        "tag": ["forecasting"],
    },
    {
        "description": "Repeat the most recent seasonal pattern from a time series to produce a seasonal naïve forecast.",
        "name": "repeat_last_season_forecast",
        "optional_parameters": [],
        "required_parameters": [
            {"name": "x", "type": "array_like", "default": None, "description": "Historical series containing the repeating seasonal pattern."},
            {"name": "horizon", "type": "int", "default": None, "description": "Number of future samples to generate."},
            {"name": "seasonal_period", "type": "int", "default": None, "description": "Length of the seasonal cycle to repeat (must not exceed the number of finite observations)."},
        ],
        "tag": ["forecasting", "seasonality"],
    },
    {
        "description": "Perform a drift forecast by fitting a straight line between the first and last finite observations and extrapolating that trend forward.",
        "name": "linear_drift_forecast",
        "optional_parameters": [],
        "required_parameters": [
            {"name": "x", "type": "array_like", "default": None, "description": "Historical series from which to compute the linear drift."},
            {"name": "horizon", "type": "int", "default": None, "description": "Number of time steps to forecast into the future."},
        ],
        "tag": ["forecasting"],
    },
    {
        "description": "Generate a flat forecast equal to the mean of the most recent window of observations.",
        "name": "trailing_mean_forecast",
        "optional_parameters": [],
        "required_parameters": [
            {"name": "x", "type": "array_like", "default": None, "description": "Historical series used to compute the moving average."},
            {"name": "horizon", "type": "int", "default": None, "description": "Number of future points to forecast."},
            {"name": "window", "type": "int", "default": None, "description": "Number of most recent finite observations to average (recommended ≥ 1; if ≤ 0, current code effectively uses the whole series due to slicing semantics)."}
        ],
        "tag": ["forecasting", "smoothing"],
    },
    {
        "description": "Compute a constant forecast using simple exponential smoothing; uses a default smoothing factor of 2/(n+1) when none is provided.",
        "name": "simple_exp_smoothing_forecast",
        "optional_parameters": [
            {"name": "alpha", "type": "float", "default": None, "description": "Smoothing factor in (0,1]; if omitted, uses 2/(n+1) (not validated in code)."}
        ],
        "required_parameters": [
            {"name": "x", "type": "array_like", "default": None, "description": "Historical series."},
            {"name": "horizon", "type": "int", "default": None, "description": "Number of future points to forecast."}
        ],
        "tag": ["forecasting", "smoothing"],
    },
    {
        "description": "Create a constant signal equal to a specified value for every time point.",
        "name": "constant_signal",
        "optional_parameters": [],
        "required_parameters": [
            {"name": "t", "type": "array_like", "default": None, "description": "Reference time array whose shape determines the output length."},
            {"name": "value", "type": "float", "default": None, "description": "Constant value to assign to each element."},
        ],
        "tag": ["trend"],
    },
    {
        "description": "Generate a linear trend by computing intercept + slope * t element‑wise.",
        "name": "linear_trend",
        "optional_parameters": [],
        "required_parameters": [
            {"name": "t", "type": "array_like", "default": None, "description": "Time points at which to evaluate the trend."},
            {"name": "slope", "type": "float", "default": None, "description": "Slope of the line."},
            {"name": "intercept", "type": "float", "default": None, "description": "Vertical offset at t=0."},
        ],
        "tag": ["trend"],
    },
    {
        "description": "Construct a piecewise linear trend with changepoints; the slope adjusts by the specified deltas after each knot, and an optional offset can shift the baseline.",
        "name": "changepoint_linear_trend",
        "optional_parameters": [
            {"name": "offset", "type": "float", "default": 0.0, "description": "Constant baseline added after applying the changepoints."}
        ],
        "required_parameters": [
            {"name": "t", "type": "array_like", "default": None, "description": "Time points to evaluate the piecewise function."},
            {"name": "knots", "type": "sequence", "default": None, "description": "Monotonically increasing changepoint positions."},
            {"name": "deltas", "type": "sequence", "default": None, "description": "Slope changes at each knot (must match the length of knots)."},
        ],
        "tag": ["trend", "changepoints"],
    },
    {
        "description": "Generate a smooth logistic transition between two levels, controlled by the centre point and width parameters.",
        "name": "sigmoid_transition",
        "optional_parameters": [],
        "required_parameters": [
            {"name": "t", "type": "array_like", "default": None, "description": "Time points at which to evaluate the transition."},
            {"name": "center", "type": "float", "default": None, "description": "Time of the midpoint (logistic inflection)."},
            {"name": "width", "type": "float", "default": None, "description": "Controls steepness; larger values give smoother transitions (must be > 0)."},
            {"name": "low", "type": "float", "default": None, "description": "Lower asymptote before the transition."},
            {"name": "high", "type": "float", "default": None, "description": "Upper asymptote after the transition."},
        ],
        "tag": ["trend", "transition"],
    },
    {
        "description": "Compute a sinusoidal wave y(t) = amplitude * sin(2π * t / period + phase) and optionally shift it by a constant offset.",
        "name": "sine_wave",
        "optional_parameters": [
            {"name": "phase", "type": "float", "default": 0.0, "description": "Phase offset in radians; use π/2 for a cosine."},
            {"name": "offset", "type": "float", "default": 0.0, "description": "Constant value added to the sinusoid."},
        ],
        "required_parameters": [
            {"name": "t", "type": "array_like", "default": None, "description": "Time points at which to compute the sinusoid."},
            {"name": "period", "type": "float", "default": None, "description": "Length of one full cycle in the same units as t."},
            {"name": "amplitude", "type": "float", "default": None, "description": "Height of the wave."},
        ],
        "tag": ["periodic", "seasonality"],
    },
    {
        "description": "Generate a seasonal signal by summing a series of cosine and sine harmonics given their coefficients and fundamental period, with an optional constant offset.",
        "name": "fourier_series_seasonality",
        "optional_parameters": [
            {"name": "offset", "type": "float", "default": 0.0, "description": "Constant baseline added after summing harmonics."}
        ],
        "required_parameters": [
            {"name": "t", "type": "array_like", "default": None, "description": "Time points to evaluate the seasonal component."},
            {"name": "period", "type": "float", "default": None, "description": "Fundamental period of the seasonality."},
            {"name": "n_harmonics", "type": "int", "default": None, "description": "Number of harmonics to include (≥ 1)."},
            {"name": "cos_coeffs", "type": "sequence", "default": None, "description": "Sequence of cosine coefficients of length n_harmonics."},
            {"name": "sin_coeffs", "type": "sequence", "default": None, "description": "Sequence of sine coefficients of length n_harmonics."},
        ],
        "tag": ["periodic", "seasonality"],
    },
    {
        "description": "Sum multiple sinusoids with arbitrary frequencies, amplitudes and phases, then optionally add a constant offset.",
        "name": "sum_of_sinusoids",
        "optional_parameters": [
            {"name": "offset", "type": "float", "default": 0.0, "description": "Constant baseline added after summing all sinusoids."}
        ],
        "required_parameters": [
            {"name": "t", "type": "array_like", "default": None, "description": "Time points at which to evaluate the combined waveform."},
            {"name": "freqs", "type": "sequence", "default": None, "description": "Frequencies (cycles per unit time) for each sinusoid."},
            {"name": "amplitudes", "type": "sequence", "default": None, "description": "Amplitude for each sinusoid (same length as freqs)."},
            {"name": "phases", "type": "sequence", "default": None, "description": "Phase offsets in radians (same length as freqs)."},
        ],
        "tag": ["periodic", "seasonality"],
    },
    {
        "description": "Generate a repeating rectangular pulse of fixed amplitude and width within each period, with optional phase shift and baseline.",
        "name": "periodic_pulse_train",
        "optional_parameters": [
            {"name": "phase_offset", "type": "float", "default": 0.0, "description": "Phase shift in the same units as t applied before taking the modulus."},
            {"name": "offset", "type": "float", "default": 0.0, "description": "Constant baseline added to the pulse train."}
        ],
        "required_parameters": [
            {"name": "t", "type": "array_like", "default": None, "description": "Time points at which to evaluate the pulse train."},
            {"name": "period", "type": "float", "default": None, "description": "Length of each repetition interval (must be > 0)."},
            {"name": "width", "type": "float", "default": None, "description": "Active width inside each period (clipped to period, must be ≥ 0)."},
            {"name": "amplitude", "type": "float", "default": None, "description": "Pulse height when active."},
        ],
        "tag": ["pulse", "periodic"],
    },
    {
        "description": "Generate a step function that is zero before a threshold and equal to the given amplitude afterwards, with an optional constant offset.",
        "name": "delayed_step_signal",
        "optional_parameters": [
            {"name": "offset", "type": "float", "default": 0.0, "description": "Constant baseline added to the step output."},
        ],
        "required_parameters": [
            {"name": "t", "type": "array_like", "default": None, "description": "Time points at which to evaluate the step."},
            {"name": "t0", "type": "float", "default": None, "description": "Time at which the step occurs."},
            {"name": "amplitude", "type": "float", "default": None, "description": "Value of the step for t ≥ t0."},
        ],
        "tag": ["pulse"],
    },
    {
        "description": "Produce a ramp that remains zero before a specified time and increases linearly with a given slope thereafter, optionally shifted by a constant baseline.",
        "name": "delayed_ramp_signal",
        "optional_parameters": [
            {"name": "offset", "type": "float", "default": 0.0, "description": "Constant baseline added to the ramp."}
        ],
        "required_parameters": [
            {"name": "t", "type": "array_like", "default": None, "description": "Time points at which to compute the ramp."},
            {"name": "t0", "type": "float", "default": None, "description": "Onset time of the ramp."},
            {"name": "slope", "type": "float", "default": None, "description": "Rate of increase after t0."},
        ],
        "tag": ["trend", "pulse"],
    },
    {
        "description": "Return a unit-height rectangular (boxcar) pulse centred at a specified time with a given width, optionally adding a constant offset.",
        "name": "unit_boxcar_pulse",
        "optional_parameters": [
            {"name": "offset", "type": "float", "default": 0.0, "description": "Constant baseline added outside/inside the pulse."}
        ],
        "required_parameters": [
            {"name": "t", "type": "array_like", "default": None, "description": "Time points to evaluate the boxcar."},
            {"name": "center", "type": "float", "default": None, "description": "Centre of the pulse."},
            {"name": "width", "type": "float", "default": None, "description": "Full width of the pulse (must be ≥ 0)."},
        ],
        "tag": ["pulse"],
    },
    {
        "description": "Compute a Gaussian bell‑shaped pulse centred at a given mean with specified standard deviation and unit peak height, optionally shifting the baseline.",
        "name": "unit_gaussian_pulse",
        "optional_parameters": [
            {"name": "offset", "type": "float", "default": 0.0, "description": "Constant baseline added to the Gaussian pulse."}
        ],
        "required_parameters": [
            {"name": "t", "type": "array_like", "default": None, "description": "Time points to evaluate the pulse."},
            {"name": "mu", "type": "float", "default": None, "description": "Centre (mean) of the Gaussian."},
            {"name": "sigma", "type": "float", "default": None, "description": "Standard deviation (> 0)."},
        ],
        "tag": ["pulse"],
    },
    {
        "description": "Generate an exponentially decaying pulse that starts at a specified time and has unit value at onset before decaying with the given time constant, with an optional baseline shift.",
        "name": "unit_exponential_decay_pulse",
        "optional_parameters": [
            {"name": "offset", "type": "float", "default": 0.0, "description": "Constant baseline added to the decay profile."}
        ],
        "required_parameters": [
            {"name": "t", "type": "array_like", "default": None, "description": "Time points at which to compute the pulse."},
            {"name": "t0", "type": "float", "default": None, "description": "Onset time of the decay."},
            {"name": "tau", "type": "float", "default": None, "description": "Decay time constant (> 0)."},
        ],
        "tag": ["pulse"],
    },
    {
        "description": "Sum delayed kernels at fixed event times to produce a shot noise signal; the supplied kernel function is applied to the delay t−e for each event, and an optional offset can shift the baseline.",
        "name": "kernel_shot_noise_signal",
        "optional_parameters": [
            {"name": "offset", "type": "float", "default": 0.0, "description": "Constant baseline added after summing kernels."}
        ],
        "required_parameters": [
            {"name": "t", "type": "array_like", "default": None, "description": "Time grid on which to build the signal."},
            {"name": "events", "type": "sequence", "default": None, "description": "Sequence of event times at which kernels are centred."},
            {"name": "kernel_fn", "type": "callable", "default": None, "description": "Function mapping time delays to pulse values; must accept and return arrays of the same shape."},
        ],
        "tag": ["noise", "pulse"],
    },
    {
        "description": "Apply amplitude modulation by multiplying a base signal elementwise with an envelope of the same shape.",
        "name": "apply_amplitude_envelope",
        "optional_parameters": [],
        "required_parameters": [
            {"name": "x", "type": "array_like", "default": None, "description": "Base signal to modulate."},
            {"name": "envelope", "type": "array_like", "default": None, "description": "Amplitude envelope array matching the shape of x."},
        ],
        "tag": ["modulation", "envelope"],
    },
    {
        "description": "Create a deterministic on/off mask that is 1 for a fraction of each period defined by duty_cycle and 0 otherwise, optionally shifted by a phase offset and a constant baseline.",
        "name": "periodic_duty_cycle_mask",
        "optional_parameters": [
            {"name": "phase_offset", "type": "float", "default": 0.0, "description": "Time shift of the active window within the period."},
            {"name": "offset", "type": "float", "default": 0.0, "description": "Constant baseline added to the mask."},
        ],
        "required_parameters": [
            {"name": "t", "type": "array_like", "default": None, "description": "Time points to compute the mask."},
            {"name": "period", "type": "float", "default": None, "description": "Length of one cycle."},
            {"name": "duty_cycle", "type": "float", "default": None, "description": "Fraction of each period for which the mask equals 1 (0 < duty_cycle ≤ 1)."},
        ],
        "tag": ["mask", "periodic"],
    },
    {
        "description": "Return a 0/1 mask based on calendar rules (weekend, weekday, month_end, month_start, or holiday).",
        "name": "calendar_rule_mask",
        "optional_parameters": [
            {"name": "holidays", "type": "sequence", "default": None, "description": "Iterable of numpy.datetime64 values to mark as holidays when rule='holiday'."},
        ],
        "required_parameters": [
            {"name": "dates", "type": "array_like", "default": None, "description": "Array of datetime64 values to classify."},
            {"name": "rule", "type": "str", "default": None, "description": "One of 'weekend', 'weekday', 'month_end', 'month_start', or 'holiday'."},
        ],
        "tag": ["mask", "calendar"],
    },
    {
        "description": "Compute a simple moving average (boxcar filter) over a fixed window to smooth a 1‑D signal.",
        "name": "moving_average_filter",
        "optional_parameters": [],
        "required_parameters": [
            {"name": "x", "type": "array_like", "default": None, "description": "Signal to be smoothed."},
            {"name": "window", "type": "int", "default": None, "description": "Width of the averaging window (≥ 1)."},
        ],
        "tag": ["filter", "smoothing"],
    },
    {
        "description": "Apply exponential smoothing to a signal using a specified smoothing factor alpha, returning a smoothed series.",
        "name": "exponential_smoothing_filter",
        "optional_parameters": [],
        "required_parameters": [
            {"name": "x", "type": "array_like", "default": None, "description": "Input signal to smooth."},
            {"name": "alpha", "type": "float", "default": None, "description": "Smoothing parameter (0 < alpha ≤ 1)."},
        ],
        "tag": ["filter", "smoothing"],
    },
    {
        "description": "Perform linear convolution of an input signal with a kernel using numpy's convolve; the mode can be 'full', 'valid', or 'same'.",
        "name": "convolve_1d_signal",
        "optional_parameters": [
            {"name": "mode", "type": "str", "default": "same", "description": "Convolution mode: 'full', 'valid', or 'same'."},
        ],
        "required_parameters": [
            {"name": "x", "type": "array_like", "default": None, "description": "Input signal to convolve."},
            {"name": "kernel", "type": "array_like", "default": None, "description": "Convolution kernel (impulse response)."},
        ],
        "tag": ["filter"],
    },
    {
        "description": "Compute discrete differences of a signal a specified number of times; equivalent to repeated application of numpy.diff.",
        "name": "difference_signal",
        "optional_parameters": [
            {"name": "order", "type": "int", "default": 1, "description": "Number of times to apply differencing (≥ 1 and < len(x))."},
        ],
        "required_parameters": [
            {"name": "x", "type": "array_like", "default": None, "description": "Input signal from which to compute differences."},
        ],
        "tag": ["transform"],
    },
    {
        "description": "Compute the cumulative sum of a sequence, effectively performing discrete integration.",
        "name": "cumulative_sum_signal",
        "optional_parameters": [],
        "required_parameters": [
            {"name": "x", "type": "array_like", "default": None, "description": "Input increments or values to accumulate."},
        ],
        "tag": ["transform", "integration"],
    },
    {
        "description": "Downsample a signal by grouping consecutive samples into blocks and applying an aggregation function (mean, sum, max, or min). The final partial block (if any) is included.",
        "name": "blockwise_aggregate",
        "optional_parameters": [
            {"name": "agg", "type": "str", "default": "mean", "description": "Aggregation to apply to each block: 'mean', 'sum', 'max', or 'min'."}
        ],
        "required_parameters": [
            {"name": "x", "type": "array_like", "default": None, "description": "Input signal to downsample."},
            {"name": "block_size", "type": "int", "default": None, "description": "Number of samples per block (≥ 1)."}
        ],
        "tag": ["resampling"],
    },
    {
        "description": "Interpolate or extrapolate a signal onto a new set of time points using linear, nearest‑neighbour or zero‑order hold methods.",
        "name": "resample_to_time_index",
        "optional_parameters": [
            {"name": "method", "type": "str", "default": "linear", "description": "Interpolation method: 'linear', 'nearest', or 'zero'."},
        ],
        "required_parameters": [
            {"name": "x", "type": "array_like", "default": None, "description": "Values of the signal at original times."},
            {"name": "t", "type": "array_like", "default": None, "description": "Original time stamps corresponding to x (must be sorted)."},
            {"name": "new_t", "type": "array_like", "default": None, "description": "Target times to resample onto (sorted ascending)."},
        ],
        "tag": ["resampling", "interpolation"],
    },
    {
        "description": "Apply a (typically monotonic) time reparameterisation by passing the time vector through a user-supplied warp function. Monotonicity is not enforced.",
        "name": "apply_time_warp",
        "optional_parameters": [],
        "required_parameters": [
            {"name": "t", "type": "array_like", "default": None, "description": "1-D time points."},
            {"name": "warp_fn", "type": "callable", "default": None, "description": "Function mapping t -> tau. Called elementwise."}
        ],
        "tag": ["transform"],
    },
    {
        "description": "Warp time using segment-specific linear rates defined by ordered knots, yielding a deterministic irregular cadence.",
        "name": "piecewise_linear_time_warp",
        "optional_parameters": [],
        "required_parameters": [
            {"name": "t", "type": "array_like", "default": None, "description": "Sorted time points to transform."},
            {"name": "knots", "type": "sequence", "default": None, "description": "Strictly increasing breakpoints within the range of t."},
            {"name": "rates", "type": "sequence", "default": None, "description": "Positive rates for each segment (length len(knots)+1)."}
        ],
        "tag": ["transform", "time_warp"],
    },
    {
        "description": "Multiply a signal elementwise by a mask (binary 0/1 or fractional weights) to zero out or attenuate inactive segments.",
        "name": "apply_signal_mask",
        "optional_parameters": [],
        "required_parameters": [
            {"name": "x", "type": "array_like", "default": None, "description": "Signal to be masked."},
            {"name": "mask", "type": "array_like", "default": None, "description": "Mask array (same length as x). Values in [0,1] are typical; 0 = remove, 1 = keep."}
        ],
        "tag": ["mask"],
    },
    {
        "description": "Insert NaNs wherever the supplied mask is inactive (≤ 0.5), preserving values where it is active.",
        "name": "apply_nan_mask",
        "optional_parameters": [],
        "required_parameters": [
            {"name": "x", "type": "array_like", "default": None, "description": "Signal whose samples may be masked out."},
            {"name": "mask", "type": "array_like", "default": None, "description": "Mask array, same shape as x; > 0.5 keeps the value, otherwise NaN is inserted."}
        ],
        "tag": ["mask", "missingness"],
    },
    {
        "description": "Winsorize a signal by clipping it to the specified lower and upper quantiles.",
        "name": "winsorize_signal",
        "optional_parameters": [
            {"name": "lower_q", "type": "float", "default": 0.01, "description": "Lower quantile in [0,1] used as the minimum clip value."},
            {"name": "upper_q", "type": "float", "default": 0.99, "description": "Upper quantile in [0,1] used as the maximum clip value."}
        ],
        "required_parameters": [
            {"name": "x", "type": "array_like", "default": None, "description": "Signal to be winsorized."}
        ],
        "tag": ["transform", "distribution"],
    },
    {
        "description": "Apply a finite impulse response (FIR) filter by summing lagged versions of the signal weighted by the provided coefficients.",
        "name": "lag_weighted_filter",
        "optional_parameters": [],
        "required_parameters": [
            {"name": "x", "type": "array_like", "default": None, "description": "Input signal to filter."},
            {"name": "lags", "type": "sequence[int]", "default": None, "description": "Non‑negative lag offsets at which to apply the weights."},
            {"name": "weights", "type": "sequence[float]", "default": None, "description": "Coefficients corresponding to each lag (same length as lags)."}
        ],
        "tag": ["filter", "fir"],
    },
]
