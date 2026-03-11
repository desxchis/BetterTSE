"""Auto‑generated tool descriptions for editing functions in ``ts_processor.py``.

This module exposes a list named ``description`` containing metadata for each
editing function defined in ``ts_processor.py``.  These descriptions are
intended to guide a language model agent when invoking editing
tools for time series modification.  Each entry in the list
corresponds to a single function and includes its human‑readable
description, function name, a breakdown of required and optional
parameters (with defaults where applicable) and high‑level tags that
categorise the function.  Parameter types are expressed using simple
primitives (e.g. ``int``, ``float``, ``array_like``) so that
the agent can supply appropriate arguments.
"""

description = [
    {
        "description": "Apply moving average smoothing to a specific region of a time series, reducing local fluctuations while preserving overall trend.",
        "name": "smooth_region",
        "optional_parameters": [
            {
                "name": "window",
                "type": "int",
                "default": 3,
                "description": "Window size for moving average (must be >= 1). Larger windows produce smoother results."
            }
        ],
        "required_parameters": [
            {
                "name": "x",
                "type": "array_like",
                "default": None,
                "description": "Input time series to be smoothed."
            },
            {
                "name": "start_idx",
                "type": "int",
                "default": None,
                "description": "Start index of the region to smooth (inclusive, 0-based)."
            },
            {
                "name": "end_idx",
                "type": "int",
                "default": None,
                "description": "End index of the region to smooth (exclusive, 0-based)."
            }
        ],
        "tag": ["editing", "smoothing", "region"]
    },
    {
        "description": "Interpolate values in a specific region using linear, nearest-neighbor, or zero-order hold methods to fill gaps or smooth transitions.",
        "name": "interpolate_region",
        "optional_parameters": [
            {
                "name": "method",
                "type": "str",
                "default": "linear",
                "description": "Interpolation method: 'linear' for smooth transition, 'nearest' for step-like, 'zero' for forward-fill."
            }
        ],
        "required_parameters": [
            {
                "name": "x",
                "type": "array_like",
                "default": None,
                "description": "Input time series."
            },
            {
                "name": "start_idx",
                "type": "int",
                "default": None,
                "description": "Start index of the region to interpolate (inclusive, 0-based)."
            },
            {
                "name": "end_idx",
                "type": "int",
                "default": None,
                "description": "End index of the region to interpolate (exclusive, 0-based)."
            }
        ],
        "tag": ["editing", "interpolation", "region"]
    },
    {
        "description": "Detect and remove anomalies in a specific region using z-score or IQR methods, then fill the removed points with interpolated values, mean, or median.",
        "name": "remove_anomalies_in_region",
        "optional_parameters": [
            {
                "name": "method",
                "type": "str",
                "default": "zscore",
                "description": "Anomaly detection method: 'zscore' for standard deviations, 'iqr' for interquartile range."
            },
            {
                "name": "threshold",
                "type": "float",
                "default": 3.0,
                "description": "Threshold for anomaly detection. Higher values are more permissive."
            },
            {
                "name": "fill_method",
                "type": "str",
                "default": "interpolate",
                "description": "Method to fill removed anomalies: 'interpolate', 'mean', or 'median'."
            }
        ],
        "required_parameters": [
            {
                "name": "x",
                "type": "array_like",
                "default": None,
                "description": "Input time series."
            },
            {
                "name": "start_idx",
                "type": "int",
                "default": None,
                "description": "Start index of the region (inclusive, 0-based)."
            },
            {
                "name": "end_idx",
                "type": "int",
                "default": None,
                "description": "End index of the region (exclusive, 0-based)."
            }
        ],
        "tag": ["editing", "anomaly_detection", "region"]
    },
    {
        "description": "Apply a linear trend to a specific region, allowing you to create upward or downward slopes within that segment.",
        "name": "apply_trend_in_region",
        "optional_parameters": [
            {
                "name": "offset",
                "type": "float",
                "default": 0.0,
                "description": "Vertical offset at the start of the region. Useful for aligning with existing data."
            }
        ],
        "required_parameters": [
            {
                "name": "x",
                "type": "array_like",
                "default": None,
                "description": "Input time series."
            },
            {
                "name": "start_idx",
                "type": "int",
                "default": None,
                "description": "Start index of the region (inclusive, 0-based)."
            },
            {
                "name": "end_idx",
                "type": "int",
                "default": None,
                "description": "End index of the region (exclusive, 0-based)."
            },
            {
                "name": "slope",
                "type": "float",
                "default": None,
                "description": "Slope of the linear trend. Positive for upward, negative for downward."
            }
        ],
        "tag": ["editing", "trend", "region"]
    },
    {
        "description": "Scale values in a specific region by a multiplicative factor, optionally centering around the region mean to preserve relative position.",
        "name": "scale_region",
        "optional_parameters": [
            {
                "name": "center",
                "type": "bool",
                "default": False,
                "description": "If True, scale around the region mean (preserves baseline). If False, scale from zero."
            }
        ],
        "required_parameters": [
            {
                "name": "x",
                "type": "array_like",
                "default": None,
                "description": "Input time series."
            },
            {
                "name": "start_idx",
                "type": "int",
                "default": None,
                "description": "Start index of the region (inclusive, 0-based)."
            },
            {
                "name": "end_idx",
                "type": "int",
                "default": None,
                "description": "End index of the region (exclusive, 0-based)."
            },
            {
                "name": "scale_factor",
                "type": "float",
                "default": None,
                "description": "Scaling factor to apply. Values > 1 amplify, values < 1 attenuate."
            }
        ],
        "tag": ["editing", "scaling", "region"]
    },
    {
        "description": "Increase the trend (slope) in a specific region of a time series by multiplying the current trend by a factor. Useful for making an upward trend steeper or a downward trend more negative.",
        "name": "increase_trend",
        "optional_parameters": [
            {
                "name": "factor",
                "type": "float",
                "default": 1.5,
                "description": "Factor to increase the trend by. Values > 1 make the trend steeper."
            }
        ],
        "required_parameters": [
            {
                "name": "x",
                "type": "array_like",
                "default": None,
                "description": "Input time series."
            },
            {
                "name": "start_idx",
                "type": "int",
                "default": None,
                "description": "Start index of region (inclusive, 0-based)."
            },
            {
                "name": "end_idx",
                "type": "int",
                "default": None,
                "description": "End index of region (exclusive, 0-based)."
            }
        ],
        "tag": ["editing", "trend", "increase"]
    },
    {
        "description": "Decrease the trend (slope) in a specific region of a time series by multiplying the current trend by a factor. Useful for flattening steep trends.",
        "name": "decrease_trend",
        "optional_parameters": [
            {
                "name": "factor",
                "type": "float",
                "default": 0.5,
                "description": "Factor to decrease the trend by (0 < factor < 1). Smaller values make the trend flatter."
            }
        ],
        "required_parameters": [
            {
                "name": "x",
                "type": "array_like",
                "default": None,
                "description": "Input time series."
            },
            {
                "name": "start_idx",
                "type": "int",
                "default": None,
                "description": "Start index of region (inclusive, 0-based)."
            },
            {
                "name": "end_idx",
                "type": "int",
                "default": None,
                "description": "End index of region (exclusive, 0-based)."
            }
        ],
        "tag": ["editing", "trend", "decrease"]
    },
    {
        "description": "Increase the volatility (variance) in a specific region of a time series by multiplying deviations from the mean by a factor. Useful for making fluctuations more pronounced.",
        "name": "increase_volatility",
        "optional_parameters": [
            {
                "name": "factor",
                "type": "float",
                "default": 1.5,
                "description": "Factor to increase volatility by. Values > 1 make fluctuations larger."
            }
        ],
        "required_parameters": [
            {
                "name": "x",
                "type": "array_like",
                "default": None,
                "description": "Input time series."
            },
            {
                "name": "start_idx",
                "type": "int",
                "default": None,
                "description": "Start index of region (inclusive, 0-based)."
            },
            {
                "name": "end_idx",
                "type": "int",
                "default": None,
                "description": "End index of region (exclusive, 0-based)."
            }
        ],
        "tag": ["editing", "volatility", "increase"]
    },
    {
        "description": "Decrease the volatility (variance) in a specific region of a time series by multiplying deviations from the mean by a factor. Useful for smoothing out fluctuations.",
        "name": "decrease_volatility",
        "optional_parameters": [
            {
                "name": "factor",
                "type": "float",
                "default": 0.5,
                "description": "Factor to decrease volatility by (0 < factor < 1). Smaller values make fluctuations smaller."
            }
        ],
        "required_parameters": [
            {
                "name": "x",
                "type": "array_like",
                "default": None,
                "description": "Input time series."
            },
            {
                "name": "start_idx",
                "type": "int",
                "default": None,
                "description": "Start index of region (inclusive, 0-based)."
            },
            {
                "name": "end_idx",
                "type": "int",
                "default": None,
                "description": "End index of region (exclusive, 0-based)."
            }
        ],
        "tag": ["editing", "volatility", "decrease"]
    }
]