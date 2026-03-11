"""
tool_specs.py
JSON-schema definitions for every tool in the tsa_toolbox.
Keep arguments scalar (string, int, float, boolean, short enum) so an LLM
can populate them deterministically.
"""

TOOLBOX = [

    # ─────────────────────────────── Data Hygiene ──────────────────────────────
    {
        "name": "describe",
        "description": "Return basic stats (n, mean, σ, min, max, %missing, "
                       "inferred frequency, start & end timestamps).",
        "parameters": {
            "type": "object",
            "properties": {
                "handle": {"type": "string"}
            },
            "required": ["handle"]
        }
    },
    {
        "name": "fill_missing",
        "description": "Impute gaps with a chosen method.",
        "parameters": {
            "type": "object",
            "properties": {
                "handle": {"type": "string"},
                "method":  {"type": "string",
                            "enum": ["ffill", "bfill", "linear", "median", "knn"]},
                "limit":   {"type": "integer", "minimum": 1}
            },
            "required": ["handle", "method"]
        }
    },
    {
        "name": "remove_outliers",
        "description": "Clip or drop extreme values using Z-score or IQR.",
        "parameters": {
            "type": "object",
            "properties": {
                "handle":   {"type": "string"},
                "method":   {"type": "string", "enum": ["zscore", "iqr"]},
                "threshold":{"type": "number", "default": 3.0}
            },
            "required": ["handle"]
        }
    },
    {
        "name": "standardise",
        "description": "Scale series to zero-mean/Unit-σ or robust "
                       "(median/IQR) scale.",
        "parameters": {
            "type": "object",
            "properties": {
                "handle": {"type": "string"},
                "method": {"type": "string", "enum": ["zscore", "robust"]}
            },
            "required": ["handle"]
        }
    },
    {
        "name": "deduplicate",
        "description": "Merge duplicate timestamps (keep first / last / mean).",
        "parameters": {
            "type": "object",
            "properties": {
                "handle":   {"type": "string"},
                "strategy": {"type": "string",
                             "enum": ["first", "last", "mean"], "default": "first"}
            },
            "required": ["handle"]
        }
    },

    # ──────────────────────────────── Transforms ───────────────────────────────
    {
        "name": "difference",
        "description": "Regular or seasonal differencing.",
        "parameters": {
            "type": "object",
            "properties": {
                "handle":          {"type": "string"},
                "order":           {"type": "integer", "minimum": 1, "default": 1},
                "seasonal_period": {"type": "integer", "minimum": 1}
            },
            "required": ["handle"]
        }
    },
    {
        "name": "moving_average",
        "description": "Rolling mean / sum / median smoothing.",
        "parameters": {
            "type": "object",
            "properties": {
                "handle":  {"type": "string"},
                "window":  {"type": "integer", "minimum": 1},
                "centered":{"type": "boolean", "default": False},
                "stat":    {"type": "string", "enum": ["mean", "sum", "median"],
                            "default": "mean"}
            },
            "required": ["handle", "window"]
        }
    },
    {
        "name": "decompose",
        "description": "STL trend-seasonal-residual split.",
        "parameters": {
            "type": "object",
            "properties": {
                "handle": {"type": "string"},
                "period": {"type": "integer", "minimum": 2},
                "robust": {"type": "boolean", "default": True}
            },
            "required": ["handle", "period"]
        }
    },
    {
        "name": "boxcox",
        "description": "Box-Cox power transform; λ can be inferred or supplied.",
        "parameters": {
            "type": "object",
            "properties": {
                "handle": {"type": "string"},
                "lmbda":  {"type": "number"}   # optional
            },
            "required": ["handle"]
        }
    },
    {
        "name": "log_transform",
        "description": "Natural log (x + ε) transformation.",
        "parameters": {
            "type": "object",
            "properties": {
                "handle":  {"type": "string"},
                "epsilon": {"type": "number", "default": 1e-6}
            },
            "required": ["handle"]
        }
    },
    {
        "name": "cumsum",
        "description": "Cumulative sum (integration) of a differenced series.",
        "parameters": {
            "type": "object",
            "properties": {
                "handle": {"type": "string"}
            },
            "required": ["handle"]
        }
    },

    # ─────────────────────────────── Diagnostics ───────────────────────────────
    {
        "name": "acf_pacf",
        "description": "Return ACF & PACF arrays; optionally PNG plot.",
        "parameters": {
            "type": "object",
            "properties": {
                "handle": {"type": "string"},
                "lags":   {"type": "integer", "minimum": 1, "default": 20},
                "plot":   {"type": "boolean", "default": False}
            },
            "required": ["handle"]
        }
    },
    {
        "name": "stationarity_test",
        "description": "Augmented-Dickey–Fuller test.",
        "parameters": {
            "type": "object",
            "properties": {
                "handle": {"type": "string"},
                "alpha":  {"type": "number", "default": 0.05}
            },
            "required": ["handle"]
        }
    },
    {
        "name": "seasonality_test",
        "description": "Simple seasonality strength test via "
                       "autocorrelation at lag = period.",
        "parameters": {
            "type": "object",
            "properties": {
                "handle": {"type": "string"},
                "period": {"type": "integer", "minimum": 2}
            },
            "required": ["handle", "period"]
        }
    },
    {
        "name": "heteroscedasticity_test",
        "description": "ARCH-LM test for changing variance.",
        "parameters": {
            "type": "object",
            "properties": {
                "handle": {"type": "string"},
                "lags":   {"type": "integer", "minimum": 1, "default": 12}
            },
            "required": ["handle"]
        }
    },
    {
        "name": "normality_test",
        "description": "Jarque–Bera or Shapiro–Wilk normality test.",
        "parameters": {
            "type": "object",
            "properties": {
                "handle": {"type": "string"},
                "method": {"type": "string", "enum": ["jarque_bera", "shapiro"],
                           "default": "jarque_bera"}
            },
            "required": ["handle"]
        }
    },

    # ─────────────────────────────── Forecasting ───────────────────────────────
    {
        "name": "repeat_last_value_forecast",
        "description": "Repeat the last observed value for every horizon step.",
        "parameters": {
            "type": "object",
            "properties": {
                "handle":  {"type": "string"},
                "horizon": {"type": "integer", "minimum": 1}
            },
            "required": ["handle", "horizon"]
        }
    },
    {
        "name": "repeat_last_season_forecast",
        "description": "Repeat the value from exactly s steps ago.",
        "parameters": {
            "type": "object",
            "properties": {
                "handle":          {"type": "string"},
                "horizon":         {"type": "integer", "minimum": 1},
                "seasonal_period": {"type": "integer", "minimum": 1}
            },
            "required": ["handle", "horizon", "seasonal_period"]
        }
    },
    {
        "name": "linear_drift_forecast",
        "description": "Linear drift extrapolation through first & last point.",
        "parameters": {
            "type": "object",
            "properties": {
                "handle":  {"type": "string"},
                "horizon": {"type": "integer", "minimum": 1}
            },
            "required": ["handle", "horizon"]
        }
    },
    {
        "name": "trailing_mean_forecast",
        "description": "Forecast is the mean of the last k points.",
        "parameters": {
            "type": "object",
            "properties": {
                "handle":  {"type": "string"},
                "horizon": {"type": "integer", "minimum": 1},
                "window":  {"type": "integer", "minimum": 1}
            },
            "required": ["handle", "horizon", "window"]
        }
    },
    {
        "name": "simple_exp_smoothing_forecast",
        "description": "Simple Exponential Smoothing with analytic alpha = "
                       "2/(n+1) unless user supplies one.",
        "parameters": {
            "type": "object",
            "properties": {
                "handle":  {"type": "string"},
                "horizon": {"type": "integer", "minimum": 1},
                "alpha":   {"type": "number"}   # optional
            },
            "required": ["handle", "horizon"]
        }
    },

    # ───────────────────────────── Anomaly Detection ───────────────────────────
    {
        "name": "zscore_anomaly",
        "description": "Flag points with |z| > threshold (global mean/σ).",
        "parameters": {
            "type": "object",
            "properties": {
                "handle":    {"type": "string"},
                "threshold": {"type": "number", "default": 3.0}
            },
            "required": ["handle"]
        }
    },
    {
        "name": "rolling_zscore",
        "description": "Compute z-score relative to a rolling window.",
        "parameters": {
            "type": "object",
            "properties": {
                "handle":    {"type": "string"},
                "window":    {"type": "integer", "minimum": 2},
                "threshold": {"type": "number", "default": 3.0},
                "min_periods":{"type": "integer"}
            },
            "required": ["handle", "window"]
        }
    },
    {
        "name": "iqr_anomaly",
        "description": "Flag points outside [Q1−k·IQR, Q3+k·IQR].",
        "parameters": {
            "type": "object",
            "properties": {
                "handle": {"type": "string"},
                "k":      {"type": "number", "default": 1.5}
            },
            "required": ["handle"]
        }
    },
    {
        "name": "mad_anomaly",
        "description": "Median ± k·MAD rule for heavy-tailed distributions.",
        "parameters": {
            "type": "object",
            "properties": {
                "handle": {"type": "string"},
                "k":      {"type": "number", "default": 3.5}
            },
            "required": ["handle"]
        }
    },
    {
        "name": "stl_residual_iqr",
        "description": "STL decompose then apply IQR on residual component.",
        "parameters": {
            "type": "object",
            "properties": {
                "handle": {"type": "string"},
                "period": {"type": "integer", "minimum": 2},
                "k":      {"type": "number", "default": 1.5}
            },
            "required": ["handle", "period"]
        }
    },
]
