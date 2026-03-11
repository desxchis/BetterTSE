# ts_descriptors_description.py
#
# This module defines metadata descriptions for each time_series_descriptors
# function.  Each entry is a dictionary containing the name of the
# descriptor function, a human‑readable description, specification of
# required and optional parameters, tags that group the descriptor by
# type, and guidance on interpreting the output.  The definitions here
# mirror the actual functions implemented in time_series_descriptors.py.

description = [
    {
        "name": "length",
        "description": "Return the number of observations in the series (including missing values).",
        "required_parameters": [
            {"name": "values", "type": "list",
                "description": "Sequence of observations (any type)."}
        ],
        "optional_parameters": [],
        "tag": ["ts_descriptor", "universal", "temporal_structure"],
        "interpretation": [{
            "output": "JSON string encoding Integer ≥ 0",
            "range": "[0, ∞)",
            "higher_is": "longer series and potentially more reliable estimates",
            "how_to_read": "Use to gauge sample size; compare with time_span to assess sampling density.",
            "edge_cases": "Empty list returns 0."
        }],
    },
    {
        "name": "summary_stats",
        "description": "Return (mean, std, min, max) for a numeric series, ignoring missing values. Non-numeric inputs are coerced to numeric (invalid entries become NaN).",
        "required_parameters": [
            {"name": "values", "type": "list",
                "description": "Input series (numeric or coercible to numeric).", "default": None}
        ],
        "optional_parameters": [],
        "tag": ["ts_descriptor", "universal", "distribution", "scale"],
        "interpretation": [{
            "output": "JSON string encoding (mean, std, min, max) — 4 floats",
            "range": "std >= 0; min <= mean <= max (when any valid values exist)",
            "higher_is": "N/A (multi-output)",
            "how_to_read": "mean describes central tendency (same units as values); std quantifies dispersion (same units) using a population denominator (ddof=0); min/max show observed extremes. NaN in any field indicates that component is undefined (e.g., no finite values).",
            "edge_cases": "If there are no finite numeric values, returns (NaN, NaN, NaN, NaN). Strings or mixed types are coerced; unparsable tokens become NaN and are ignored."
        }],
    },
    {
        "name": "time_span",
        "description": "Return the elapsed time between the minimum and maximum timestamp in seconds.",
        "required_parameters": [
            {"name": "timestamps", "type": "list",
                "description": "Sequence of datetime‑like values."}
        ],
        "optional_parameters": [],
        "tag": ["ts_descriptor", "universal", "temporal_structure"],
        "interpretation": [{
            "output": "JSON string encoding Float ≥ 0 (seconds)",
            "range": "[0, ∞)",
            "higher_is": "wider calendar coverage",
            "how_to_read": "Large time_span with small length implies sparse/irregular sampling.",
            "edge_cases": "If only one timestamp, time_span = 0."
        }],
    },
    {
        "name": "mean_gap",
        "description": "Return the mean interval in seconds between consecutive timestamps.",
        "required_parameters": [
            {"name": "timestamps", "type": "list",
                "description": "Sequence of datetime‑like values."}
        ],
        "optional_parameters": [],
        "tag": ["ts_descriptor", "universal", "temporal_structure"],
        "interpretation": [{
            "output": "JSON string encoding Float; NaN if <2 timestamps",
            "range": "(0, ∞) when defined",
            "higher_is": "coarser sampling cadence",
            "how_to_read": "Compare with median_gap and gap_cv to assess irregularity and typical cadence.",
            "edge_cases": "Undefined (NaN) when fewer than two timestamps."
        }],
    },
    {
        "name": "median_gap",
        "description": "Return the median interval in seconds between consecutive timestamps.",
        "required_parameters": [
            {"name": "timestamps", "type": "list",
                "description": "Sequence of datetime‑like values."}
        ],
        "optional_parameters": [],
        "tag": ["ts_descriptor", "universal", "temporal_structure"],
        "interpretation": [{
            "output": "JSON string encoding Float; NaN if <2 timestamps",
            "range": "(0, ∞) when defined",
            "higher_is": "coarser typical cadence (robust to outliers)",
            "how_to_read": "Use as the nominal sampling period; helpful for normalising other indicators.",
            "edge_cases": "Undefined (NaN) when fewer than two timestamps."
        }],
    },
    {
        "name": "gap_cv",
        "description": "Coefficient of variation of sampling gaps: standard deviation divided by mean gap.",
        "required_parameters": [
            {"name": "timestamps", "type": "list",
                "description": "Sequence of datetime‑like values."}
        ],
        "optional_parameters": [],
        "tag": ["ts_descriptor", "universal", "temporal_structure", "irregularity"],
        "interpretation": [{
            "output": "JSON string encoding Float ≥ 0; NaN if <2 timestamps or mean gap is zero",
            "range": "[0, ∞)",
            "higher_is": "more irregular sampling",
            "how_to_read": "≈0 indicates nearly regular sampling; >1 indicates highly unequal gaps.",
            "edge_cases": "Undefined (NaN) when fewer than two timestamps or when mean gap equals zero."
        }],
    },
    {
        "name": "max_gap",
        "description": "Return the maximum interval in seconds between consecutive timestamps.",
        "required_parameters": [
            {"name": "timestamps", "type": "list",
                "description": "Sequence of datetime‑like values."}
        ],
        "optional_parameters": [],
        "tag": ["ts_descriptor", "universal", "temporal_structure"],
        "interpretation": [{
            "output": "JSON string encoding Float; NaN if <2 timestamps",
            "range": "[0, ∞) when defined",
            "higher_is": "larger outages or sparse stretches",
            "how_to_read": "Large max_gap relative to median_gap indicates long blackouts; affects seasonality detection.",
            "edge_cases": "Undefined (NaN) when fewer than two timestamps."
        }],
    },
    {
        "name": "gap_entropy",
        "description": "Shannon entropy (base 2) of binned gap sizes.",
        "required_parameters": [
            {"name": "timestamps", "type": "list",
                "description": "Sequence of datetime‑like values."}
        ],
        "optional_parameters": [],
        "tag": ["ts_descriptor", "universal", "temporal_structure", "irregularity"],
        "interpretation": [{
            "output": "JSON string encoding Float entropy ≥ 0 (bits)",
            "range": "[0, log2(10)] approximately",
            "higher_is": "more diverse gap sizes (less regular)",
            "how_to_read": "Near 0 implies nearly fixed cadence; higher values imply varied sampling intervals.",
            "edge_cases": "Undefined (NaN) when fewer than two gaps or all gaps are identical."
        }],
    },
    {
        "name": "missing_rate",
        "description": "Return the proportion of missing values (NaN or None) in the series.",
        "required_parameters": [
            {"name": "values", "type": "list",
                "description": "Input series of observations."}
        ],
        "optional_parameters": [],
        "tag": ["ts_descriptor", "universal", "quality"],
        "interpretation": [{
            "output": "JSON string encoding Float in [0,1]",
            "range": "[0,1]",
            "higher_is": "more missing values",
            "how_to_read": "0 means no missing values; a high missing_rate indicates poor data completeness.",
            "edge_cases": "Empty series returns 0."
        }],
    },
    {
        "name": "missing_runs",
        "description": "Number of contiguous runs of missing values in the series.",
        "required_parameters": [
            {"name": "values", "type": "list",
                "description": "Input series of observations."}
        ],
        "optional_parameters": [],
        "tag": ["ts_descriptor", "universal", "quality"],
        "interpretation": [{
            "output": "JSON string encoding Non‑negative integer",
            "range": "[0, ∞)",
            "higher_is": "more fragmented missingness",
            "how_to_read": "Distinguishes many small gaps from a few long missing segments.",
            "edge_cases": "Series with all values missing returns 1."
        }],
    },
    {
        "name": "longest_missing_run",
        "description": "Return the length (number of points) of the longest run of missing values.",
        "required_parameters": [
            {"name": "values", "type": "list",
                "description": "Input series of observations."}
        ],
        "optional_parameters": [],
        "tag": ["ts_descriptor", "universal", "quality"],
        "interpretation": [{
            "output": "JSON string encoding Non‑negative integer",
            "range": "[0, ∞)",
            "higher_is": "longer single blackout",
            "how_to_read": "Use with missing_runs: few runs + large longest_missing_run implies rare but severe outages.",
            "edge_cases": "Returns 0 if there are no missing values."
        }],
    },
    {
        "name": "alphabet_size",
        "description": "Number of unique states after applying default quantile symbolisation to numeric values.",
        "required_parameters": [
            {"name": "values", "type": "list",
                "description": "Sequence of observations (numeric or categorical)."}
        ],
        "optional_parameters": [],
        "tag": ["ts_descriptor", "universal", "distribution"],
        "interpretation": [{
            "output": "JSON string encoding Non‑negative integer",
            "range": "[0, min(T, 10)] approximately",
            "higher_is": "greater value/state diversity",
            "how_to_read": "Low values suggest discretisation or limited variety; high values indicate rich variation.",
            "edge_cases": "Returns 0 if all values are missing or identical after binning."
        }],
    },
    {
        "name": "mode_probability",
        "description": "Maximum probability of any state in the symbolised series.",
        "required_parameters": [
            {"name": "values", "type": "list",
                "description": "Sequence of observations (numeric or categorical)."}
        ],
        "optional_parameters": [],
        "tag": ["ts_descriptor", "universal", "distribution"],
        "interpretation": [{
            "output": "JSON string encoding Float in [0,1]; NaN if no valid states",
            "range": "[0,1]",
            "higher_is": "more dominance of a single state",
            "how_to_read": "≈1 implies near constant series; ≈1/K for evenly balanced K states.",
            "edge_cases": "Undefined (NaN) when there are no non‑missing values."
        }],
    },
    {
        "name": "shannon_entropy",
        "description": "Shannon entropy (base 2) of the symbolised value distribution.",
        "required_parameters": [
            {"name": "values", "type": "list",
                "description": "Sequence of observations (numeric or categorical)."}
        ],
        "optional_parameters": [],
        "tag": ["ts_descriptor", "universal", "distribution", "information"],
        "interpretation": [{
            "output": "JSON string encoding Float ≥ 0 (bits); NaN if undefined",
            "range": "[0, log2(K)]",
            "higher_is": "more uniform spread over states",
            "how_to_read": "0 if all values identical; maximum when states are equally likely.",
            "edge_cases": "Undefined (NaN) when there are no non‑missing values."
        }],
    },
    {
        "name": "gini_simpson_index",
        "description": "Gini–Simpson diversity index 1 − ∑ p_i^2 over states.",
        "required_parameters": [
            {"name": "values", "type": "list",
                "description": "Sequence of observations (numeric or categorical)."}
        ],
        "optional_parameters": [],
        "tag": ["ts_descriptor", "universal", "distribution"],
        "interpretation": [{
            "output": "JSON string encoding Float in [0,1); NaN if undefined",
            "range": "[0,1)",
            "higher_is": "more diversity / less dominance",
            "how_to_read": "0 => all mass in one state; increases as distribution spreads across states.",
            "edge_cases": "Undefined (NaN) when there are no non‑missing values."
        }],
    },
    {
        "name": "evenness",
        "description": "Pielou's evenness: Shannon entropy divided by log2(K).",
        "required_parameters": [
            {"name": "values", "type": "list",
                "description": "Sequence of observations (numeric or categorical)."}
        ],
        "optional_parameters": [],
        "tag": ["ts_descriptor", "universal", "distribution"],
        "interpretation": [{
            "output": "JSON string encoding Float in [0,1]; NaN if undefined",
            "range": "[0,1]",
            "higher_is": "more even value distribution",
            "how_to_read": "1 indicates perfectly even usage of states; near 0 indicates heavy skew.",
            "edge_cases": "Undefined (NaN) if only one state or no non‑missing values."
        }],
    },
    {
        "name": "change_rate",
        "description": "Fraction of adjacent non‑missing observations that differ (X_t != X_{t-1}).",
        "required_parameters": [
            {"name": "values", "type": "list",
                "description": "Sequence of observations (numeric or categorical)."}
        ],
        "optional_parameters": [],
        "tag": ["ts_descriptor", "universal", "sequential"],
        "interpretation": [{
            "output": "JSON string encoding Float in [0,1]; NaN if undefined",
            "range": "[0,1]",
            "higher_is": "choppier sequence",
            "how_to_read": "0 implies constant series; close to 1 implies frequent switching or high volatility.",
            "edge_cases": "Undefined (NaN) when fewer than two non‑missing values."
        }],
    },
    {
        "name": "persistence",
        "description": "Probability that adjacent non‑missing values are equal (1 − change_rate).",
        "required_parameters": [
            {"name": "values", "type": "list",
                "description": "Sequence of observations (numeric or categorical)."}
        ],
        "optional_parameters": [],
        "tag": ["ts_descriptor", "universal", "sequential"],
        "interpretation": [{
            "output": "JSON string encoding Float in [0,1]; NaN if undefined",
            "range": "[0,1]",
            "higher_is": "more stickiness / longer runs",
            "how_to_read": "Useful to detect regime stability; complement of change_rate.",
            "edge_cases": "Undefined (NaN) when fewer than two non‑missing values."
        }],
    },
    {
        "name": "mean_run_length",
        "description": "Average length of runs of identical non‑missing values.",
        "required_parameters": [
            {"name": "values", "type": "list",
                "description": "Sequence of observations (numeric or categorical)."}
        ],
        "optional_parameters": [],
        "tag": ["ts_descriptor", "universal", "sequential"],
        "interpretation": [{
            "output": "JSON string encoding Float ≥ 1; NaN if undefined",
            "range": "[1, T] approximately",
            "higher_is": "longer regimes",
            "how_to_read": "Works with persistence; high mean_run_length suggests low switching.",
            "edge_cases": "Undefined (NaN) when there are no non‑missing values."
        }],
    },
    {
        "name": "longest_run",
        "description": "Maximum length of any run of identical non‑missing values.",
        "required_parameters": [
            {"name": "values", "type": "list",
                "description": "Sequence of observations (numeric or categorical)."}
        ],
        "optional_parameters": [],
        "tag": ["ts_descriptor", "universal", "sequential"],
        "interpretation": [{
            "output": "JSON string encoding Integer ≥ 0",
            "range": "[0, T]",
            "higher_is": "dominant plateau/regime",
            "how_to_read": "Large value indicates sustained constancy; useful for anomaly detection (stuck sensors).",
            "edge_cases": "Returns 0 when there are no non‑missing values."
        }],
    },
    {
        "name": "transition_matrix",
        "description": "Empirical first‑order transition matrix of the symbolised series.",
        "required_parameters": [
            {"name": "values", "type": "list",
                "description": "Sequence of observations (numeric or categorical)."}
        ],
        "optional_parameters": [],
        "tag": ["ts_descriptor", "universal", "sequential"],
        "interpretation": [{
            "output": "JSON string encoding 2D array (K×K) of transition probabilities",
            "range": "Probabilities in [0,1] per row",
            "higher_is": "N/A",
            "how_to_read": "Row i gives distribution of next state given current=i. Diagonal dominance implies persistence.",
            "edge_cases": "States never observed as current have undefined rows (NaN)."
        }],
    },
    {
        "name": "entropy_rate",
        "description": "First‑order entropy rate of the symbolised series in bits.",
        "required_parameters": [
            {"name": "values", "type": "list",
                "description": "Sequence of observations (numeric or categorical)."}
        ],
        "optional_parameters": [],
        "tag": ["ts_descriptor", "universal", "sequential", "information"],
        "interpretation": [{
            "output": "JSON string encoding Float ≥ 0; NaN if undefined",
            "range": "[0, log2(K)]",
            "higher_is": "more uncertainty in the next step given current state",
            "how_to_read": "0 if transitions are deterministic; larger values when transitions are more diffuse or uniform.",
            "edge_cases": "Undefined (NaN) when fewer than two non‑missing values."
        }],
    },
    {
        "name": "lagged_mutual_information",
        "description": "Mutual information (in bits) between X_t and X_{t-k} for a symbolised series.",
        "required_parameters": [
            {"name": "values", "type": "list",
                "description": "Sequence of observations (numeric or categorical)."}
        ],
        "optional_parameters": [
            {"name": "k", "type": "int",
                "description": "Positive lag k ≥ 1 (default 1).", "default": 1}
        ],
        "tag": ["ts_descriptor", "universal", "sequential", "information"],
        "interpretation": [{
            "output": "JSON string encoding Float ≥ 0; NaN if undefined",
            "range": "[0, +∞)",
            "higher_is": "stronger dependency between X_t and X_{t-k}",
            "how_to_read": "Helps choose lags; MI=0 implies independence at lag k (given discretisation).",
            "edge_cases": "Undefined (NaN) when the series has ≤k non‑missing values or k<1."
        }],
    },
    {
        "name": "dominant_period",
        "description": "Estimate the dominant period of a numeric series using the peak of its power spectrum.",
        "required_parameters": [
            {"name": "values", "type": "list",
                "description": "Numeric time series values."}
        ],
        "optional_parameters": [
            {"name": "timestamps", "type": "list",
                "description": "Datetime‑like timestamps associated with the values (optional).", "default": None}
        ],
        "tag": ["ts_descriptor", "universal", "frequency", "periodicity"],
        "interpretation": [{
            "output": "JSON string encoding Float > 0 (seconds) or NaN",
            "range": "(0, +∞) when defined",
            "higher_is": "longer cycle length",
            "how_to_read": "Use with seasonal_strength; NaN if the series is constant or too short to estimate a peak.",
            "edge_cases": "Requires at least two samples; uses FFT; assumes uniform sampling if timestamps are not provided."
        }],
    },
    {
        "name": "seasonal_strength",
        "description": "Relative strength of the dominant seasonal frequency in the power spectrum.",
        "required_parameters": [
            {"name": "values", "type": "list",
                "description": "Numeric time series values."}
        ],
        "optional_parameters": [
            {"name": "timestamps", "type": "list",
                "description": "Datetime‑like timestamps associated with the values (optional).", "default": None}
        ],
        "tag": ["ts_descriptor", "universal", "frequency", "periodicity"],
        "interpretation": [{
            "output": "JSON string encoding Float in [0,1] or NaN",
            "range": "[0,1] when defined",
            "higher_is": "stronger seasonality",
            "how_to_read": "≈0 means little seasonal signal; values >0.5 indicate seasonality dominates residual variance.",
            "edge_cases": "Requires at least two samples and a non‑constant series."
        }],
    },
    {
        "name": "spectral_entropy",
        "description": "Normalized Shannon entropy of the power spectral density.",
        "required_parameters": [
            {"name": "values", "type": "list",
                "description": "Numeric time series values."}
        ],
        "optional_parameters": [
            {"name": "timestamps", "type": "list",
                "description": "Datetime‑like timestamps (optional).", "default": None}
        ],
        "tag": ["ts_descriptor", "universal", "frequency", "information"],
        "interpretation": [{
            "output": "JSON string encoding Float in [0,1] or NaN",
            "range": "[0,1]",
            "higher_is": "flatter spectrum (noise‑like)",
            "how_to_read": "0 indicates a single frequency; values approaching 1 indicate a broad, flat spectrum.",
            "edge_cases": "Undefined (NaN) for constant series or very short sequences."
        }],
    },
    {
        "name": "spectral_flatness",
        "description": "Spectral flatness, the ratio of geometric to arithmetic mean of the power spectrum.",
        "required_parameters": [
            {"name": "values", "type": "list",
                "description": "Numeric time series values."}
        ],
        "optional_parameters": [
            {"name": "timestamps", "type": "list",
                "description": "Datetime‑like timestamps (optional).", "default": None}
        ],
        "tag": ["ts_descriptor", "universal", "frequency"],
        "interpretation": [{
            "output": "JSON string encoding Float in [0,1] or NaN",
            "range": "[0,1]",
            "higher_is": "whiter/noisier signal",
            "how_to_read": "≈0 indicates tonal or peaky spectrum; ≈1 indicates flat/white‑like noise.",
            "edge_cases": "Undefined (NaN) when the power spectrum has zero or negative values or the series is too short."
        }],
    },
    {
        "name": "distributional_change_points",
        "description": "Detect distributional change points using a sliding window chi‑square test on the symbolised series.",
        "required_parameters": [
            {"name": "values", "type": "list",
                "description": "Sequence of observations (numeric or categorical)."}
        ],
        "optional_parameters": [
            {"name": "window", "type": "int",
                "description": "Size of the sliding window for local distribution comparisons (default 50).", "default": 50},
            {"name": "threshold", "type": "float",
                "description": "Threshold multiplier on the median chi‑square statistic (default 1.0).", "default": 1.0}
        ],
        "tag": ["ts_descriptor", "universal", "changepoint"],
        "interpretation": [{
            "output": "JSON string encoding (n_change_points, max_statistic) where max_statistic is float",
            "range": "n_change_points ≥ 0; max_statistic ≥ 0",
            "higher_is": "more regime shifts (for n_change_points) or a stronger single shift (for max_statistic)",
            "how_to_read": "Use n_change_points to count how many change points exceed the threshold; max_statistic gauges the most extreme distributional shift.",
            "edge_cases": "Returns (0, NaN) if the series is too short to compute statistics (length < window)."
        }],
    },
    {
        "name": "rare_state_rate",
        "description": "Fraction of observations in states whose empirical probability falls below the default rarity threshold tau = 1/(K^2).",
        "required_parameters": [
            {"name": "values", "type": "list",
                "description": "Sequence of observations (numeric or categorical)."}
        ],
        "optional_parameters": [],
        "tag": ["ts_descriptor", "universal", "anomaly"],
        "interpretation": [{
            "output": "JSON string encoding Float in [0,1] or NaN",
            "range": "[0,1]",
            "higher_is": "more time spent in rare/unusual states",
            "how_to_read": "Useful as an anomaly frequency proxy; threshold tightens automatically as the number of states grows.",
            "edge_cases": "Undefined (NaN) when there are no non‑missing values."
        }],
    },
    {
        "name": "trend_strength",
        "description": "Fit a simple linear regression to the series and return the slope and coefficient of determination (R²).",
        "required_parameters": [
            {"name": "values", "type": "list",
                "description": "Sequence of observations (numeric or coercible to numeric)."}
        ],
        "optional_parameters": [
            {"name": "timestamps", "type": "list", "default": None,
                "description": "Optional datetime-like timestamps; defaults to index order if omitted."}
        ],
        "tag": ["ts_descriptor", "trend", "regression"],
        "interpretation": [{
            "output": "JSON string encoding {'slope': float, 'r2': float}",
            "range": "slope ∈ ℝ; r2 ∈ [0,1] when defined",
            "higher_is": "For r2, closer to 1 implies a stronger linear trend; slope sign/magnitude show direction and steepness.",
            "how_to_read": "Use slope to gauge monotonic drift (per unit time) and r2 to judge how well a straight line explains the series.",
            "edge_cases": "Returns {slope: 0.0, r2: 0.0} when timepoints have zero variance; both become NaN when fewer than two valid samples."
        }],
    },
    {
        "name": "fourier_coefficients",
        "description": "Solve for the DC offset and cosine/sine coefficients of the first N harmonics for a specified period.",
        "required_parameters": [
            {"name": "values", "type": "list",
                "description": "Sequence of observations (numeric or coercible to numeric)."},
            {"name": "period", "type": "float",
                "description": "Fundamental period in the same units as the timeline."},
            {"name": "n_harmonics", "type": "int",
                "description": "Number of harmonic pairs (cos/sin) to fit (≥ 1)."}
        ],
        "optional_parameters": [
            {"name": "timestamps", "type": "list", "default": None,
                "description": "Optional datetime-like timestamps to use instead of positional indices."}
        ],
        "tag": ["ts_descriptor", "seasonality", "frequency"],
        "interpretation": [{
            "output": "JSON string encoding {'offset': float, 'cos': [float], 'sin': [float]}",
            "range": "Offset ∈ ℝ; cosine/sine vectors length = n_harmonics",
            "higher_is": "Coefficient magnitudes indicate contribution strength of each harmonic.",
            "how_to_read": "Combine cos/sin coefficients with matching composer tools to rebuild the seasonal component.",
            "edge_cases": "Returns NaNs for all coefficients when no valid points remain after filtering."
        }],
    },
    {
        "name": "autocorrelation_profile",
        "description": "Compute Pearson autocorrelation coefficients for the requested lags.",
        "required_parameters": [
            {"name": "values", "type": "list",
                "description": "Sequence of observations (numeric or coercible to numeric)."}
        ],
        "optional_parameters": [
            {"name": "lags", "type": "list", "default": "(1, 2, 3, 7, 14)",
                "description": "Iterable of non-negative lags to evaluate (0 -> 1.0 by definition)."}
        ],
        "tag": ["ts_descriptor", "dependence", "autocorrelation"],
        "interpretation": [{
            "output": "JSON string encoding list[float] with one entry per requested lag",
            "range": "Each coefficient ∈ [-1,1] or NaN if undefined",
            "higher_is": "Large |value| indicates strong linear dependence at that lag.",
            "how_to_read": "Use positive spikes to justify repeating patterns and negative spikes for alternating behaviour.",
            "edge_cases": "Lags ≥ length of the valid series return NaN; lag 0 returns 1.0."
        }],
    },
    {
        "name": "spike_index",
        "description": "Detect spikes using a rolling MAD heuristic and report their frequency and average magnitude.",
        "required_parameters": [
            {"name": "values", "type": "list",
                "description": "Sequence of observations (numeric or coercible to numeric)."}
        ],
        "optional_parameters": [
            {"name": "window", "type": "int", "default": 24,
                "description": "Size of the centred rolling window used for the median/MAD baseline (≥ 1)."},
            {"name": "threshold", "type": "float", "default": 3.0,
                "description": "MAD-based z-score threshold for flagging spikes (> 0)."}
        ],
        "tag": ["ts_descriptor", "quality", "anomaly"],
        "interpretation": [{
            "output": "JSON string encoding {'spike_rate': float, 'avg_magnitude': float}",
            "range": "spike_rate ∈ [0,1]; avg_magnitude ≥ 0",
            "higher_is": "Higher spike_rate means more bursts; larger avg_magnitude means stronger excursions.",
            "how_to_read": "Use spike_rate to decide whether to add explicit pulse components and avg_magnitude to set their amplitude.",
            "edge_cases": "Returns {NaN, NaN} when the series has no valid observations; avg_magnitude is 0 when no spikes exceed the threshold."
        }],
    },
    {
        "name": "missing_segments",
        "description": "Identify contiguous runs of missing values and return them as [start, length] pairs.",
        "required_parameters": [
            {"name": "values", "type": "list",
                "description": "Sequence of observations (any type)."}
        ],
        "optional_parameters": [],
        "tag": ["ts_descriptor", "quality", "missingness"],
        "interpretation": [{
            "output": "JSON string encoding list[[int, int], ...] (start index, run length)",
            "range": "start ≥ 0, length ≥ 1; empty list if no missing runs",
            "higher_is": "Longer lengths imply prolonged outages; more entries mean fragmented missingness.",
            "how_to_read": "Feed directly into masking composers to recreate observed gaps.",
            "edge_cases": "Returns [] when there are no missing samples."
        }],
    },
    {
        "name": "pairwise_error",
        "description": "Compare two aligned series and compute MAE, RMSE, bias and Pearson correlation.",
        "required_parameters": [
            {"name": "y_true", "type": "list",
                "description": "Reference/ground-truth series."},
            {"name": "y_pred", "type": "list",
                "description": "Comparison or forecast series (must match y_true length)."}
        ],
        "optional_parameters": [],
        "tag": ["ts_descriptor", "evaluation", "comparison"],
        "interpretation": [{
            "output": "JSON string encoding {'mae': float, 'rmse': float, 'bias': float, 'corr': float}",
            "range": "mae, rmse ≥ 0; bias ∈ ℝ; corr ∈ [-1,1] or NaN",
            "higher_is": "Lower mae/rmse/bias is better; corr closer to 1 indicates stronger alignment.",
            "how_to_read": "Use to validate composer outputs against history or other baselines.",
            "edge_cases": "All metrics become NaN when no overlapping finite values exist; corr is NaN when either series is constant."
        }],
    },
    {
        "name": "lempel_ziv_complexity",
        "description": "Normalized Lempel–Ziv complexity: number of distinct substrings divided by series length.",
        "required_parameters": [
            {"name": "values", "type": "list",
                "description": "Sequence of observations (numeric or categorical)."}
        ],
        "optional_parameters": [],
        "tag": ["ts_descriptor", "universal", "complexity", "information"],
        "interpretation": [{
            "output": "JSON string encoding Float in (0,1] approximately; NaN if undefined",
            "range": "(0,1]",
            "higher_is": "more novel patterns / higher algorithmic complexity",
            "how_to_read": "Low values indicate repetitive structure; high values indicate randomness or rich structure.",
            "edge_cases": "Undefined (NaN) when the sequence is empty."
        }],
    },
    {
        "name": "normalized_compression_ratio",
        "description": "Ratio of compressed size to raw size for the string representation of the series.",
        "required_parameters": [
            {"name": "values", "type": "list",
                "description": "Sequence of observations (numeric or categorical)."}
        ],
        "optional_parameters": [],
        "tag": ["ts_descriptor", "universal", "complexity", "information"],
        "interpretation": [{
            "output": "JSON string encoding Float ≥ 0; NaN if undefined",
            "range": "[0, +∞) (typically ≤1 for numeric bytes)",
            "higher_is": "less compressible (more random/noisy)",
            "how_to_read": "<1 indicates compressible structure; ~1 indicates little redundancy.",
            "edge_cases": "Undefined (NaN) when the series is empty."
        }],
    },
    {
        "name": "block_entropy_growth",
        "description": "Slope of the block entropy growth curve using blocks of length 1–5 after default symbolisation.",
        "required_parameters": [
            {"name": "values", "type": "list",
                "description": "Sequence of observations (numeric or categorical)."}
        ],
        "optional_parameters": [],
        "tag": ["ts_descriptor", "universal", "complexity", "information"],
        "interpretation": [{
            "output": "JSON string encoding Float ≥ 0; NaN if undefined",
            "range": "[0, +∞)",
            "higher_is": "richer higher‑order structure or randomness",
            "how_to_read": "Near 0 implies deterministic/repetitive sequences; high values imply complex dependencies.",
            "edge_cases": "Undefined (NaN) when the series is too short to compute block entropies."
        }],
    },
]
