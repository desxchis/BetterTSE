"""
time_series_descriptors.py
-----------------------------------

This module provides a collection of functions for computing
model‑agnostic descriptors of time series data.  The functions
implemented here are based on the universal descriptors outlined
in the conversation.  Each descriptor operates on either the
timestamps associated with a series, the observed values, or
both.  All functions are self contained and rely only on
NumPy and pandas, avoiding any heavy external dependencies to
ensure broad compatibility.

To use these functions you should pass in either a pandas Series
or numpy array of values.  For descriptors requiring timestamps
(e.g. sampling irregularity), pass a second pandas Series or
numpy array containing the timestamps.  Missing values should
be represented as ``np.nan`` in numeric series.  Categorical
series are supported by symbolization using quantile bins when
necessary.

Example usage::

    import pandas as pd
    import numpy as np
    from time_series_descriptors import length, mean_gap

    values = pd.Series([1.0, 2.0, np.nan, 4.0],
                       index=pd.to_datetime(["2020-01-01", "2020-01-02",
                                            "2020-01-04", "2020-01-07"]))
    # Length of the series
    n = length(values)
    # Compute mean gap between timestamps
    m_gap = mean_gap(values.index)

The module also exposes helper functions for symbolizing numeric
series and constructing transition matrices.  Wherever possible
the functions gracefully handle missing data and irregular
sampling.  For LLM consumption, every public descriptor returns a
JSON-formatted string representation of its result.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import json
import math
import zlib
from typing import Any, Iterable, List, Tuple, Dict


DEFAULT_DISCRETIZATION_BINS = 10
DEFAULT_BLOCK_LENGTH = 5


def _to_serializable(value: Any) -> Any:
    """Convert common scientific Python objects into JSON-serialisable forms."""
    if isinstance(value, (np.generic,)):
        return value.item()
    if isinstance(value, np.ndarray):
        return [_to_serializable(v) for v in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [_to_serializable(v) for v in value]
    if isinstance(value, dict):
        return {k: _to_serializable(v) for k, v in value.items()}
    if isinstance(value, pd.Series):
        return [_to_serializable(v) for v in value.tolist()]
    if isinstance(value, pd.Index):
        return [_to_serializable(v) for v in value.tolist()]
    if isinstance(value, (pd.Timestamp, pd.Timedelta)):
        return value.isoformat()
    return value


def _format_tool_output(value: Any) -> str:
    """Convert tool outputs to string representations suitable for LLM prompts."""
    if isinstance(value, pd.DataFrame):
        return value.to_json(orient='split', date_format='iso')
    if isinstance(value, pd.Series):
        serialisable = _to_serializable(value)
    else:
        serialisable = _to_serializable(value)
    try:
        return json.dumps(serialisable, allow_nan=True)
    except TypeError:
        return str(serialisable)

###############################################################################
# Helper functions
###############################################################################


def _to_numpy_array(x: Any) -> np.ndarray:
    """Convert various sequence types to a NumPy array.

    Parameters
    ----------
    x : Any
        A pandas Series, numpy array, or list‑like object.

    Returns
    -------
    np.ndarray
        A one‑dimensional NumPy array.
    """
    if isinstance(x, pd.Series):
        return x.to_numpy()
    elif isinstance(x, np.ndarray):
        return x
    else:
        return np.asarray(x)


def _clean_values(values: Any) -> np.ndarray:
    """Convert values to a 1‑D NumPy array and ensure missing values are ``np.nan``.

    This helper will convert pandas Series, lists and numpy arrays into a
    consistent form.  Non numeric values will be left as is; numeric
    values will have missing entries coerced to ``np.nan``.

    Parameters
    ----------
    values : Any
        Input series or array.

    Returns
    -------
    np.ndarray
        Cleaned array.
    """
    arr = _to_numpy_array(values)
    # If dtype is object, attempt to coerce numeric values and leave
    # categorical values unchanged.  Using pandas to_numeric will
    # convert uncoercible values to NaN.
    if arr.dtype == object:
        # Try converting to float; non convertible values become NaN
        try:
            arr_num = pd.to_numeric(arr, errors='coerce').to_numpy()
            # if there are both strings and numbers, arr_num will contain NaNs
            # but original arr still contains categorical states.  In that
            # case return arr unchanged.
            if np.isnan(arr_num).sum() > 0 and np.any(pd.isna(arr) == False):
                return arr
            return arr_num
        except Exception:
            return arr
    return arr


def _compute_gaps(timestamps: Any) -> np.ndarray:
    """Compute successive time differences from a sequence of timestamps.

    Parameters
    ----------
    timestamps : pandas.Series, numpy.ndarray or list
        Timestamps must be coercible to pandas Timedelta or numeric
        differences.  The ordering of timestamps is preserved.

    Returns
    -------
    np.ndarray
        Array of successive differences converted to seconds.
    """
    # Convert to pandas datetime if necessary
    if isinstance(timestamps, pd.DatetimeIndex):
        dt_index = timestamps
    elif isinstance(timestamps, pd.Series) and isinstance(timestamps.dtype, pd.DatetimeTZDtype):
        dt_index = timestamps
    elif isinstance(timestamps, (pd.Series, list, np.ndarray)):
        dt_index = pd.to_datetime(timestamps)
    else:
        raise TypeError(
            "timestamps must be datetime‑like or convertible to datetime")

    if len(dt_index) < 2:
        return np.array([])
    deltas = np.diff(dt_index.astype('int64'))  # nanoseconds
    # Convert nanoseconds to seconds for interpretability
    return deltas.astype(np.float64) / 1e9


def _timestamps_to_numeric(timestamps: Any) -> np.ndarray:
    """Convert timestamps to seconds since epoch as a float array."""
    if timestamps is None:
        raise ValueError("timestamps must not be None")
    if isinstance(timestamps, pd.DatetimeIndex):
        dt_index = timestamps
        valid_mask = ~pd.isna(dt_index)
        numeric = dt_index.view('int64').to_numpy(dtype=np.float64, copy=False) / 1e9
    else:
        dt_series = pd.to_datetime(timestamps)
        valid_mask = ~dt_series.isna()
        numeric = dt_series.astype('int64').to_numpy(dtype=np.float64, copy=False) / 1e9
    mask_array = valid_mask.to_numpy() if hasattr(valid_mask, "to_numpy") else np.asarray(valid_mask)
    numeric[~mask_array] = np.nan
    return numeric


def _symbolize_numeric_series(values: np.ndarray) -> np.ndarray:
    """Discretize a numeric series into quantile bins.

    The function uses pandas `qcut` to assign each numeric value to one of
    ``bins`` equally populated buckets.  Missing values are represented
    as ``None`` and remain unchanged.

    Parameters
    ----------
    values : np.ndarray
        Numeric values to discretize.  Missing values should be ``np.nan``.
    Returns
    -------
    np.ndarray
        Array of bin labels (integers from 0 to DEFAULT_DISCRETIZATION_BINS‑1)
        or None for missing.
    """
    # Separate missing and non missing indices
    arr = _clean_values(values)
    is_nan = pd.isna(arr)
    # If the series is entirely missing, return array of None
    if is_nan.all():
        return np.array([None] * len(arr), dtype=object)
    # Use pandas qcut to compute quantile bins; duplicates are handled by ranking
    ser = pd.Series(arr[~is_nan].astype(float))
    # If there are fewer unique values than bins, reduce bins
    n_unique = ser.nunique(dropna=True)
    actual_bins = (min(DEFAULT_DISCRETIZATION_BINS, n_unique)
                   if n_unique > 0 else DEFAULT_DISCRETIZATION_BINS)
    # qcut might throw if there are too few unique values; fallback to cut
    try:
        labels, bin_edges = pd.qcut(
            ser, q=actual_bins, retbins=True, labels=False, duplicates='drop')
    except ValueError:
        # Fall back to equal width bins
        labels, bin_edges = pd.cut(
            ser, bins=actual_bins, labels=False, retbins=True)
    # Create output array and fill with bin labels
    out = np.array([None] * len(arr), dtype=object)
    out[~is_nan] = labels.to_numpy()
    out[is_nan] = None
    return out


def _get_state_sequence(values: Any) -> np.ndarray:
    """Convert a series of numeric or categorical values to a sequence of discrete states.

    Numeric values are discretized using quantile bins via
    :func:`_symbolize_numeric_series`.  Categorical values (dtype object or
    string) are left unchanged.  Missing values are represented as None.

    Parameters
    ----------
    values : array‑like
        Series of values to convert.
    Returns
    -------
    np.ndarray
        Array of discrete states (strings, ints or None).
    """
    arr = _clean_values(values)
    # If dtype is numeric
    if np.issubdtype(arr.dtype, np.number):
        return _symbolize_numeric_series(arr)
    # Categorical values: convert to string representation; treat missing as None
    states = []
    for v in arr:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            states.append(None)
        else:
            states.append(str(v))
    return np.array(states, dtype=object)


def _compress_bytes(data: bytes) -> int:
    """Return the length of zlib compressed bytes for normalized compression ratio.

    Parameters
    ----------
    data : bytes
        Raw bytes to compress.

    Returns
    -------
    int
        Length of compressed byte string.
    """
    return len(zlib.compress(data))


def _unique_substrings(seq: str) -> int:
    """Count the number of distinct substrings in a string using Lempel–Ziv78 algorithm.

    The Lempel–Ziv complexity measures the number of new substrings
    encountered when scanning through the sequence from left to right
    and adding each new substring to a dictionary.  It is used as
    a proxy for complexity or novelty of the sequence.  This
    implementation follows a simple version of the LZ78 algorithm.

    Parameters
    ----------
    seq : str
        Input string.

    Returns
    -------
    int
        Number of distinct substrings encountered.
    """
    dictionary: Dict[str, int] = {}
    w = ""
    count = 0
    for c in seq:
        wc = w + c
        if wc not in dictionary:
            dictionary[wc] = count
            count += 1
            w = ""
        else:
            w = wc
    # Add last substring if nonempty
    if w:
        count += 1
    return count


def _block_entropy(values: List[Any], max_block: int = 5) -> float:
    """Compute the slope of block entropies for increasing block lengths.

    For m = 1..max_block, this function computes the Shannon entropy
    H(m) of contiguous blocks of length m in the symbolized sequence,
    then estimates the slope (linear regression) of H(m) vs m.  The
    slope serves as a simple measure of higher‑order structure: a
    higher slope indicates more complex dependency between symbols.

    Parameters
    ----------
    values : List[Any]
        Sequence of discrete values (states).
    max_block : int
        Maximum block length to consider (default 5).

    Returns
    -------
    float
        Slope of H(m) vs m for m >= 1.
    """
    # Convert sequence to string tokens separated by spaces to avoid collisions
    seq = [str(v) for v in values if v is not None]
    if len(seq) == 0:
        return float('nan')
    entropies = []
    lengths = []
    for m in range(1, min(max_block, len(seq)) + 1):
        blocks = [tuple(seq[i:i + m]) for i in range(len(seq) - m + 1)]
        if not blocks:
            continue
        # Count occurrences of each block
        counts = pd.Series(blocks).value_counts(normalize=True)
        probs = counts.values
        H_m = -np.sum(probs * np.log2(probs))
        entropies.append(H_m)
        lengths.append(m)
    if len(lengths) < 2:
        return float('nan')
    # Linear regression slope
    x = np.array(lengths)
    y = np.array(entropies)
    # Fit least squares line y = a + b x
    A = np.vstack([x, np.ones(len(x))]).T
    b, a = np.linalg.lstsq(A, y, rcond=None)[0]
    return b


###############################################################################
# Core descriptors applicable to all time series
###############################################################################

def summary_stats(values: Any) -> str:
    """Return (mean, std, min, max) for a numeric series, ignoring missing values.

    Non-numeric inputs are coerced to numeric (invalid entries become NaN).
    The standard deviation is computed with ddof=0 (population estimate).
    If there are no valid numeric values, all results are NaN.

    Parameters
    ----------
    values : array-like
        Input series (numeric or coercible to numeric).

    Returns
    -------
    (float, float, float, float)
        Tuple containing (mean, std, min, max). Each element is NaN if undefined.
    """
    arr = _clean_values(values)
    # Coerce non-numeric to numeric; invalid parse -> NaN
    if not np.issubdtype(arr.dtype, np.number):
        arr = pd.to_numeric(arr, errors='coerce').to_numpy()

    vals = arr.astype(float)
    # If no finite values, return all NaNs
    finite_mask = np.isfinite(vals)
    if finite_mask.sum() == 0:
        nan = float('nan')
        return (nan, nan, nan, nan)

    mean_val = float(np.nanmean(vals))
    std_val = float(np.nanstd(vals, ddof=0))
    min_val = float(np.nanmin(vals))
    max_val = float(np.nanmax(vals))
    result = (mean_val, std_val, min_val, max_val)
    return _format_tool_output(result)


def length(values: Any) -> str:
    """Return the number of observations in the series.

    Parameters
    ----------
    values : array‑like
        Input time series values.

    Returns
    -------
    int
        Number of observations (including missing values).
    """
    result = len(_to_numpy_array(values))
    return _format_tool_output(result)


def time_span(timestamps: Any) -> str:
    """Return the time span of a series in seconds.

    The time span is defined as the difference between the maximum and
    minimum timestamp.  If there is only a single timestamp, the span is
    zero.

    Parameters
    ----------
    timestamps : array‑like
        Sequence of datetime‑like timestamps.

    Returns
    -------
    float
        Span in seconds.
    """
    # Convert to datetime index
    if isinstance(timestamps, pd.DatetimeIndex):
        dt_index = timestamps
    else:
        dt_index = pd.to_datetime(timestamps)
    if len(dt_index) == 0:
        return _format_tool_output(0.0)
    span = (dt_index.max() - dt_index.min()).total_seconds()
    return _format_tool_output(span)


def mean_gap(timestamps: Any) -> str:
    """Return the mean gap between successive timestamps in seconds.

    Parameters
    ----------
    timestamps : array‑like
        Sequence of datetime‑like timestamps.

    Returns
    -------
    float
        Mean of successive differences (in seconds); returns NaN if fewer than
        two timestamps.
    """
    gaps = _compute_gaps(timestamps)
    if gaps.size == 0:
        return _format_tool_output(float('nan'))
    result = float(np.nanmean(gaps))
    return _format_tool_output(result)


def median_gap(timestamps: Any) -> str:
    """Return the median gap between successive timestamps in seconds.

    Parameters
    ----------
    timestamps : array‑like
        Sequence of datetime‑like timestamps.

    Returns
    -------
    float
        Median of successive differences (in seconds); returns NaN if fewer than
        two timestamps.
    """
    gaps = _compute_gaps(timestamps)
    if gaps.size == 0:
        return _format_tool_output(float('nan'))
    result = float(np.nanmedian(gaps))
    return _format_tool_output(result)


def gap_cv(timestamps: Any) -> str:
    """Return the coefficient of variation of sampling gaps.

    The coefficient of variation is defined as the ratio of the standard
    deviation to the mean of gaps.  It measures irregularity in sampling.

    Parameters
    ----------
    timestamps : array‑like
        Sequence of datetime‑like timestamps.

    Returns
    -------
    float
        Coefficient of variation; returns NaN if fewer than two timestamps or
        if the mean gap is zero.
    """
    gaps = _compute_gaps(timestamps)
    if gaps.size == 0:
        return _format_tool_output(float('nan'))
    mean_gap_val = np.nanmean(gaps)
    if mean_gap_val == 0:
        return _format_tool_output(float('nan'))
    result = float(np.nanstd(gaps) / mean_gap_val)
    return _format_tool_output(result)


def max_gap(timestamps: Any) -> str:
    """Return the maximum gap between successive timestamps in seconds.

    Parameters
    ----------
    timestamps : array‑like
        Sequence of datetime‑like timestamps.

    Returns
    -------
    float
        Maximum gap (seconds); returns NaN if fewer than two timestamps.
    """
    gaps = _compute_gaps(timestamps)
    if gaps.size == 0:
        return _format_tool_output(float('nan'))
    result = float(np.nanmax(gaps))
    return _format_tool_output(result)


def gap_entropy(timestamps: Any) -> str:
    """Return the Shannon entropy of binned gap sizes.

    Gaps are binned into a default number of equal-frequency buckets; the
    entropy measures diversity of sampling intervals.  Returns NaN if
    there are fewer than two timestamps or if all gaps are identical.

    Parameters
    ----------
    timestamps : array‑like
        Sequence of datetime‑like timestamps.
    Returns
    -------
    float
        Shannon entropy (log base 2).
    """
    gaps = _compute_gaps(timestamps)
    if gaps.size == 0:
        return _format_tool_output(float('nan'))
    # Discretize gaps into bins using qcut
    try:
        categories = pd.qcut(gaps, q=min(DEFAULT_DISCRETIZATION_BINS, len(gaps)),
                             labels=False, duplicates='drop')
    except ValueError:
        categories = pd.cut(gaps, bins=DEFAULT_DISCRETIZATION_BINS, labels=False)
    counts = pd.Series(categories).value_counts(normalize=True)
    probs = counts.values
    if len(probs) == 0:
        return _format_tool_output(float('nan'))
    result = float(-np.sum(probs * np.log2(probs)))
    return _format_tool_output(result)


def _missing_rate_numeric(values: Any) -> float:
    arr = _to_numpy_array(values)
    if len(arr) == 0:
        return 0.0
    missing_mask = pd.isna(arr)
    return float(missing_mask.sum() / len(arr))


def missing_rate(values: Any) -> str:
    """Return the proportion of missing values in the series.

    Missing values are defined as ``np.nan`` for numeric series or ``None``
    for object series.

    Parameters
    ----------
    values : array‑like
        Input series.

    Returns
    -------
    float
        Fraction of missing values; 0 if there are no observations.
    """
    result = _missing_rate_numeric(values)
    return _format_tool_output(result)


def missing_runs(values: Any) -> str:
    """Return the number of contiguous runs of missing values.

    A missing run is one or more consecutive missing values separated by
    non missing values.  If the series is entirely missing the run
    count is one.

    Parameters
    ----------
    values : array‑like
        Input series.

    Returns
    -------
    int
        Number of missing runs.
    """
    arr = _to_numpy_array(values)
    if len(arr) == 0:
        return _format_tool_output(0)
    missing_mask = pd.isna(arr)
    runs = 0
    prev_missing = False
    for m in missing_mask:
        m_bool = bool(m)
        if m_bool and not prev_missing:
            runs += 1
        prev_missing = m_bool
    return _format_tool_output(int(runs))


def longest_missing_run(values: Any) -> str:
    """Return the length (number of points) of the longest missing run.

    Parameters
    ----------
    values : array‑like
        Input series.

    Returns
    -------
    int
        Maximum length of consecutive missing values; zero if no missing.
    """
    arr = _to_numpy_array(values)
    if len(arr) == 0:
        return _format_tool_output(0)
    missing_mask = pd.isna(arr)
    max_run = 0
    current_run = 0
    for m in missing_mask:
        if bool(m):
            current_run += 1
            if current_run > max_run:
                max_run = current_run
        else:
            current_run = 0
    return _format_tool_output(int(max_run))


###############################################################################
# Value distribution descriptors (requires discretization for numeric series)
###############################################################################

def alphabet_size(values: Any) -> str:
    """Return the number of unique states in the symbolized series.

    Parameters
    ----------
    values : array‑like
        Numeric or categorical values.
    Returns
    -------
    int
        Number of unique states (excluding missing values).  Returns 0 if the
        series is entirely missing.
    """
    states = _get_state_sequence(values)
    unique_states = {s for s in states if s is not None}
    result = len(unique_states)
    return _format_tool_output(result)


def mode_probability(values: Any) -> str:
    """Return the probability of the most common state in the symbolized series.

    Parameters
    ----------
    values : array‑like
        Numeric or categorical values.
    Returns
    -------
    float
        Maximum state probability; returns NaN if the series has no non missing
        values.
    """
    states = _get_state_sequence(values)
    valid = [s for s in states if s is not None]
    if not valid:
        return _format_tool_output(float('nan'))
    counts = pd.Series(valid).value_counts(normalize=True)
    return _format_tool_output(float(counts.max()))


def shannon_entropy(values: Any) -> str:
    """Return the Shannon entropy of the symbolized value distribution.

    Parameters
    ----------
    values : array‑like
        Numeric or categorical values.
    Returns
    -------
    float
        Shannon entropy (log2).  Returns NaN if there are no non missing values.
    """
    states = _get_state_sequence(values)
    valid = [s for s in states if s is not None]
    if not valid:
        return _format_tool_output(float('nan'))
    counts = pd.Series(valid).value_counts(normalize=True)
    probs = counts.values
    result = float(-np.sum(probs * np.log2(probs)))
    return _format_tool_output(result)


def gini_simpson_index(values: Any) -> str:
    """Return the Gini–Simpson diversity index of the symbolized series.

    The index is defined as 1 − ∑_i p_i^2, where p_i are state
    probabilities.

    Parameters
    ----------
    values : array‑like
        Numeric or categorical values.
    Returns
    -------
    float
        Gini–Simpson index; returns NaN if there are no non missing values.
    """
    states = _get_state_sequence(values)
    valid = [s for s in states if s is not None]
    if not valid:
        return _format_tool_output(float('nan'))
    counts = pd.Series(valid).value_counts(normalize=True)
    probs = counts.values
    result = float(1.0 - np.sum(probs ** 2))
    return _format_tool_output(result)


def evenness(values: Any) -> str:
    """Return Pielou's evenness of the symbolized distribution.

    The evenness is computed as H / log(K), where H is the Shannon
    entropy and K the number of non empty states.  It is undefined
    (returns NaN) if the series has no non missing values.

    Parameters
    ----------
    values : array‑like
        Numeric or categorical values.
    Returns
    -------
    float
        Evenness index between 0 and 1; NaN if undefined.
    """
    states = _get_state_sequence(values)
    valid = [s for s in states if s is not None]
    if not valid:
        return _format_tool_output(float('nan'))
    counts = pd.Series(valid).value_counts(normalize=True)
    probs = counts.values
    H = -np.sum(probs * np.log2(probs))
    K = len(counts)
    if K <= 1:
        return _format_tool_output(float('nan'))
    result = float(H / np.log2(K))
    return _format_tool_output(result)


###############################################################################
# Sequential structure descriptors (markovian behaviour)
###############################################################################

def _change_rate_numeric(values: Any) -> float:
    states = _get_state_sequence(values)
    # Filter out missing values for change detection
    non_missing = [s for s in states if s is not None]
    if len(non_missing) < 2:
        return float('nan')
    diff_count = sum(1 for i in range(1, len(non_missing))
                     if non_missing[i] != non_missing[i - 1])
    return float(diff_count / (len(non_missing) - 1))


def change_rate(values: Any) -> str:
    """Return the proportion of adjacent observations that differ.

    Parameters
    ----------
    values : array‑like
        Numeric or categorical series.
    Returns
    -------
    float
        Fraction of adjacent pairs where X_t != X_{t-1}.  Returns NaN if fewer
        than two non missing values.
    """
    result = _change_rate_numeric(values)
    return _format_tool_output(result)


def persistence(values: Any) -> str:
    """Return the probability that adjacent states are equal (1 − change_rate).

    Parameters
    ----------
    values : array‑like
        Numeric or categorical series.
    Returns
    -------
    float
        Probability of consecutive states being equal.  Returns NaN if fewer
        than two non missing values.
    """
    cr = _change_rate_numeric(values)
    if math.isnan(cr):
        return _format_tool_output(float('nan'))
    result = 1.0 - cr
    return _format_tool_output(result)


def mean_run_length(values: Any) -> str:
    """Return the average length of runs of constant state.

    A run is defined as a maximal sequence of identical states.

    Parameters
    ----------
    values : array‑like
        Numeric or categorical series.
    Returns
    -------
    float
        Average run length; returns NaN if there are no non missing values.
    """
    states = _get_state_sequence(values)
    non_missing = [s for s in states if s is not None]
    if not non_missing:
        return _format_tool_output(float('nan'))
    runs = []
    current_run = 1
    for i in range(1, len(non_missing)):
        if non_missing[i] == non_missing[i - 1]:
            current_run += 1
        else:
            runs.append(current_run)
            current_run = 1
    runs.append(current_run)
    result = float(np.mean(runs))
    return _format_tool_output(result)


def longest_run(values: Any) -> str:
    """Return the length of the longest run of identical states.

    Parameters
    ----------
    values : array‑like
        Numeric or categorical series.
    Returns
    -------
    int
        Length of the longest run; returns 0 if there are no non missing values.
    """
    states = _get_state_sequence(values)
    non_missing = [s for s in states if s is not None]
    if not non_missing:
        return _format_tool_output(0)
    max_run = 1
    current_run = 1
    for i in range(1, len(non_missing)):
        if non_missing[i] == non_missing[i - 1]:
            current_run += 1
            if current_run > max_run:
                max_run = current_run
        else:
            current_run = 1
    return _format_tool_output(int(max_run))


def _transition_matrix_from_states(states: List[Any]) -> pd.DataFrame:
    if len(states) < 2:
        return pd.DataFrame(dtype=float)
    uniq = sorted(set(states))
    counts = {s: {t: 0 for t in uniq} for s in uniq}
    for i in range(len(states) - 1):
        s = states[i]
        t = states[i + 1]
        counts[s][t] += 1
    df = pd.DataFrame.from_dict(counts, orient='index', columns=uniq, dtype=float)
    df = df.div(df.sum(axis=1).replace(0, np.nan), axis=0)
    return df


def transition_matrix(values: Any) -> str:
    """Return the empirical first order transition matrix for the symbolized series."""
    states = _get_state_sequence(values)
    non_missing = [s for s in states if s is not None]
    df = _transition_matrix_from_states(non_missing)
    return _format_tool_output(df)


def entropy_rate(values: Any) -> str:
    """Return the first order entropy rate of the symbolized series.

    The entropy rate H(X_t|X_{t-1}) is computed as

        H = -∑_{i,j} p(i) p(j|i) log2 p(j|i),

    where p(i) is the stationary distribution estimated by relative
    frequencies of states and p(j|i) is the transition probability
    obtained from the empirical transition matrix.

    Parameters
    ----------
    values : array‑like
        Numeric or categorical series.
    Returns
    -------
    float
        Entropy rate in bits; returns NaN if there are fewer than two
        non missing values.
    """
    states = _get_state_sequence(values)
    non_missing = [s for s in states if s is not None]
    if len(non_missing) < 2:
        return _format_tool_output(float('nan'))
    # Compute state probabilities
    p_state = pd.Series(non_missing).value_counts(normalize=True)
    # Compute conditional probabilities p(j|i)
    P = _transition_matrix_from_states(non_missing)
    H = 0.0
    for i, p_i in p_state.items():
        # Skip states with zero transitions
        if i not in P.index:
            continue
        row = P.loc[i]
        probs = row.dropna().to_numpy()
        probs = probs[probs > 0]
        # Skip rows with zero total probability
        if probs.size == 0:
            continue
        H_i = -np.sum(probs * np.log2(probs))
        H += p_i * H_i
    return _format_tool_output(float(H))


def lagged_mutual_information(values: Any, k: int = 1) -> str:
    """Compute mutual information between X_t and X_{t-k} for a symbolized series.

    This function discretizes the series if numeric and returns the mutual
    information between the sequence and itself shifted by k.  Mutual
    information is defined as

        MI = ∑_{i,j} p(i,j) log2 [ p(i,j) / (p(i)p(j)) ].

    Parameters
    ----------
    values : array‑like
        Numeric or categorical series.
    k : int, optional
        Lag to compute mutual information (default 1).  Positive k uses
        X_{t-k}.
    Returns
    -------
    float
        Mutual information in bits; returns NaN if series is too short.
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    states = _get_state_sequence(values)
    non_missing = [s for s in states if s is not None]
    if len(non_missing) <= k:
        return _format_tool_output(float('nan'))
    # Construct pairs (X_t, X_{t-k})
    x_t = non_missing[k:]
    x_tm = non_missing[:-k]
    df = pd.DataFrame({'x_t': x_t, 'x_tm': x_tm})
    joint_counts = df.value_counts(normalize=True)
    # Compute marginal distributions
    p_x = pd.Series(x_t).value_counts(normalize=True)
    p_y = pd.Series(x_tm).value_counts(normalize=True)
    MI = 0.0
    for (i, j), p_ij in joint_counts.items():
        MI += p_ij * np.log2(p_ij / (p_x[i] * p_y[j]))
    return _format_tool_output(float(MI))


###############################################################################
# Frequency and periodicity descriptors
###############################################################################

def dominant_period(values: Any, timestamps: Any = None) -> str:
    """Estimate the dominant period of a numeric time series.

    This function computes the dominant cycle length by locating the peak of
    the Fourier power spectrum of the detrended series.  If timestamps are
    provided, they are used to convert the dominant frequency to seconds;
    when timestamps are absent the sampling interval is assumed to be
    uniform and equal to one unit.  Missing values are linearly
    interpolated before spectral analysis.  The routine falls back to a
    simple autocorrelation method if the FFT computation fails.  The
    dominant period is returned in seconds.

    Parameters
    ----------
    values : array‑like
        Numeric time series values.  Missing values are linearly
        interpolated.
    timestamps : array‑like, optional
        Sequence of datetime‑like timestamps corresponding to the values.

    Returns
    -------
    float
        Dominant period in seconds; returns NaN if the series is constant or
        too short to estimate.
    """
    arr = _clean_values(values)
    # Remove missing values and align timestamps if provided
    if timestamps is not None:
        ts = pd.to_datetime(timestamps)
        ser = pd.Series(arr, index=ts).astype(float)
        ser = ser.sort_index()
        # Interpolate missing values linearly
        ser = ser.interpolate(method='time').bfill().ffill()
        y = ser.to_numpy()
        t = (ser.index - ser.index[0]).total_seconds().to_numpy()
    else:
        ser = pd.Series(arr).astype(float)
        ser = ser.interpolate().bfill().ffill()
        y = ser.to_numpy()
        # Assume unit sampling interval for lack of timestamps
        t = np.arange(len(y), dtype=float)
    if len(y) < 2:
        return _format_tool_output(float('nan'))
    # Detrend by subtracting mean
    y_d = y - np.mean(y)
    if np.allclose(y_d, 0):
        return _format_tool_output(float('nan'))
    # Use FFT to estimate the dominant frequency; fall back to autocorrelation if FFT fails
    try:
        # Determine Nyquist frequency
        dt = np.median(np.diff(t))
        fs = 1.0 / dt
        # Compute next power of two for zero padding
        nfft = int(2 ** np.ceil(np.log2(len(y_d))))
        freqs = np.fft.rfftfreq(nfft, d=dt)
        fft_vals = np.fft.rfft(y_d, n=nfft)
        psd = np.abs(fft_vals) ** 2
        # Ignore zero frequency (DC)
        if len(freqs) <= 1:
            return _format_tool_output(float('nan'))
        dominant_idx = np.argmax(psd[1:]) + 1
        dominant_freq = freqs[dominant_idx]
        if dominant_freq == 0:
            return _format_tool_output(float('nan'))
        return _format_tool_output(float(1.0 / dominant_freq))
    except Exception:
        # Fallback: simple autocorrelation method
        acf = np.correlate(y_d, y_d, mode='full')
        acf = acf[acf.size // 2:]
        # find first local maximum beyond lag 0
        if len(acf) < 3:
            return _format_tool_output(float('nan'))
        # Normalize
        acf = acf / acf[0]
        # Find peaks
        diffs = np.diff(acf)
        # first zero crossing of derivative
        peaks = np.where((diffs[:-1] > 0) & (diffs[1:] <= 0))[0] + 1
        if len(peaks) == 0:
            return _format_tool_output(float('nan'))
        # Convert lag to time
        lag = peaks[0]
        return _format_tool_output(float(dt * lag))


def seasonal_strength(values: Any, timestamps: Any = None) -> str:
    """Estimate the relative strength of the dominant seasonal frequency.

    This is computed as the ratio of spectral power at the dominant
    frequency to the total power.  Returns NaN if the series is constant
    or too short.

    Parameters
    ----------
    values : array‑like
        Numeric time series values.
    timestamps : array‑like, optional
        Sequence of datetime‑like timestamps.

    Returns
    -------
    float
        Seasonal strength between 0 and 1; NaN if undefined.
    """
    arr = _clean_values(values)
    if timestamps is not None:
        ts = pd.to_datetime(timestamps)
        ser = pd.Series(arr, index=ts).astype(float)
        ser = ser.sort_index().interpolate(method='time').bfill().ffill()
        y = ser.to_numpy()
        t = (ser.index - ser.index[0]).total_seconds().to_numpy()
    else:
        ser = pd.Series(arr).astype(float).interpolate().bfill().ffill()
        y = ser.to_numpy()
        t = np.arange(len(y), dtype=float)
    if len(y) < 2:
        return _format_tool_output(float('nan'))
    y_d = y - np.mean(y)
    if np.allclose(y_d, 0):
        return _format_tool_output(float('nan'))
    # Compute FFT to approximate periodogram
    dt = np.median(np.diff(t))
    nfft = int(2 ** np.ceil(np.log2(len(y_d))))
    freqs = np.fft.rfftfreq(nfft, d=dt)
    fft_vals = np.fft.rfft(y_d, n=nfft)
    psd = np.abs(fft_vals) ** 2
    # Exclude zero frequency
    if len(psd) <= 1:
        return _format_tool_output(float('nan'))
    total_power = np.sum(psd[1:])
    if total_power == 0:
        return _format_tool_output(float('nan'))
    dominant_idx = np.argmax(psd[1:]) + 1
    dominant_power = psd[dominant_idx]
    return _format_tool_output(float(dominant_power / total_power))


def spectral_entropy(values: Any, timestamps: Any = None) -> str:
    """Compute the normalized spectral entropy of a numeric series.

    The spectral entropy is the Shannon entropy of the normalized power
    spectral density of the detrended series.  After normalizing the
    power spectrum to sum to one, the entropy is divided by log2(K)
    (where K is the number of non‑DC frequency bins) so that the result
    lies in the interval [0, 1].  A value of 0 indicates that the power
    is concentrated at a single frequency (perfectly periodic), whereas
    values approaching 1 indicate a broad, flat spectrum (noise‑like).

    Parameters
    ----------
    values : array‑like
        Numeric time series values.
    timestamps : array‑like, optional
        Sequence of datetime‑like timestamps.

    Returns
    -------
    float
        Normalized spectral entropy; NaN if undefined.
    """
    arr = _clean_values(values)
    if timestamps is not None:
        ts = pd.to_datetime(timestamps)
        ser = pd.Series(arr, index=ts).astype(float).sort_index()
        ser = ser.interpolate(method='time').bfill().ffill()
        y = ser.to_numpy()
        t = (ser.index - ser.index[0]).total_seconds().to_numpy()
    else:
        ser = pd.Series(arr).astype(float).interpolate().bfill().ffill()
        y = ser.to_numpy()
        t = np.arange(len(y), dtype=float)
    if len(y) < 2:
        return _format_tool_output(float('nan'))
    y_d = y - np.mean(y)
    if np.allclose(y_d, 0):
        return _format_tool_output(float('nan'))
    dt = np.median(np.diff(t))
    nfft = int(2 ** np.ceil(np.log2(len(y_d))))
    fft_vals = np.fft.rfft(y_d, n=nfft)
    psd = np.abs(fft_vals) ** 2
    psd = psd[1:]  # drop DC component
    if psd.size == 0 or np.all(psd == 0):
        return _format_tool_output(float('nan'))
    # Normalize to probability distribution
    probs = psd / np.sum(psd)
    H = -np.sum(probs * np.log2(probs))
    maxH = np.log2(len(probs))
    if maxH == 0:
        return _format_tool_output(float('nan'))
    return _format_tool_output(float(H / maxH))


def spectral_flatness(values: Any, timestamps: Any = None) -> str:
    """Compute the spectral flatness (geometric mean / arithmetic mean of power spectrum).

    Spectral flatness quantifies how noise‑like a signal is.  Values near 0
    indicate tonal signals (power concentrated at few frequencies), while
    values near 1 indicate flat (white) noise.

    Parameters
    ----------
    values : array‑like
        Numeric time series values.
    timestamps : array‑like, optional
        Sequence of datetime‑like timestamps.

    Returns
    -------
    float
        Spectral flatness ratio; NaN if undefined.
    """
    arr = _clean_values(values)
    if timestamps is not None:
        ts = pd.to_datetime(timestamps)
        ser = pd.Series(arr, index=ts).astype(float).sort_index()
        ser = ser.interpolate(method='time').bfill().ffill()
        y = ser.to_numpy()
        t = (ser.index - ser.index[0]).total_seconds().to_numpy()
    else:
        ser = pd.Series(arr).astype(float).interpolate().bfill().ffill()
        y = ser.to_numpy()
        t = np.arange(len(y), dtype=float)
    if len(y) < 2:
        return _format_tool_output(float('nan'))
    y_d = y - np.mean(y)
    if np.allclose(y_d, 0):
        return _format_tool_output(float('nan'))
    dt = np.median(np.diff(t))
    nfft = int(2 ** np.ceil(np.log2(len(y_d))))
    psd = np.abs(np.fft.rfft(y_d, n=nfft)) ** 2
    psd = psd[1:]
    if psd.size == 0 or np.any(psd <= 0):
        return _format_tool_output(float('nan'))
    gm = np.exp(np.mean(np.log(psd)))
    am = np.mean(psd)
    return _format_tool_output(float(gm / am))


###############################################################################
# Change point and anomaly descriptors
###############################################################################

def distributional_change_points(values: Any, window: int = 50, threshold: float = 1.0) -> str:
    """Detect simple distributional change points using sliding window chi‑square tests.

    This heuristic approach slides a window of size ``window`` across the
    symbolized series and compares the distribution of states in the left
    and right halves.  A chi‑square statistic is computed at each split
    point and change points are counted where the statistic exceeds
    ``threshold`` times the global median.  The maximum chi‑square
    statistic is also returned.

    Parameters
    ----------
    values : array‑like
        Numeric or categorical series.
    window : int, optional
        Window size for local distribution comparisons (default 50).  Must
        be >= 2.
    threshold : float, optional
        Threshold multiplier on the median chi‑square statistic to flag a
        change point (default 1.0).

    Returns
    -------
    (int, float)
        A tuple containing (number of change points, maximum chi‑square
        statistic).  If the series is too short, returns (0, NaN).
    """
    if window < 2:
        raise ValueError("window must be >= 2")
    states = _get_state_sequence(values)
    non_missing = [s for s in states if s is not None]
    n = len(non_missing)
    if n < window:
        return _format_tool_output((0, float('nan')))
    # Unique states
    uniq = list(set(non_missing))
    m = len(uniq)
    chi_stats = []
    # Precompute counts for sliding window using cumulative counts
    # For each state, compute cumulative count up to each position
    indices = {u: i for i, u in enumerate(uniq)}
    cum_counts = np.zeros((m, n + 1), dtype=int)
    for idx, s in enumerate(non_missing):
        row = indices[s]
        cum_counts[:, idx + 1] = cum_counts[:, idx]
        cum_counts[row, idx + 1] += 1
    for pos in range(window, n - window + 1):
        # Left half [pos-window, pos)
        left_counts = cum_counts[:, pos] - cum_counts[:, pos - window]
        right_counts = cum_counts[:, pos + window] - cum_counts[:, pos]
        # Expected counts under null: both halves share the same distribution
        total_counts = left_counts + right_counts
        # Skip positions with zero counts
        if total_counts.sum() == 0:
            chi_stats.append(0.0)
            continue
        # Expected counts for left half
        expected_left = total_counts * (window / (2 * window))
        expected_right = total_counts * (window / (2 * window))
        # Compute chi‑square statistic: ∑ (obs‑exp)^2 / exp
        chi = 0.0
        # avoid division by zero
        for obs, exp in zip(np.concatenate([left_counts, right_counts]),
                            np.concatenate([expected_left, expected_right])):
            if exp > 0:
                chi += ((obs - exp) ** 2) / exp
        chi_stats.append(chi)
    if not chi_stats:
        return _format_tool_output((0, float('nan')))
    chi_stats = np.array(chi_stats)
    median_chi = np.median(chi_stats)
    # Identify change points where chi exceeds threshold * median
    change_points = np.sum(chi_stats > threshold * median_chi)
    max_chi = float(np.max(chi_stats))
    result = (int(change_points), max_chi)
    return _format_tool_output(result)


def rare_state_rate(values: Any) -> str:
    """Return the fraction of observations in states with probability < tau.

    Rare states are identified using a fixed threshold ``tau = 1/(K^2)``,
    where ``K`` is the number of unique states in the symbolized series.

    Parameters
    ----------
    values : array‑like
        Numeric or categorical series.

    Returns
    -------
    float
        Fraction of observations in rare states; NaN if there are no
        non missing values.
    """
    states = _get_state_sequence(values)
    valid = [s for s in states if s is not None]
    n = len(valid)
    if n == 0:
        return _format_tool_output(float('nan'))
    counts = pd.Series(valid).value_counts(normalize=True)
    K = len(counts)
    if K == 0:
        return _format_tool_output(float('nan'))
    tau = 1.0 / (K ** 2)
    rare_states = counts[counts < tau].index.tolist()
    rare_count = sum(1 for s in valid if s in rare_states)
    return _format_tool_output(float(rare_count / n))


###############################################################################
# Trend and correlation descriptors
###############################################################################


def trend_strength(values: Any, timestamps: Any = None) -> str:
    """Return the slope and R² of a best-fit line through the series."""
    arr = _clean_values(values)
    numeric = np.asarray(pd.to_numeric(arr, errors='coerce'), dtype=float)
    n = numeric.size
    if timestamps is None:
        times = np.arange(n, dtype=float)
    else:
        times = _timestamps_to_numeric(timestamps)
        if times.shape[0] != n:
            raise ValueError("timestamps and values must have the same length")
    mask = np.isfinite(numeric) & np.isfinite(times)
    if mask.sum() < 2:
        result = {"slope": float('nan'), "r2": float('nan')}
        return _format_tool_output(result)
    x = times[mask]
    y = numeric[mask]
    x_mean = float(np.mean(x))
    y_mean = float(np.mean(y))
    x_centered = x - x_mean
    denom = float(np.dot(x_centered, x_centered))
    if denom == 0.0:
        slope = 0.0
        r2 = 0.0
    else:
        slope = float(np.dot(x_centered, y - y_mean) / denom)
        intercept = float(y_mean - slope * x_mean)
        y_pred = intercept + slope * x
        ss_res = float(np.sum((y - y_pred) ** 2))
        ss_tot = float(np.sum((y - y_mean) ** 2))
        if ss_tot == 0.0:
            r2 = 0.0
        else:
            r2 = float(max(0.0, 1.0 - ss_res / ss_tot))
    result = {"slope": float(slope), "r2": r2}
    return _format_tool_output(result)


def fourier_coefficients(values: Any, period: float, n_harmonics: int, timestamps: Any = None) -> str:
    """Return Fourier offset, cosine and sine coefficients for a given period."""
    if period <= 0:
        raise ValueError("period must be positive")
    if n_harmonics < 1:
        raise ValueError("n_harmonics must be at least 1")
    arr = _clean_values(values)
    numeric = np.asarray(pd.to_numeric(arr, errors='coerce'), dtype=float)
    n = numeric.size
    if timestamps is None:
        times = np.arange(n, dtype=float)
    else:
        times = _timestamps_to_numeric(timestamps)
        if times.shape[0] != n:
            raise ValueError("timestamps and values must have the same length")
    mask = np.isfinite(numeric) & np.isfinite(times)
    if mask.sum() == 0:
        result = {
            "offset": float('nan'),
            "cos": [float('nan')] * n_harmonics,
            "sin": [float('nan')] * n_harmonics,
        }
        return _format_tool_output(result)
    x = times[mask]
    y = numeric[mask]
    omega = 2.0 * np.pi / period
    cols = 2 * n_harmonics + 1
    design = np.ones((x.shape[0], cols), dtype=float)
    for idx in range(n_harmonics):
        h = idx + 1
        angle = omega * h * x
        design[:, 1 + idx] = np.cos(angle)
        design[:, 1 + n_harmonics + idx] = np.sin(angle)
    coeffs, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    offset = float(coeffs[0])
    cos_coeffs = [float(c) for c in coeffs[1:1 + n_harmonics]]
    sin_coeffs = [float(s) for s in coeffs[1 + n_harmonics:1 + 2 * n_harmonics]]
    result = {"offset": offset, "cos": cos_coeffs, "sin": sin_coeffs}
    return _format_tool_output(result)


def autocorrelation_profile(values: Any, lags: Iterable[int] = (1, 2, 3, 7, 14)) -> str:
    """Return the autocorrelation coefficients for the requested lags."""
    arr = _clean_values(values)
    numeric = np.asarray(pd.to_numeric(arr, errors='coerce'), dtype=float)
    valid = numeric[np.isfinite(numeric)]
    lag_list = list(lags)
    if valid.size == 0:
        results = [float('nan')] * len(lag_list)
        return _format_tool_output(results)
    results: List[float] = []
    for lag in lag_list:
        if lag < 0:
            raise ValueError("lags must be non-negative")
        if lag == 0:
            results.append(1.0)
            continue
        if lag >= valid.size:
            results.append(float('nan'))
            continue
        x = valid[:-lag]
        y = valid[lag:]
        if x.size < 2:
            results.append(float('nan'))
            continue
        x_dev = x - x.mean()
        y_dev = y - y.mean()
        denom = np.sqrt(np.sum(x_dev ** 2) * np.sum(y_dev ** 2))
        if denom == 0.0:
            results.append(float('nan'))
        else:
            results.append(float(np.dot(x_dev, y_dev) / denom))
    return _format_tool_output(results)


###############################################################################
# Event and quality descriptors
###############################################################################


def spike_index(values: Any, window: int = 24, threshold: float = 3.0) -> str:
    """Return spike rate and average spike magnitude based on rolling MAD."""
    if window < 1:
        raise ValueError("window must be at least 1")
    if threshold <= 0:
        raise ValueError("threshold must be positive")
    arr = _clean_values(values)
    numeric = np.asarray(pd.to_numeric(arr, errors='coerce'), dtype=float)
    series = pd.Series(numeric)
    valid_mask = series.notna()
    n_valid = int(valid_mask.sum())
    if n_valid == 0:
        result = {"spike_rate": float('nan'), "avg_magnitude": float('nan')}
        return _format_tool_output(result)
    rolling_median = series.rolling(window=window, center=True, min_periods=1).median()
    abs_dev = (series - rolling_median).abs()
    mad = abs_dev.rolling(window=window, center=True, min_periods=1).median()
    scale = 1.4826 * mad.fillna(0.0)
    eps = 1e-8
    spike_mask = (abs_dev > threshold * (scale + eps)) & valid_mask
    spike_count = int(spike_mask.sum())
    if spike_count == 0:
        avg_mag = 0.0
    else:
        avg_mag = float(abs_dev[spike_mask].mean())
    spike_rate = float(spike_count / n_valid)
    result = {"spike_rate": spike_rate, "avg_magnitude": avg_mag}
    return _format_tool_output(result)


def missing_segments(values: Any) -> str:
    """Return contiguous segments of missing values as [start, length] pairs."""
    arr = _to_numpy_array(values)
    missing_mask = pd.isna(arr)
    segments: List[List[int]] = []
    idx = 0
    while idx < len(missing_mask):
        if missing_mask[idx]:
            start = idx
            while idx < len(missing_mask) and missing_mask[idx]:
                idx += 1
            segments.append([int(start), int(idx - start)])
        else:
            idx += 1
    return _format_tool_output(segments)


###############################################################################
# Evaluation descriptors
###############################################################################


def pairwise_error(y_true: Any, y_pred: Any) -> str:
    """Return MAE, RMSE, bias and correlation between reference and prediction."""
    true_arr = _to_numpy_array(y_true)
    pred_arr = _to_numpy_array(y_pred)
    if true_arr.shape[0] != pred_arr.shape[0]:
        raise ValueError("y_true and y_pred must have the same length")
    true_num = np.asarray(pd.to_numeric(true_arr, errors='coerce'), dtype=float)
    pred_num = np.asarray(pd.to_numeric(pred_arr, errors='coerce'), dtype=float)
    mask = np.isfinite(true_num) & np.isfinite(pred_num)
    if mask.sum() == 0:
        result = {
            "mae": float('nan'),
            "rmse": float('nan'),
            "bias": float('nan'),
            "corr": float('nan'),
        }
        return _format_tool_output(result)
    true_vals = true_num[mask]
    pred_vals = pred_num[mask]
    diff = pred_vals - true_vals
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    bias = float(np.mean(diff))
    if true_vals.size < 2:
        corr = float('nan')
    else:
        true_dev = true_vals - true_vals.mean()
        pred_dev = pred_vals - pred_vals.mean()
        denom = np.sqrt(np.sum(true_dev ** 2) * np.sum(pred_dev ** 2))
        corr = float(np.dot(true_dev, pred_dev) / denom) if denom > 0 else float('nan')
    result = {"mae": mae, "rmse": rmse, "bias": bias, "corr": corr}
    return _format_tool_output(result)


###############################################################################
# Complexity and information descriptors
###############################################################################

def lempel_ziv_complexity(values: Any) -> str:
    """Return the normalized Lempel–Ziv complexity of the symbolized series.

    The complexity is defined as the number of distinct substrings
    (dictionary size) encountered when scanning the sequence, divided by
    the sequence length.  A higher value indicates greater complexity.

    Parameters
    ----------
    values : array‑like
        Numeric or categorical series.
    Returns
    -------
    float
        Normalized Lempel–Ziv complexity; NaN if series is empty.
    """
    states = _get_state_sequence(values)
    seq = ''.join(['#' if s is None else str(s) + '|' for s in states])
    if len(seq) == 0:
        return _format_tool_output(float('nan'))
    lzc = _unique_substrings(seq)
    return _format_tool_output(float(lzc / len(states)))


def normalized_compression_ratio(values: Any) -> str:
    """Return the normalized compression ratio of the raw byte representation of the series.

    This metric measures compressibility by comparing the length of
    compressed data to the original raw byte length.  Lower values
    indicate more compressible (less complex) sequences.

    Parameters
    ----------
    values : array‑like
        Numeric or categorical series.

    Returns
    -------
    float
        Normalized compression ratio (compressed size / original size);
        NaN if the series is empty.
    """
    arr = _to_numpy_array(values)
    if len(arr) == 0:
        return _format_tool_output(float('nan'))
    # Convert to bytes; use string representation separated by commas
    raw_str = ','.join(['' if pd.isna(v) else str(v) for v in arr])
    raw_bytes = raw_str.encode('utf-8')
    compressed_length = _compress_bytes(raw_bytes)
    if len(raw_bytes) == 0:
        return _format_tool_output(float('nan'))
    return _format_tool_output(float(compressed_length / len(raw_bytes)))


def block_entropy_growth(values: Any) -> str:
    """Return the slope of the block entropy growth curve of the symbolized series.

    Uses block entropies for block lengths 1 through 5 and performs a
    simple linear regression to estimate the rate of increase.  A higher
    slope suggests stronger higher‑order dependencies.

    Parameters
    ----------
    values : array‑like
        Numeric or categorical series.
    Returns
    -------
    float
        Slope of block entropy vs block length; NaN if series is too short.
    """
    states = _get_state_sequence(values)
    seq = [s for s in states if s is not None]
    result = float(_block_entropy(seq, max_block=DEFAULT_BLOCK_LENGTH))
    return _format_tool_output(result)


###############################################################################
# End of module
###############################################################################
