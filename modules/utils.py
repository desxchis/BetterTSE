"""Utility functions for time series parsing, conversion, and visualization."""

from __future__ import annotations

import importlib
import json
import os
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from langchain_core.messages.base import get_msg_title_repr
from langchain_core.utils.interactive_env import is_interactive_env

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def parse_user_input(user_input: str) -> tuple[dict[str, object], dict[str, object], str]:
    """Parse the simulated payload and return history, forecast, and optional context."""

    payload = json.loads(user_input)

    history_df = pd.DataFrame(
        {
            "timestamp": payload.get("history", {}).get("timestamps"),
            "value": payload.get("history", {}).get("values"),
        }
    )

    history_df["timestamp"] = pd.to_datetime(
        history_df["timestamp"], utc=True, errors="coerce")
    history_df = history_df.dropna(subset=["timestamp"])
    history_df = history_df.sort_values("timestamp")
    history_timestamps = history_df["timestamp"].dt.strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    ).tolist()
    numeric_values = pd.to_numeric(
        history_df["value"], errors="coerce", downcast="float"
    ).astype("float64").tolist()

    forecast_section = payload.get("forecast") or {}
    forecast_timestamps = forecast_section.get("timestamps") or []

    if forecast_section.get("values") is not None:
        forecast_numeric_values = pd.to_numeric(
            forecast_section["values"], errors="coerce", downcast="float"
        ).astype("float64").tolist()
    else:
        # Initialize the forecast values to the mean of the historical values
        mean_value = np.nanmean(numeric_values)
        forecast_numeric_values = [mean_value] * len(forecast_timestamps)

    history: dict[str, object] = {
        "values": numeric_values,
        "timestamps": history_timestamps,
    }

    forecast: dict[str, object] = {
        "values": forecast_numeric_values,
        "timestamps": forecast_timestamps
    }

    context_text = payload.get("context") or payload.get(
        "series_description") or ""

    return history, forecast, context_text


def timestamps_to_numeric(timestamps: list[str]) -> np.ndarray:
    """Convert ISO timestamp strings to a numeric offset series."""
    if not timestamps:
        return np.array([], dtype=float)
    try:
        dt = np.array([np.datetime64(ts)
                      for ts in timestamps], dtype="datetime64[ns]")
        return ((dt - dt[0]) / np.timedelta64(1, "s")).astype(float)
    except Exception:
        return np.arange(len(timestamps), dtype=float)


def to_python(value: Any) -> Any:
    """Convert NumPy scalars/arrays to native Python structures."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {k: to_python(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_python(v) for v in value]
    return value


def pretty_print(message, printout=True):
    """Format and optionally print a LangChain message."""
    if isinstance(message, tuple):
        title = message
    elif isinstance(message.content, list):
        title = get_msg_title_repr(message.type.title(
        ).upper() + " Message", bold=is_interactive_env())
        if message.name is not None:
            title += f"\nName: {message.name}"

        for i in message.content:
            if i["type"] == "text":
                title += f"\n{i['text']}\n"
            elif i["type"] == "tool_use":
                title += f"\nTool: {i['name']}"
                title += f"\nInput: {i['input']}"
        if printout:
            print(f"{title}")
    else:
        title = get_msg_title_repr(
            message.type.title() + " Message", bold=is_interactive_env())
        if message.name is not None:
            title += f"\nName: {message.name}"
        title += f"\n\n{message.content}"
        if printout:
            print(f"{title}")
    return title


def textify_api_dict(api_dict):
    """Convert a nested API dictionary to a nicely formatted string."""
    lines = []
    for category, methods in api_dict.items():
        lines.append("=" * (len("Import file: ") + len(category)))
        lines.append(f"Import file: {category}")
        lines.append("for the following methods:")
        for method in methods:
            lines.append(f"Method: {method.get('name', 'N/A')}")
            lines.append(
                f"  Description: {method.get('description', 'No description provided.')}")

            # Process required parameters
            req_params = method.get("required_parameters")
            if req_params:
                lines.append("  Required Parameters:")
                for param in req_params:
                    param_name = param.get("name", "N/A")
                    param_type = param.get("type", "N/A")
                    param_desc = param.get("description", "No description")
                    param_default = param.get("default", "None")
                    lines.append(
                        f"    - {param_name} ({param_type}): {param_desc} [Default: {param_default}]")

            # Process optional parameters
            opt_params = method.get("optional_parameters")
            if opt_params:
                lines.append("  Optional Parameters:")
                for param in opt_params:
                    param_name = param.get("name", "N/A")
                    param_type = param.get("type", "N/A")
                    param_desc = param.get("description", "No description")
                    param_default = param.get("default", "None")
                    lines.append(
                        f"    - {param_name} ({param_type}): {param_desc} [Default: {param_default}]")

            lines.append("")  # Empty line between methods
        lines.append("")  # Extra empty line after each category

    return "\n".join(lines)


def read_module2api():
    """Load tool description modules and return a mapping of module names to API specs."""
    fields = [
        "ts_describers",
        "ts_composers",
    ]

    module2api = {}
    for field in fields:
        module_name = f"tool.tool_description.{field}"
        module = importlib.import_module(module_name)
        module2api[f"tool.{field}"] = module.description
    return module2api


def plot_series(filename, input_ts, output_ts, predicted_ts, save_folder):
    """Plot input series, ground truth, and prediction."""
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(input_ts)), input_ts,
             label="Input Time Series", marker='o')
    plt.plot(range(len(input_ts), len(input_ts) + len(output_ts)),
             output_ts, label="Ground Truth", marker='o')
    plt.plot(range(len(input_ts), len(input_ts) + len(output_ts)),
             output_ts, label="Predicted", linestyle='dashed')
    plt.legend()
    plt.title(f"Prediction for {filename}")
    plt.xlabel("Time Steps")
    plt.ylabel("Value")
    plt.grid()
    plt.savefig(os.path.join(save_folder, filename+'.png'))
    plt.close()


def plot_forecast_steps(
    filename,
    input_ts,
    output_ts,
    payload_forecast,
    pipeline_outputs,
    save_folder,
):
    """Plot successive forecast refinements alongside the history and target."""
    forecast_versions = []

    def _normalize(values, target_length):
        if values is None:
            return None
        cleaned = []
        for value in values:
            if value is None:
                cleaned.append(np.nan)
                continue
            try:
                cleaned.append(float(value))
            except (TypeError, ValueError):
                cleaned.append(np.nan)

        if not cleaned:
            return None

        cleaned = np.asarray(cleaned, dtype=float)
        if np.isnan(cleaned).all():
            return None

        if target_length and len(cleaned) != target_length:
            source_idx = np.linspace(0, 1, len(cleaned))
            target_idx = np.linspace(0, 1, target_length)
            cleaned = np.interp(target_idx, source_idx, cleaned)

        return np.asarray(cleaned, dtype=float)

    horizon_length = len(output_ts)
    initial_forecast = None
    if isinstance(payload_forecast, dict):
        initial_forecast = _normalize(
            payload_forecast.get("values"), horizon_length)
    if initial_forecast is not None:
        forecast_versions.append(("Initial Forecast", initial_forecast))

    if isinstance(pipeline_outputs, list):
        step_idx = 1
        for event in pipeline_outputs:
            if not isinstance(event, dict):
                continue
            if event.get("type") != "composer.output":
                continue
            values = _normalize(event.get("forecast_values"), horizon_length)
            if values is None:
                continue
            tool_name = event.get("tool")
            label = f"Composer Step {step_idx}"
            if tool_name:
                label += f" ({tool_name})"
            forecast_versions.append((label, values))
            step_idx += 1

    if not forecast_versions:
        return

    os.makedirs(save_folder, exist_ok=True)
    plt.figure(figsize=(12, 6))

    history_indices = np.arange(len(input_ts))
    forecast_indices = np.arange(len(input_ts), len(input_ts) + horizon_length)

    plt.plot(history_indices, input_ts, label="Input Time Series", marker="o")
    plt.plot(forecast_indices, output_ts, label="Ground Truth", marker="o")

    cmap = plt.get_cmap("viridis")
    color_positions = np.linspace(0.2, 0.9, len(forecast_versions))

    for color_pos, (label, forecast_values) in zip(color_positions, forecast_versions):
        if len(forecast_values) != horizon_length:
            continue
        plt.plot(
            forecast_indices,
            forecast_values,
            label=label,
            color=cmap(color_pos),
            linestyle="--",
        )

    plt.title(f"Forecast Refinement for {filename}")
    plt.xlabel("Time Steps")
    plt.ylabel("Value")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()

    save_path = os.path.join(save_folder, f"{filename}_forecast_steps.png")
    plt.savefig(save_path)
    plt.close()
