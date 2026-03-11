# BetterTSF

BetterTSF is an agentic time series forecasting workflow that combines a language model planner with deterministic time series tools. The core idea is to keep the LLM focused on choosing *what* to do (which tool, where to apply it, and how strongly), while the actual series updates are performed by transparent, reproducible functions. The system loops over three stages - describe, plan, compose - until the forecast is good enough.

## How it works (end-to-end)

1) **Input parsing**
`modules.utils.parse_user_input` accepts a JSON payload containing a history window, a forecast horizon, and optional context. It normalizes timestamps, coerces numeric values, and (if forecast values are omitted) seeds the forecast with the mean of the historical values.

2) **Describe**
The agent first computes time series descriptors from `tool/ts_describers.py` (length, missingness, gaps, entropy, seasonality, trend strength, etc.). This step produces a compact, structured summary of both history and the current forecast. The results are recorded in the pipeline trace so every subsequent decision can reference them. The amount of context the planner sees is controlled by `planner_context_mode` (`series_only`, `descriptors_only`, or `hybrid`).

3) **Plan**
The planner prompt is built in `agent/prompts.py`. It injects the current state (series values and/or descriptor summaries) and a catalog of available composer tools (`tool/tool_description/ts_composers.py`). The LLM must output either:
- a `<step>{...}</step>` JSON payload naming a tool, time window, and blend coefficient, or
- a `<solution>...</solution>` explaining why the forecast should stop changing.

4) **Compose**
The composer node in `agent/nodes.py` executes the selected function from `tool/ts_composers.py`, producing a candidate series `ts_tmp`. It then blends that candidate into the current forecast with:

```
forecast = beta * ts_tmp + (1 - beta) * forecast
```

The update is applied only over the requested window, and diff statistics are captured so the planner can see the impact of each iteration.

5) **Loop or finish**
The workflow is implemented with LangGraph (`StateGraph`) and cycles through describe -> plan -> compose until the planner emits a solution or exceeds retry limits. Each run yields both the final forecast and a detailed pipeline trace.

## How the pieces connect

- **`agent/`** is the control plane:
  - `agent.py` defines the `A1` class that configures the LangGraph workflow, manages state, and provides the main `go()` entry point.
  - `nodes.py` contains the three workflow nodes (`node_describer`, `node_planner`, `node_composer`) that implement the describe/plan/compose logic.
  - `prompts.py` generates planner prompts and collects descriptor outputs.
- **`tool/ts_describers.py`** provides the analytics layer (descriptors). These outputs become the evidence the planner cites when choosing the next action.
- **`tool/ts_composers.py`** provides the synthesis layer (forecast builders, seasonal signals, filters, masks). The planner selects from these tools; the composer executes them deterministically.
- **`tool/tool_description/`** mirrors the tool functions with JSON-style metadata. The planner uses these descriptions to decide which tool fits the current gap in the forecast.
- **`tool/ts_processor.py`** is a preprocessing toolbox (imputation, outlier handling, transforms). It is not invoked by the main agent loop today, but it can be used to prepare inputs or extend the toolset.
- **`modules/llm.py`** abstracts LLM access. It supports OpenAI-style APIs and local vLLM endpoints, and normalizes LangChain message formats.
- **`modules/utils.py`** provides parsing, timestamp handling, visualization, and conversion utilities that keep inputs and tool outputs consistent.
- **`tryout.py`** is a runnable example that feeds a sample series into the agent, then plots the forecast and the step-by-step updates.

## Input format

The agent expects a JSON string shaped like:

```
{
  "series_id": "optional-id",
  "context": "optional text",
  "history": {
    "timestamps": ["ISO-8601", ...],
    "values": [float | null, ...]
  },
  "forecast": {
    "timestamps": ["ISO-8601", ...],
    "values": [float | null, ...]   // optional; if missing, mean(history) is used
  }
}
```

## Example run

`tryout.py` builds a local agent, feeds it a fridge temperature series, and saves plots under `visualizations/tryout/`.

```
python tryout.py
```

## Outputs you can inspect

- **Final forecast**: `agent.ts_forecast` (values + horizon timestamps).
- **Pipeline trace**: `agent.latest_pipeline_snapshot` or the `pipeline` yields from `A1.go(...)`.
- **Workflow graph**: `workflow_graph.png` (auto-generated when `A1` configures its LangGraph).
- **Visualizations**: in `visualizations/tryout/` when running `tryout.py`.
