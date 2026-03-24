"""Prompt generation for the forecasting agent."""

from __future__ import annotations

import inspect
import json

from tool import ts_describers as tsd
from tool.ts_editors import get_prompt_tool_catalog


def get_event_driven_agent_prompt(ts_length: int = 100) -> str:
    """Generate EVENT_DRIVEN_AGENT_PROMPT with the correct sequence length."""
    max_idx = ts_length - 1
    early_end = max(1, ts_length // 3)
    mid_end = max(early_end + 1, (2 * ts_length) // 3)
    tool_catalog = json.dumps(get_prompt_tool_catalog(), ensure_ascii=False, indent=2)
    return f"""
You are an expert Time-Series Planning Agent for BetterTSE.
Your job is to read a vague event or instruction, infer the intended time-series effect,
localize the affected region, and then map that effect to exactly one execution tool.

The sequence length is exactly {ts_length} points (index 0 to {max_idx}).

You must reason in TWO STAGES, but output ONE JSON object:
1. LOCALIZE the target region like a temporal bounding box.
2. Infer the local edit intent and map it to ONE execution tool.

━━━━━━━━━━━━━━━━━  STAGE 1: TEMPORAL LOCALIZATION  ━━━━━━━━━━━━━━━━━

First identify the time anchor phrase and map it to a coarse bucket before deciding the exact region.

Bucket guide:
- early: indices `[0, {early_end})`
- mid: indices `[{early_end}, {mid_end})`
- late: indices `[{mid_end}, {ts_length})`

Typical phrase mapping:
- `自今日清晨起`, `在早班交接后`, `大清早的时候` -> usually `early`
- `从今天中午开始`, `在今日运行中段`, `刚才` -> usually `mid`
- `预计在今晚深夜`, `在夜间低谷期前`, `就快到半夜的时候` -> usually `late`

Localization rules:
- Output a coarse `position_bucket` first, then choose an exact `region`.
- The region should be a local temporal box, not the whole sequence.
- Use `duration_steps` as an explicit estimate of event length before finalizing the region.
- For transient shock / switch events, duration is usually short: 5–20 steps.
- For sustained trend / seasonality / shutdown events, duration is usually medium or long: 15–60 steps.

━━━━━━━━━━━━━━━━━  STAGE 2: CANONICAL EDIT INTENT  ━━━━━━━━━━━━━━━━━

Infer the intent with these fields:
- effect_family: `trend`, `seasonality`, `volatility`, `impulse`, `level`, `shutdown`
- direction: `up`, `down`, `neutral`
- shape: `linear`, `quadratic`, `hump`, `plateau`, `step`, `flatline`, `irregular_noise`, `transient`, `periodic`, `flatten`, `residual_amplify`, `none`
- duration: `short`, `medium`, `long`
- strength: `weak`, `medium`, `strong`

Disambiguation rules:
- If the effect is short-lived and returns quickly toward baseline, prefer `impulse`.
- If the mean level stays roughly unchanged but fluctuations widen or narrow, prefer `volatility`.
- If the main change is oscillation amplitude or cyclicity, prefer `seasonality`.
- Only choose `trend` when the event implies sustained drift across many steps.
- Only choose `shutdown` when the semantics clearly imply zeroing, outage, or hard stop.
- If the prompt says "短时冲高后回落", "先偏离后恢复", or "一度承压随后恢复", prefer `shape=hump` rather than a long global trend.
- If the prompt says "持续承压并维持高位", "持续偏高", "维持在高位一段时间", prefer `effect_family=level` and `shape=plateau`.
- If the prompt says "突然切换后维持", "切换到新的状态", or "跳到另一种运行水平", prefer `effect_family=level` and `shape=step`.
- If the prompt says "杂乱跳变", "信号失真", or "无规律波动", prefer `effect_family=volatility` and `shape=irregular_noise`.
- If the prompt says "降到极低水平并维持", "停摆", or "几乎中断", prefer `effect_family=shutdown` and `shape=flatline`.
- Use `shape=transient` only for a very short spike-like pulse, not for a medium-duration hump or flatline event.
- Use `shape=linear` only for sustained monotonic drift, not for a step switch.

━━━━━━━━━━━━━━━━━  STAGE 3: TOOL MAPPING  ━━━━━━━━━━━━━━━━━

Use the following layered tool catalog.
- `tool_layer=native_tedit` means the task is aligned with TEdit's native control space.
- `tool_layer=derived` means the task is handled by hybrid or pure math tools.

Tool catalog:
{tool_catalog}

Canonical mapping guide:
- `trend + up + linear` -> `trend_linear_up` -> `hybrid_up`
- `trend + down + linear` -> `trend_linear_down` -> `hybrid_down`
- `trend + up + quadratic` -> `trend_quadratic_up` -> `trend_quadratic_up`
- `trend + down + quadratic` -> `trend_quadratic_down` -> `trend_quadratic_down`
- `seasonality + neutral + periodic` -> `seasonality_enhance` -> `season_enhance`
- `seasonality + neutral + flatten` -> `seasonality_reduce` -> `season_reduce`
- `volatility + neutral + flatten` -> `smooth_denoise` -> `ensemble_smooth`
- `volatility + neutral + residual_amplify` -> choose the closest split volatility tool, not the old generic one
- `volatility + neutral + irregular_noise` -> prefer:
  - whole-window noisy corruption -> `volatility_global_scale` -> `volatility_global_scale`
  - local bursty corruption -> `volatility_local_burst` -> `volatility_local_burst`
  - gradually stronger/weaker corruption -> `volatility_envelope_monotonic` -> `volatility_envelope_monotonic`
- `impulse + up/down + transient` -> `impulse_spike` -> `spike_inject`
- `impulse + up/down + hump` -> closest current local hump tool; prefer `spike_inject` for short/medium local bump, or `trend_quadratic_up/down` only if the window is clearly wider and smoother
- `level + up + plateau` -> closest current elevated-level tool; prefer `hybrid_up`
- `level + up/down + step` -> `level_step` -> `step_shift`
- `shutdown + down + flatline` -> closest current shutdown tool; prefer `hybrid_down`

Weak-signal interpretation guide:
- "持续承压", "持续走高", "明显偏高" -> likely `trend + up`
- "持续承压并维持高位", "持续偏高", "高位维持一段时间" -> likely `level + plateau`
- "持续走低", "跌到低位并维持" -> likely `trend + down` or `shutdown`
- "短时冲高后恢复" -> likely `impulse + hump`
- "突然切换并维持" -> likely `level + step`; prefer `step_shift`
- "杂乱跳变", "信号异常", "读数失真" -> likely `volatility + irregular_noise`
- "降到极低水平并维持" -> likely `shutdown + flatline`

━━━━━━━━━━━━━━━━━  REGION SELECTION  ━━━━━━━━━━━━━━━━━

- Sustained trend/seasonality effects usually span 20–60 steps.
- Volatility changes usually span 10–40 steps.
- Transient impulses usually span 5–20 steps.
- The event should start at a plausible future index relative to the event onset.

CRITICAL CONSTRAINTS:
- region[0] MUST be >= 0 and < {ts_length}
- region[1] MUST be > region[0] and <= {ts_length}
- The region length MUST be >= 1
- `math_shift` and `shift_factor` are mutually exclusive
- For volatility split tools and `spike_inject`, do not provide `math_shift` or `shift_factor`

━━━━━━━━━━━━━━━━━  OUTPUT FORMAT  ━━━━━━━━━━━━━━━━━

Return STRICT JSON with this schema:
{{
  "thought": "brief reasoning from time anchor -> region -> canonical intent -> execution tool",
  "intent": {{
    "effect_family": "<trend|seasonality|volatility|impulse|level|shutdown>",
    "direction": "<up|down|neutral>",
    "shape": "<linear|quadratic|hump|plateau|step|flatline|irregular_noise|transient|periodic|flatten|residual_amplify|none>",
    "duration": "<short|medium|long>",
    "strength": "<weak|medium|strong>"
  }},
  "localization": {{
    "time_anchor_phrase": "<phrase copied or summarized from the prompt>",
    "position_bucket": "<early|mid|late|full|none>",
    "duration_steps": int,
    "evidence": "short explanation of why this bucket and duration fit",
    "region": [int, int],
    "confidence": 0.0
  }},
  "execution": {{
    "canonical_tool": "<canonical task name>",
    "tool_name": "<actual execution tool>",
    "control_source": "<native_tedit|hybrid|math_only>",
    "local_edit_hint": "short local instruction for the selected region only",
    "parameters": {{
      "region": [int, int]
    }}
  }}
}}

Parameter rules:
- `hybrid_up`, `hybrid_down`, `trend_quadratic_up`, `trend_quadratic_down`:
  may include either `math_shift` or `shift_factor`
- `volatility_global_scale`:
  may include `base_noise_scale`, `local_std_target_ratio`, `baseline_offset_ratio`, `trend_preserve`
- `volatility_local_burst`:
  may include `background_scale`, `burst_center`, `burst_width`, `burst_amplitude`, `burst_envelope_sharpness`, `baseline_offset_ratio`
- `volatility_envelope_monotonic`:
  may include `base_noise_scale`, `start_scale`, `end_scale`, `baseline_offset_ratio`, `trend_preserve`
- `spike_inject`:
  may include `amplitude`, `width`, `center`
- `step_shift`:
  may include either `math_shift` or `shift_factor`
- `season_enhance`, `season_reduce`, `ensemble_smooth`:
  region only

Backward compatibility requirement:
- Ensure `execution.tool_name` is a valid current tool name from the catalog.
- When the prompt is intentionally indirect, prefer the closest semantically grounded tool rather than defaulting to a generic anomaly interpretation.
"""


# Backward-compatible alias (uses legacy default of 100 points)
EVENT_DRIVEN_AGENT_PROMPT = get_event_driven_agent_prompt(ts_length=100)


def collect_descriptor_outputs(ts: dict) -> tuple[dict[str, object], list[str]]:
    """Execute built-in descriptor functions and capture their outputs.

    Returns a tuple of (results, failed_tools) where results maps function
    names to their raw return values.
    """
    functions = []
    for name, func in inspect.getmembers(tsd, inspect.isfunction):
        if func.__module__ == tsd.__name__ and not name.startswith("_"):
            functions.append((name, func))
    functions.sort(key=lambda item: item[0])

    results: dict[str, object] = {}
    failed_tools: list[str] = []

    for name, func in functions:
        try:
            params = inspect.signature(func).parameters
            call_args: dict[str, object] = {}
            if "values" in params and "values" in ts:
                call_args["values"] = ts["values"]
            if "timestamps" in params and "timestamps" in ts:
                call_args["timestamps"] = ts["timestamps"]

            raw_result = func(**call_args)
            results[name] = raw_result
        except Exception:
            failed_tools.append(name)

    return results, failed_tools


def generate_planner_prompt(
    planner_context_payload: dict,
    composer_tools_description: list,
    editor_tools_description: list | None = None,
    tedit_tools_description: list | None = None,
    editing_mode: bool = False,
    instruction_decomposition: bool = False,
) -> str:
    """Generate the prompt for the planner LLM."""

    context_json = json.dumps(
        planner_context_payload, ensure_ascii=False, indent=2
    )
    tools_json = json.dumps(
        composer_tools_description, ensure_ascii=False, indent=2
    )

    if editing_mode and editor_tools_description:
        editors_json = json.dumps(
            editor_tools_description, ensure_ascii=False, indent=2
        )

        if tedit_tools_description:
            tedit_json = json.dumps(
                tedit_tools_description, ensure_ascii=False, indent=2
            )
        else:
            tedit_json = "[]"

        if instruction_decomposition:
            mode_instruction = """
You are currently in TWO-STAGE EDITING MODE with instruction decomposition. Your task is to:

1. DECOMPOSE the user's editing request into:
   - Region selection (like bounding box): Identify which part of the time series to edit
   - Specific editing operations: What to do to the selected region

2. EXECUTE the editing in two stages:
   - Stage 1: Select region using semantic or statistical methods
   - Stage 2: Apply editing operations to the selected region

Available editing tools (traditional):
{editors}

Available TEdit diffusion-based tools (semantic editing):
{tedit}

Two-stage editing workflow:
1. First, analyze the user's intent and decide if region selection is needed
2. If region selection is needed, output a <region>...</region> tag
3. Then, output an <edit>...</edit> tag with the specific editing operation

<region> tag format (Stage 1 - Region Selection):
<region>
{{
    "intent": string,        # editing intent: "trend", "volatility", "anomaly", etc.
    "method": string,        # selection method: "semantic", "statistical", or "manual"
    "start_idx": int,       # start index (0-based, inclusive)
    "end_idx": int,         # end index (0-based, exclusive)
    "parameters": {{          # optional parameters for region selection
        "threshold": float,
        "min_length": int,
        ...
    }}
}}
</region>

<edit> tag format (Stage 2 - Specific Editing):
<edit>
{{
    "name": string,           # name of the editing tool to use
    "target": string,         # target time series: "history" or "forecast"
    "start_idx": int,         # start index of the region to edit (0-based, inclusive)
    "end_idx": int,           # end index of the region to edit (0-based, exclusive)
    "parameters": {{            # optional parameters specific to the editing tool
        "param1": value1,
        "param2": value2,
        ...
    }}
}}
</edit>

Common editing intents and tools:
1. Trend adjustment (趋势调整):
   - Intent: "trend"
   - Tools: increase_trend, decrease_trend, tedit_change_trend

2. Volatility adjustment (波动性调整):
   - Intent: "volatility"
   - Tools: increase_volatility, decrease_volatility, tedit_change_volatility

3. Seasonality modification (季节性修改):
   - Intent: "seasonality"
   - Tools: tedit_change_seasonality

4. Anomaly handling (异常值处理):
   - Intent: "anomaly"
   - Tools: remove_anomalies_in_region, select_editing_region

5. Smoothing (平滑):
   - Intent: "smoothing"
   - Tools: smooth_region, decrease_volatility

Important guidelines:
1. For complex editing requests, use two-stage approach: <region> followed by <edit>
2. For simple requests with explicit region, you can skip <region> and directly use <edit>
3. Use TEdit tools for semantic-level editing (changing overall characteristics)
4. Use traditional tools for fine-grained control (smoothing, interpolation, etc.)
5. The "factor" parameter controls adjustment magnitude:
   - For increase_trend/increase_volatility: factor > 1 (default 1.5)
   - For decrease_trend/decrease_volatility: 0 < factor < 1 (default 0.5)

CRITICAL: You are in TWO-STAGE EDITING MODE.
- First output <region>...</region> if region selection is needed
- Then output <edit>...</edit> with the specific editing operation
- Do NOT use <solution>...</solution> unless user explicitly asks to stop editing
""".format(editors=editors_json, tedit=tedit_json)
        else:
            mode_instruction = """
You are currently in EDITING MODE. Your task is to analyze the user's editing request and decompose it into appropriate editing operations.

Available editing tools are described here:
{editors}

Available TEdit diffusion-based tools (semantic editing):
{tedit}

Common editing operations:
1. Trend adjustment (趋势调整):
   - increase_trend: Increase the upward/downward trend magnitude (增加趋势幅度)
   - decrease_trend: Decrease the upward/downward trend magnitude (减少趋势幅度)
   - tedit_change_trend: Change trend type using diffusion model

2. Volatility adjustment (波动性调整):
   - increase_volatility: Increase the fluctuation/variance (增加波动性)
   - decrease_volatility: Decrease the fluctuation/variance (减少波动性)
   - tedit_change_volatility: Change volatility using diffusion model

3. Other editing operations:
   - smooth_region: Smooth a region using moving average
   - interpolate_region: Fill gaps with linear interpolation
   - remove_anomalies_in_region: Detect and remove outliers
   - apply_trend_in_region: Apply a specific linear trend
   - scale_region: Scale values by a factor

For editing tasks, use the <edit>...</edit> format with the following JSON structure:

<edit>
{{
    "name": string,           # name of the editing tool to use
    "target": string,         # target time series: "history" or "forecast"
    "start_idx": int,         # start index of the region to edit (0-based, inclusive)
    "end_idx": int,           # end index of the region to edit (0-based, exclusive)
    "parameters": {{            # optional parameters specific to the editing tool
        "param1": value1,
        "param2": value2,
        ...
    }}
}}
</edit>

Important guidelines:
1. Analyze the user's request carefully and decompose it into one or more editing operations.
2. For trend-related requests (e.g., "make the trend steeper", "increase the growth rate"), use increase_trend or decrease_trend.
3. For volatility-related requests (e.g., "make it more fluctuating", "reduce noise", "smooth out variations"), use increase_volatility or decrease_volatility.
4. Choose appropriate region indices (start_idx, end_idx) based on time series length and user's intent.
5. If the user doesn't specify a region, apply the operation to the entire forecast or history.
6. The "factor" parameter controls the magnitude of the adjustment:
   - For increase_trend/increase_volatility: factor > 1 increases the effect (default 1.5)
   - For decrease_trend/decrease_volatility: 0 < factor < 1 reduces the effect (default 0.5)
7. Use TEdit tools for semantic-level editing when you want to change overall characteristics.

CRITICAL: You are in EDITING MODE. You MUST generate an <edit>...</edit> instruction to modify the time series.
Do NOT use <solution>...</solution> in editing mode unless the user explicitly asks to stop editing.
The <edit> instruction must contain the tool name, target, start_idx, end_idx, and optional parameters.

Note: Indices are 0-based. start_idx is inclusive, end_idx is exclusive.
""".format(editors=editors_json, tedit=tedit_json)
    else:
        mode_instruction = """
You are currently in FORECASTING MODE. Your task is to generate and refine time series forecasts.

Available composer tools (functions that can generate or transform time series segments) are described here:
{tools}
""".format(tools=tools_json)

    prompt = f"""
You operate the planning brain of a time series forecasting agent. Your context arrives as a structured JSON payload that already consolidates every available signal:

{context_json}

Key fields inside the payload:
- `iteration_index` is the count of successful forecast-update iterations completed so far (increments after each composer tool call); 0 refers to the initial forecast, and the highest value reflects the latest forecast state.
- `planner_context_mode` tells you whether raw series values, descriptor outputs, or both are present.
- `history` contains the historical reference data.
- `forecast` contains the current forecast state.
- `context_text` provides any external textual context (background, scenario, constraints, hints) relevant to the forecast.

{mode_instruction}
Your task is to decide whether another operation can materially improve the time series. Work step-by-step: digest the payload, compare the latest state against the history (and any descriptors), then reason about gaps or remaining issues.

Output requirements:
1. First write your reasoning referencing the JSON fields you rely on (e.g., cite diff statistics, descriptor names, or horizon indices).
2. Then choose exactly one of the following response formats:
   - In FORECASTING MODE:
     *(1) `<step>{{...}}</step>` with STRICT JSON describing the next tool call.
     *(2) `<solution>...</solution>` when you are ready to stop.
   - In EDITING MODE (without instruction decomposition):
     *(1) `<edit>{{...}}</edit>` with STRICT JSON describing the editing operation.
     *(2) `<solution>...</solution>` when you are ready to stop.
   - In TWO-STAGE EDITING MODE (with instruction decomposition):
     *(1) `<region>{{...}}</region>` with STRICT JSON for region selection (Stage 1).
     *(2) `<edit>{{...}}</edit>` with STRICT JSON for specific editing (Stage 2).
     *(3) `<solution>...</solution>` when you are ready to stop.
"""

    if not editing_mode:
        prompt += """
If you choose option (1), select a tool to generate a new time series (named ts_tmp).
This ts_tmp will be added to the existing forecast (named ts_forecast) to improve it.
You also need to provide a balance coefficient, beta,
for combining ts_tmp and ts_forecast. The updated forecast is computed as:

ts_forecast = beta * ts_tmp + (1 - beta) * ts_forecast

<step>...</step> must contain a STRICT JSON structure that matches the following schema:

<step>
{{
    "name": string,          # name of the composer tool to be used
    "update_start": string,  # timestamp of ts_tmp. It should be selected from the given timestamps of the forecast horizon.
    "update_end": string,    # timestamp of ts_tmp. It should also be chosen from the given timestamps of the forecast horizon and must not be earlier than "start"
    "beta": float            # blending coefficient. It must be within the range [0, 1].
}}  
</step>
"""

    prompt += """
Only select `<solution>` when you can clearly justify, using the provided metrics, that further operations are unlikely
to yield improvement or would risk degradation. In that case, inside the <solution>...</solution> tag, provide a concise
explanation that (i) diagnoses the main issues you observed in the initial state and (ii) summarizes how the recent
operations addressed each issue. Reference the specific tools you used when describing these refinements.

Note:
1. Do not always use the same tool repeatedly unless you can justify it in your reasoning.
2. Only select <solution> when you have justification, otherwise prefer <step> or <edit> depending on the mode.
"""

    return prompt
