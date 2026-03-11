"""LangGraph workflow nodes for the forecasting agent."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Literal, Optional

import numpy as np
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from tool import ts_composers, ts_processor
from tool.tool_description.ts_composers import description as tsc_description
from tool.tool_description.ts_editors import description as tse_description
from tool.tool_description.tedit_tools import description as tedit_description
from modules.utils import timestamps_to_numeric, to_python

from .prompts import collect_descriptor_outputs, generate_planner_prompt

def node_describer(agent: "A1", state: "AgentState") -> "AgentState":
    """计算时间序列描述符并构建规划器上下文负载。
    
    这个函数是工作流的第一个节点，负责分析历史数据和当前预测的特征，
    生成结构化的上下文信息，为后续的规划决策提供基础。

    参数:
        agent (A1): 智能预测代理实例，包含时间序列数据和配置信息
        state (AgentState): 当前工作流状态，包含上下文消息和管道输出

    返回:
        AgentState: 更新后的工作流状态，包含新的上下文消息和事件记录
    """
    # 检查描述符功能是否启用（根据planner_context_mode配置）
    descriptors_enabled = agent._planner_includes_descriptors()

    # 计算历史数据的描述符（仅在启用描述符且未计算过的情况下）
    if descriptors_enabled and agent.history_descriptor_results is None:
        history_results, history_failures = collect_descriptor_outputs(agent.ts_history)
        agent.history_descriptor_results = history_results  # 存储历史数据描述符结果
        agent.history_descriptor_failures = history_failures  # 记录执行失败的描述符
    elif not descriptors_enabled:
        # 若未启用描述符，清空相关结果
        agent.history_descriptor_results = None
        agent.history_descriptor_failures = []

    # 计算预测数据的描述符
    if descriptors_enabled:
        forecast_results, forecast_failures = collect_descriptor_outputs(agent.ts_forecast)
    else:
        # 若未启用描述符，使用空结果
        forecast_results, forecast_failures = None, []
    agent.forecast_descriptor_results = forecast_results  # 存储预测数据描述符结果
    agent.forecast_descriptor_failures = forecast_failures  # 记录执行失败的描述符

    # 构建历史数据的上下文负载（包含原始数据和描述符结果）
    history_payload = agent._build_ts_context_payload(
        agent.ts_history,
        agent.history_descriptor_results,
        agent.history_descriptor_failures,
    )

    # 记录历史数据结构化事件到管道输出
    agent._record_pipeline_event(
        state,
        {
            "type": "describer.history_structured",  # 事件类型
            "payload": agent._snapshot(history_payload),  # 历史数据负载快照
            "failures": agent._snapshot(agent.history_descriptor_failures),  # 失败描述符列表
        },
    )

    # 构建完整的规划器上下文负载（包含历史和预测的所有相关信息）
    planner_payload = agent._build_planner_context_payload()
    # 将规划器上下文转换为格式化的JSON字符串
    planner_blob = json.dumps(planner_payload, ensure_ascii=False, indent=2)
    
    # 创建包含规划器上下文的HumanMessage
    context_message = HumanMessage(
        content=(
            f"Planner context payload (iteration {agent.iteration_index}):\n\n"
            + planner_blob
        ),
        tool_call_id="describer",  # 标记消息来源为描述器节点
    )
    # 将新的上下文消息添加到状态中
    state["context_messages"].append(context_message)

    # 清理之前的描述器消息，避免重复发送相同的历史数据块
    sanitized_messages = []
    for msg in state["context_messages"]:
        # 只处理来自描述器且不是当前消息的消息
        if getattr(msg, "tool_call_id", None) == "describer" and msg is not context_message:
            parts = str(msg.content).split("\n\n", 1)
            if len(parts) == 2:
                prefix, payload_text = parts
                try:
                    payload = json.loads(payload_text)
                    if isinstance(payload, dict):
                        # 只保留迭代索引、上下文模式和预测数据
                        iteration_val = payload.get("iteration_index")
                        sanitized_payload = {
                            key: value
                            for key, value in {
                                "iteration_index": iteration_val,
                                "planner_context_mode": payload.get("planner_context_mode"),
                                "forecast": payload.get("forecast"),
                            }.items()
                            if value is not None
                        }
                        # 生成简化的消息内容
                        label = (
                            f"Prior forecast snapshot (iteration {iteration_val})"
                            if iteration_val is not None
                            else "Prior forecast snapshot"
                        )
                        stripped_content = label + "\n\n" + json.dumps(
                            sanitized_payload, ensure_ascii=False, indent=2
                        )
                        # 替换原消息为简化版本
                        msg = HumanMessage(
                            content=stripped_content,
                            tool_call_id=getattr(msg, "tool_call_id", None),
                        )
                except Exception:
                    # 解析失败时保留原消息
                    pass
        sanitized_messages.append(msg)
    # 更新上下文消息列表为清理后的版本
    state["context_messages"] = sanitized_messages

    # 记录预测数据结构化事件到管道输出
    agent._record_pipeline_event(
        state,
        {
            "type": "describer.forecast_structured",  # 事件类型
            "payload": agent._snapshot(planner_payload),  # 规划器负载快照
            "failures": agent._snapshot(forecast_failures),  # 失败描述符列表
        },
    )

    # 设置下一个工作流步骤为"plan"（规划阶段）
    agent._set_next_step(state, "plan")
    # 返回更新后的状态
    return state


def node_planner(agent: "A1", state: "AgentState") -> "AgentState":
    """规划下一步操作或完成预测任务。
    
    该函数是工作流的核心决策节点，基于当前预测状态和历史数据，
    通过调用LLM来决定是继续优化预测（调用工具）还是完成预测任务。
    支持两种模式：预测模式和编辑模式。

    参数:
        agent (A1): 智能预测代理实例，包含时间序列数据和配置信息
        state (AgentState): 当前工作流状态，包含上下文消息和管道输出

    返回:
        AgentState: 更新后的工作流状态，包含新的上下文消息和事件记录
    """
    # 构建规划器上下文负载（包含历史数据、当前预测和描述符结果）
    planner_payload = agent._build_planner_context_payload()
    
    # 生成规划器提示，包含上下文和可用工具信息
    prompt_planner = generate_planner_prompt(
        planner_context_payload=planner_payload,  # 规划器上下文负载
        composer_tools_description=tsc_description,  # 可用的合成器工具描述
        editor_tools_description=tse_description,  # 可用的编辑工具描述
        tedit_tools_description=tedit_description if agent.enable_tedit else None,  # TEdit工具描述
        editing_mode=state.get("editing_mode", False),  # 当前是否处于编辑模式
        instruction_decomposition=state.get("instruction_decomposition", False),  # 是否启用指令分解
    )

    # 记录规划器提示事件到管道输出
    agent._record_pipeline_event(
        state,
        {
            "type": "planner.prompt",  # 事件类型：规划器提示
            "prompt": prompt_planner,  # 完整的提示内容
            "payload": agent._snapshot(planner_payload),  # 上下文负载快照
            "forecast_message_index": len(state["context_messages"]) - 1,  # 最近预测消息的索引
        },
    )

    # 构建发送给LLM的消息列表
    messages = [HumanMessage(content=prompt_planner)]  # 规划器提示作为最新消息
    messages.extend(state["context_messages"][:-1])  # 添加之前的上下文消息（除了最新的描述器消息）

    # 调用LLM并获取响应
    response_msg, state = agent._call_llm(messages, state)
    msg = str(response_msg.content).strip()  # 提取响应内容

    # 自动闭合不完整的标签（容错处理）
    if "<step>" in msg and "</step>" not in msg:
        msg += "</step>"
    if "<solution>" in msg and "</solution>" not in msg:
        msg += "</solution>"
    if "<edit>" in msg and "</edit>" not in msg:
        msg += "</edit>"

    # 使用正则表达式提取step、solution和edit内容
    step_match = re.search(r"<step>(.*?)</step>", msg, re.DOTALL)  # 提取工具调用步骤
    solution_match = re.search(r"<solution>(.*?)</solution>", msg, re.DOTALL)  # 提取完成解决方案
    edit_match = re.search(r"<edit>(.*?)</edit>", msg, re.DOTALL)  # 提取编辑任务

    # 记录规划器响应事件到管道输出
    agent._record_pipeline_event(
        state,
        {
            "type": "planner.response",  # 事件类型：规划器响应
            "raw_content": msg,  # 原始响应内容
            "has_step": bool(step_match),  # 是否包含step
            "has_solution": bool(solution_match),  # 是否包含solution
            "has_edit": bool(edit_match),  # 是否包含edit
            "solution_excerpt": solution_match.group(1).strip() if solution_match else None,  # solution内容摘要
            "editing_mode": state.get("editing_mode", False),  # 记录当前编辑模式
        },
    )
    
    # 添加详细的调试日志
    print(f"[DEBUG] Planner LLM Response:\n{msg}\n")
    print(f"[DEBUG] Has step: {bool(step_match)}")
    print(f"[DEBUG] Has solution: {bool(solution_match)}")
    print(f"[DEBUG] Has edit: {bool(edit_match)}")
    print(f"[DEBUG] Editing mode: {state.get('editing_mode', False)}")
    
    # 格式化响应消息，添加思考前缀
    msg = "Thoughts on whether the forecast needs further updates:\n\n" + msg
    response_msg = agent._clone_ai_message(response_msg, msg)  # 克隆并更新AI消息
    state["context_messages"].append(response_msg)  # 添加到上下文消息列表

    # 处理解决方案响应（完成预测任务）
    if solution_match:
        agent._set_next_step(state, "done")  # 设置下一步为"done"（完成）
        
        solution_text = solution_match.group(1).strip()  # 提取解决方案文本
        # 生成最终预测结果的消息
        solution_content = "The final forecast is:\n\n" + json.dumps(
            agent.ts_forecast, ensure_ascii=False, indent=2
        )
        state["context_messages"].append(AIMessage(content=solution_content))  # 添加到上下文消息
        
        # 记录解决方案事件到管道输出
        agent._record_pipeline_event(
            state,
            {
                "type": "planner.solution",  # 事件类型：规划器解决方案
                "forecast": agent._snapshot(agent.ts_forecast),  # 最终预测结果快照
                "message_index": len(state["context_messages"]) - 1,  # 最终预测消息的索引
                "explanation": solution_text,  # 解决方案解释
            },
        )
    
    # 处理编辑任务响应
    elif edit_match:
        raw_edit = edit_match.group(1).strip()  # 提取edit内容
        parse_error = None
        
        # 解析edit内容为JSON
        try:
            parsed = json.loads(raw_edit)
            if not isinstance(parsed, dict):
                parse_error = "Expected a JSON object."
        except Exception as exc:
            parsed = None
            parse_error = str(exc)  # 解析错误：记录异常信息

        # 处理解析错误
        if parse_error:
            # 生成错误提示消息
            warning_message = HumanMessage(
                content=(
                    "The <edit> payload must be valid JSON with keys "
                    '{"name", "target", "start_idx", "end_idx", "parameters"}. '
                    "Please resend the plan using a proper JSON object."
                )
            )
            state["context_messages"].append(warning_message)  # 添加到上下文消息
            
            # 记录解析错误事件
            agent._record_pipeline_event(
                state,
                {
                    "type": "planner.edit_parse_error",  # 事件类型：编辑解析错误
                    "raw_edit": raw_edit,  # 原始edit内容
                    "error": parse_error,  # 错误信息
                },
            )
            
            state["edit_task"] = None  # 清空编辑任务
            
            # 重试逻辑：最多重试3次
            if state.get("cnt_retry_planner") < 3:
                agent._set_next_step(state, "plan")  # 设置下一步为"plan"（重新规划）
                state["cnt_retry_planner"] += 1  # 增加重试计数
            else:
                agent._set_next_step(state, "done")  # 重试次数过多，设置为"done"（完成）
            
            return state  # 返回更新后的状态

        # 解析成功：保存编辑任务内容
        state["edit_task"] = parsed
        
        # 记录解析成功的编辑任务事件
        agent._record_pipeline_event(
            state,
            {
                "type": "planner.edit_json",  # 事件类型：编辑任务JSON
                "raw_edit": raw_edit,  # 原始edit内容
                "parsed": agent._snapshot(parsed),  # 解析后的edit内容快照
            },
        )
        
        agent._set_next_step(state, "edit")  # 设置下一步为"edit"（执行编辑）
    
    # 处理工具调用响应（继续优化预测）
    elif step_match:
        raw_step = step_match.group(1).strip()  # 提取step内容
        parse_error = None
        
        # 解析step内容为JSON
        try:
            parsed = json.loads(raw_step)
            if not isinstance(parsed, dict):
                parse_error = "Expected a JSON object."  # 类型错误：期望JSON对象
        except Exception as exc:
            parsed = None
            parse_error = str(exc)  # 解析错误：记录异常信息

        # 处理解析错误
        if parse_error:
            # 生成错误提示消息
            warning_message = HumanMessage(
                content=(
                    "The <step> payload must be valid JSON with keys "
                    '{"name", "update_start", "update_end", "beta"}. '
                    "Please resend the plan using a proper JSON object."
                )
            )
            state["context_messages"].append(warning_message)  # 添加到上下文消息
            
            # 记录解析错误事件
            agent._record_pipeline_event(
                state,
                {
                    "type": "planner.step_parse_error",  # 事件类型：步骤解析错误
                    "raw_step": raw_step,  # 原始step内容
                    "error": parse_error,  # 错误信息
                },
            )
            
            state["step_content"] = None  # 清空步骤内容
            
            # 重试逻辑：最多重试3次
            if state.get("cnt_retry_planner") < 3:
                agent._set_next_step(state, "plan")  # 设置下一步为"plan"（重新规划）
                state["cnt_retry_planner"] += 1  # 增加重试计数
            else:
                agent._set_next_step(state, "done")  # 重试次数过多，设置为"done"（完成）
            
            return state  # 返回更新后的状态

        # 解析成功：保存步骤内容
        state["step_content"] = parsed
        
        # 记录解析成功的步骤事件
        agent._record_pipeline_event(
            state,
            {
                "type": "planner.step_json",  # 事件类型：步骤JSON
                "raw_step": raw_step,  # 原始step内容
                "parsed": agent._snapshot(parsed),  # 解析后的step内容快照
            },
        )
        
        agent._set_next_step(state, "compose")  # 设置下一步为"compose"（执行工具）
    
    # 处理无效响应（既不是step也不是solution也不是edit）
    else:
        # 生成警告消息
        warning_message = HumanMessage(
            content=(
                "In PLANNING mode, output exactly one of: <solution>...</solution>, "
                "<edit>...</edit> for editing tasks, "
                "or a JSON plan inside <step>...</step> for forecasting. Try again."
            )
        )
        state["context_messages"].append(warning_message)  # 添加到上下文消息
        
        # 记录重试警告事件
        agent._record_pipeline_event(
            state,
            {
                "type": "planner.retry_warning",  # 事件类型：重试警告
                "message_index": len(state["context_messages"]) - 1,  # 警告消息的索引
                "content": warning_message.content,  # 警告内容
            },
        )
        
        # 重试逻辑：最多重试3次
        if state.get("cnt_retry_planner") < 3:
            agent._set_next_step(state, "plan")  # 设置下一步为"plan"（重新规划）
            state["cnt_retry_planner"] += 1  # 增加重试计数
        else:
            # 重试次数过多：终止预测任务
            agent._set_next_step(state, "done")
            
            # 生成终止消息
            termination_message = AIMessage(
                content="Planning terminated due to repeated parsing errors in the answer."
            )
            state["context_messages"].append(termination_message)  # 添加到上下文消息
            
            # 记录终止事件
            agent._record_pipeline_event(
                state,
                {
                    "type": "planner.termination",  # 事件类型：规划器终止
                    "reason": "parsing_errors",  # 终止原因：解析错误
                    "message_index": len(state["context_messages"]) - 1,  # 终止消息的索引
                },
            )

    # 返回更新后的工作流状态
    return state


def node_composer(agent: "A1", state: "AgentState") -> "AgentState":
    """Execute the selected composer tool and blend into forecast."""

    step = state.get("step_content") or {}
    tool_name = step["name"]
    spec = agent.composer_specs[tool_name]

    try:
        horizon_timestamps = agent.ts_forecast["timestamps"]
        update_start = step.get("update_start", horizon_timestamps[0])
        update_end = step.get("update_end", horizon_timestamps[-1])

        start_idx = horizon_timestamps.index(update_start)
        end_idx = horizon_timestamps.index(update_end) + 1
        window_slice = slice(start_idx, end_idx)

        full_t_numeric = timestamps_to_numeric(horizon_timestamps)

        history = agent.ts_history.get("values")
        x_values = [float(v) if v is not None else np.nan for v in history]

        current_forecast_values = agent.ts_forecast.get("values")
        agent._forecast_before_last_update = agent._snapshot(current_forecast_values)
        has_existing_forecast = current_forecast_values is not None
        if not has_existing_forecast:
            current_forecast_values = [0.0] * len(horizon_timestamps)
        forecast = np.asarray(current_forecast_values, dtype=float)
        prev_forecast_snapshot = forecast.copy()

        initial_forecast_array: np.ndarray | None = None
        if (
            agent._initial_forecast_values is not None
            and len(agent._initial_forecast_values) == forecast.shape[0]
        ):
            initial_forecast_array = np.asarray(agent._initial_forecast_values, dtype=float)

        # Build call arguments from spec
        param_names = {
            p["name"]
            for section in ("required_parameters", "optional_parameters")
            for p in spec.get(section)
        }

        call_args: Dict[str, Any] = {}
        if "x" in param_names:
            call_args["x"] = x_values
        if "t" in param_names:
            call_args["t"] = full_t_numeric.tolist()
        if "horizon" in param_names:
            call_args["horizon"] = len(horizon_timestamps)
        if "offset" in param_names:
            offset_value: Optional[float] = None
            if has_existing_forecast:
                window_values = forecast[window_slice]
                finite_window = window_values[np.isfinite(window_values)]
                if finite_window.size:
                    offset_value = float(np.mean(finite_window))
            if offset_value is None:
                history_arr = np.asarray(x_values, dtype=float)
                finite_history = history_arr[np.isfinite(history_arr)]
                if finite_history.size:
                    offset_value = float(finite_history[-1])
                else:
                    offset_value = 0.0
            call_args["offset"] = offset_value

        beta_value = float(np.clip(step.get("beta", 1.0), 0.0, 1.0))

        missing_required = [
            param["name"]
            for param in spec.get("required_parameters")
            if param["name"] not in call_args
        ]
        missing_optional = [
            param["name"]
            for param in spec.get("optional_parameters")
            if param["name"] not in call_args
        ]

        synthesized_values: Dict[str, Any] = {}
        applied_optional: Dict[str, Any] = {}
        ignored_optional: Dict[str, str] = {}

        # Infer missing parameters via LLM if needed
        if missing_required or missing_optional:
            call_args, synthesized_values, applied_optional, ignored_optional = (
                _infer_missing_parameters(
                    agent, state, spec, call_args, missing_required, missing_optional,
                    beta_value, update_start, update_end, horizon_timestamps,
                    start_idx, end_idx, prev_forecast_snapshot, window_slice
                )
            )

        if applied_optional:
            synthesized_values.update(applied_optional)

        agent._record_pipeline_event(
            state,
            {
                "type": "composer.call",
                "tool": tool_name,
                "update_start": update_start,
                "update_end": update_end,
                "beta": beta_value,
                "call_args": agent._snapshot(to_python(call_args)),
            },
        )

        # Execute the tool
        function = getattr(ts_composers, tool_name)
        output = function(**call_args)
        ts_tmp = np.asarray(output, dtype=float).reshape(len(horizon_timestamps))

        # Blend into forecast
        forecast[window_slice] = (
            beta_value * ts_tmp[window_slice] + (1.0 - beta_value) * forecast[window_slice]
        )
        agent.ts_forecast["values"] = forecast.tolist()

        compose_info = {
            "tool": tool_name,
            "arguments": to_python(list(call_args.keys())),
            "synthesized_argument_values": synthesized_values,
            "beta": beta_value,
            "update_start": update_start,
            "update_end": update_end,
        }

        update_summary = agent._summarize_forecast_update(
            iteration_index=agent.iteration_index + 1,
            tool_name=tool_name,
            beta=beta_value,
            update_start=update_start,
            update_end=update_end,
            horizon_timestamps=horizon_timestamps,
            prev_forecast=prev_forecast_snapshot,
            new_forecast=forecast,
            initial_forecast=initial_forecast_array,
            window_slice=window_slice,
            synthesized_values=synthesized_values,
        )
        agent.iteration_index += 1
        agent._previous_forecast_values = agent._snapshot(agent.ts_forecast.get("values"))

        agent._record_pipeline_event(
            state,
            {"type": "composer.update_summary", "summary": agent._snapshot(update_summary)},
        )

        agent._record_pipeline_event(
            state,
            {
                "type": "composer.output",
                "tool": tool_name,
                "beta": beta_value,
                "update_start": update_start,
                "update_end": update_end,
                "forecast_values": agent._snapshot(agent.ts_forecast["values"]),
            },
        )

        tool_message = HumanMessage(
            content=f"The tools and parameters used to compose the time series are:\n{json.dumps(compose_info, ensure_ascii=False, indent=2)}\n",
            tool_call_id=tool_name,
        )
        state["context_messages"].append(tool_message)
        agent._record_pipeline_event(
            state,
            {
                "type": "composer.tool_message",
                "tool": tool_name,
                "content": tool_message.content,
                "message_index": len(state["context_messages"]) - 1,
            },
        )

        agent._set_next_step(state, "describe")
        state["step_content"] = None
        return state

    except Exception as exc:
        error_content = f"Tool '{tool_name}' failed: {exc}. Forecast unchanged."
        error_message = HumanMessage(content=error_content, tool_call_id=tool_name)
        state["context_messages"].append(error_message)
        agent._record_pipeline_event(
            state,
            {
                "type": "composer.error",
                "tool": tool_name,
                "error": str(exc),
                "message_index": len(state["context_messages"]) - 1,
            },
        )
        agent._set_next_step(state, "plan")
        state["step_content"] = None
        return state


def _infer_missing_parameters(
    agent: "A1",
    state: "AgentState",
    spec: dict,
    call_args: Dict[str, Any],
    missing_required: list,
    missing_optional: list,
    beta_value: float,
    update_start: str,
    update_end: str,
    horizon_timestamps: list,
    start_idx: int,
    end_idx: int,
    prev_forecast_snapshot: np.ndarray,
    window_slice: slice,
) -> tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, str]]:
    """Use LLM to infer missing tool parameters."""

    window_values = prev_forecast_snapshot[window_slice]
    window_serialised = [float(v) if np.isfinite(v) else None for v in window_values]
    finite_window = window_values[np.isfinite(window_values)]
    window_mean = float(np.mean(finite_window)) if finite_window.size else None

    blend_context = {
        "beta": beta_value,
        "blend_rule": "ts_forecast = beta * ts_tmp + (1 - beta) * ts_forecast",
        "update_start": update_start,
        "update_end": update_end,
        "window_timestamps": horizon_timestamps[start_idx:end_idx],
        "current_forecast_window": window_serialised,
        "current_window_mean": window_mean,
    }

    required_desc = []
    optional_desc = []
    for param in spec.get("required_parameters"):
        if param["name"] in missing_required:
            required_desc.append(f"- {param['name']} ({param['type']}): {param['description']}")
    for param in spec.get("optional_parameters"):
        if param["name"] in missing_optional:
            optional_desc.append(f"- {param['name']} ({param['type']}): {param['description']}")

    system_message = SystemMessage(
        content=(
            "You are a coding assistant. "
            "Infer missing function arguments from the recent reasoning log. "
            "Return a single JSON object containing the requested keys. "
            "Include all REQUIRED keys. Include OPTIONAL keys only if you are confident; otherwise omit them."
        )
    )

    user_message_lines = [
        f"Target function: {spec['name']}",
        f"Purpose: {spec['description']}",
        "Required parameters needing values:",
        "\n".join(required_desc) if required_desc else "(none)",
        "",
        "Optional parameters you may include if confident:",
        "\n".join(optional_desc) if optional_desc else "(none)",
        "",
        f"Arguments already available: {json.dumps(list(call_args.keys()), ensure_ascii=False)}",
        "Return JSON with keys drawn only from:",
        ", ".join(missing_required + missing_optional),
        "",
        "Context for the current blend (JSON):",
        json.dumps(blend_context, ensure_ascii=False, indent=2),
        "",
        "Remember: ts_forecast = beta * ts_tmp + (1 - beta) * ts_forecast. "
        "Choose parameters that make sense under this weighting.",
    ]

    user_message = HumanMessage(content="\n".join(user_message_lines))
    llm_messages = [system_message, user_message]
    llm_messages.extend(state["context_messages"][-1:])

    response, _ = agent._call_llm(llm_messages, state)
    try:
        payload_all = json.loads(response.content if response else "{}")
    except Exception as exc:
        raise ValueError(f"Unable to parse parameter inference payload as JSON: {exc}") from exc

    if not isinstance(payload_all, dict):
        raise ValueError(f"Parameter inference response must be a JSON object, got: {type(payload_all)}")

    missing_keys = [key for key in missing_required if key not in payload_all]
    if missing_keys:
        raise ValueError(f"LLM response missing required keys {missing_keys} for tool {spec['name']}")

    synthesized_values: Dict[str, Any] = {}
    applied_optional: Dict[str, Any] = {}
    ignored_optional: Dict[str, str] = {}

    for key in missing_required:
        try:
            converted = to_python({key: payload_all[key]})
        except Exception as exc:
            raise ValueError(f"Failed to convert required parameter '{key}': {exc}") from exc
        call_args[key] = converted[key]
        synthesized_values[key] = converted[key]

    for key in missing_optional:
        if key in payload_all:
            try:
                converted = to_python({key: payload_all[key]})
            except Exception as exc:
                ignored_optional[key] = str(exc)
                continue
            call_args[key] = converted[key]
            applied_optional[key] = converted[key]

    agent._record_pipeline_event(
        state,
        {
            "type": "composer.argument_inference",
            "tool": spec["name"],
            "missing_required": missing_required,
            "missing_optional": missing_optional,
            "beta": beta_value,
            "applied_required": agent._snapshot(synthesized_values),
            "applied_optional": agent._snapshot(applied_optional),
            "ignored_optional": agent._snapshot(ignored_optional),
        },
    )

    return call_args, synthesized_values, applied_optional, ignored_optional


def route_main(state) -> Literal["describe", "plan", "compose", "edit", "done"]:
    """Route to the next node based on state['next_step']."""
    next_step = state.get("next_step")
    if next_step == "describe":
        return "describe"
    elif next_step == "plan":
        return "plan"
    elif next_step == "compose":
        return "compose"
    elif next_step == "edit":
        return "edit"
    elif next_step == "done":
        return "done"
    else:
        raise ValueError(f"Unexpected next_step: {next_step}")


def node_editor(agent: "A1", state: "AgentState") -> "AgentState":
    """Execute the selected editing tool and apply changes to time series.

    This node handles editing operations on time series data, such as smoothing,
    interpolation, anomaly removal, trend application, and scaling within
    specified regions. Also supports TEdit diffusion-based editing tools.

    Args:
        agent (A1): Intelligent forecasting agent instance
        state (AgentState): Current workflow state

    Returns:
        AgentState: Updated workflow state with edited time series
    """
    edit_task = state.get("edit_task") or {}
    tool_name = edit_task.get("name")

    if not tool_name:
        error_message = HumanMessage(
            content="No editing tool specified in edit_task."
        )
        state["context_messages"].append(error_message)
        agent._record_pipeline_event(
            state,
            {
                "type": "editor.error",
                "error": "No tool name provided",
            },
        )
        agent._set_next_step(state, "plan")
        return state

    try:
        target_ts_key = edit_task.get("target", "forecast")
        if target_ts_key == "forecast":
            target_ts = agent.ts_forecast
        elif target_ts_key == "history":
            target_ts = agent.ts_history
        else:
            raise ValueError(f"Invalid target: {target_ts_key}")

        timestamps = target_ts.get("timestamps", [])
        values = target_ts.get("values", [])

        if not values:
            raise ValueError(f"No values found in {target_ts_key}")

        values_arr = np.asarray(values, dtype=float)
        original_values = values_arr.copy()

        start_idx = edit_task.get("start_idx", 0)
        end_idx = edit_task.get("end_idx", len(values))

        if start_idx < 0:
            start_idx = 0
        if end_idx > len(values):
            end_idx = len(values)
        if start_idx >= end_idx:
            raise ValueError(f"Invalid region: start_idx={start_idx}, end_idx={end_idx}")

        is_tedit_tool = tool_name.startswith("tedit_")

        if is_tedit_tool:
            edited_values = _execute_tedit_tool(
                agent, tool_name, values_arr, start_idx, end_idx, edit_task
            )
        else:
            edit_params = {
                "x": values_arr,
                "start_idx": start_idx,
                "end_idx": end_idx,
            }

            optional_params = edit_task.get("parameters", {})
            edit_params.update(optional_params)

            function = getattr(ts_processor, tool_name)
            edited_values = function(**edit_params)

        if target_ts_key == "forecast":
            agent.ts_forecast["values"] = edited_values.tolist()
        else:
            agent.ts_history["values"] = edited_values.tolist()

        edit_info = {
            "tool": tool_name,
            "target": target_ts_key,
            "start_idx": start_idx,
            "end_idx": end_idx,
            "parameters": edit_task.get("parameters", {}),
            "is_tedit": is_tedit_tool,
        }

        agent._record_pipeline_event(
            state,
            {
                "type": "editor.edit_summary",
                "summary": agent._snapshot(edit_info),
                "original_values": agent._snapshot(original_values[start_idx:end_idx].tolist()),
                "edited_values": agent._snapshot(edited_values[start_idx:end_idx].tolist()),
            },
        )

        tool_message = HumanMessage(
            content=f"Applied editing tool '{tool_name}' to {target_ts_key}:\n{json.dumps(edit_info, ensure_ascii=False, indent=2)}\n",
            tool_call_id=tool_name,
        )
        state["context_messages"].append(tool_message)

        agent._record_pipeline_event(
            state,
            {
                "type": "editor.tool_message",
                "tool": tool_name,
                "content": tool_message.content,
                "message_index": len(state["context_messages"]) - 1,
            },
        )

        agent._set_next_step(state, "describe")
        state["edit_task"] = None

        return state

    except Exception as exc:
        error_content = f"Editing tool '{tool_name}' failed: {exc}. Time series unchanged."
        error_message = HumanMessage(content=error_content, tool_call_id=tool_name)
        state["context_messages"].append(error_message)
        agent._record_pipeline_event(
            state,
            {
                "type": "editor.error",
                "tool": tool_name,
                "error": str(exc),
                "message_index": len(state["context_messages"]) - 1,
            },
        )
        agent._set_next_step(state, "plan")
        state["edit_task"] = None
        return state


def _execute_tedit_tool(
    agent: "A1",
    tool_name: str,
    values_arr: np.ndarray,
    start_idx: int,
    end_idx: int,
    edit_task: Dict[str, Any],
) -> np.ndarray:
    """Execute TEdit-based editing tool.

    Args:
        agent: Agent instance
        tool_name: Name of TEdit tool
        values_arr: Time series values
        start_idx: Start index of region
        end_idx: End index of region
        edit_task: Edit task dictionary

    Returns:
        Edited time series values
    """
    try:
        from tool.tedit_wrapper import get_tedit_instance
    except ImportError:
        raise ImportError(
            "TEdit wrapper not available. Make sure TEdit is properly configured."
        )

    tedit = get_tedit_instance()

    if not tedit.is_loaded:
        raise RuntimeError(
            "TEdit model is not loaded. Please load a model checkpoint first."
        )

    parameters = edit_task.get("parameters", {})
    n_samples = parameters.get("n_samples", 1)
    sampler = parameters.get("sampler", "ddim")

    if tool_name == "tedit_edit":
        src_attrs = parameters.get("src_attrs", [0, 0])
        tgt_attrs = parameters.get("tgt_attrs", [1, 0])
        edited = tedit.edit_time_series(
            values_arr, src_attrs, tgt_attrs, n_samples, sampler
        )
        return edited[0]

    elif tool_name == "tedit_edit_region":
        src_attrs = parameters.get("src_attrs", [0, 0])
        tgt_attrs = parameters.get("tgt_attrs", [1, 0])
        edited = tedit.edit_region(
            values_arr, start_idx, end_idx, src_attrs, tgt_attrs, n_samples, sampler
        )
        return edited

    elif tool_name == "tedit_change_trend":
        trend_type_idx = parameters.get("trend_type_idx", 1)
        src_attrs = [0, 0]
        tgt_attrs = [trend_type_idx, 0]
        edited = tedit.edit_time_series(
            values_arr, src_attrs, tgt_attrs, n_samples, sampler
        )
        return edited[0]

    elif tool_name == "tedit_change_seasonality":
        seasonality_type_idx = parameters.get("seasonality_type_idx", 1)
        src_attrs = [0, 0]
        tgt_attrs = [0, seasonality_type_idx]
        edited = tedit.edit_time_series(
            values_arr, src_attrs, tgt_attrs, n_samples, sampler
        )
        return edited[0]

    elif tool_name == "tedit_change_volatility":
        volatility_type_idx = parameters.get("volatility_type_idx", 1)
        src_attrs = [0, 0]
        tgt_attrs = [0, volatility_type_idx]
        edited = tedit.edit_time_series(
            values_arr, src_attrs, tgt_attrs, n_samples, sampler
        )
        return edited[0]

    elif tool_name == "tedit_two_stage_edit":
        intent = parameters.get("intent", "general")
        src_attrs = parameters.get("src_attrs", [0, 0])
        tgt_attrs = parameters.get("tgt_attrs", [1, 0])

        try:
            from tool.region_selector import get_selector
            selector = get_selector()
            region_result = selector.select_region(
                values_arr, intent, method="semantic"
            )
            region_start = region_result["start_idx"]
            region_end = region_result["end_idx"]
        except Exception as e:
            region_start = start_idx
            region_end = end_idx

        edited = tedit.edit_region(
            values_arr, region_start, region_end, src_attrs, tgt_attrs, n_samples, sampler
        )
        return edited

    else:
        raise ValueError(f"Unknown TEdit tool: {tool_name}")


def execute_llm_tool(
    base_ts: np.ndarray,
    plan_dict: dict,
    tedit_instance,
    use_soft_boundary: bool = True,
) -> np.ndarray:
    """Execute LLM-planned editing tool on time series.

    This function serves as the "hands" of the agent, executing
    the tool call decided by the LLM brain.

    Args:
        base_ts: Input time series (numpy array)
        plan_dict: LLM-generated plan with keys: tool_name, parameters
        tedit_instance: TEdit model instance
        use_soft_boundary: Whether to use soft-boundary injection (default: True)

    Returns:
        Edited time series
    """
    from tool.ts_editors import execute_llm_tool as _execute_tool

    edited_ts, _ = _execute_tool(
        plan=plan_dict,
        ts=base_ts,
        tedit=tedit_instance,
        n_ensemble=15,
        use_soft_boundary=use_soft_boundary,
    )

    return edited_ts