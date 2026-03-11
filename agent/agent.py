"""Main forecasting agent class."""

from __future__ import annotations

import copy                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     
import json
import os
from pathlib import Path
import time
from typing import Any, Dict, Literal, TypedDict

import numpy as np
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from modules.llm import get_llm
from modules.utils import parse_user_input, pretty_print, to_python
from tool.tool_description.ts_composers import description as tsc_description
from tool.tool_description.ts_editors import description as tse_description
from tool.tool_description.tedit_tools import description as tedit_description

from .nodes import node_describer, node_planner, node_composer, node_editor, route_main

# 加载环境变量（如果存在.env文件）
if os.path.exists(".env"):
    load_dotenv(".env", override=False)
    print("Loaded environment variables from .env")


class AgentState(TypedDict):
    """
    工作流节点之间传递的状态数据结构。
    
    Attributes:
        context_messages: 上下文字消息列表，用于在节点间传递对话历史
        pipeline_outputs: 管道输出列表，记录工作流执行过程中的所有事件
        next_step: 下一个要执行的步骤名称
        step_content: 当前步骤的内容，通常是解析后的JSON数据
        input: 用户输入数据
        forecast: 预测结果
        cnt_retry_planner: 规划器重试次数计数器
        editing_mode: 标识当前是否处于编辑模式
        edit_task: 存储编辑任务信息（操作类型、区域、参数）
    """
    context_messages: list[BaseMessage]
    pipeline_outputs: list[dict[str, Any]]
    next_step: str | None
    step_content: dict | str | None
    input: dict | None
    forecast: dict | None
    cnt_retry_planner: int
    editing_mode: bool
    edit_task: dict[str, Any] | None


class A1:
    """
    智能时间序列预测代理类。
    
    使用描述→规划→合成的循环工作流，通过LLM引导工具选择和执行。
    
    Attributes:
        llm: 语言模型实例
        composer_specs: 合成器工具规范字典
        latest_pipeline_snapshot: 最新的管道执行快照
        planner_context_mode: 规划器上下文模式
        history_descriptor_results: 历史数据描述符结果
        forecast_descriptor_results: 预测数据描述符结果
        history_descriptor_failures: 历史数据描述符失败列表
        forecast_descriptor_failures: 预测数据描述符失败列表
        context_text: 上下文文本
        _initial_forecast_values: 初始预测值
        _previous_forecast_values: 上一次预测值
        _forecast_before_last_update: 更新前的预测值
        iteration_index: 迭代索引
        ts_history: 历史时间序列数据
        ts_forecast: 预测时间序列数据
        app: 编译后的LangGraph应用
        checkpointer: 检查点存储器
    """

    def __init__(
        self,
        llm_name: str = "claude-sonnet-4-20250514",
        base_url: str | None = None,
        api_key: str = "EMPTY",
        source: str | None = None,
        planner_context_mode: Literal["series_only", "descriptors_only", "hybrid"] = "series_only",
        enable_tedit: bool = False,
        enable_instruction_decomposition: bool = False,
        tedit_model_path: str | None = None,
        tedit_config_path: str | None = None,
        tedit_device: str = "cuda:0",
    ):
        """
        初始化预测代理。
        
        Args:
            llm_name: 语言模型名称
            base_url: 语言模型API基础URL
            api_key: 语言模型API密钥
            source: 语言模型源（如OpenAI、vLLM等）
            planner_context_mode: 规划器上下文模式，可选值：series_only（仅序列）、
                                descriptors_only（仅描述符）、hybrid（混合模式）
            enable_tedit: 是否启用TEdit模型
            enable_instruction_decomposition: 是否启用指令分解
            tedit_model_path: TEdit模型检查点路径
            tedit_config_path: TEdit模型配置文件路径
            tedit_device: TEdit模型设备
        """
        # 初始化语言模型
        self.llm = get_llm(
            llm_name,
            stop_sequences=["</execute>", "</solution>", "</step>", "</region>"],
            base_url=base_url,
            api_key=api_key,
            source=source,
        )

        # 加载合成器工具规范
        self.composer_specs: Dict[str, dict[str, Any]] = {
            item["name"]: item for item in tsc_description
        }
        self.editor_specs: Dict[str, dict[str, Any]] = {
            item["name"]: item for item in tse_description
        }
        self.tedit_specs: Dict[str, dict[str, Any]] = {
            item["name"]: item for item in tedit_description
        } if enable_tedit else {}
        self.latest_pipeline_snapshot: list[dict[str, Any]] = []

        # 验证规划器上下文模式
        allowed_modes = {"series_only", "descriptors_only", "hybrid"}
        if planner_context_mode not in allowed_modes:
            raise ValueError(f"planner_context_mode must be one of {sorted(allowed_modes)}")
        self.planner_context_mode = planner_context_mode

        # TEdit配置
        self.enable_tedit = enable_tedit
        self.enable_instruction_decomposition = enable_instruction_decomposition
        self.tedit_model_path = tedit_model_path
        self.tedit_config_path = tedit_config_path
        self.tedit_device = tedit_device

        # 初始化TEdit模型（如果启用）
        self.tedit_wrapper = None
        if enable_tedit and tedit_model_path and tedit_config_path:
            try:
                from tool.tedit_wrapper import get_tedit_instance
                self.tedit_wrapper = get_tedit_instance(
                    model_path=tedit_model_path,
                    config_path=tedit_config_path,
                    device=tedit_device
                )
                print(f"TEdit model loaded from {tedit_model_path}")
            except Exception as e:
                print(f"Warning: Failed to load TEdit model: {e}")
                self.enable_tedit = False

        # 描述符结果（工作流执行过程中计算）
        self.history_descriptor_results: dict[str, Any] | None = None
        self.forecast_descriptor_results: dict[str, Any] | None = None
        self.history_descriptor_failures: list[str] = []
        self.forecast_descriptor_failures: list[str] = []

        # 预测跟踪
        self.context_text: str = ""
        self._initial_forecast_values: list[float] | None = None
        self._previous_forecast_values: list[float] | None = None
        self._forecast_before_last_update: list[float] | None = None
        self.iteration_index: int = 0
        self.editing_mode: bool = False

        # 时间序列数据（在go()方法中设置）
        self.ts_history: dict[str, Any] = {}
        self.ts_forecast: dict[str, Any] = {}
        # 配置工作流
        self._configure_workflow()

    def _configure_workflow(self):
        """构建并编译LangGraph工作流。"""
        # 创建状态图
        workflow = StateGraph(AgentState)

        # 添加工作流节点，绑定当前代理实例
        workflow.add_node("describer", lambda state: node_describer(self, state))
        workflow.add_node("planner", lambda state: node_planner(self, state))
        workflow.add_node("composer", lambda state: node_composer(self, state))
        workflow.add_node("editor", lambda state: node_editor(self, state))

        # 定义工作流边
        workflow.add_edge(START, "describer")
        workflow.add_edge("describer", "planner")
        
        # 添加条件边，根据规划器的输出决定下一个节点
        workflow.add_conditional_edges(
            "planner",
            route_main,
            path_map={
                "describe": "describer",
                "plan": "planner",
                "compose": "composer",
                "edit": "editor",
                "done": END,
            },
        )
        workflow.add_edge("composer", "describer")
        workflow.add_edge("editor", "describer")

        # 编译工作流
        self.app = workflow.compile()
        self.checkpointer = MemorySaver()
        self.app.checkpointer = self.checkpointer

        # 保存工作流可视化图
        graph_path = Path(__file__).resolve().parents[1] / "workflow_graph.png"
        graph_path.write_bytes(self.app.get_graph().draw_mermaid_png())

    # -------------------------------------------------------------------------
    # 状态和管道辅助方法
    # -------------------------------------------------------------------------

    def _record_pipeline_event(self, state: AgentState, event: dict[str, Any]) -> None:
        """
        将结构化事件添加到管道跟踪中。
        
        Args:
            state: 当前代理状态
            event: 要记录的事件数据
        """
        event = {"timestamp": time.time(), **event}
        state["pipeline_outputs"].append(event)
        self.latest_pipeline_snapshot = state["pipeline_outputs"]

    def _set_next_step(self, state: AgentState, next_step: str | None) -> None:
        """
        设置下一个工作流步骤并记录状态转换。
        
        Args:
            state: 当前代理状态
            next_step: 下一个步骤的名称
        """
        state["next_step"] = next_step
        self._record_pipeline_event(state, {"type": "state.transition", "next_step": next_step})

    def _snapshot(self, data: Any) -> Any:
        """
        深拷贝数据以创建不可变快照。
        
        Args:
            data: 要创建快照的数据
            
        Returns:
            数据的深拷贝，如果深拷贝失败则返回原始数据
        """
        try:
            return copy.deepcopy(data)
        except Exception:
            return data

    # -------------------------------------------------------------------------
    # 上下文模式辅助方法
    # -------------------------------------------------------------------------

    def _planner_includes_series(self) -> bool:
        """
        检查规划器上下文中是否包含序列数据。
        
        Returns:
            如果包含序列数据则返回True，否则返回False
        """
        return self.planner_context_mode in {"series_only", "hybrid"}

    def _planner_includes_descriptors(self) -> bool:
        """
        检查规划器上下文中是否包含描述符数据。
        
        Returns:
            如果包含描述符数据则返回True，否则返回False
        """
        return self.planner_context_mode in {"descriptors_only", "hybrid"}

    # -------------------------------------------------------------------------
    # 负载构建方法
    # -------------------------------------------------------------------------

    def _build_ts_context_payload(
        self,
        ts: dict[str, Any],
        descriptor_results: dict[str, Any] | None,
        descriptor_failures: list[str],
    ) -> dict[str, Any]:
        """
        为时间序列构建上下文负载。
        
        Args:
            ts: 时间序列数据
            descriptor_results: 描述符结果
            descriptor_failures: 描述符失败列表
            
        Returns:
            构建好的上下文负载
        """
        include_series = self._planner_includes_series()
        include_descriptors = self._planner_includes_descriptors()

        payload: dict[str, Any] = {}

        # 如果包含序列数据或当前是预测数据，则添加时间戳摘要
        if include_series or ts is self.ts_forecast:
            payload["timestamps_summary"] = self._summarize_timestamps(ts.get("timestamps"))

        # 如果包含序列数据，则添加值
        if include_series:
            payload["values"] = self._snapshot(ts.get("values"))

        # 如果包含描述符数据且有结果，则添加描述符
        if include_descriptors and descriptor_results is not None:
            payload["descriptors"] = self._snapshot(descriptor_results)
            if descriptor_failures:
                payload["descriptor_failures"] = self._snapshot(descriptor_failures)

        return payload

    def _build_planner_context_payload(self) -> dict[str, Any]:
        """
        为规划器构建完整的上下文负载。
        
        Returns:
            构建好的规划器上下文负载
        """
        return {
            "iteration_index": self.iteration_index,
            "planner_context_mode": self.planner_context_mode,
            "history": self._build_ts_context_payload(
                self.ts_history,
                self.history_descriptor_results,
                self.history_descriptor_failures,
            ),
            "forecast": self._build_ts_context_payload(
                self.ts_forecast,
                self.forecast_descriptor_results,
                self.forecast_descriptor_failures,
            ),
            "context_text": self.context_text,
        }

    # -------------------------------------------------------------------------
    # 总结辅助方法
    # -------------------------------------------------------------------------

    def _summarize_timestamps(self, timestamps: list[str] | None) -> dict[str, Any]:
        """
        生成时间戳的紧凑摘要。
        
        Args:
            timestamps: 时间戳列表
            
        Returns:
            时间戳摘要字典，包含数量、开始和结束时间
        """
        if not timestamps:
            return {"count": 0, "start": None, "end": None}
        return {
            "count": len(timestamps),
            "start": timestamps[0],
            "end": timestamps[-1],
        }

    def _summarize_window(self, arr: np.ndarray) -> dict[str, Any]:
        """
        生成窗口数据的统计摘要。
        
        Args:
            arr: 窗口数据数组
            
        Returns:
            统计摘要字典，包含均值、标准差、最小值、最大值等
        """
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            return {"has_values": False}

        return {
            "has_values": True,
            "mean": float(np.mean(finite)),
            "std": float(np.std(finite)),
            "min": float(np.min(finite)),
            "max": float(np.max(finite)),
            "last_value": float(finite[-1]),
        }

    def _calculate_diff_stats(self, diff: np.ndarray) -> dict[str, Any]:
        """
        计算预测差异的统计信息。
        
        Args:
            diff: 预测差异数组
            
        Returns:
            差异统计信息字典，包含平均变化、平均绝对变化、最大绝对变化等
        """
        finite = diff[np.isfinite(diff)]
        if finite.size == 0:
            return {
                "mean_change": None,
                "mean_abs_change": None,
                "max_abs_change": None,
                "changed_points": 0,
                "positive_points": 0,
                "negative_points": 0,
            }

        abs_diff = np.abs(finite)
        nonzero_mask = np.abs(diff) > 1e-9
        return {
            "mean_change": float(np.mean(finite)),
            "mean_abs_change": float(np.mean(abs_diff)),
            "max_abs_change": float(np.max(abs_diff)),
            "changed_points": int(np.sum(nonzero_mask)),
            "positive_points": int(np.sum(diff > 1e-9)),
            "negative_points": int(np.sum(diff < -1e-9)),
        }

    def _summarize_forecast_update(
        self,
        *,
        iteration_index: int,
        tool_name: str,
        beta: float,
        update_start: str,
        update_end: str,
        horizon_timestamps: list[str],
        prev_forecast: np.ndarray,
        new_forecast: np.ndarray,
        initial_forecast: np.ndarray | None,
        window_slice: slice,
        synthesized_values: dict[str, Any],
    ) -> dict[str, Any]:
        """
        为管道跟踪生成预测更新摘要。
        
        Args:
            iteration_index: 迭代索引
            tool_name: 使用的工具名称
            beta: 混合参数
            update_start: 更新窗口开始时间
            update_end: 更新窗口结束时间
            horizon_timestamps: 预测时间戳列表
            prev_forecast: 更新前的预测值
            new_forecast: 更新后的预测值
            initial_forecast: 初始预测值
            window_slice: 更新窗口的切片
            synthesized_values: 合成参数
            
        Returns:
            预测更新摘要字典
        """
        # 计算窗口内的预测差异
        prev_window = prev_forecast[window_slice]
        new_window = new_forecast[window_slice]
        diff_prev = new_window - prev_window
        diff_prev_stats = self._calculate_diff_stats(diff_prev)

        # 计算与初始预测的差异（如果有）
        diff_initial_stats: dict[str, Any] | None = None
        if initial_forecast is not None and initial_forecast.shape == new_forecast.shape:
            initial_window = initial_forecast[window_slice]
            diff_initial = new_window - initial_window
            diff_initial_stats = self._calculate_diff_stats(diff_initial)

        # 构建摘要
        summary = {
            "iteration": iteration_index,
            "tool": tool_name,
            "beta": beta,
            "update_window": {
                "start": update_start,
                "end": update_end,
                "indices": [
                    horizon_timestamps.index(update_start),
                    horizon_timestamps.index(update_end),
                ],
            },
            "forecast_window_summary": self._summarize_window(new_window),
            "diff_vs_previous": diff_prev_stats,
            "diff_vs_initial": diff_initial_stats,
        }

        # 添加合成参数（如果有）
        if synthesized_values:
            summary["synthesized_parameters"] = to_python(synthesized_values)

        return summary

    # -------------------------------------------------------------------------
    # LLM交互方法
    # -------------------------------------------------------------------------

    def _call_llm(
        self, messages: list[BaseMessage], state: AgentState
    ) -> tuple[AIMessage | None, AgentState]:
        """
        调用语言模型，包含速率限制重试逻辑。
        
        Args:
            messages: 消息列表
            state: 当前代理状态
            
        Returns:
            语言模型响应和更新后的代理状态
        """
        response = None
        max_retries = 5
        base_delay = 1

        # 尝试调用语言模型，最多重试5次
        for attempt in range(max_retries):
            try:
                response = self.llm.invoke(messages)
                break
            except Exception as e:
                # 处理速率限制错误
                if '429' in str(e) and attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    print(f"Rate limit error detected. Retrying in {delay} seconds...")
                    time.sleep(delay)
                else:
                    # 其他错误，记录并返回错误消息
                    print(f"LLM call failed after retries or due to a non-retriable error: {e}")
                    error_message = "The language model is currently unavailable. Please try again later."
                    self._record_pipeline_event(
                        state,
                        {"type": "llm.error", "exception": str(e), "attempt": attempt},
                    )
                    self._set_next_step(state, "done")
                    return AIMessage(content=error_message), state

        return response, state

    def _clone_ai_message(self, message: AIMessage, content: str) -> AIMessage:
        """
        克隆AIMessage并替换内容，保留元数据。
        
        Args:
            message: 要克隆的AIMessage
            content: 新的内容
            
        Returns:
            克隆后的AIMessage，包含新内容和原始元数据
        """
        return AIMessage(
            content=content,
            additional_kwargs=getattr(message, "additional_kwargs", {}),
            response_metadata=getattr(message, "response_metadata", {}),
            tool_calls=(getattr(message, "tool_calls", []) or []),
        )

    # -------------------------------------------------------------------------
    # 主要入口点
    # -------------------------------------------------------------------------

    def set_editing_mode(self, enabled: bool = True):
        """
        设置编辑模式。

        Args:
            enabled: 是否启用编辑模式
        """
        self.editing_mode = enabled

    def go(self, user_input: str):
        """
        执行预测工作流。
        
        Args:
            user_input: 包含历史和预测规范的JSON字符串
            
        Yields:
            每次迭代产生的日志、最后一条消息和管道输出
        """
        # 重置状态
        self.history_descriptor_results = None
        self.history_descriptor_failures = []
        self.forecast_descriptor_results = None
        self.forecast_descriptor_failures = []
        self.iteration_index = 0

        # 解析用户输入
        self.ts_history, self.ts_forecast, self.context_text = parse_user_input(user_input)
        self.ts_history = to_python(self.ts_history)
        self.ts_forecast = to_python(self.ts_forecast)

        # 初始化预测跟踪
        self._initial_forecast_values = self._snapshot(self.ts_forecast.get("values"))
        self._previous_forecast_values = self._snapshot(self.ts_forecast.get("values"))
        self._forecast_before_last_update = None

        # 初始化工作流状态
        inputs = {
            "context_messages": [],
            "pipeline_outputs": [],
            "next_step": None,
            "step_content": None,
            "cnt_retry_planner": 0,
            "editing_mode": self.editing_mode,
            "edit_task": None,
            "instruction_decomposition": self.enable_instruction_decomposition,
        }
        self.latest_pipeline_snapshot = []

        # 配置工作流
        config = {"recursion_limit": 500, "configurable": {"thread_id": 42}}
        self.log = []

        prev_context_len = 0
        last_message: BaseMessage | None = None
        latest_pipeline: list[dict[str, Any]] = []

        # 执行工作流
        for s in self.app.stream(inputs, stream_mode="values", config=config):
            context_messages = s["context_messages"]
            # 记录新消息
            new_messages = context_messages[prev_context_len:]
            if new_messages:
                self.log.extend(pretty_print(message) for message in new_messages)
            prev_context_len = len(context_messages)
            if context_messages:
                last_message = context_messages[-1]

            # 更新管道输出
            pipeline_outputs = s.get("pipeline_outputs", [])
            latest_pipeline = pipeline_outputs
            self.latest_pipeline_snapshot = pipeline_outputs

            # 产生当前状态
            yield self.log, last_message, pipeline_outputs

        # 返回最终结果
        final_content = last_message.content if last_message else None
        return self.log, final_content, latest_pipeline