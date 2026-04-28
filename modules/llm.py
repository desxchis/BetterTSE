import os
import time
from typing import Literal, Optional, List, Optional, Dict, Any

from openai import OpenAI, APIStatusError

try:
    from langchain_openai import ChatOpenAI
except ImportError:  # pragma: no cover - optional dependency
    ChatOpenAI = None

try:
    from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
    from langchain_core.language_models.chat_models import BaseChatModel
except ImportError:  # pragma: no cover - optional dependency
    class BaseMessage:  # type: ignore[override]
        def __init__(self, content: str = "", **kwargs: Any):
            self.content = content
            for key, value in kwargs.items():
                setattr(self, key, value)

    class SystemMessage(BaseMessage):
        pass

    class HumanMessage(BaseMessage):
        pass

    class ToolMessage(BaseMessage):
        pass

    class AIMessage(BaseMessage):
        def __init__(self, content: str = "", additional_kwargs: Dict[str, Any] | None = None, **kwargs: Any):
            super().__init__(content=content, additional_kwargs=additional_kwargs or {}, **kwargs)

    class BaseChatModel:  # type: ignore[override]
        pass
from modules.pure_editing_volatility import (
    infer_volatility_subtype_from_text,
    resolve_volatility_subtype_route,
)
from modules.region_localizer import infer_shape_hint, localize_region
from modules.strength_parser import parse_strength_text
from tool.ts_editors import normalize_llm_plan


SourceType = Literal["OpenAI", "DashScope", "InterWeb", "Ollama", "vLLM"]


def _infer_direction_hint(text: str) -> str:
    normalized = text or ""
    upward_hints = ("冲高", "抬升", "走高", "偏高", "激增", "飙升", "高位")
    if any(token in normalized for token in upward_hints):
        return "up"
    downward_hints = ("降至极低", "跌到低位", "停摆", "中断", "掉到特别低", "走低")
    if any(token in normalized for token in downward_hints):
        return "down"
    return ""


def _apply_volatility_route(plan: Dict[str, Any], instruction_text: str, ts_length: int | None = None) -> Dict[str, Any]:
    normalized = normalize_llm_plan(plan, ts_length=ts_length)
    intent = normalized.setdefault("intent", {})
    execution = normalized.setdefault("execution", {})
    parameters = normalized.setdefault("parameters", {})
    localization = normalized.setdefault("localization", {})

    shape = str(intent.get("shape") or infer_shape_hint(instruction_text) or "")
    effect_family = str(intent.get("effect_family", ""))
    if shape != "irregular_noise" and effect_family != "volatility":
        return normalized

    region = parameters.get("region")
    if not isinstance(region, list):
        region = localization.get("region")
    routed = resolve_volatility_subtype_route(
        text=instruction_text,
        proposed_subtype=normalized.get("volatility_subtype") or intent.get("volatility_subtype"),
        region=region if isinstance(region, list) else None,
        ts_length=ts_length,
    )
    normalized["volatility_subtype"] = routed["final_subtype"]
    intent["volatility_subtype"] = routed["proposed_subtype"]
    normalized["volatility_routing"] = {
        "proposed_subtype": routed["proposed_subtype"],
        "guarded_subtype": routed["guarded_subtype"],
        "final_subtype": routed["final_subtype"],
        "guard_reason": routed["guard_reason"],
        "is_preview": routed["is_preview"],
    }
    execution["canonical_tool"] = routed["canonical_tool"]
    execution["tool_name"] = routed["tool_name"]
    execution["volatility_subtype"] = routed["final_subtype"]
    execution.setdefault("parameters", {})
    execution["parameters"].update(routed["parameters"])
    parameters.update(routed["parameters"])
    normalized["canonical_tool"] = routed["canonical_tool"]
    normalized["tool_name"] = routed["tool_name"]
    return normalize_llm_plan(normalized, ts_length=ts_length)


def _apply_explicit_prompt_hints(plan: Dict[str, Any], instruction_text: str, ts_length: int | None = None) -> Dict[str, Any]:
    """Override obviously wrong intent/tool fields using explicit lexical cues.

    This is intentionally narrow: only strong, high-precision text hints should
    be allowed to overrule the LLM plan.
    """
    normalized = normalize_llm_plan(plan, ts_length=ts_length)
    intent = normalized.setdefault("intent", {})
    execution = normalized.setdefault("execution", {})
    params = normalized.setdefault("parameters", {})

    strength_parse = parse_strength_text(instruction_text)
    normalized["strength_parse"] = strength_parse
    if str(intent.get("strength", "")).lower() not in {"weak", "medium", "strong"}:
        intent["strength"] = str(strength_parse["strength_text"])
    params.setdefault("strength_label", int(strength_parse["strength_label"]))

    shape_hint = infer_shape_hint(instruction_text)
    direction_hint = _infer_direction_hint(instruction_text)

    if not shape_hint and not direction_hint:
        return normalized

    if shape_hint:
        intent["shape"] = shape_hint
        if shape_hint == "flatline":
            intent["effect_family"] = "shutdown"
            if not direction_hint:
                direction_hint = "down"
            execution["canonical_tool"] = "trend_linear_down"
            execution["tool_name"] = "hybrid_down"
        elif shape_hint == "step":
            intent["effect_family"] = "level"
            execution["canonical_tool"] = "level_step"
            execution["tool_name"] = "step_shift"
        elif shape_hint == "irregular_noise":
            intent["effect_family"] = "volatility"
            intent["direction"] = "neutral"
            proposed_subtype = normalized.get("volatility_subtype") or intent.get("volatility_subtype")
            if not proposed_subtype:
                proposed_subtype = infer_volatility_subtype_from_text(instruction_text)
            normalized["volatility_subtype"] = proposed_subtype
            intent["volatility_subtype"] = proposed_subtype
        elif shape_hint == "hump":
            intent["effect_family"] = "impulse"
            execution["canonical_tool"] = "impulse_spike"
            execution["tool_name"] = "spike_inject"

    seasonality_enhance_hints = ("周期更明显", "周期性更强", "节律更明显", "峰谷更清晰", "增强季节性", "增强周期性")
    seasonality_reduce_hints = ("周期减弱", "周期性减弱", "节律变弱", "峰谷被压平", "削弱季节性", "削弱周期性", "平滑周期")
    if any(token in instruction_text for token in seasonality_enhance_hints):
        intent["effect_family"] = "seasonality"
        intent["direction"] = "neutral"
        intent["shape"] = "periodic"
        execution["canonical_tool"] = "seasonality_enhance"
        execution["tool_name"] = "season_enhance"
    elif any(token in instruction_text for token in seasonality_reduce_hints):
        intent["effect_family"] = "seasonality"
        intent["direction"] = "neutral"
        intent["shape"] = "flatten"
        execution["canonical_tool"] = "seasonality_reduce"
        execution["tool_name"] = "season_reduce"

    plateau_hints = ("持续承压", "持续偏高", "维持高位", "高位维持", "持续处于高位")
    if any(token in instruction_text for token in plateau_hints):
        intent["effect_family"] = "level"
        intent["shape"] = "plateau"
        if not direction_hint:
            direction_hint = "up"
        execution["canonical_tool"] = "trend_linear_up"
        execution["tool_name"] = "hybrid_up"

    if direction_hint:
        intent["direction"] = direction_hint
        if execution.get("tool_name") in {"hybrid_up", "hybrid_down"}:
            execution["tool_name"] = "hybrid_down" if direction_hint == "down" else "hybrid_up"
            execution["canonical_tool"] = "trend_linear_down" if direction_hint == "down" else "trend_linear_up"

    normalized["tool_name"] = execution.get("tool_name", normalized.get("tool_name"))
    normalized["canonical_tool"] = execution.get("canonical_tool", normalized.get("canonical_tool"))
    execution.setdefault("parameters", params)
    normalized = normalize_llm_plan(normalized, ts_length=ts_length)
    return _apply_volatility_route(normalized, instruction_text, ts_length=ts_length)


# Define a customized LLM client implementing the invoke method
class CustomLLMClient:
    """
    A lightweight client with a single `invoke(messages)` method.
    - Prefers OpenAI Responses API.
    - Falls back to Chat Completions if /responses isn't supported.
    - Maps LangChain messages correctly (incl. tool calls).
    """

    def __init__(
        self,
        model_name: str,
        temperature: float = 0.4,
        stop_sequences: Optional[List[str]] = None,
        base_url: Optional[str] = None,
        api_key: str = "EMPTY",
        prefer_responses: bool = False,       # set False if your server lacks /responses
        # optional function tools
        tools: Optional[List[Dict[str, Any]]] = None,
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.stop_sequences = stop_sequences
        self.base_url = base_url
        self.api_key = api_key
        self.prefer_responses = prefer_responses
        self.tools = tools

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,   # e.g., "http://localhost:8000/v1" for vLLM
        )

    # ------- Formatting helpers -------
    def _format_for_responses(self, messages: List[BaseMessage]):
        """Split LC messages into `instructions` (system) + `input` (others)."""
        instructions_chunks = []
        input_messages = []
        for m in messages:
            if isinstance(m, SystemMessage):
                input_messages.append(
                    {"role": "developer", "content": m.content})
            # if isinstance(m, SystemMessage):
            #     instructions_chunks.append(m.content)
            elif isinstance(m, HumanMessage):
                input_messages.append({"role": "user", "content": m.content})
            elif isinstance(m, AIMessage):
                input_messages.append(
                    {"role": "assistant", "content": m.content})
            elif isinstance(m, ToolMessage):
                item = {"role": "tool", "content": m.content}
                # If LC ToolMessage carries tool_call_id, keep it
                if hasattr(m, "tool_call_id") and m.tool_call_id:
                    item["tool_call_id"] = m.tool_call_id
                input_messages.append(item)
            else:
                raise ValueError(f"Unsupported message type: {type(m)}")

        instructions = "\n".join(
            instructions_chunks) if instructions_chunks else None
        return instructions, input_messages

    def _format_for_chat(self, messages: List[BaseMessage]):
        """Map LC messages to Chat Completions format (max compatibility)."""
        formatted = []
        for m in messages:
            if isinstance(m, SystemMessage):
                # Stick with 'system' for widest support (incl. vLLM)
                role = "system"
            elif isinstance(m, HumanMessage):
                role = "user"
            elif isinstance(m, AIMessage):
                role = "assistant"
            elif isinstance(m, ToolMessage):
                role = "tool"
            else:
                raise ValueError(f"Unsupported message type: {type(m)}")

            msg: Dict[str, Any] = {"role": role, "content": m.content}
            # Preserve tool_call_id on tool messages if present
            if role == "tool" and hasattr(m, "tool_call_id") and m.tool_call_id:
                msg["tool_call_id"] = m.tool_call_id
            formatted.append(msg)
        return formatted

    # ------- Public API -------
    def invoke(self, messages: List[BaseMessage]) -> AIMessage:
        # 1) Try Responses API (preferred on OpenAI cloud)
        if self.prefer_responses:
            try:
                instructions, input_messages = self._format_for_responses(
                    messages)
                resp = self.client.responses.create(
                    model=self.model_name,
                    input=input_messages if input_messages else "",
                    instructions=instructions,
                    temperature=self.temperature,
                    text={"stop": self.stop_sequences},
                )
                text = getattr(resp, "output_text", None) or ""
                return AIMessage(content=text)
            except APIStatusError as e:
                # Graceful fallback if server doesn’t support /responses
                if e.status_code in {404, 400, 405, 501}:
                    pass  # fall through to Chat Completions
                else:
                    raise

        # 2) Fallback: Chat Completions (widely supported, incl. vLLM)
        formatted = self._format_for_chat(messages)
        resp = self.client.chat.completions.create(
            model=self.model_name,
            messages=formatted,
            temperature=self.temperature,
            stop=self.stop_sequences,
        )
        choice = resp.choices[0].message
        # Preserve tool calls if any (LangChain can route them)
        extra = {}
        if getattr(choice, "tool_calls", None):
            extra["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in choice.tool_calls
            ]
        return AIMessage(content=choice.content or "", additional_kwargs=extra)


def get_llm(
    model_name: str = "claude-3-5-sonnet-20241022",
    temperature: float = 0.4,
    stop_sequences: list[str] | None = None,
    source: SourceType | None = None,
    base_url: str | None = None,
    api_key: str = "EMPTY",
):
    if source is None:
        source = "vLLM"

    # Create appropriate model based on source
    if source == "vLLM":
        llm = CustomLLMClient(
            model_name=model_name,
            temperature=temperature,
            stop_sequences=stop_sequences,
            base_url=base_url,
            api_key=api_key,
        )
        return llm
    elif source == "OpenAI":
        if ChatOpenAI is None:
            raise ImportError("langchain_openai is required for source='OpenAI'. Install langchain-openai or use source='vLLM'.")
        llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL") or None,  # optional override
        )
        return llm
    else:
        raise ValueError(
            f"Invalid source: {source}. Valid options are 'OpenAI', 'AzureOpenAI', 'Anthropic', 'Gemini', 'Bedrock', or 'Ollama'"
        )


def call_llm(
    llm: BaseChatModel,
    messages: list[BaseMessage],
    state=None
):
    response = None
    max_retries = 5
    base_delay = 1

    for attempt in range(max_retries):
        try:
            response = llm.invoke(messages)
            break
        except Exception as e:
            if '429' in str(e) and attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                print(f"Rate limit error detected. Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                print(f"LLM call failed after retries or due to a non-retriable error: {e}")
                error_message = "The language model is currently unavailable. Please try again later."
                if state is not None:
                    state["messages"].append(AIMessage(content=error_message))
                    state["next_step"] = "done"
                return None, state

    return response, state


def get_event_driven_plan(
    news_text: str,
    instruction_text: str,
    ts_length: int = 100,
    client=None,
    model: str = "gpt-4o",
    llm=None,
    system_prompt: str = None,
) -> dict:
    """
    Get structured editing plan from LLM based on event-driven instruction.

    Supports two calling patterns:
    1. Direct OpenAI client: get_event_driven_plan(news, instruction, client, model)
    2. LangChain LLM: get_event_driven_plan(news, instruction, llm=llm, system_prompt=prompt)

    Args:
        news_text: Breaking news or event description
        instruction_text: Specific editing instruction
        ts_length: Total length of the time series (default: 100). Used to inject
                   the correct sequence length into the system prompt and user message.
        client: OpenAI client instance (for direct API calls)
        model: Model name for direct API calls (default: "gpt-4o")
        llm: LangChain LLM client (alternative to client)
        system_prompt: Custom system prompt (uses default if None; length-aware prompt
                       is generated automatically when ts_length is provided)

    Returns:
        dict: Parsed JSON plan with keys: thought, tool_name, parameters
    """
    import json
    import re

    if system_prompt is None:
        from agent.prompts import get_event_driven_agent_prompt
        system_prompt = get_event_driven_agent_prompt(ts_length=ts_length)

    early_end = max(1, ts_length // 3)
    mid_end = max(early_end + 1, (2 * ts_length) // 3)
    user_message = (
        f"News: {news_text}\n\n"
        f"Instruction: {instruction_text}\n\n"
        f"[Sequence Info] The time series has exactly {ts_length} timesteps "
        f"(index 0 to {ts_length - 1}). "
        f"You MUST ensure region[0] >= 0 and region[1] <= {ts_length}.\n"
        f"[Allowed Effect Families] trend, seasonality, volatility, impulse, level, shutdown\n"
        f"[Localization Hint] Long trends usually span 20-60 steps; transient shocks usually span 5-20 steps.\n"
        f"[Temporal Buckets] early=[0,{early_end}), mid=[{early_end},{mid_end}), late=[{mid_end},{ts_length})\n"
        f"[Anchor Mapping] 清晨/早班/大清早->early; 中午/运行中段/刚才->mid; 深夜/夜间低谷前/半夜->late"
    )
    
    if client is not None:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            temperature=0.3,
        )
        content = response.choices[0].message.content
    elif llm is not None:
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_message)
        ]
        response, _ = call_llm(llm, messages)
        if response is None:
            raise ValueError("LLM call failed - no response received")
        content = response.content
    else:
        raise ValueError("Either 'client' or 'llm' must be provided")
    
    json_match = re.search(r'\{[\s\S]*\}', content)
    if json_match:
        try:
            plan = json.loads(json_match.group())
            normalized = normalize_llm_plan(plan, ts_length=ts_length)
            normalized = _apply_explicit_prompt_hints(normalized, instruction_text, ts_length=ts_length)
            refined_localization = localize_region(
                prompt_text=instruction_text,
                ts_length=ts_length,
                llm_plan=normalized,
            )
            normalized.setdefault("localization", {}).update(refined_localization)
            normalized.setdefault("parameters", {})["region"] = refined_localization["region"]
            normalized.setdefault("execution", {}).setdefault("parameters", {})
            normalized["execution"]["parameters"]["region"] = refined_localization["region"]
            normalized = _apply_volatility_route(normalized, instruction_text, ts_length=ts_length)
            return normalize_llm_plan(normalized, ts_length=ts_length)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse LLM response as JSON: {e}\nResponse: {content}")
    else:
        raise ValueError(f"No JSON found in LLM response: {content}")
