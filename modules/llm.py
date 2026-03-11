import os
import time
from typing import Literal, Optional, List, Optional, Dict, Any

from langchain_openai import ChatOpenAI
from openai import OpenAI, APIStatusError

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.language_models.chat_models import BaseChatModel


SourceType = Literal["OpenAI", "DashScope", "InterWeb", "Ollama", "vLLM"]


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
        client: OpenAI client instance (for direct API calls)
        model: Model name for direct API calls (default: "gpt-4o")
        llm: LangChain LLM client (alternative to client)
        system_prompt: Custom system prompt (uses default if None)
    
    Returns:
        dict: Parsed JSON plan with keys: thought, tool_name, parameters
    """
    import json
    import re
    
    if system_prompt is None:
        from agent.prompts import EVENT_DRIVEN_AGENT_PROMPT
        system_prompt = EVENT_DRIVEN_AGENT_PROMPT
    
    user_message = f"News: {news_text}\n\nInstruction: {instruction_text}"
    
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
            return plan
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse LLM response as JSON: {e}\nResponse: {content}")
    else:
        raise ValueError(f"No JSON found in LLM response: {content}")
