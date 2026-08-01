"""LangGraph loop for one LLM decision step (optional message memory)."""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from alphaduel.agents.llm.parse import ParsedLLMAction, parse_llm_response


class LLMGraphState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    market_state: str
    user_prompt: str
    raw_response: str
    parsed: ParsedLLMAction


def build_llm_graph(llm: Any):
    """Compile a tiny graph: inject user turn → call model → parse actions."""

    def write_user(state: LLMGraphState) -> dict:
        return {"messages": [HumanMessage(content=state["user_prompt"])]}

    def call_model(state: LLMGraphState) -> dict:
        response = llm.invoke(state["messages"])
        # Prefer LangChain's unified `.text` (handles Anthropic content blocks).
        text_prop = getattr(response, "text", None)
        if isinstance(text_prop, str) and text_prop:
            content = text_prop
        else:
            content = getattr(response, "content", None)
            if content is None:
                content = str(response)
            elif isinstance(content, list):
                # Some chat models return content blocks (e.g. Anthropic tool/thinking).
                parts = []
                for block in content:
                    if isinstance(block, dict) and "text" in block:
                        parts.append(str(block["text"]))
                    else:
                        parts.append(str(block))
                content = "".join(parts)
        return {
            "messages": [AIMessage(content=content)],
            "raw_response": content,
        }

    def parse_node(state: LLMGraphState) -> dict:
        return {"parsed": parse_llm_response(state.get("raw_response", ""))}

    graph = StateGraph(LLMGraphState)
    graph.add_node("write_user", write_user)
    graph.add_node("call_model", call_model)
    graph.add_node("parse", parse_node)
    graph.add_edge(START, "write_user")
    graph.add_edge("write_user", "call_model")
    graph.add_edge("call_model", "parse")
    graph.add_edge("parse", END)
    return graph.compile()


def system_message(text: str) -> SystemMessage:
    return SystemMessage(content=text)
