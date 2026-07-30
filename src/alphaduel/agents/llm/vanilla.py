"""Vanilla off-the-shelf LLM trading agent (injected chat model + LangGraph).

Build the chat model outside the agent (see :mod:`alphaduel.agents.llm.factory`),
then pass it in::

    from alphaduel.agents.llm import LLMHandler, VanillaLLMAgent

    llm = LLMHandler.create("qwen3.5:2b", temperature=0.0)
    agent = VanillaLLMAgent(llm=llm)

The agent reads a structured market brief, produces free-form natural language, and
embeds share-delta actions in a fenced JSON block::

    ```json
    {"actions": {"AAPL": 2, "MSFT": -1}}
    ```

Omitted tickers are held. A parse failure becomes a no-trade with ``last_parse_ok=False``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from langchain_core.messages import BaseMessage, SystemMessage

from alphaduel.agents.base import Agent
from alphaduel.agents.llm.graph import build_llm_graph
from alphaduel.agents.llm.parse import ParsedLLMAction
from alphaduel.agents.llm.state_format import (
    format_market_state,
    share_deltas_to_target_weights,
)
from alphaduel.prompts import prompt_manager


class VanillaLLMAgent(Agent):
    """Off-the-shelf LLM agent driven by Jinja prompts + an injected chat model."""

    name = "llm_vanilla"

    def __init__(
        self,
        llm: Any,
        *,
        use_memory: bool = False,
        max_memory_turns: int = 8,
        system_prompt_key: str = "base_agent_system",
        user_prompt_key: str = "base_agent_user",
        symbols: list[str] | None = None,
        allow_short: bool = False,
    ) -> None:
        if llm is None:
            raise TypeError(
                "VanillaLLMAgent requires an `llm` instance; "
                "build one with LLMHandler.create(...) / create_llm(...)."
            )
        self.llm = llm
        self.use_memory = bool(use_memory)
        self.max_memory_turns = int(max_memory_turns)
        self.system_prompt_key = system_prompt_key
        self.user_prompt_key = user_prompt_key
        self.symbols = list(symbols) if symbols else None
        self.allow_short = bool(allow_short)

        self.last_thoughts: str | None = None
        self.last_parse_ok: bool = False
        self.last_actions: dict[str, int] = {}
        self.last_raw_response: str = ""

        self._graph = build_llm_graph(self.llm)
        self._memory: list[BaseMessage] = []

    # ------------------------------------------------------------------ Agent API

    def reset(self) -> None:
        self._memory.clear()
        self.last_thoughts = None
        self.last_parse_ok = False
        self.last_actions = {}
        self.last_raw_response = ""

    def act(self, observation: np.ndarray, info: dict) -> np.ndarray:
        symbols = self._symbols_from(info)
        market_state = format_market_state(observation, info, symbols=symbols)
        system_text = prompt_manager.get(self.system_prompt_key).content
        user_text = prompt_manager.get(self.user_prompt_key).render(
            market_state=market_state,
            symbols=symbols,
        )

        prior: list[BaseMessage] = [SystemMessage(content=system_text)]
        if self.use_memory:
            prior.extend(self._memory)

        result = self._graph.invoke(
            {
                "messages": prior,
                "market_state": market_state,
                "user_prompt": user_text,
                "raw_response": "",
                "parsed": ParsedLLMAction(),
            }
        )
        parsed: ParsedLLMAction = result["parsed"]
        self.last_raw_response = parsed.raw
        self.last_thoughts = parsed.rationale
        self.last_parse_ok = parsed.parse_ok
        self.last_actions = dict(parsed.actions) if parsed.parse_ok else {}

        if self.use_memory:
            # Keep only the newest human/AI turns produced by the graph (skip system).
            new_msgs = [m for m in result["messages"] if not isinstance(m, SystemMessage)]
            self._memory = new_msgs[-(2 * self.max_memory_turns) :]

        shares, prices, equity = self._portfolio_arrays(info, symbols)
        if not parsed.parse_ok:
            # Do nothing: target current weights (or zeros if flat).
            if equity <= 0:
                return np.zeros(len(symbols), dtype=np.float32)
            current = (shares * prices) / equity
            current = np.clip(current, 0.0, 1.0)
            total = float(current.sum())
            if total > 1.0:
                current = current / total
            return current.astype(np.float32)

        weights = share_deltas_to_target_weights(
            parsed.actions,
            symbols=symbols,
            shares=shares,
            prices=prices,
            equity=equity,
            allow_short=self.allow_short,
        )
        return weights if len(symbols) > 1 else weights[:1]

    # ----------------------------------------------------------------- internals

    def _symbols_from(self, info: dict) -> list[str]:
        if self.symbols:
            return list(self.symbols)
        if "symbols" in info:
            return [str(s) for s in info["symbols"]]
        return ["ASSET"]

    def _portfolio_arrays(
        self, info: dict, symbols: list[str]
    ) -> tuple[np.ndarray, np.ndarray, float]:
        n = len(symbols)
        if "prices" in info:
            prices = np.asarray(info["prices"], dtype=np.float64).ravel()[:n]
        else:
            prices = np.full(n, float(info.get("price", 1.0)), dtype=np.float64)
        if prices.size < n:
            prices = np.pad(prices, (0, n - prices.size), constant_values=1.0)

        raw_shares = info.get("shares", 0)
        shares = np.asarray(raw_shares, dtype=np.float64).ravel()
        if shares.size == 0:
            shares = np.zeros(n, dtype=np.float64)
        elif shares.size == 1 and n > 1:
            padded = np.zeros(n, dtype=np.float64)
            padded[0] = shares[0]
            shares = padded
        elif shares.size < n:
            shares = np.pad(shares, (0, n - shares.size))
        else:
            shares = shares[:n]

        equity = float(
            info.get("equity")
            or float(np.dot(shares, prices) + float(info.get("cash", 0.0)))
        )
        return shares, prices, equity


#: Backward-compatible alias used by the registry / SPEC naming.
OffShelfLLMAgent = VanillaLLMAgent
