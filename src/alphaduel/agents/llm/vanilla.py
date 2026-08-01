"""Vanilla off-the-shelf LLM trading agent (injected chat model + LangGraph).

Build the chat model outside the agent (see :mod:`alphaduel.agents.llm.factory`),
then pass it in::

    from alphaduel.agents.llm import LLMHandler, VanillaLLMAgent

    llm = LLMHandler.create("qwen3.5:2b", temperature=0.0)
    agent = VanillaLLMAgent(llm=llm)

The agent reads a structured market brief with **anonymized** tickers (ASS1, ASS2, …),
produces free-form natural language, and embeds share-delta actions in a fenced JSON
block::

    ```json
    {"actions": {"ASS1": 2, "ASS2": -1}}
    ```

Actions are unmasked back to real symbols before execution. ``last_actions`` exposes
the **real** ticker deltas for the dashboard. Omitted tickers are held. A parse failure
becomes a no-trade with ``last_parse_ok=False``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from langchain_core.messages import BaseMessage, SystemMessage

from alphaduel.agents.base import Agent
from alphaduel.agents.llm.graph import build_llm_graph
from alphaduel.agents.llm.masking import SymbolMask
from alphaduel.agents.llm.parse import ParsedLLMAction
from alphaduel.agents.llm.state_format import (
    format_market_state,
    share_deltas_to_target_weights,
)
from alphaduel.agents.llm.trajectory import serialize_messages, to_jsonable
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
        mask_symbols: bool = True,
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
        self.mask_symbols = bool(mask_symbols)

        self.last_thoughts: str | None = None
        self.last_parse_ok: bool = False
        self.last_actions: dict[str, int] = {}
        self.last_masked_actions: dict[str, int] = {}
        self.last_raw_response: str = ""
        self.last_symbol_mask: SymbolMask | None = None
        self.last_trace: dict[str, Any] | None = None

        self._graph = build_llm_graph(self.llm)
        self._memory: list[BaseMessage] = []

    # ------------------------------------------------------------------ Agent API

    def reset(self) -> None:
        self._memory.clear()
        self.last_thoughts = None
        self.last_parse_ok = False
        self.last_actions = {}
        self.last_masked_actions = {}
        self.last_raw_response = ""
        self.last_symbol_mask = None
        self.last_trace = None

    def act(self, observation: np.ndarray, info: dict) -> np.ndarray:
        real_symbols = self._symbols_from(info)
        mask = SymbolMask.from_symbols(real_symbols) if self.mask_symbols else None
        llm_symbols = list(mask.masked) if mask is not None else list(real_symbols)
        self.last_symbol_mask = mask

        market_state = format_market_state(observation, info, symbols=llm_symbols)
        system_text = prompt_manager.get(self.system_prompt_key).content
        user_text = prompt_manager.get(self.user_prompt_key).render(
            market_state=market_state,
            symbols=llm_symbols,
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

        if parsed.parse_ok and mask is not None:
            self.last_masked_actions = {
                str(k).upper(): int(v) for k, v in parsed.actions.items()
            }
            self.last_actions = mask.unmask_actions(parsed.actions)
        elif parsed.parse_ok:
            self.last_masked_actions = {}
            self.last_actions = dict(parsed.actions)
        else:
            self.last_masked_actions = {}
            self.last_actions = {}

        if self.use_memory:
            # Keep only the newest human/AI turns produced by the graph (skip system).
            new_msgs = [m for m in result["messages"] if not isinstance(m, SystemMessage)]
            self._memory = new_msgs[-(2 * self.max_memory_turns) :]

        shares, prices, equity = self._portfolio_arrays(info, real_symbols)
        if not parsed.parse_ok:
            # Do nothing: target current weights (or zeros if flat).
            if equity <= 0:
                weights = np.zeros(len(real_symbols), dtype=np.float32)
            else:
                current = (shares * prices) / equity
                current = np.clip(current, 0.0, 1.0)
                total = float(current.sum())
                if total > 1.0:
                    current = current / total
                weights = current.astype(np.float32)
        else:
            weights = share_deltas_to_target_weights(
                self.last_actions,
                symbols=real_symbols,
                shares=shares,
                prices=prices,
                equity=equity,
                allow_short=self.allow_short,
            )
            if len(real_symbols) == 1:
                weights = weights[:1]

        # Full context the model saw + a clean single-turn SFT triple.
        full_messages = serialize_messages(list(result.get("messages") or []))
        sft_messages = [
            {"role": "system", "content": system_text},
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": self.last_raw_response or ""},
        ]
        symbol_map = {}
        if mask is not None:
            symbol_map = {m: r for m, r in zip(mask.masked, mask.real)}

        self.last_trace = {
            "market_state": market_state,
            "system_prompt_key": self.system_prompt_key,
            "user_prompt_key": self.user_prompt_key,
            "system_prompt": system_text,
            "user_prompt": user_text,
            "messages": full_messages,
            "sft_messages": sft_messages,
            "assistant_raw": self.last_raw_response,
            "parse_ok": bool(self.last_parse_ok),
            "actions_masked": dict(self.last_masked_actions),
            "actions_real": dict(self.last_actions),
            "symbols_real": list(real_symbols),
            "symbols_llm": list(llm_symbols),
            "symbol_map": symbol_map,
            "mask_symbols": bool(self.mask_symbols),
            "use_memory": bool(self.use_memory),
            "observation": to_jsonable(observation),
            "info_before": {
                k: to_jsonable(info[k])
                for k in (
                    "timestamp",
                    "equity",
                    "cash",
                    "shares",
                    "prices",
                    "price",
                    "weights",
                    "symbols",
                    "feature_names",
                )
                if k in info
            },
            "target_weights": to_jsonable(weights),
        }
        return weights

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
