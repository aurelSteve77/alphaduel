"""LLM trading policy: free-form text reply with a fenced JSON action block.

The model reasons in natural language, then ends with::

    ```json
    {"actions": {"AAPL": 2, "MSFT": -1}}
    ```

The full reply is kept as the rationale; share deltas are parsed from that JSON.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from pydantic import BaseModel, Field

from alphaduel.logger import get_logger
from alphaduel.prompt_manager import PromptManager
from alphaduel.strategies.formatting import build_prompt_payload
from alphaduel.strategies.parsing import message_content, parse_freeform_response
from alphaduel.strategies.schemas import TradingDecision

log = get_logger(__name__)


class LLMPolicyConfig(BaseModel):
    """Knobs for :class:`LLMPolicy` (also loadable from ``project.yaml``)."""

    model: str = Field("qwen3.5:0.8b", description="Ollama model name")
    temperature: float = Field(0.0, ge=0.0, description="Sampling temperature")
    reasoning: bool = Field(False, description="Enable reasoning")
    base_url: str | None = Field(
        None,
        description="Optional Ollama base URL (default: local daemon)",
    )
    prompt_group: str = Field("llm_policy", description="PromptManager group name")
    max_history: int | None = Field(
        None,
        ge=0,
        description=(
            "Max prior user/ai turn pairs to keep in the chat (None = unlimited). "
            "Oldest turns are dropped first."
        ),
    )

    model_config = {"extra": "forbid"}

    @classmethod
    def from_project_yaml(cls, **overrides: object) -> LLMPolicyConfig:
        from alphaduel.configuration import Configuration

        cfg = Configuration()
        payload = {
            "model": cfg.get("llm.model", "qwen3.5:2b"),
            "temperature": cfg.get("llm.temperature", 0.0),
            "base_url": cfg.get("llm.base_url"),
            "reasoning": cfg.get("llm.reasoning", False),
            "prompt_group": cfg.get("llm.prompt_group", "llm_policy"),
            "max_history": cfg.get("llm.max_history"),
        }
        payload.update(overrides)
        return cls.model_validate(payload)


class LLMPolicy:
    """Gymnasium-compatible strategy driven by a chat LLM (free-form + JSON).

    Example::

        from alphaduel import TradingEnv, TradingEnvConfig
        from alphaduel.strategies import LLMPolicy

        env = TradingEnv(config=TradingEnvConfig(...), prices=prices)
        policy = LLMPolicy.from_env(env)

        obs, info = env.reset(seed=0)
        action = policy.act(obs, info)
        obs, reward, terminated, truncated, info = env.step(action)
    """

    def __init__(
        self,
        tickers: Sequence[str],
        *,
        max_shares: int,
        allow_short: bool = False,
        feature_names: Sequence[str] | None = None,
        config: LLMPolicyConfig | None = None,
        llm: Any | None = None,
        chain: Any | None = None,
        prompts: PromptManager | None = None,
    ) -> None:
        if max_shares < 1:
            raise ValueError("max_shares must be >= 1")
        self.tickers = [str(t).strip().upper() for t in tickers]
        if not self.tickers:
            raise ValueError("tickers must be non-empty")
        self.max_shares = int(max_shares)
        self.allow_short = bool(allow_short)
        self.feature_names = [str(n) for n in (feature_names or [])]
        self.config = config or LLMPolicyConfig.from_project_yaml()
        self._prompts = prompts or PromptManager()
        self._last_decision: TradingDecision | None = None
        self._last_raw: str | None = None
        # Alternating prior turns: user state → ai reply → user → ai → ...
        self._history: list[dict[str, str]] = []

        # ``chain`` kept as an alias for injectable fakes / custom callables.
        self._llm = chain if chain is not None else self._build_llm(llm)

    @classmethod
    def from_env(cls, env: Any, **kwargs: Any) -> LLMPolicy:
        """Build a policy aligned with a :class:`~alphaduel.environments.TradingEnv`."""
        return cls(
            tickers=list(env.tickers),
            max_shares=int(env.config.max_shares),
            allow_short=bool(env.config.allow_short),
            feature_names=list(env.feature_names),
            **kwargs,
        )

    @property
    def last_decision(self) -> TradingDecision | None:
        """Most recent parsed decision (``None`` before the first ``act``)."""
        return self._last_decision

    @property
    def last_raw(self) -> str | None:
        """Most recent raw model text (``None`` before the first ``act``)."""
        return self._last_raw

    @property
    def history(self) -> list[dict[str, str]]:
        """Copy of prior ``user`` / ``ai`` turns (excludes the live system + user)."""
        return [dict(m) for m in self._history]

    def reset(self) -> None:
        """Clear chat history and the last decision (call on ``env.reset``)."""
        self._history.clear()
        self._last_decision = None
        self._last_raw = None

    def act(self, obs: dict[str, np.ndarray], info: dict[str, Any]) -> np.ndarray:
        """Query the LLM, parse the free-form reply, return share deltas."""
        if int(info.get("step", 0)) == 0:
            self.reset()

        messages = self.build_messages(obs, info)
        raw = message_content(self._llm.invoke(messages))
        decision = parse_freeform_response(raw)
        self._last_raw = raw
        self._last_decision = decision
        action = decision.to_action(self.tickers, max_shares=self.max_shares)

        self._append_turn(messages[-1], raw)

        log.info(
            "LLM decision date=%s action=%s rationale_chars=%d",
            info.get("date"),
            action.tolist(),
            len(decision.rationale),
        )
        return action

    def __call__(self, obs: dict[str, np.ndarray], info: dict[str, Any]) -> np.ndarray:
        return self.act(obs, info)

    def build_messages(
        self, obs: dict[str, np.ndarray], info: dict[str, Any]
    ) -> list[dict[str, str]]:
        """Render ``[system] + history + [current user]`` for the LLM call."""
        payload = build_prompt_payload(
            obs,
            info,
            tickers=self.tickers,
            feature_names=self.feature_names,
            max_shares=self.max_shares,
            allow_short=self.allow_short,
        )
        group = self._prompts[self.config.prompt_group]
        return [
            {"role": "system", "content": group["system"].render(**payload)},
            *self._history,
            {"role": "user", "content": group["user"].render(**payload)},
        ]

    def _append_turn(self, user_message: dict[str, str], ai_text: str) -> None:
        """Record the user state and the free-form AI reply in chat history."""
        self._history.append({"role": "user", "content": user_message["content"]})
        self._history.append({"role": "ai", "content": ai_text})
        self._trim_history()

    def _trim_history(self) -> None:
        """Keep at most ``max_history`` user/ai pairs (2 messages each)."""
        limit = self.config.max_history
        if limit is None:
            return
        max_messages = int(limit) * 2
        if len(self._history) > max_messages:
            self._history = self._history[-max_messages:]

    def _build_llm(self, llm: Any | None) -> Any:
        if llm is not None:
            return llm
        from langchain_ollama import ChatOllama

        kwargs: dict[str, Any] = {
            "model": self.config.model,
            "temperature": self.config.temperature,
            "reasoning": self.config.reasoning,
        }
        if self.config.base_url:
            kwargs["base_url"] = self.config.base_url
        return ChatOllama(**kwargs)
