"""Agent registry: build an ``Agent`` from an ``AgentConfig``."""

from __future__ import annotations

from alphaduel.agents.base import Agent
from alphaduel.agents.baselines import (
    BuyAndHoldAgent,
    MeanReversionAgent,
    MomentumAgent,
    RandomAgent,
)
from alphaduel.config.schema import AgentConfig

_BASELINES = {
    "buy_and_hold": BuyAndHoldAgent,
    "random": RandomAgent,
    "momentum": MomentumAgent,
    "mean_reversion": MeanReversionAgent,
}


def build_agent(config: AgentConfig) -> Agent:
    if config.kind == "baseline":
        cls = _BASELINES.get(config.name)
        if cls is None:
            raise ValueError(f"Unknown baseline agent: {config.name!r}")
        agent = cls(**config.params)
        agent.name = config.name
        return agent

    if config.kind == "rl":
        from alphaduel.agents.rl import SB3Agent

        agent = SB3Agent(**config.params)
        agent.name = config.name
        return agent

    if config.kind == "llm":
        from alphaduel.agents.llm import OffShelfLLMAgent

        agent = OffShelfLLMAgent(**config.params)
        agent.name = config.name
        return agent

    raise ValueError(f"Unknown agent kind: {config.kind!r}")
