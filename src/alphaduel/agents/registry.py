"""Agent registry: build an ``Agent`` from an ``AgentConfig``."""

from __future__ import annotations

from alphaduel.agents.base import Agent
from alphaduel.agents.baselines import (
    BuyAndHoldAgent,
    EqualWeightAgent,
    InverseVolatilityAgent,
    MeanReversionAgent,
    MomentumAgent,
    RandomAgent,
    RandomWeightsAgent,
    VolatilityTargetAgent,
)
from alphaduel.config.schema import AgentConfig

_BASELINES = {
    "buy_and_hold": BuyAndHoldAgent,
    "random": RandomAgent,
    "momentum": MomentumAgent,
    "mean_reversion": MeanReversionAgent,
    "volatility_target": VolatilityTargetAgent,
    "equal_weight": EqualWeightAgent,
    "random_weights": RandomWeightsAgent,
    "inverse_volatility": InverseVolatilityAgent,
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

    if config.kind == "generative":
        from alphaduel.agents.generative import GenPortfolioAgent

        agent = GenPortfolioAgent(**config.params)
        agent.name = config.name
        return agent

    raise ValueError(f"Unknown agent kind: {config.kind!r}")
