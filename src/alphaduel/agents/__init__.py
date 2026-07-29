"""Agents: shared interface + baseline / RL / LLM families."""

from alphaduel.agents.base import Agent
from alphaduel.agents.baselines import (
    BuyAndHoldAgent,
    MeanReversionAgent,
    MomentumAgent,
    RandomAgent,
)
from alphaduel.agents.registry import build_agent

__all__ = [
    "Agent",
    "BuyAndHoldAgent",
    "MeanReversionAgent",
    "MomentumAgent",
    "RandomAgent",
    "build_agent",
]
