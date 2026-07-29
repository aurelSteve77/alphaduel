"""Agent interface shared by baselines, RL policies, and LLM agents.

Every agent maps an observation to an action in the env's action space (a target
weight vector). Agents may optionally emit ``thoughts`` (used by LLM agents and shown
in the dashboard) via ``last_thoughts``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Agent(ABC):
    #: Human-readable name used in reports / MLflow.
    name: str = "agent"

    #: Optional free-text rationale for the last action (LLM agents populate this).
    last_thoughts: str | None = None

    def reset(self) -> None:  # noqa: B027 - optional hook
        """Called at the start of each episode."""

    @abstractmethod
    def act(self, observation: np.ndarray, info: dict) -> np.ndarray:
        """Return an action for the given observation."""

    def train(self, env) -> None:  # noqa: B027 - optional hook
        """Optional training phase (no-op for baselines)."""
