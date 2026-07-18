"""The shared portfolio environment (core simulator + Gymnasium wrapper)."""

from __future__ import annotations

from .gym_env import AlphaDuelEnv
from .portfolio import PortfolioSimulator, StepResult, project_long_only

# Register the Gymnasium id once, so users can `gymnasium.make("AlphaDuel-v0")`.
try:  # pragma: no cover - registration is a side effect
    from gymnasium.envs.registration import register

    register(id="AlphaDuel-v0", entry_point="alphaduel.env.gym_env:AlphaDuelEnv")
except Exception:  # noqa: BLE001 - already registered / gymnasium missing
    pass

__all__ = ["AlphaDuelEnv", "PortfolioSimulator", "StepResult", "project_long_only"]
