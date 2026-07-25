"""Portfolio trading environments."""

from __future__ import annotations

import gymnasium as gym

from alphaduel.environments.config import TradingEnvConfig
from alphaduel.environments.trading_env import AlphaDuelEnv, TradingEnv, make_trading_env

__all__ = [
    "AlphaDuelEnv",
    "TradingEnv",
    "TradingEnvConfig",
    "make_trading_env",
]

_ENV_ID = "AlphaDuel-v0"

if _ENV_ID not in gym.envs.registry:
    gym.register(
        id=_ENV_ID,
        entry_point="alphaduel.environments.trading_env:make_trading_env",
    )
