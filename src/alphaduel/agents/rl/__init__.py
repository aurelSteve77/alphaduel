"""RL agents (P1). Thin wrapper over Stable-Baselines3.

Kept import-light: SB3/torch are optional deps (``pip install -e '.[rl]'``). The wrapper
adapts an SB3 policy to the ``Agent`` interface so RL and baselines are evaluated
identically.
"""

from __future__ import annotations

import numpy as np

from alphaduel.agents.base import Agent


class SB3Agent(Agent):
    name = "sb3"

    def __init__(self, algorithm: str = "PPO", **params) -> None:
        self.algorithm = algorithm
        self.params = params
        self._model = None

    def train(self, env) -> None:
        try:
            import stable_baselines3 as sb3
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "RL extras not installed. Run: pip install -e '.[rl]'"
            ) from exc

        algo_cls = getattr(sb3, self.algorithm)
        total_timesteps = self.params.pop("total_timesteps", 100_000)
        policy = self.params.pop("policy", "MlpPolicy")
        self._model = algo_cls(policy, env, **self.params)
        self._model.learn(total_timesteps=total_timesteps)

    def act(self, observation: np.ndarray, info: dict) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("SB3Agent must be trained (or loaded) before acting.")
        action, _ = self._model.predict(observation, deterministic=True)
        return np.asarray(action, dtype=np.float32)
