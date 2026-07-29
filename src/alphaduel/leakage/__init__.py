"""Leakage checks: automated guards that no future data reaches the agent.

The headline test asserts that perturbing any *future* value in the price/feature panel
cannot change the observation the agent sees at time ``t``. If it can, a causal feature
or the env's execution timing is leaking. See SPEC §4.1.
"""

from __future__ import annotations

import numpy as np

from alphaduel.envs.alphaduel_gym import AlphaDuelGym
from alphaduel.features.store import MarketPanel


def assert_observation_is_causal(panel: MarketPanel, env: AlphaDuelGym, t: int) -> None:
    """Assert the observation at ``t`` is independent of data strictly after ``t``.

    We snapshot the observation, corrupt all future feature rows, rebuild, and require the
    observation at ``t`` to be unchanged.
    """
    env.reset(seed=0)
    env.t = t
    baseline_obs = env._observation().copy()

    corrupted = panel.features.copy()
    corrupted[t + 1 :] = np.nan  # destroy the future
    env.panel = MarketPanel(
        timestamps=panel.timestamps,
        open=panel.open,
        close=panel.close,
        features=corrupted,
        feature_names=panel.feature_names,
    )
    env.t = t
    corrupted_obs = env._observation()

    if not np.allclose(baseline_obs, corrupted_obs, equal_nan=True):
        raise AssertionError(
            f"Leakage detected at t={t}: observation depends on future feature values."
        )
