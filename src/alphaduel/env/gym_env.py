"""Gymnasium wrapper around :class:`PortfolioSimulator`.

The observation is a flat vector of: a rolling window of past log-returns per
asset (features known at time ``t``), the current portfolio weights, the cash
fraction, and the normalized time within the episode. The action is a continuous
per-asset position delta in ``[-1, 1]``.

Later, real numeric features and verbalized news will replace the synthetic
return-window observation, but the interface stays the same so models written
against v1 keep working.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ..utils.prices import generate_gbm_prices
from .portfolio import PortfolioSimulator


class AlphaDuelEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        prices: np.ndarray | None = None,
        *,
        budget: float = 100_000.0,
        cost_bps: float = 5.0,
        max_weight: float = 0.30,
        rebalance_every: int = 1,
        obs_window: int = 21,
        horizon: int = 252,
        n_assets: int = 10,
        n_days: int = 1500,
        seed: int = 0,
    ) -> None:
        super().__init__()

        if prices is None:
            # Synthetic prices so the env is runnable before the data pipeline exists.
            prices = generate_gbm_prices(n_days, n_assets, seed=seed)
        self.prices = np.asarray(prices, dtype=float)
        self.n_days, self.n_assets = self.prices.shape

        # Log-returns aligned to prices (row 0 is zero-padded).
        self.log_returns = np.vstack(
            [np.zeros((1, self.n_assets)), np.diff(np.log(self.prices), axis=0)]
        )

        self.obs_window = int(obs_window)
        self.horizon = int(horizon)
        self.sim = PortfolioSimulator(
            self.prices,
            budget=budget,
            cost_bps=cost_bps,
            max_weight=max_weight,
            rebalance_every=rebalance_every,
            horizon=horizon,
        )

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.n_assets,), dtype=np.float32
        )
        obs_dim = self.obs_window * self.n_assets + self.n_assets + 2
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        self._rng = np.random.default_rng(seed)
        self._start = self.obs_window

    def _build_obs(self) -> np.ndarray:
        t = self.sim._t
        window = self.log_returns[t - self.obs_window + 1 : t + 1]  # [W, K]
        state = self.sim.state()
        time_frac = (t - self._start) / max(1, self.horizon)
        obs = np.concatenate(
            [
                window.flatten(),
                state["weights"],
                np.array([state["cash_weight"], time_frac]),
            ]
        )
        return obs.astype(np.float32)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        latest_start = self.n_days - self.horizon - 1
        low = self.obs_window
        start = int(self._rng.integers(low, max(low + 1, latest_start)))
        self._start = start
        self.sim.reset(start)
        return self._build_obs(), {"portfolio_value": self.sim.portfolio_value}

    def step(self, action: np.ndarray):
        result = self.sim.step(action)
        obs = self._build_obs()
        info = {
            "portfolio_value": result.portfolio_value,
            "turnover": result.turnover,
            "cost": result.cost,
        }
        return obs, float(result.reward), bool(result.terminated), False, info
