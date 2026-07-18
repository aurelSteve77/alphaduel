"""Gymnasium wrapper around :class:`PortfolioSimulator`.

Two observation modes share the same action space (per-asset position deltas in
``[-1, 1]``):

- **feature mode** (real data): the observation is the flattened point-in-time
  feature vector ``features[t]`` (info known at ``t``) plus the current weights,
  cash fraction and normalized time. Build it with :meth:`from_market_data`.
- **synthetic mode** (no data yet): the observation is a rolling window of past
  log-returns generated from GBM prices, so the env runs out of the box.

No lookahead: the observation at ``t`` uses information up to ``t``; the trade
executes at ``prices[t]`` and its reward is the return realized over
``[t, t + rebalance_every]`` (strictly future).
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
        features: np.ndarray | None = None,
        feature_names: list[str] | None = None,
        budget: float = 100_000.0,
        cost_bps: float = 5.0,
        max_weight: float = 0.30,
        rebalance_every: int = 1,
        obs_window: int = 21,
        horizon: int = 252,
        n_assets: int = 10,
        n_days: int = 1500,
        seed: int = 0,
        start_index: int | None = None,
        end_index: int | None = None,
    ) -> None:
        super().__init__()

        if prices is None:
            prices = generate_gbm_prices(n_days, n_assets, seed=seed)
        self.prices = np.asarray(prices, dtype=float)
        self.n_days, self.n_assets = self.prices.shape

        self.features = None if features is None else np.asarray(features, dtype=float)
        self.feature_names = feature_names
        self.use_features = self.features is not None

        # Only needed in synthetic (return-window) mode.
        self.log_returns = np.vstack(
            [np.zeros((1, self.n_assets)), np.diff(np.log(self.prices), axis=0)]
        )

        self.obs_window = int(obs_window)
        self.horizon = int(horizon)
        default_low = 0 if self.use_features else self.obs_window
        self.start_index = int(default_low if start_index is None else start_index)
        self.end_index = int(self.n_days if end_index is None else end_index)

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
        if self.use_features:
            obs_dim = self.n_assets * self.features.shape[2] + self.n_assets + 2
        else:
            obs_dim = self.obs_window * self.n_assets + self.n_assets + 2
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        self._rng = np.random.default_rng(seed)
        self._start = self.start_index

    @classmethod
    def from_market_data(
        cls,
        market,
        *,
        split: str | None = None,
        train_end: str | None = None,
        val_end: str | None = None,
        horizon: int = 252,
        **kwargs,
    ) -> AlphaDuelEnv:
        """Build a feature-mode env, optionally restricted to a chronological split."""
        if split is not None and train_end is not None and val_end is not None:
            lo, hi = market.split_indices(train_end, val_end)[split]
        else:
            lo, hi = market.valid_from, market.n_steps
        return cls(
            prices=market.prices,
            features=market.features,
            feature_names=market.feature_names,
            horizon=horizon,
            start_index=max(lo, market.valid_from),
            end_index=hi,
            **kwargs,
        )

    def _build_obs(self) -> np.ndarray:
        t = self.sim._t
        state = self.sim.state()
        time_frac = (t - self._start) / max(1, self.horizon)
        if self.use_features:
            market_obs = self.features[t].reshape(-1)
        else:
            market_obs = self.log_returns[t - self.obs_window + 1 : t + 1].flatten()
        obs = np.concatenate(
            [market_obs, state["weights"], np.array([state["cash_weight"], time_frac])]
        )
        return obs.astype(np.float32)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        latest_start = self.end_index - self.horizon - 1
        low = self.start_index
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
