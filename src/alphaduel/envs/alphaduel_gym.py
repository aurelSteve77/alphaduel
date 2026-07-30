"""AlphaDuelGym: single-asset, weight-action trading environment (Gymnasium API).

Rules of the game (shared by every agent):
- The agent observes features + portfolio state at bar ``t`` (built from past data only).
- Its action is a target asset weight in ``[0, 1]`` (remainder is cash).
- The order executes at the **open of bar ``t + execution_lag``** (leakage guard), as an
  integer number of shares, net of transaction costs.
- Equity is marked at the close of the new bar; reward is the configured signal.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from alphaduel.config.schema import EnvConfig
from alphaduel.envs.costs import TransactionCostModel
from alphaduel.envs.portfolio import Portfolio
from alphaduel.envs.rewards import make_reward
from alphaduel.features.store import MarketPanel

_PORTFOLIO_STATE_DIM = 4  # asset_weight, cash_fraction, unrealized_pnl, steps_remaining


class AlphaDuelGym(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        panel: MarketPanel,
        config: EnvConfig,
        include_portfolio_state: bool = True,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        self.panel = panel
        self.config = config
        self.include_portfolio_state = include_portfolio_state
        self._rng = np.random.default_rng(seed)

        obs_dim = panel.n_features + (_PORTFOLIO_STATE_DIM if include_portfolio_state else 0)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        # Single asset: one target weight in [0, 1]; cash is the remainder.
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)

        self._costs = TransactionCostModel(config.costs)
        self._reward_fn = make_reward(config.reward)
        self._max_start = panel.n_steps - config.episode_length - config.execution_lag - 1
        if self._max_start < 1:
            raise ValueError("Not enough data for one episode; extend the date range.")

    # ------------------------------------------------------------------ gym API

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self.t = int(self._rng.integers(1, self._max_start)) if self.config.random_start else 1
        self.step_count = 0
        self.portfolio = Portfolio(
            cash=self.config.initial_cash,
            costs=self._costs,
            allow_short=self.config.allow_short,
        )
        self.prev_equity = self.config.initial_cash
        self.peak_equity = self.config.initial_cash
        self._reward_fn.reset()
        return self._observation(), self._info(fill_shares=0, fill_cost=0.0)

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        target_w = float(np.clip(np.asarray(action, dtype=np.float64).ravel()[0], 0.0, 1.0))

        exec_idx = self.t + self.config.execution_lag
        exec_price = float(self.panel.open[exec_idx])
        delta = self.portfolio.target_weight_to_order(target_w, exec_price)
        fill = self.portfolio.execute(delta, exec_price, adv=None)
        turnover = abs(delta) * exec_price / self.prev_equity if self.prev_equity > 0 else 0.0

        self.t += 1
        self.step_count += 1
        mark_price = float(self.panel.close[self.t])
        equity = self.portfolio.equity(mark_price)
        self.peak_equity = max(self.peak_equity, equity)
        drawdown = (self.peak_equity - equity) / self.peak_equity if self.peak_equity > 0 else 0.0

        terminated = (
            self.step_count >= self.config.episode_length
            or (self.t + self.config.execution_lag) >= self.panel.n_steps
        )
        reward = self._reward_fn.step(self.prev_equity, equity, turnover, drawdown, terminated)
        self.prev_equity = equity
        return (
            self._observation(),
            float(reward),
            bool(terminated),
            False,
            self._info(fill_shares=fill.shares, fill_cost=fill.cost),
        )

    # --------------------------------------------------------------- internals

    def _observation(self) -> np.ndarray:
        feats = self.panel.features[self.t]
        if not self.include_portfolio_state:
            return feats.astype(np.float32)

        price = float(self.panel.close[self.t])
        eq = self.portfolio.equity(price)
        asset_weight = (self.portfolio.shares * price) / eq if eq > 0 else 0.0
        cash_frac = self.portfolio.cash / eq if eq > 0 else 0.0
        unrealized = eq / self.config.initial_cash - 1.0
        steps_remaining = 1.0 - self.step_count / self.config.episode_length
        port = np.array(
            [asset_weight, cash_frac, unrealized, steps_remaining], dtype=np.float32
        )
        return np.concatenate([feats, port]).astype(np.float32)

    def _info(self, fill_shares: int, fill_cost: float) -> dict:
        price = float(self.panel.close[self.t])
        symbol = getattr(self.panel, "symbol", None) or "ASSET"
        eq = self.portfolio.equity(price)
        weight = (self.portfolio.shares * price) / eq if eq > 0 else 0.0
        return {
            "timestamp": self.panel.timestamps[self.t],
            "equity": eq,
            "cash": self.portfolio.cash,
            "shares": self.portfolio.shares,
            "price": price,
            "prices": np.array([price], dtype=np.float64),
            "symbols": [symbol],
            "weights": np.array([weight], dtype=np.float32),
            "asset_features": self.panel.features[self.t].reshape(1, -1),
            "feature_names": list(self.panel.feature_names),
            "fill_shares": fill_shares,
            "fill_cost": fill_cost,
        }
