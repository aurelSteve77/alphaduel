"""MultiAssetGym: N-asset, weight-vector trading environment (Gymnasium API).

Generalizes ``AlphaDuelGym`` to a portfolio over N assets. Same leakage guard (decide on
bar ``t``, execute at the open of ``t + execution_lag``), realistic per-asset costs, and
pluggable reward. The action is a target weight vector over the N assets; the remainder
(``1 - sum(weights)``) is held in cash. Weights summing above 1 are renormalized to 1.

This is the env target for the P5 GenPortfolio agent (see SPEC §15), but any multi-asset
agent (e.g. a multi-asset PPO) can train and be benchmarked here identically.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from alphaduel.config.schema import EnvConfig
from alphaduel.envs.costs import TransactionCostModel
from alphaduel.envs.portfolio import MultiAssetPortfolio
from alphaduel.envs.rewards import make_reward
from alphaduel.features.store import MultiAssetPanel


class MultiAssetGym(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        panel: MultiAssetPanel,
        config: EnvConfig,
        include_portfolio_state: bool = True,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        self.panel = panel
        self.config = config
        self.include_portfolio_state = include_portfolio_state
        self.n_assets = panel.n_assets
        self._rng = np.random.default_rng(seed)

        # Portfolio state appended to the flat observation: per-asset weights + cash + pnl + t.
        port_dim = (self.n_assets + 2) if include_portfolio_state else 0
        obs_dim = panel.n_assets * panel.n_features + port_dim
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        # One target weight per asset in [0, 1]; cash is the remainder.
        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.n_assets,), dtype=np.float32
        )

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
        self.portfolio = MultiAssetPortfolio(
            cash=self.config.initial_cash,
            costs=self._costs,
            n_assets=self.n_assets,
            allow_short=self.config.allow_short,
        )
        self.prev_equity = self.config.initial_cash
        self.peak_equity = self.config.initial_cash
        self._reward_fn.reset()
        return self._observation(), self._info(fills=np.zeros(self.n_assets), cost=0.0)

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        target_w = np.clip(np.asarray(action, dtype=np.float64).ravel(), 0.0, 1.0)
        total = target_w.sum()
        if total > 1.0:  # renormalize an over-allocated book to fully invested
            target_w = target_w / total

        exec_idx = self.t + self.config.execution_lag
        exec_prices = self.panel.open[exec_idx]
        deltas = self.portfolio.target_weights_to_orders(target_w, exec_prices)
        cost = self.portfolio.execute(deltas, exec_prices, adv=None)
        traded_notional = float(np.abs(deltas) @ exec_prices)
        turnover = traded_notional / self.prev_equity if self.prev_equity > 0 else 0.0

        self.t += 1
        self.step_count += 1
        mark_prices = self.panel.close[self.t]
        equity = self.portfolio.equity(mark_prices)
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
            self._info(fills=deltas, cost=cost),
        )

    # --------------------------------------------------------------- internals

    def _weights(self) -> np.ndarray:
        prices = self.panel.close[self.t]
        eq = self.portfolio.equity(prices)
        if eq <= 0:
            return np.zeros(self.n_assets, dtype=np.float32)
        return (self.portfolio.shares * prices / eq).astype(np.float32)

    def _observation(self) -> np.ndarray:
        feats = self.panel.features[self.t].reshape(-1)  # (N*F,)
        if not self.include_portfolio_state:
            return feats.astype(np.float32)

        prices = self.panel.close[self.t]
        eq = self.portfolio.equity(prices)
        weights = self._weights()
        cash_frac = self.portfolio.cash / eq if eq > 0 else 0.0
        unrealized = eq / self.config.initial_cash - 1.0
        extra = np.array([cash_frac, unrealized], dtype=np.float32)
        return np.concatenate([feats, weights, extra]).astype(np.float32)

    def _info(self, fills: np.ndarray, cost: float) -> dict:
        prices = self.panel.close[self.t]
        return {
            "timestamp": self.panel.timestamps[self.t],
            "equity": self.portfolio.equity(prices),
            "cash": self.portfolio.cash,
            "shares": self.portfolio.shares.copy(),
            "prices": prices.copy(),
            "symbols": self.panel.symbols,
            "weights": self._weights(),
            "asset_features": self.panel.features[self.t].copy(),  # (N, F)
            "feature_names": self.panel.feature_names,
            "fill_shares": int(np.count_nonzero(fills)),
            "fill_cost": cost,
        }
