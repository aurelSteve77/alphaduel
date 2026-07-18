"""Framework-agnostic, long-only portfolio simulator.

This is the single source of truth for how an allocation turns into P&L. Both
branches of the study (LLM and quant) drive *this* object, which is what makes
the comparison fair.

No lookahead by construction: a decision taken at step ``t`` is executed at
``prices[t]`` and its reward is realized over ``[t, t + rebalance_every]``. The
caller is responsible for building observations from information available up to
``t`` only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class StepResult:
    reward: float
    portfolio_value: float
    turnover: float
    cost: float
    terminated: bool


def project_long_only(weights: np.ndarray, max_weight: float) -> np.ndarray:
    """Project raw target weights onto the feasible long-only set.

    Constraints: ``w_i >= 0``, ``w_i <= max_weight`` and ``sum(w) <= 1`` (the
    remainder is held in cash). This is a simple clip-then-renormalize, which is
    good enough for v1; an exact simplex projection can come later.
    """
    w = np.clip(np.asarray(weights, dtype=float), 0.0, max_weight)
    total = w.sum()
    if total > 1.0:
        w = w / total
    return w


class PortfolioSimulator:
    def __init__(
        self,
        prices: np.ndarray,
        *,
        budget: float = 100_000.0,
        cost_bps: float = 5.0,
        max_weight: float = 0.30,
        rebalance_every: int = 1,
        horizon: int | None = None,
    ) -> None:
        self.prices = np.asarray(prices, dtype=float)
        if self.prices.ndim != 2:
            raise ValueError("prices must be a 2D array of shape [n_steps, n_assets]")
        self.n_steps, self.n_assets = self.prices.shape
        self.budget = float(budget)
        self.cost_rate = cost_bps / 1e4
        self.max_weight = float(max_weight)
        self.rebalance_every = int(rebalance_every)
        self.horizon = horizon
        self.reset(0)

    def reset(self, start: int = 0) -> dict:
        self._t = int(start)
        self._start = int(start)
        self.weights = np.zeros(self.n_assets)
        self.cash_weight = 1.0
        self.portfolio_value = self.budget
        return self.state()

    def state(self) -> dict:
        return {
            "t": self._t,
            "weights": self.weights.copy(),
            "cash_weight": self.cash_weight,
            "portfolio_value": self.portfolio_value,
        }

    def step(self, delta_weights: np.ndarray) -> StepResult:
        """Apply a continuous change in target weights and advance time.

        ``delta_weights`` is the per-asset position delta the agent wants (buy if
        positive, sell if negative). It is added to the current weights and then
        projected onto the feasible long-only set.
        """
        delta = np.asarray(delta_weights, dtype=float)
        target = project_long_only(self.weights + delta, self.max_weight)

        turnover = float(np.abs(target - self.weights).sum())
        cost = turnover * self.cost_rate * self.portfolio_value
        value_after_cost = self.portfolio_value - cost

        t0 = self._t
        t1 = min(t0 + self.rebalance_every, self.n_steps - 1)
        asset_ret = self.prices[t1] / self.prices[t0] - 1.0
        port_ret = float(np.dot(target, asset_ret))  # cash earns 0
        new_value = value_after_cost * (1.0 + port_ret)

        reward = new_value / self.portfolio_value - 1.0

        self.portfolio_value = new_value
        self.weights = target
        self.cash_weight = 1.0 - float(target.sum())
        self._t = t1

        return StepResult(
            reward=reward,
            portfolio_value=new_value,
            turnover=turnover,
            cost=cost,
            terminated=self._is_terminated(),
        )

    def _is_terminated(self) -> bool:
        if self._t >= self.n_steps - 1:
            return True
        if self.horizon is not None and (self._t - self._start) >= self.horizon:
            return True
        return False
