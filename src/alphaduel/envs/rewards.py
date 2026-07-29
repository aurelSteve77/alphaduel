"""Reward functions.

- ``log_return``: dense per-step log growth of equity (default, stable for RL).
- ``differential_sharpe``: online risk-adjusted reward (Moody & Saffell 1998).
- ``terminal_pnl``: sparse; only the final step is non-zero (hard for RL).

Optional drawdown / turnover penalties are applied on top of the base signal.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod

from alphaduel.config.schema import RewardConfig


class RewardFunction(ABC):
    def __init__(self, config: RewardConfig) -> None:
        self.config = config

    def reset(self) -> None:  # noqa: B027 - optional hook
        pass

    @abstractmethod
    def step(
        self, prev_equity: float, equity: float, turnover: float, drawdown: float, done: bool
    ) -> float: ...

    def _penalty(self, turnover: float, drawdown: float) -> float:
        return (
            self.config.turnover_penalty * turnover
            + self.config.drawdown_penalty * drawdown
        )


class LogReturnReward(RewardFunction):
    def step(self, prev_equity, equity, turnover, drawdown, done) -> float:
        base = math.log(equity / prev_equity) if prev_equity > 0 and equity > 0 else 0.0
        return base - self._penalty(turnover, drawdown)


class DifferentialSharpeReward(RewardFunction):
    """Online Sharpe via EMAs of return (A) and squared return (B)."""

    def reset(self) -> None:
        self.a = 0.0
        self.b = 0.0

    def step(self, prev_equity, equity, turnover, drawdown, done) -> float:
        r = (equity - prev_equity) / prev_equity if prev_equity > 0 else 0.0
        eta = self.config.dsr_eta
        d_a = r - self.a
        d_b = r * r - self.b
        denom = (self.b - self.a**2) ** 1.5
        dsr = (self.b * d_a - 0.5 * self.a * d_b) / denom if denom > 1e-12 else 0.0
        self.a += eta * d_a
        self.b += eta * d_b
        return dsr - self._penalty(turnover, drawdown)


class TerminalPnLReward(RewardFunction):
    def __init__(self, config: RewardConfig) -> None:
        super().__init__(config)
        self._initial: float | None = None

    def reset(self) -> None:
        self._initial = None

    def step(self, prev_equity, equity, turnover, drawdown, done) -> float:
        if self._initial is None:
            self._initial = prev_equity
        if not done:
            return -self._penalty(turnover, drawdown)
        return (equity - self._initial) / self._initial - self._penalty(turnover, drawdown)


def make_reward(config: RewardConfig) -> RewardFunction:
    mapping = {
        "log_return": LogReturnReward,
        "differential_sharpe": DifferentialSharpeReward,
        "terminal_pnl": TerminalPnLReward,
    }
    reward = mapping[config.kind](config)
    reward.reset()
    return reward
