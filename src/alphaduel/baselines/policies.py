"""Baseline policies.

A policy maps ``(market, t, state)`` to a vector of **target weights** (fractions
of portfolio value per asset). The backtester turns the target into a position
delta and feeds it to the shared simulator, so baselines and learned agents are
scored on exactly the same substrate.

Only information available at ``t`` is used (``market.prices[:t+1]``), preserving
the no-lookahead guarantee.
"""

from __future__ import annotations

import numpy as np


class Policy:
    name = "policy"

    def reset(self) -> None:  # noqa: D401 - stateful policies override this
        """Reset any per-backtest internal state."""

    def act(self, market, t: int, state: dict) -> np.ndarray:
        raise NotImplementedError


class EqualWeight(Policy):
    """Hold an equally weighted, fully invested portfolio (rebalanced each step)."""

    name = "equal_weight"

    def __init__(self, max_weight: float = 0.30) -> None:
        self.max_weight = max_weight

    def act(self, market, t: int, state: dict) -> np.ndarray:
        k = market.n_assets
        return np.full(k, min(1.0 / k, self.max_weight))


class BuyAndHold(Policy):
    """Allocate equally once, then never trade again (turnover ≈ 0 afterwards)."""

    name = "buy_and_hold"

    def __init__(self, max_weight: float = 0.30) -> None:
        self.max_weight = max_weight
        self._allocated = False

    def reset(self) -> None:
        self._allocated = False

    def act(self, market, t: int, state: dict) -> np.ndarray:
        if not self._allocated:
            self._allocated = True
            k = market.n_assets
            return np.full(k, min(1.0 / k, self.max_weight))
        return state["weights"]  # hold => zero delta


class Momentum(Policy):
    """Cross-sectional momentum: long the top-N assets by trailing return."""

    name = "momentum"

    def __init__(self, lookback: int = 63, top_n: int = 3, max_weight: float = 0.30) -> None:
        self.lookback = lookback
        self.top_n = top_n
        self.max_weight = max_weight

    def act(self, market, t: int, state: dict) -> np.ndarray:
        k = market.n_assets
        weights = np.zeros(k)
        if t - self.lookback < 0:
            return weights

        mom = market.prices[t] / market.prices[t - self.lookback] - 1.0
        ranked = np.argsort(mom)[::-1][: self.top_n]
        selected = [i for i in ranked if mom[i] > 0]
        if selected:
            w = min(1.0 / len(selected), self.max_weight)
            weights[selected] = w
        return weights
