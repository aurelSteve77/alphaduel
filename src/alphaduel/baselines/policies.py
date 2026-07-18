"""Baseline policies (the honest bar the learned agents must beat).

A policy maps ``(market, t, state)`` to a vector of **target weights** (fractions
of portfolio value per asset). The backtester turns the target into a position
delta and feeds it to the shared simulator, so baselines and learned agents are
scored on exactly the same substrate.

Only information available at ``t`` is used (``market.prices[:t+1]``), preserving
the no-lookahead guarantee. Every policy caps weights with ``project_long_only``.
"""

from __future__ import annotations

import numpy as np

from ..env.portfolio import project_long_only

_EPS = 1e-12


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
            weights[selected] = min(1.0 / len(selected), self.max_weight)
        return weights


class MeanReversion(Policy):
    """Cross-sectional reversal: long the biggest short-term losers."""

    name = "mean_reversion"

    def __init__(self, lookback: int = 5, bottom_n: int = 3, max_weight: float = 0.30) -> None:
        self.lookback = lookback
        self.bottom_n = bottom_n
        self.max_weight = max_weight

    def act(self, market, t: int, state: dict) -> np.ndarray:
        k = market.n_assets
        weights = np.zeros(k)
        if t - self.lookback < 0:
            return weights

        ret = market.prices[t] / market.prices[t - self.lookback] - 1.0
        ranked = np.argsort(ret)[: self.bottom_n]  # most negative first
        selected = [i for i in ranked if ret[i] < 0]
        if selected:
            weights[selected] = min(1.0 / len(selected), self.max_weight)
        return weights


class InverseVolatility(Policy):
    """Risk-based: weight each asset inversely to its recent volatility."""

    name = "inverse_vol"

    def __init__(self, lookback: int = 21, max_weight: float = 0.30) -> None:
        self.lookback = lookback
        self.max_weight = max_weight

    def act(self, market, t: int, state: dict) -> np.ndarray:
        k = market.n_assets
        if t - self.lookback < 1:
            return np.zeros(k)

        window = market.prices[t - self.lookback : t + 1]
        log_ret = np.diff(np.log(window), axis=0)
        vol = log_ret.std(axis=0)
        inv = 1.0 / (vol + _EPS)
        raw = inv / inv.sum()
        return project_long_only(raw, self.max_weight)


class MovingAverageTrend(Policy):
    """Trend-following: hold assets trading above their moving average, equally."""

    name = "ma_trend"

    def __init__(self, window: int = 50, max_weight: float = 0.30) -> None:
        self.window = window
        self.max_weight = max_weight

    def act(self, market, t: int, state: dict) -> np.ndarray:
        k = market.n_assets
        weights = np.zeros(k)
        if t - self.window < 0:
            return weights

        sma = market.prices[t - self.window + 1 : t + 1].mean(axis=0)
        in_trend = np.where(market.prices[t] > sma)[0]
        if in_trend.size:
            weights[in_trend] = min(1.0 / in_trend.size, self.max_weight)
        return weights


class RandomAllocation(Policy):
    """Random long-only allocation — a floor baseline for sanity checks."""

    name = "random"

    def __init__(self, max_weight: float = 0.30, seed: int = 0) -> None:
        self.max_weight = max_weight
        self.seed = seed
        self._rng = np.random.default_rng(seed)

    def reset(self) -> None:
        self._rng = np.random.default_rng(self.seed)

    def act(self, market, t: int, state: dict) -> np.ndarray:
        raw = self._rng.random(market.n_assets)
        raw = raw / raw.sum()
        return project_long_only(raw, self.max_weight)


# ---------------------------------------------------------------------------
# Registry — a single source of truth for names, tunable params and factories.
# ---------------------------------------------------------------------------

STRATEGY_SPECS: dict[str, dict] = {
    "equal_weight": {},
    "buy_and_hold": {},
    "momentum": {"lookback": 63, "top_n": 3},
    "mean_reversion": {"lookback": 5, "bottom_n": 3},
    "inverse_vol": {"lookback": 21},
    "ma_trend": {"window": 50},
    "random": {"seed": 0},
}

_FACTORIES = {
    "equal_weight": EqualWeight,
    "buy_and_hold": BuyAndHold,
    "momentum": Momentum,
    "mean_reversion": MeanReversion,
    "inverse_vol": InverseVolatility,
    "ma_trend": MovingAverageTrend,
    "random": RandomAllocation,
}


def build_policy(name: str, max_weight: float = 0.30, **params) -> Policy:
    """Instantiate a policy by name, merging defaults with any overrides."""
    if name not in _FACTORIES:
        raise KeyError(f"unknown strategy '{name}'. Available: {list(_FACTORIES)}")
    merged = {**STRATEGY_SPECS[name], **params}
    return _FACTORIES[name](max_weight=max_weight, **merged)


def default_policies(max_weight: float = 0.30) -> list[Policy]:
    """One instance of every registered strategy with default parameters."""
    return [build_policy(name, max_weight) for name in STRATEGY_SPECS]
