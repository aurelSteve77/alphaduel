"""Naive baselines — mandatory reference points for any serious agent.

These are deliberately simple and self-contained: they act from the price stream in
``info`` rather than from feature ordering, so they work regardless of the feature set.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from alphaduel.agents.base import Agent


class BuyAndHoldAgent(Agent):
    name = "buy_and_hold"

    def act(self, observation: np.ndarray, info: dict) -> np.ndarray:
        return np.array([1.0], dtype=np.float32)  # fully invested at all times


class RandomAgent(Agent):
    name = "random"

    def __init__(self, seed: int = 0) -> None:
        self._rng = np.random.default_rng(seed)

    def reset(self) -> None:
        pass

    def act(self, observation: np.ndarray, info: dict) -> np.ndarray:
        return np.array([self._rng.uniform(0.0, 1.0)], dtype=np.float32)


class MomentumAgent(Agent):
    name = "momentum"

    def __init__(self, lookback: int = 20) -> None:
        self.lookback = lookback
        self._prices: deque[float] = deque(maxlen=lookback + 1)

    def reset(self) -> None:
        self._prices.clear()

    def act(self, observation: np.ndarray, info: dict) -> np.ndarray:
        self._prices.append(float(info.get("price", np.nan)))
        if len(self._prices) <= self.lookback:
            return np.array([1.0], dtype=np.float32)
        signal = self._prices[-1] / self._prices[0] - 1.0
        return np.array([1.0 if signal > 0 else 0.0], dtype=np.float32)


class MeanReversionAgent(Agent):
    name = "mean_reversion"

    def __init__(self, lookback: int = 20) -> None:
        self.lookback = lookback
        self._prices: deque[float] = deque(maxlen=lookback)

    def reset(self) -> None:
        self._prices.clear()

    def act(self, observation: np.ndarray, info: dict) -> np.ndarray:
        self._prices.append(float(info.get("price", np.nan)))
        if len(self._prices) < self.lookback:
            return np.array([0.5], dtype=np.float32)
        mean = float(np.mean(self._prices))
        last = self._prices[-1]
        # Below the mean -> buy; above -> lighten.
        return np.array([1.0 if last < mean else 0.0], dtype=np.float32)


class VolatilityTargetAgent(Agent):
    """Single-asset baseline: scale exposure so realized vol tracks ``target_vol``."""

    name = "volatility_target"

    def __init__(self, target_vol: float = 0.15, lookback: int = 20) -> None:
        self.target_vol = target_vol
        self.lookback = lookback
        self._prices: deque[float] = deque(maxlen=lookback + 1)

    def reset(self) -> None:
        self._prices.clear()

    def act(self, observation: np.ndarray, info: dict) -> np.ndarray:
        self._prices.append(float(info.get("price", np.nan)))
        if len(self._prices) <= self.lookback:
            return np.array([1.0], dtype=np.float32)
        prices = np.asarray(self._prices, dtype=np.float64)
        rets = prices[1:] / prices[:-1] - 1.0
        realized = float(rets.std(ddof=1) * np.sqrt(252))
        weight = self.target_vol / realized if realized > 1e-9 else 1.0
        return np.array([np.clip(weight, 0.0, 1.0)], dtype=np.float32)


class EqualWeightAgent(Agent):
    """Multi-asset baseline: split capital equally across all assets (info-driven length)."""

    name = "equal_weight"

    def act(self, observation: np.ndarray, info: dict) -> np.ndarray:
        n = len(info["symbols"])
        return np.full(n, 1.0 / n, dtype=np.float32)


class RandomWeightsAgent(Agent):
    """Multi-asset baseline: random weights over the simplex (Dirichlet)."""

    name = "random_weights"

    def __init__(self, seed: int = 0) -> None:
        self._rng = np.random.default_rng(seed)

    def act(self, observation: np.ndarray, info: dict) -> np.ndarray:
        n = len(info["symbols"])
        return self._rng.dirichlet(np.ones(n)).astype(np.float32)


class InverseVolatilityAgent(Agent):
    """Multi-asset baseline: risk-parity-lite, weights ∝ 1/σ_i (fully invested)."""

    name = "inverse_volatility"

    def __init__(self, lookback: int = 20) -> None:
        self.lookback = lookback
        self._prices: deque[np.ndarray] = deque(maxlen=lookback + 1)

    def reset(self) -> None:
        self._prices.clear()

    def act(self, observation: np.ndarray, info: dict) -> np.ndarray:
        n = len(info["symbols"])
        self._prices.append(np.asarray(info["prices"], dtype=np.float64))
        if len(self._prices) <= self.lookback:
            return np.full(n, 1.0 / n, dtype=np.float32)
        prices = np.vstack(self._prices)
        rets = prices[1:] / prices[:-1] - 1.0
        vol = rets.std(axis=0, ddof=1)
        inv = np.where(vol > 1e-9, 1.0 / vol, 0.0)
        total = inv.sum()
        weights = inv / total if total > 1e-9 else np.full(n, 1.0 / n)
        return weights.astype(np.float32)

