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
