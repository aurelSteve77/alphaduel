"""Simple non-LLM baselines used for eval smoke tests and comparisons."""

from __future__ import annotations

from typing import Any

import numpy as np


class HoldStrategy:
    """Always hold — emit a zero share-delta vector."""

    def __init__(self, n_assets: int) -> None:
        if n_assets < 1:
            raise ValueError("n_assets must be >= 1")
        self.n_assets = int(n_assets)

    def act(self, obs: dict[str, np.ndarray], info: dict[str, Any]) -> np.ndarray:
        return np.zeros(self.n_assets, dtype=np.int32)


class RandomStrategy:
    """Uniform random integer share deltas in ``[-max_shares, max_shares]``."""

    def __init__(self, n_assets: int, max_shares: int, *, seed: int | None = None) -> None:
        if n_assets < 1:
            raise ValueError("n_assets must be >= 1")
        if max_shares < 1:
            raise ValueError("max_shares must be >= 1")
        self.n_assets = int(n_assets)
        self.max_shares = int(max_shares)
        self._rng = np.random.default_rng(seed)

    def act(self, obs: dict[str, np.ndarray], info: dict[str, Any]) -> np.ndarray:
        return self._rng.integers(
            -self.max_shares, self.max_shares + 1, size=self.n_assets, dtype=np.int32
        )


class BuyAndHoldStrategy:
    """On the first step, buy up to ``max_shares`` of each ticker; then hold."""

    def __init__(self, n_assets: int, max_shares: int) -> None:
        if n_assets < 1:
            raise ValueError("n_assets must be >= 1")
        self.n_assets = int(n_assets)
        self.max_shares = int(max_shares)
        self._done_entry = False

    def reset(self) -> None:
        self._done_entry = False

    def act(self, obs: dict[str, np.ndarray], info: dict[str, Any]) -> np.ndarray:
        if self._done_entry or int(info.get("step", 0)) > 0:
            return np.zeros(self.n_assets, dtype=np.int32)
        self._done_entry = True
        return np.full(self.n_assets, self.max_shares, dtype=np.int32)
