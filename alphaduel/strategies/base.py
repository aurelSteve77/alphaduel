"""Strategy interface shared by LLM and quantitative agents."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Strategy(Protocol):
    """Map an environment observation (+ info) to an integer share-delta action."""

    def act(self, obs: dict[str, np.ndarray], info: dict[str, Any]) -> np.ndarray:
        """Return a ``(n_assets,)`` int array of share deltas."""
        ...
