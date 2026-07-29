"""Feature interface.

A ``Feature`` transforms a raw per-symbol price/macro panel into one or more named
columns. Each feature declares a ``warmup`` (bars of history required before its first
valid value) so the store can guard against NaN leakage and align windows correctly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class FeatureContext:
    """Parameters shared across features (from ``FeatureConfig``)."""

    technical_windows: tuple[int, ...] = (5, 10, 20)
    rsi_period: int = 14


class Feature(ABC):
    #: Feature group used by the ablation toggles ("technical" | "macro" | "text").
    group: str = "technical"

    @property
    @abstractmethod
    def warmup(self) -> int:
        """Number of leading rows that will be NaN/invalid for this feature."""

    @abstractmethod
    def compute(self, prices: pd.DataFrame, ctx: FeatureContext) -> pd.DataFrame:
        """Return a DataFrame of named feature columns aligned to ``prices`` index."""
