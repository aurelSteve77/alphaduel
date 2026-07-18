"""Data pipeline: download, point-in-time features, splits, and the shared
``MarketData`` container."""

from __future__ import annotations

from .dataset import MarketData
from .features import compute_features
from .splits import split_indices

__all__ = ["MarketData", "compute_features", "split_indices"]
