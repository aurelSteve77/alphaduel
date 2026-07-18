"""Rule-based baseline policies (the honest bar the learned agents must beat)."""

from __future__ import annotations

from .policies import (
    STRATEGY_SPECS,
    BuyAndHold,
    EqualWeight,
    InverseVolatility,
    MeanReversion,
    Momentum,
    MovingAverageTrend,
    Policy,
    RandomAllocation,
    build_policy,
    default_policies,
)

__all__ = [
    "Policy",
    "EqualWeight",
    "BuyAndHold",
    "Momentum",
    "MeanReversion",
    "InverseVolatility",
    "MovingAverageTrend",
    "RandomAllocation",
    "STRATEGY_SPECS",
    "build_policy",
    "default_policies",
]
