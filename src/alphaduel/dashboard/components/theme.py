"""Shared palette and small formatting helpers for dashboard components."""

from __future__ import annotations

_PALETTE = [
    "#168740", "#0ea5e9", "#059669", "#fbbf24", "#e45756",
    "#b279a2", "#3d3a2a", "#72b7b2", "#9d755d", "#f58518",
]

# Prettier labels for known agents.
_LABELS = {
    "buy_and_hold": "Buy & Hold",
    "momentum": "Momentum",
    "mean_reversion": "Mean Reversion",
    "volatility_target": "Vol Target",
    "random": "Random",
    "equal_weight": "Equal Weight",
    "inverse_volatility": "Inverse Vol",
    "random_weights": "Random Weights",
    "genportfolio": "GenPortfolio",
    "llm_vanilla": "LLM Vanilla",
}


def color_map(names: list[str]) -> dict[str, str]:
    """Assign a stable color to each agent name."""
    return {name: _PALETTE[i % len(_PALETTE)] for i, name in enumerate(names)}


def label(name: str) -> str:
    return _LABELS.get(name, name.replace("_", " ").title())


def pct(value: float) -> str:
    return f"{value * 100:,.2f}%"


def num(value: float, digits: int = 2) -> str:
    return f"{value:,.{digits}f}"
