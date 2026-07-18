"""Typed configuration loaded from YAML.

A single ``Config`` object is passed around the whole project so that the data
pipeline, the environment, the models and the evaluation harness all agree on
the same universe, budget, horizon and costs.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class DataConfig(BaseModel):
    """Universe and date range for the study."""

    tickers: list[str] = Field(
        default_factory=lambda: [
            "AAPL", "MSFT", "JNJ", "JPM", "XOM",
            "PG", "KO", "WMT", "HD", "CVX",
        ]
    )
    start: str = "2015-01-01"
    end: str = "2024-12-31"
    # Chronological split (no shuffling — avoids lookahead).
    train_end: str = "2021-12-31"
    val_end: str = "2022-12-31"


class EnvConfig(BaseModel):
    """Parameters of the shared portfolio environment.

    One environment step advances ``rebalance_every`` trading days. The base
    clock is daily so that daily-timestamped news can be aligned to each state.
    """

    budget: float = 100_000.0
    # Round-trip transaction cost applied to turnover, in basis points.
    cost_bps: float = 5.0
    # Long-only cap per asset (fraction of portfolio value).
    max_weight: float = 0.30
    allow_short: bool = False
    # Δ — number of trading days advanced per decision (1 = daily).
    rebalance_every: int = 1
    # Length of the past-returns window exposed in the observation.
    obs_window: int = 21
    # Episode length in steps.
    horizon: int = 252


class Config(BaseModel):
    seed: int = 42
    data: DataConfig = Field(default_factory=DataConfig)
    env: EnvConfig = Field(default_factory=EnvConfig)


def load_config(path: str | Path) -> Config:
    """Load a :class:`Config` from a YAML file."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return Config(**(raw or {}))
