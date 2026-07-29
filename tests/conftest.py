"""Shared pytest fixtures: a synthetic MarketPanel so tests need no network."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alphaduel.features.store import MarketPanel


@pytest.fixture
def synthetic_panel() -> MarketPanel:
    n = 300
    rng = np.random.default_rng(0)
    # Geometric random walk for prices.
    rets = rng.normal(0.0005, 0.01, size=n)
    close = 100.0 * np.exp(np.cumsum(rets))
    open_ = close * (1.0 + rng.normal(0.0, 0.001, size=n))
    ts = pd.date_range("2020-01-01", periods=n, freq="B", tz="UTC")
    features = np.column_stack(
        [rets, np.roll(rets, 1), np.abs(rets)]  # 3 toy causal-ish features
    ).astype(np.float32)
    return MarketPanel(
        timestamps=ts,
        open=open_,
        close=close,
        features=features,
        feature_names=["f0", "f1", "f2"],
    )
