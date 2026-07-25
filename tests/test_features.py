"""Tests for point-in-time feature calculators."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alphaduel.data import features


def test_resolve_unknown_feature():
    with pytest.raises(KeyError, match="Unknown feature"):
        features.resolve_features(["not_a_feature"])


def test_build_feature_tensor_shapes(price_panel):
    # Longer panel so RSI / vol lookbacks produce finite values.
    idx = pd.bdate_range("2024-01-02", periods=40)
    rng = np.random.default_rng(0)
    prices = pd.DataFrame(
        {
            "AAPL": 100 * np.cumprod(1 + rng.normal(0, 0.01, len(idx))),
            "MSFT": 200 * np.cumprod(1 + rng.normal(0, 0.01, len(idx))),
        },
        index=idx,
    )
    tensor, names, dates, tickers = features.build_feature_tensor(
        prices, ["close", "pct_change_1d", "rsi_14", "volatility_20d"]
    )
    assert tensor.shape == (len(idx), 2, 4)
    assert names == ["close", "pct_change_1d", "rsi_14", "volatility_20d"]
    assert tickers == ["AAPL", "MSFT"]
    assert len(dates) == len(idx)
    # First close equals raw price; late RSI is finite.
    np.testing.assert_allclose(tensor[0, 0, 0], prices.iloc[0]["AAPL"], rtol=1e-5)
    assert np.isfinite(tensor[-1]).all()


def test_news_count_panel(price_panel):
    news_daily = pd.DataFrame(
        {
            "ticker": ["AAPL"],
            "date": [pd.Timestamp("2024-01-02")],
            "news_count": [3],
            "headlines": ["x"],
            "summaries": ["y"],
        }
    )
    panel = features.news_long_to_count_panel(news_daily, price_panel)
    assert panel.loc[pd.Timestamp("2024-01-02"), "AAPL"] == 3.0
    assert panel.loc[pd.Timestamp("2024-01-02"), "MSFT"] == 0.0
