"""Tests for observation → prompt payload formatting."""

from __future__ import annotations

import numpy as np

from alphaduel.strategies.formatting import build_prompt_payload, format_features


def test_format_features_rounds():
    feats = format_features(
        ["AAPL"],
        ["close", "rsi_14"],
        np.array([[123.45678, 55.12345]]),
        precision=2,
    )
    assert feats["AAPL"] == {"close": 123.46, "rsi_14": 55.12}


def test_build_prompt_payload():
    obs = {
        "cash": np.array([1000.0], dtype=np.float32),
        "holdings": np.array([2.0, 0.0], dtype=np.float32),
        "portfolio_value": np.array([1200.0], dtype=np.float32),
        "features": np.array([[100.0, 0.01], [200.0, -0.02]], dtype=np.float32),
    }
    info = {
        "date": "2024-01-02",
        "tickers": ["AAPL", "MSFT"],
        "feature_names": ["close", "pct_change_1d"],
        "step": 3,
        "news": {"AAPL": {"news_count": 1, "headlines": "Hello"}},
    }
    payload = build_prompt_payload(
        obs,
        info,
        tickers=["AAPL", "MSFT"],
        feature_names=["close", "pct_change_1d"],
        max_shares=5,
        allow_short=False,
    )
    assert payload["date"] == "2024-01-02"
    assert payload["holdings"]["AAPL"] == 2.0
    assert payload["features"]["MSFT"]["close"] == 200.0
    assert payload["news"]["AAPL"].startswith("[1]")
    assert payload["max_shares"] == 5
