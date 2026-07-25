"""Shared pytest fixtures for the alphaduel test suite."""

from __future__ import annotations

import pandas as pd
import pytest


@pytest.fixture
def price_panel() -> pd.DataFrame:
    """A small ``[dates x tickers]`` adjusted-close panel."""
    idx = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
    return pd.DataFrame(
        {"AAPL": [100.0, 101.0, 102.0], "MSFT": [200.0, 201.0, 202.0]},
        index=idx,
    )


@pytest.fixture
def news_panel() -> pd.DataFrame:
    """A small long-format news panel matching :data:`NEWS_COLUMNS`."""
    return pd.DataFrame(
        [
            {
                "ticker": "AAPL",
                "published_at": pd.Timestamp("2024-01-02 14:00", tz="UTC"),
                "source": "Yahoo",
                "category": "news",
                "headline": "Apple ships thing",
                "summary": "A summary",
                "url": "http://example.com/1",
            },
            {
                "ticker": "AAPL",
                "published_at": pd.Timestamp("2024-01-02 20:00", tz="UTC"),
                "source": "CNBC",
                "category": "news",
                "headline": "Apple ships another thing",
                "summary": "",
                "url": "http://example.com/2",
            },
        ]
    )
