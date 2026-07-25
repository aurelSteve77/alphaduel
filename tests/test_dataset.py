"""Tests for the merged market + news dataset builder."""

from __future__ import annotations

import pandas as pd

from alphaduel.data import dataset


def test_market_to_long(price_panel):
    long = dataset._market_to_long(price_panel)
    assert set(long.columns) == {"date", "ticker", "close"}
    assert len(long) == 6  # 3 dates x 2 tickers
    assert long["date"].dt.tz is None


def test_news_to_daily_aggregates(news_panel):
    daily = dataset._news_to_daily(news_panel)
    assert len(daily) == 1  # both items are same ticker + day
    row = daily.iloc[0]
    assert row["news_count"] == 2
    assert "Apple ships thing" in row["headlines"]
    assert "||" in row["headlines"]


def test_build_dataset_merges(monkeypatch, price_panel, news_panel):
    monkeypatch.setattr(dataset, "download_ohlcv", lambda t, s, e, c: price_panel)
    monkeypatch.setattr(dataset, "download_news", lambda t, s, e, c: news_panel)

    df = dataset.build_dataset(["AAPL", "MSFT"], "2024-01-01", "2024-01-31")
    assert list(df.columns) == dataset.DATASET_COLUMNS
    assert len(df) == 6

    aapl_jan2 = df[(df["ticker"] == "AAPL") & (df["date"] == pd.Timestamp("2024-01-02"))]
    assert aapl_jan2.iloc[0]["news_count"] == 2

    # A day/ticker without news is filled with zeros/empty text.
    msft_jan2 = df[(df["ticker"] == "MSFT") & (df["date"] == pd.Timestamp("2024-01-02"))]
    assert msft_jan2.iloc[0]["news_count"] == 0
    assert msft_jan2.iloc[0]["headlines"] == ""


def test_build_dataset_saves(tmp_path, monkeypatch, price_panel, news_panel):
    monkeypatch.setattr(dataset, "download_ohlcv", lambda t, s, e, c: price_panel)
    monkeypatch.setattr(dataset, "download_news", lambda t, s, e, c: news_panel)

    out = tmp_path / "merged.parquet"
    dataset.build_dataset(["AAPL"], "2024-01-01", "2024-01-31", save=True, out_path=out)
    assert out.exists()
    reloaded = pd.read_parquet(out)
    assert list(reloaded.columns) == dataset.DATASET_COLUMNS
