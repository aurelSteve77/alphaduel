"""Tests for market-data download and caching (no network)."""

from __future__ import annotations

import pandas as pd
import pytest

from alphaduel.data import download


def test_cache_hit_reads_parquet(tmp_path, price_panel):
    # Pre-seed the per-ticker cache the way download_ohlcv writes it.
    for ticker in ("AAPL", "MSFT"):
        price_panel[ticker].to_frame("close").to_parquet(tmp_path / f"{ticker}_market.parquet")

    out = download.download_ohlcv(["AAPL", "MSFT"], "2024-01-01", "2024-01-31", cache_dir=tmp_path)
    assert list(out.columns) == ["AAPL", "MSFT"]
    assert out.loc["2024-01-02", "AAPL"] == 100.0


def test_fetch_writes_cache(tmp_path, monkeypatch):
    series = pd.Series(
        [10.0, 11.0],
        index=pd.to_datetime(["2024-01-02", "2024-01-03"]),
        name="AAPL",
    )
    monkeypatch.setattr(download, "_fetch_one", lambda t, s, e: series)

    out = download.download_ohlcv(["AAPL"], "2024-01-01", "2024-01-31", cache_dir=tmp_path)
    assert (tmp_path / "AAPL_market.parquet").exists()
    assert out["AAPL"].tolist() == [10.0, 11.0]


def test_no_data_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(download, "_fetch_one", lambda t, s, e: None)
    with pytest.raises(RuntimeError):
        download.download_ohlcv(["ZZZZ"], "2024-01-01", "2024-01-31", cache_dir=tmp_path)


def test_cache_dir_defaults_to_config(monkeypatch):
    captured = {}

    def fake_mkdir(self, *a, **k):
        captured["path"] = str(self)

    monkeypatch.setattr(download.Path, "mkdir", fake_mkdir)
    monkeypatch.setattr(download, "_fetch_one", lambda t, s, e: None)
    # Use a ticker that can never have a cached parquet so the default path is
    # exercised regardless of any real cache on disk.
    with pytest.raises(RuntimeError):
        download.download_ohlcv(["__NO_SUCH_TICKER__"], "2024-01-01", "2024-01-31")
    assert captured["path"] == "data/raw/market"
