"""Tests for news/filings download, parsing and caching (no network)."""

from __future__ import annotations

import sys
import types

import pandas as pd

from alphaduel.data import download_news
from alphaduel.data.download_news import NEWS_COLUMNS, download_news as run_download_news


def _fake_requests(payload, capture=None):
    """Build a fake ``requests`` module returning ``payload`` from ``get``."""
    module = types.ModuleType("requests")

    class _Resp:
        def raise_for_status(self):
            return None

        def json(self):
            return payload

    def _get(url, params=None, headers=None, timeout=None):
        if capture is not None:
            capture["url"] = url
            capture["params"] = params
        return _Resp()

    module.get = _get
    return module


def test_finnhub_parsing(monkeypatch):
    payload = [
        {
            "datetime": 1704204000,  # 2024-01-02 14:00 UTC
            "source": "Yahoo",
            "category": "company news",
            "headline": "Apple does something",
            "summary": "details",
            "url": "http://example.com/x",
        }
    ]
    monkeypatch.setitem(sys.modules, "requests", _fake_requests(payload))
    monkeypatch.setenv("FINNHUB_API_KEY", "dummy")

    df = download_news._fetch_finnhub("AAPL", "2024-01-01", "2024-01-31")
    assert df is not None
    assert list(df.columns) == NEWS_COLUMNS
    assert df.iloc[0]["headline"] == "Apple does something"
    assert df.iloc[0]["ticker"] == "AAPL"


def test_finnhub_skipped_without_key(monkeypatch):
    monkeypatch.delenv("FINNHUB_API_KEY", raising=False)
    assert download_news._fetch_finnhub("AAPL", "2024-01-01", "2024-01-31") is None


def test_finnhub_empty_payload(monkeypatch):
    monkeypatch.setitem(sys.modules, "requests", _fake_requests([]))
    monkeypatch.setenv("FINNHUB_API_KEY", "dummy")
    assert download_news._fetch_finnhub("AAPL", "2024-01-01", "2024-01-31") is None


def test_download_news_writes_and_reads_cache(tmp_path, monkeypatch, news_panel):
    calls = {"n": 0}

    def fake_fetch_one(ticker, start, end):
        calls["n"] += 1
        return news_panel[news_panel["ticker"] == ticker]

    monkeypatch.setattr(download_news, "_fetch_one", fake_fetch_one)

    first = run_download_news(["AAPL"], "2024-01-01", "2024-01-31", cache_dir=tmp_path)
    assert (tmp_path / "AAPL_news.parquet").exists()
    assert len(first) == 2

    # Second call must hit the cache, not fetch again.
    second = run_download_news(["AAPL"], "2024-01-01", "2024-01-31", cache_dir=tmp_path)
    assert len(second) == 2
    assert calls["n"] == 1


def test_download_news_empty_returns_empty_frame(tmp_path, monkeypatch):
    monkeypatch.setattr(download_news, "_fetch_one", lambda t, s, e: None)
    out = run_download_news(["ZZZZ"], "2024-01-01", "2024-01-31", cache_dir=tmp_path)
    assert out.empty
    assert list(out.columns) == NEWS_COLUMNS


def test_download_news_sorted_by_published_at(tmp_path, monkeypatch):
    rows = pd.DataFrame(
        [
            {
                "ticker": "AAPL",
                "published_at": pd.Timestamp("2024-01-05", tz="UTC"),
                "source": "s",
                "category": "news",
                "headline": "late",
                "summary": "",
                "url": "",
            },
            {
                "ticker": "AAPL",
                "published_at": pd.Timestamp("2024-01-01", tz="UTC"),
                "source": "s",
                "category": "news",
                "headline": "early",
                "summary": "",
                "url": "",
            },
        ]
    )
    monkeypatch.setattr(download_news, "_fetch_one", lambda t, s, e: rows)
    out = run_download_news(["AAPL"], "2024-01-01", "2024-01-31", cache_dir=tmp_path)
    assert out.iloc[0]["headline"] == "early"
