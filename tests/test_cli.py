"""Tests for the alphaduel-download CLI entry point."""

from __future__ import annotations

import pandas as pd
import pytest

from alphaduel import cli


def test_market_only(monkeypatch):
    captured = {}

    def fake_ohlcv(tickers, start, end):
        captured["args"] = (tickers, start, end)
        return pd.DataFrame({"AAPL": [1.0]})

    monkeypatch.setattr("alphaduel.data.download.download_ohlcv", fake_ohlcv)
    rc = cli.download_main(["--tickers", "AAPL", "--start", "2024-01-01", "--end", "2024-02-01", "--market-only"])
    assert rc == 0
    assert captured["args"] == (["AAPL"], "2024-01-01", "2024-02-01")


def test_main_dispatches_download(monkeypatch):
    captured = {}

    def fake_news(tickers, start, end):
        captured["args"] = (tickers, start, end)
        return pd.DataFrame()

    monkeypatch.setattr("alphaduel.data.download_news.download_news", fake_news)
    rc = cli.main(["download", "--tickers", "AAPL", "--news-only"])
    assert rc == 0
    assert captured["args"][0] == ["AAPL"]


def test_main_requires_subcommand():
    with pytest.raises(SystemExit):
        cli.main([])


def test_news_only(monkeypatch):
    captured = {}

    def fake_news(tickers, start, end):
        captured["args"] = (tickers, start, end)
        return pd.DataFrame()

    monkeypatch.setattr("alphaduel.data.download_news.download_news", fake_news)
    rc = cli.download_main(["--tickers", "MSFT", "--news-only"])
    assert rc == 0
    assert captured["args"][0] == ["MSFT"]


def test_defaults_from_config(monkeypatch):
    captured = {}

    def fake_build(tickers, start, end, *, save, out_path):
        captured["tickers"] = tickers
        captured["start"] = start
        captured["save"] = save
        return pd.DataFrame(columns=["date", "ticker"])

    monkeypatch.setattr("alphaduel.data.dataset.build_dataset", fake_build)
    rc = cli.download_main([])
    assert rc == 0
    # Falls back to project.yaml values.
    from alphaduel.configuration import Configuration

    cfg = Configuration()
    assert captured["tickers"] == cfg.get("data.tickers")
    assert captured["start"] == cfg.get("data.start")


def test_merged_save_flag(monkeypatch):
    captured = {}

    def fake_build(tickers, start, end, *, save, out_path):
        captured["save"] = save
        captured["out_path"] = out_path
        return pd.DataFrame(columns=["date", "ticker"])

    monkeypatch.setattr("alphaduel.data.dataset.build_dataset", fake_build)
    cli.download_main(["--tickers", "AAPL", "--no-save"])
    assert captured["save"] is False

    cli.download_main(["--tickers", "AAPL", "--out", "x.parquet"])
    assert captured["save"] is True
    assert captured["out_path"] == "x.parquet"
