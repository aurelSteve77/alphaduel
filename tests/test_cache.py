"""Tests for ParquetCache listing / clearing helpers used by the Data dashboard page."""

from __future__ import annotations

import pandas as pd

from alphaduel.data.storage import ParquetCache


def test_cache_list_delete_and_clear(tmp_path):
    cache = ParquetCache(tmp_path)
    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2020-01-02", "2020-01-03"], utc=True),
            "open": [1.0, 1.1],
            "close": [1.05, 1.15],
            "symbol": ["AAPL", "AAPL"],
        }
    )
    params = {"symbols": ["AAPL"], "start": "2020-01-01", "end": "2020-02-01"}
    cache.put("prices_yfinance", params, df)

    entries = cache.list_entries()
    assert len(entries) == 1
    assert entries[0]["namespace"] == "prices_yfinance"
    assert entries[0]["symbols"] == ["AAPL"]
    assert entries[0]["rows"] == 2

    assert cache.delete("prices_yfinance", params) is True
    assert cache.list_entries() == []

    cache.put("prices_yfinance", params, df)
    cache.put("macro_fred", {"series": ["VIXCLS"], "start": "2020-01-01", "end": "2020-02-01"}, df)
    assert cache.clear_namespace("prices_yfinance") == 1
    assert cache.clear_all() == 1
    assert cache.list_entries() == []
