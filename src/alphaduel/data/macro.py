"""Macro data via FRED (Federal Reserve Economic Data).

FRED is reliable and free. We stamp ``available_at`` with the series' release logic:
for the P0 scaffold we approximate availability by the observation date (many daily
market series like VIX/rates are same-day). For releases with a lag (CPI), a real PIT
build should use FRED ALFRED vintages; noted as a P1 refinement.
"""

from __future__ import annotations

from datetime import date

import pandas as pd

from alphaduel.data.base import DataSource
from alphaduel.data.storage import ParquetCache


class FredMacroSource(DataSource):
    name = "macro_fred"

    def __init__(self, cache: ParquetCache, api_key: str | None, series: list[str]) -> None:
        self.cache = cache
        self.api_key = api_key
        self.series = series

    def fetch(self, symbols: list[str], start: date, end: date) -> pd.DataFrame:
        # ``symbols`` is unused for macro; the configured FRED series ids drive the fetch.
        params = {"series": sorted(self.series), "start": start, "end": end}
        cached = self.cache.get(self.name, params)
        if cached is not None:
            return cached

        if not self.api_key:
            raise RuntimeError("FRED_API_KEY is not set; add it to .env to fetch macro data.")

        from fredapi import Fred  # lazy import

        fred = Fred(api_key=self.api_key)
        cols = {}
        for sid in self.series:
            s = fred.get_series(sid, observation_start=str(start), observation_end=str(end))
            cols[sid] = s
        wide = pd.DataFrame(cols)
        wide.index = pd.to_datetime(wide.index, utc=True)
        wide.index.name = "timestamp"
        wide = wide.reset_index()
        wide["available_at"] = wide["timestamp"]  # P1: replace with ALFRED vintage dates
        self.cache.put(self.name, params, wide)
        return wide
