"""Price data via yfinance.

LIMITATIONS (documented on purpose):
- yfinance history is split/dividend adjusted retroactively, which introduces a subtle
  look-ahead: past adjusted prices reflect future corporate actions. For rigorous PIT
  work, prefer a point-in-time provider (Polygon/Tiingo) via a future adapter. For the
  P0 MVP we accept this and cache aggressively.
- The endpoint is unofficial and occasionally rate-limits or returns gaps.
"""

from __future__ import annotations

from datetime import date

import pandas as pd

from alphaduel.data.base import DataSource
from alphaduel.data.storage import ParquetCache

_OHLCV = ["open", "high", "low", "close", "volume"]


class YahooPriceSource(DataSource):
    name = "prices_yfinance"

    def __init__(self, cache: ParquetCache) -> None:
        self.cache = cache

    def fetch(self, symbols: list[str], start: date, end: date) -> pd.DataFrame:
        params = {"symbols": sorted(symbols), "start": start, "end": end, "interval": "1d"}
        cached = self.cache.get(self.name, params)
        if cached is not None:
            return cached

        import yfinance as yf  # imported lazily so the pkg installs without network

        frames: list[pd.DataFrame] = []
        for sym in symbols:
            raw = yf.download(
                sym, start=str(start), end=str(end), interval="1d",
                auto_adjust=True, progress=False,
            )
            if raw.empty:
                continue
            df = raw.rename(columns=str.lower)[_OHLCV].copy()
            df.index = pd.to_datetime(df.index, utc=True)
            df.index.name = "timestamp"
            df["symbol"] = sym
            # A daily bar becomes actionable at the *next* session's open; we stamp
            # availability at the bar close and rely on env execution_lag to trade later.
            df["available_at"] = df.index
            frames.append(df.reset_index())

        result = (
            pd.concat(frames, ignore_index=True)
            if frames
            else pd.DataFrame(columns=["timestamp", *_OHLCV, "symbol", "available_at"])
        )
        self.cache.put(self.name, params, result)
        return result
