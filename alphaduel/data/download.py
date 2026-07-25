"""Market-data download with caching.

Primary source is yfinance (adjusted close); Stooq is used as a per-ticker
fallback. Each ticker is cached to ``data/raw/<TICKER>.parquet`` so repeated runs
are offline and reproducible. Requires the ``data`` extra:

"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from alphaduel.configuration import Configuration
from alphaduel.logger import get_logger

log = get_logger(__name__)

_MARKET_CACHE_KEY = "data.market.cache_dir"
_DEFAULT_MARKET_CACHE = "data/raw/market"


def _resolve_cache_dir(cache_dir: str | Path | None) -> Path:
    """Use the explicit ``cache_dir`` if given, else read it from project.yaml."""
    if cache_dir is None:
        cache_dir = Configuration().get(_MARKET_CACHE_KEY, _DEFAULT_MARKET_CACHE)
    return Path(cache_dir)


def _fetch_one(ticker: str, start: str, end: str) -> pd.Series | None:
    """Fetch a single ticker's adjusted close; try yfinance then Stooq."""
    try:
        import yfinance as yf

        df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
        if df is not None and len(df):
            close = df["Close"]
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
            return close.rename(ticker)
    except Exception:  # noqa: BLE001 - fall through to the next source
        pass

    try:
        from pandas_datareader import data as pdr

        df = pdr.DataReader(ticker, "stooq", start, end).sort_index()
        if df is not None and len(df):
            return df["Close"].rename(ticker)
    except Exception:  # noqa: BLE001 - no source succeeded
        pass

    return None


def download_ohlcv(
    tickers: list[str],
    start: str,
    end: str,
    cache_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Return a ``[dates x tickers]`` adjusted-close panel, using the cache.

    ``cache_dir`` defaults to ``data.market.cache_dir`` from ``project.yaml``.
    """
    cache = _resolve_cache_dir(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)

    series: dict[str, pd.Series] = {}
    for ticker in tickers:
        fp = cache / f"{ticker}_market.parquet"
        if fp.exists():
            series[ticker] = pd.read_parquet(fp)["close"]
            continue
        fetched = _fetch_one(ticker, start, end)
        if fetched is not None:
            fetched.to_frame("close").to_parquet(fp)
            series[ticker] = fetched

    if not series:
        raise RuntimeError(
            "No market data could be downloaded. Install the data extra "
            "(`uv sync --extra data`) and check your network connection."
        )
    return pd.DataFrame(series).sort_index()


def load_prices(cache_dir: str | Path | None = None) -> pd.DataFrame:
    """Convenience wrapper reading universe/date range from ``project.yaml``."""
    cfg = Configuration()
    tickers = cfg.get("data.tickers", [])
    start = cfg.get("data.start", "2000-01-01")
    end = cfg.get("data.end", datetime.now(tz=UTC).strftime("%Y-%m-%d"))
    return download_ohlcv(tickers, start, end, cache_dir)