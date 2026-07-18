"""Market-data download with caching.

Primary source is yfinance (adjusted close); Stooq is used as a per-ticker
fallback. Each ticker is cached to ``data/raw/<TICKER>.parquet`` so repeated runs
are offline and reproducible. Requires the ``data`` extra:

    uv sync --extra data
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


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
    cache_dir: str | Path = "data/raw",
) -> pd.DataFrame:
    """Return a ``[dates x tickers]`` adjusted-close panel, using the cache."""
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)

    series: dict[str, pd.Series] = {}
    for ticker in tickers:
        fp = cache / f"{ticker}.parquet"
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


def load_prices(config, cache_dir: str | Path = "data/raw") -> pd.DataFrame:
    """Convenience wrapper that reads the universe/date range from a ``Config``."""
    return download_ohlcv(config.data.tickers, config.data.start, config.data.end, cache_dir)
