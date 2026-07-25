"""Build the unified market + news panel used by both study branches.

:func:`build_dataset` downloads the adjusted-close market panel and the
point-in-time news/filings, collapses the news to one row per ``(ticker, date)``
and left-joins it onto the price panel, yielding a single long DataFrame with one
row per traded ``(date, ticker)``.

Requires the ``data`` extra::

    uv sync --extra data
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from alphaduel.configuration import Configuration
from alphaduel.data.download import download_ohlcv
from alphaduel.data.download_news import download_news
from alphaduel.logger import get_logger

log = get_logger(__name__)

DATASET_COLUMNS = [
    "date",
    "ticker",
    "close",
    "news_count",
    "headlines",
    "summaries",
]

_MERGED_CACHE_KEY = "data.merged.cache_dir"
_DEFAULT_MERGED_CACHE = "data/processed"


def _market_to_long(prices: pd.DataFrame) -> pd.DataFrame:
    """Melt a ``[dates x tickers]`` close panel into long ``date, ticker, close``."""
    if prices.empty:
        return pd.DataFrame(columns=["date", "ticker", "close"])
    long = (
        prices.rename_axis("date")
        .reset_index()
        .melt(id_vars="date", var_name="ticker", value_name="close")
        .dropna(subset=["close"])
    )
    long["date"] = pd.to_datetime(long["date"]).dt.normalize().dt.tz_localize(None)
    return long


def _news_to_daily(news: pd.DataFrame) -> pd.DataFrame:
    """Collapse long news to one row per ``(ticker, date)`` with joined text."""
    cols = ["ticker", "date", "news_count", "headlines", "summaries"]
    if news.empty:
        return pd.DataFrame(columns=cols)

    n = news.copy()
    published = pd.to_datetime(n["published_at"], utc=True)
    n["date"] = published.dt.normalize().dt.tz_localize(None)

    def _join(series: pd.Series) -> str:
        return " || ".join(str(x) for x in series if str(x).strip())

    daily = (
        n.groupby(["ticker", "date"])
        .agg(
            news_count=("headline", "size"),
            headlines=("headline", _join),
            summaries=("summary", _join),
        )
        .reset_index()
    )
    return daily


def build_dataset(
    tickers: list[str],
    start: str,
    end: str,
    *,
    market_cache: str | Path | None = None,
    news_cache: str | Path | None = None,
    save: bool = False,
    out_path: str | Path | None = None,
) -> pd.DataFrame:
    """Download market + news for ``tickers`` and merge into one long DataFrame.

    Columns: ``date, ticker, close, news_count, headlines, summaries``. Days with
    no news get ``news_count=0`` and empty text. When ``save`` is set, the panel
    is written to ``out_path`` (or ``data.merged.cache_dir/dataset.parquet``).
    """
    prices = download_ohlcv(tickers, start, end, market_cache)
    news = download_news(tickers, start, end, news_cache)

    market_long = _market_to_long(prices)
    news_daily = _news_to_daily(news)

    merged = market_long.merge(news_daily, on=["date", "ticker"], how="left")
    merged["news_count"] = merged["news_count"].fillna(0).astype(int)
    merged["headlines"] = merged["headlines"].fillna("")
    merged["summaries"] = merged["summaries"].fillna("")
    merged = (
        merged[DATASET_COLUMNS]
        .sort_values(["date", "ticker"])
        .reset_index(drop=True)
    )

    log.info(
        "Built dataset: %d rows, %d tickers, %d rows with news",
        len(merged),
        merged["ticker"].nunique(),
        int((merged["news_count"] > 0).sum()),
    )

    if save:
        if out_path is None:
            cache_dir = Path(Configuration().get(_MERGED_CACHE_KEY, _DEFAULT_MERGED_CACHE))
            cache_dir.mkdir(parents=True, exist_ok=True)
            out_path = cache_dir / "dataset.parquet"
        else:
            out_path = Path(out_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
        merged.to_parquet(out_path, index=False)
        log.info("Wrote merged dataset to %s", out_path)

    return merged


def load_dataset(*, save: bool = False, out_path: str | Path | None = None) -> pd.DataFrame:
    """Build the merged dataset using the universe/date range from ``project.yaml``."""
    cfg = Configuration()
    tickers = cfg.get("data.tickers", [])
    start = cfg.get("data.start", "2000-01-01")
    end = cfg.get("data.end", datetime.now(tz=UTC).strftime("%Y-%m-%d"))
    return build_dataset(tickers, start, end, save=save, out_path=out_path)
