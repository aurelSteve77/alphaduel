"""Textual data (news / filings) download with caching.

Mirrors :mod:`alphaduel.data.download`: a primary source (Finnhub company news)
with a fallback (SEC EDGAR filings), each ticker cached to
``data/raw_news/<TICKER>.parquet`` so repeated runs are offline and reproducible.

Every item carries a UTC ``published_at`` timestamp so the study can enforce its
no-lookahead rule: only expose items with ``published_at`` at or before the
decision time. Requires the ``data`` extra::

    uv sync --extra data
"""

from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from alphaduel.configuration import Configuration
from alphaduel.logger import get_logger

log = get_logger(__name__)

NEWS_COLUMNS = ["ticker", "published_at", "source", "category", "headline", "summary", "url"]

_SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
_SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
_SEC_ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{document}"


def _empty_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=NEWS_COLUMNS)


def _fetch_finnhub(ticker: str, start: str, end: str) -> pd.DataFrame | None:
    """Fetch company news from Finnhub; needs an API key in the environment."""
    cfg = Configuration()
    api_key_env = cfg.get("data.news.finnhub.api_key_env", "FINNHUB_API_KEY")
    token = os.environ.get(api_key_env)
    if not token:
        log.debug("Finnhub skipped for %s: %s not set", ticker, api_key_env)
        return None

    try:
        import requests

        resp = requests.get(
            "https://finnhub.io/api/v1/company-news",
            params={"symbol": ticker, "from": start, "to": end, "token": token},
            timeout=30,
        )
        resp.raise_for_status()
        payload = resp.json()
        if not payload:
            return None

        rows = [
            {
                "ticker": ticker,
                "published_at": pd.to_datetime(item.get("datetime", 0), unit="s", utc=True),
                "source": str(item.get("source", "finnhub")),
                "category": str(item.get("category", "news")),
                "headline": str(item.get("headline", "")),
                "summary": str(item.get("summary", "")),
                "url": str(item.get("url", "")),
            }
            for item in payload
        ]
        log.info("Finnhub returned %d items for %s", len(rows), ticker)
        return pd.DataFrame(rows, columns=NEWS_COLUMNS)
    except Exception:  # noqa: BLE001 - fall through to the next source
        log.warning("Finnhub fetch failed for %s", ticker, exc_info=True)
        return None


def _sec_ticker_to_cik(ticker: str, user_agent: str) -> int | None:
    """Resolve a ticker to its zero-padded SEC CIK via the public mapping."""
    import requests

    resp = requests.get(_SEC_TICKERS_URL, headers={"User-Agent": user_agent}, timeout=30)
    resp.raise_for_status()
    for entry in resp.json().values():
        if str(entry.get("ticker", "")).upper() == ticker.upper():
            return int(entry["cik_str"])
    return None


def _fetch_edgar(ticker: str, start: str, end: str) -> pd.DataFrame | None:
    """Fetch recent SEC filings (dated events) as a text fallback."""
    cfg = Configuration()
    user_agent = cfg.get("data.news.edgar.user_agent", "alphaduel research")
    forms = set(cfg.get("data.news.edgar.forms", ["8-K", "10-Q", "10-K"]))
    start_ts = pd.Timestamp(start, tz="UTC")
    end_ts = pd.Timestamp(end, tz="UTC")

    try:
        import requests

        cik = _sec_ticker_to_cik(ticker, user_agent)
        if cik is None:
            log.debug("EDGAR: no CIK for %s", ticker)
            return None

        resp = requests.get(
            _SEC_SUBMISSIONS_URL.format(cik=cik),
            headers={"User-Agent": user_agent},
            timeout=30,
        )
        resp.raise_for_status()
        recent = resp.json().get("filings", {}).get("recent", {})

        rows = []
        for form, filing_date, accession, document, description in zip(
            recent.get("form", []),
            recent.get("filingDate", []),
            recent.get("accessionNumber", []),
            recent.get("primaryDocument", []),
            recent.get("primaryDocDescription", []),
            strict=False,
        ):
            if forms and form not in forms:
                continue
            published_at = pd.Timestamp(filing_date, tz="UTC")
            if not (start_ts <= published_at <= end_ts):
                continue
            accession_nodash = accession.replace("-", "")
            url = _SEC_ARCHIVE_URL.format(
                cik=cik, accession=accession_nodash, document=document
            )
            rows.append(
                {
                    "ticker": ticker,
                    "published_at": published_at,
                    "source": "sec-edgar",
                    "category": form,
                    "headline": f"{form} filing" + (f": {description}" if description else ""),
                    "summary": str(description or ""),
                    "url": url,
                }
            )

        if not rows:
            return None
        log.info("EDGAR returned %d filings for %s", len(rows), ticker)
        return pd.DataFrame(rows, columns=NEWS_COLUMNS)
    except Exception:  # noqa: BLE001 - no source succeeded
        log.warning("EDGAR fetch failed for %s", ticker, exc_info=True)
        return None


def _fetch_one(ticker: str, start: str, end: str) -> pd.DataFrame | None:
    """Fetch textual data for a ticker; try Finnhub then SEC EDGAR."""
    news = _fetch_finnhub(ticker, start, end)
    if news is not None and len(news):
        return news
    return _fetch_edgar(ticker, start, end)


def download_news(
    tickers: list[str],
    start: str,
    end: str,
    cache_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Return a long-format news panel for ``tickers`` within ``[start, end]``.

    Columns: ``ticker, published_at, source, category, headline, summary, url``.
    Sorted by ``published_at``. Uses a per-ticker parquet cache.
    """
    cfg = Configuration()
    cache = Path(cache_dir or cfg.get("data.news.cache_dir", "data/raw/news"))
    cache.mkdir(parents=True, exist_ok=True)

    frames: list[pd.DataFrame] = []
    for ticker in tickers:
        fp = cache / f"{ticker}_news.parquet"
        if fp.exists():
            frames.append(pd.read_parquet(fp))
            continue
        fetched = _fetch_one(ticker, start, end)
        if fetched is not None and len(fetched):
            fetched.to_parquet(fp, index=False)
            frames.append(fetched)

    if not frames:
        log.warning("No textual data could be downloaded for %s", tickers)
        return _empty_frame()

    panel = pd.concat(frames, ignore_index=True)
    panel = panel.sort_values("published_at").reset_index(drop=True)
    return panel


def load_news(cache_dir: str | Path | None = None) -> pd.DataFrame:
    """Convenience wrapper reading universe/date range from ``project.yaml``."""
    cfg = Configuration()
    tickers = cfg.get("data.tickers", [])
    start = cfg.get("data.start", "2000-01-01")
    end = cfg.get("data.end", datetime.now(tz=UTC).strftime("%Y-%m-%d"))
    return download_news(tickers, start, end, cache_dir)