"""Common interface for all data sources.

Every source returns a tidy, timezone-aware (UTC) DataFrame indexed by timestamp,
and declares an ``available_at`` column so downstream consumers can enforce
point-in-time (PIT) correctness (no datum used before it was knowable).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date

import pandas as pd


class DataSource(ABC):
    """Abstract data source with an idempotent, cacheable ``fetch``."""

    #: Stable identifier used as the cache namespace.
    name: str = "source"

    @abstractmethod
    def fetch(self, symbols: list[str], start: date, end: date) -> pd.DataFrame:
        """Return raw data for ``symbols`` between ``start`` and ``end`` (inclusive).

        The returned frame MUST include an ``available_at`` (UTC) column marking when
        each row first became knowable. For daily bars this is typically the bar's
        close/next-open; for macro releases it is the publication timestamp.
        """

    @staticmethod
    def _ensure_available_at(df: pd.DataFrame, ts_col: str) -> pd.DataFrame:
        """Default PIT stamp: a row is available at its own timestamp unless overridden."""
        if "available_at" not in df.columns:
            df = df.copy()
            df["available_at"] = pd.to_datetime(df[ts_col], utc=True)
        return df
