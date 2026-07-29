"""News headlines via GDELT (P2 placeholder).

GDELT is free and timestamped but noisy. Enabled in Phase 2. Kept as an interface so
the text-feature path and leakage controls (only ``published_at <= decision_time``)
can be designed now.
"""

from __future__ import annotations

from datetime import date

import pandas as pd

from alphaduel.data.base import DataSource
from alphaduel.data.storage import ParquetCache


class GdeltNewsSource(DataSource):
    name = "news_gdelt"

    def __init__(self, cache: ParquetCache) -> None:
        self.cache = cache

    def fetch(self, symbols: list[str], start: date, end: date) -> pd.DataFrame:
        raise NotImplementedError(
            "GDELT news source is scheduled for Phase 2. Returns headlines with a UTC "
            "`published_at` used as `available_at` for PIT filtering."
        )
