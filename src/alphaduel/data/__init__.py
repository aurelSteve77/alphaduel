"""Data layer: PIT-aware sources (prices/macro/news) with a Parquet cache."""

from alphaduel.data.base import DataSource
from alphaduel.data.macro import FredMacroSource
from alphaduel.data.prices import YahooPriceSource
from alphaduel.data.storage import ParquetCache

__all__ = ["DataSource", "FredMacroSource", "ParquetCache", "YahooPriceSource"]
