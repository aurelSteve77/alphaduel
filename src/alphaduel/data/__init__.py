"""Data pipeline (to be implemented in milestone M1).

Planned modules:
- ``download``  : fetch OHLCV (yfinance primary, Stooq fallback) + SPY + VIX.
- ``features``  : numeric features computed strictly <= t, cross-sectional z-score.
- ``verbalize`` : deterministic text rendering of a state for the LLM branch.
- ``labels``    : shifted forward returns for the supervised quant baseline.
- ``splits``    : chronological train/val/test with no-lookahead assertions.
"""

from __future__ import annotations

__all__: list[str] = []
