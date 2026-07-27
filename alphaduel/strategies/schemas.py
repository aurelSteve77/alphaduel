"""Schemas for LLM trading decisions (parsed from free-form text)."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from pydantic import BaseModel, Field, field_validator


class ShareOrder(BaseModel):
    """Buy/sell a whole number of shares of one ticker."""

    ticker: str = Field(description="Exact ticker symbol from the tradable universe")
    delta: int = Field(
        description="Shares to buy (positive), sell (negative), or 0 to hold"
    )

    @field_validator("ticker")
    @classmethod
    def _normalize_ticker(cls, value: str) -> str:
        cleaned = value.strip().upper()
        if not cleaned:
            raise ValueError("ticker must be non-empty")
        return cleaned


class TradingDecision(BaseModel):
    """Daily portfolio decision: free-form rationale + parsed share orders."""

    rationale: str = Field(
        default="",
        description="Full free-form model reply (including the JSON action block)",
    )
    orders: list[ShareOrder] = Field(
        default_factory=list,
        description="Share deltas parsed from the fenced JSON ``actions`` object",
    )

    def to_action(
        self,
        tickers: Sequence[str],
        *,
        max_shares: int,
    ) -> np.ndarray:
        """Project orders onto a dense ``(n_assets,)`` int32 action vector."""
        index = {str(t).upper(): i for i, t in enumerate(tickers)}
        action = np.zeros(len(tickers), dtype=np.int32)
        for order in self.orders:
            i = index.get(order.ticker)
            if i is None:
                continue
            action[i] = int(np.clip(order.delta, -max_shares, max_shares))
        return action
