"""Framework-agnostic long-only (optionally short) portfolio ledger.

Keeps cash, share holdings and fee accounting separate from the Gymnasium API
so the same simulator can be reused by baselines and evaluation harnesses.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class TradeResult:
    """Outcome of applying one multi-asset order vector."""

    executed: np.ndarray
    fees_paid: float
    cash_after: float
    holdings_after: np.ndarray


@dataclass
class Portfolio:
    """Cash + share holdings with proportional transaction costs."""

    n_assets: int
    cash: float
    fee_rate: float = 0.001
    allow_short: bool = False
    holdings: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        if self.n_assets < 1:
            raise ValueError("n_assets must be >= 1")
        if self.cash < 0:
            raise ValueError("cash must be non-negative at init")
        if self.fee_rate < 0:
            raise ValueError("fee_rate must be non-negative")
        self.holdings = np.zeros(self.n_assets, dtype=np.float64)

    def reset(self, cash: float) -> None:
        self.cash = float(cash)
        self.holdings[:] = 0.0

    def market_value(self, prices: np.ndarray) -> float:
        prices = np.asarray(prices, dtype=np.float64)
        return float(self.cash + np.dot(self.holdings, prices))

    def weights(self, prices: np.ndarray) -> np.ndarray:
        prices = np.asarray(prices, dtype=np.float64)
        value = self.market_value(prices)
        if value <= 0:
            return np.zeros(self.n_assets, dtype=np.float64)
        return (self.holdings * prices) / value

    def execute(self, orders: np.ndarray, prices: np.ndarray) -> TradeResult:
        """Buy/sell integer share deltas at ``prices``, clipped to feasibility.

        Positive orders buy, negative orders sell. Buys are limited by remaining
        cash (after fees); sells are limited by holdings unless ``allow_short``.
        """
        orders = np.asarray(orders, dtype=np.int64).reshape(-1)
        prices = np.asarray(prices, dtype=np.float64).reshape(-1)
        if orders.shape != (self.n_assets,) or prices.shape != (self.n_assets,):
            raise ValueError("orders and prices must both have shape (n_assets,)")

        executed = np.zeros(self.n_assets, dtype=np.int64)
        fees_paid = 0.0

        # Sells first — free cash before buys.
        for i, qty in enumerate(orders):
            if qty >= 0:
                continue
            sell_qty = -int(qty)
            if not self.allow_short:
                sell_qty = min(sell_qty, int(self.holdings[i]))
            if sell_qty <= 0 or not np.isfinite(prices[i]) or prices[i] <= 0:
                continue
            notional = sell_qty * prices[i]
            fee = notional * self.fee_rate
            self.holdings[i] -= sell_qty
            self.cash += notional - fee
            executed[i] = -sell_qty
            fees_paid += fee

        # Buys — spend remaining cash, greedy in ticker order.
        for i, qty in enumerate(orders):
            if qty <= 0:
                continue
            buy_qty = int(qty)
            if not np.isfinite(prices[i]) or prices[i] <= 0:
                continue
            # Each share costs price * (1 + fee_rate).
            unit_cost = prices[i] * (1.0 + self.fee_rate)
            if unit_cost <= 0:
                continue
            affordable = int(self.cash // unit_cost)
            buy_qty = min(buy_qty, affordable)
            if buy_qty <= 0:
                continue
            notional = buy_qty * prices[i]
            fee = notional * self.fee_rate
            self.holdings[i] += buy_qty
            self.cash -= notional + fee
            executed[i] = buy_qty
            fees_paid += fee

        # Numerical guard against tiny negative cash from float noise.
        if self.cash < 0 and self.cash > -1e-8:
            self.cash = 0.0

        return TradeResult(
            executed=executed,
            fees_paid=fees_paid,
            cash_after=float(self.cash),
            holdings_after=self.holdings.copy(),
        )
