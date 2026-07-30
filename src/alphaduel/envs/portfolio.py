"""Portfolio accounting for the single-asset MVP (generalizes to multi-asset later).

Tracks cash and integer share holdings, applies fills net of transaction costs, and
computes mark-to-market equity. Rebalancing to a target weight produces an integer
share order (fractional part dropped, as required).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from alphaduel.envs.costs import TransactionCostModel


@dataclass
class Fill:
    shares: int
    price: float
    cost: float


@dataclass
class Portfolio:
    cash: float
    costs: TransactionCostModel
    allow_short: bool = False
    shares: int = 0
    fills: list[Fill] = field(default_factory=list)

    def equity(self, price: float) -> float:
        return self.cash + self.shares * price

    def target_weight_to_order(self, target_w: float, price: float) -> int:
        """Integer share delta to move current asset weight toward ``target_w``.

        ``target_w`` is the desired fraction of *current equity* held in the asset.
        Only the integer part of the resulting position is taken.
        """
        eq = self.equity(price)
        target_value = max(0.0, target_w) * eq
        target_shares = int(target_value // price)  # floor -> integer shares only
        if not self.allow_short:
            target_shares = max(0, target_shares)
        return target_shares - self.shares

    def execute(self, shares_delta: int, price: float, adv: float | None = None) -> Fill:
        """Buy (>0) or sell (<0) ``shares_delta`` at ``price``, charging costs to cash."""
        if shares_delta == 0:
            fill = Fill(0, price, 0.0)
            self.fills.append(fill)
            return fill

        cost = self.costs.cost(shares_delta, price, adv)
        notional = shares_delta * price
        # Buying reduces cash by notional+cost; selling increases cash by notional-cost.
        self.cash -= notional
        self.cash -= cost
        self.shares += shares_delta
        fill = Fill(shares_delta, price, cost)
        self.fills.append(fill)
        return fill


@dataclass
class MultiAssetPortfolio:
    """Vector generalization of ``Portfolio`` for N assets (used by GenPortfolio / P5).

    Holds a cash balance and an integer share vector. Rebalancing to a target weight
    vector produces per-asset integer share orders (fractional parts dropped).
    """

    cash: float
    costs: TransactionCostModel
    n_assets: int
    allow_short: bool = False
    shares: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))

    def __post_init__(self) -> None:
        if self.shares.size == 0:
            self.shares = np.zeros(self.n_assets, dtype=np.int64)

    def equity(self, prices: np.ndarray) -> float:
        return float(self.cash + np.dot(self.shares, prices))

    def target_weights_to_orders(
        self, target_w: np.ndarray, prices: np.ndarray
    ) -> np.ndarray:
        """Integer share deltas moving each asset weight toward ``target_w``.

        ``target_w`` are fractions of *current equity*; only integer share counts are held.
        """
        eq = self.equity(prices)
        target_values = np.clip(target_w, 0.0, None) * eq
        target_shares = np.floor_divide(target_values, prices).astype(np.int64)
        if not self.allow_short:
            target_shares = np.maximum(0, target_shares)
        return target_shares - self.shares

    def execute(self, deltas: np.ndarray, prices: np.ndarray, adv: np.ndarray | None = None):
        """Apply per-asset integer share ``deltas`` at ``prices``, charging costs to cash."""
        total_cost = 0.0
        for i, delta in enumerate(deltas):
            if delta == 0:
                continue
            adv_i = float(adv[i]) if adv is not None else None
            cost = self.costs.cost(int(delta), float(prices[i]), adv_i)
            self.cash -= delta * prices[i]
            self.cash -= cost
            total_cost += cost
        self.shares = self.shares + deltas.astype(np.int64)
        return total_cost
