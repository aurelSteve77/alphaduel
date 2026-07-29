"""Portfolio accounting for the single-asset MVP (generalizes to multi-asset later).

Tracks cash and integer share holdings, applies fills net of transaction costs, and
computes mark-to-market equity. Rebalancing to a target weight produces an integer
share order (fractional part dropped, as required).
"""

from __future__ import annotations

from dataclasses import dataclass, field

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
