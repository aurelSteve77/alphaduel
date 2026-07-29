"""Transaction cost model: commission + spread + slippage/impact.

Composed so cost-sensitivity sweeps are a first-class experiment. All terms are charged
on the *executed* notional, and impact scales with the traded fraction of ADV.
"""

from __future__ import annotations

import math

from alphaduel.config.schema import CostConfig


class TransactionCostModel:
    def __init__(self, config: CostConfig) -> None:
        self.c = config

    def cost(self, shares: float, price: float, adv: float | None = None) -> float:
        """Total cost (in cash) of trading ``abs(shares)`` at ``price``.

        Parameters
        ----------
        shares: signed number of shares traded (sign ignored for cost).
        price:  execution price per share.
        adv:    average daily volume, used for the impact term (optional).
        """
        qty = abs(shares)
        if qty == 0.0:
            return 0.0
        notional = qty * price

        commission = (
            self.c.commission_flat
            + self.c.commission_per_share * qty
            + self.c.commission_bps * 1e-4 * notional
        )
        spread = self.c.half_spread_bps * 1e-4 * notional

        impact = 0.0
        if adv and adv > 0:
            frac = qty / adv
            if self.c.use_sqrt_impact:
                frac = math.sqrt(frac)
            impact = self.c.slippage_impact_coef * frac * notional

        return commission + spread + impact
