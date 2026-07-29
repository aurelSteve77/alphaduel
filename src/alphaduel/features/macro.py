"""Macro features (FRED series), forward-filled onto the trading calendar (PIT-safe)."""

from __future__ import annotations

import pandas as pd

from alphaduel.features.base import Feature, FeatureContext


class MacroLevels(Feature):
    """Expose FRED series as z-scored levels, aligned & forward-filled to prices.

    The macro panel is passed via the store (see ``FeatureStore``); this feature simply
    normalizes it. Forward-fill only uses past releases, so no leakage.
    """

    group = "macro"

    def __init__(self, macro_panel: pd.DataFrame | None) -> None:
        self._macro = macro_panel

    @property
    def warmup(self) -> int:
        return 0

    def compute(self, prices: pd.DataFrame, ctx: FeatureContext) -> pd.DataFrame:
        if self._macro is None or self._macro.empty:
            return pd.DataFrame(index=prices.index)
        macro = self._macro.set_index("timestamp").sort_index()
        macro = macro.drop(columns=[c for c in ("available_at",) if c in macro.columns])
        aligned = macro.reindex(prices.index, method="ffill")
        z = (aligned - aligned.expanding().mean()) / aligned.expanding().std()
        return z.add_prefix("macro_")
