"""Text features (P2 placeholder): news -> sentiment / embeddings.

Enabled in Phase 2. Kept as an interface so the quant side can receive the *same*
information as the LLM (news encoded numerically), which is required for a fair
text-vs-quant comparison rather than a news-vs-no-news comparison.
"""

from __future__ import annotations

import pandas as pd

from alphaduel.features.base import Feature, FeatureContext


class NewsSentiment(Feature):
    group = "text"

    @property
    def warmup(self) -> int:
        return 0

    def compute(self, prices: pd.DataFrame, ctx: FeatureContext) -> pd.DataFrame:
        raise NotImplementedError(
            "Text features are scheduled for Phase 2 (sentiment scores / embeddings from "
            "PIT-filtered GDELT headlines)."
        )
