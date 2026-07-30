"""Generative book-construction toolkit (P5 / GenPortfolio). See SPEC §15.

Exposes the domain tokenizer and mandate-constraint utilities. The transformer model is
NOT imported here so this package stays importable without torch (the agent lazy-imports
``model`` only when it needs it).
"""

from alphaduel.generative.constraints import MandateConstraints
from alphaduel.generative.tokenizer import (
    SPECIAL_TOKENS,
    PortfolioTokenizer,
    QuantBucketizer,
    SecurityVocab,
    WeightBuckets,
)

__all__ = [
    "SPECIAL_TOKENS",
    "MandateConstraints",
    "PortfolioTokenizer",
    "QuantBucketizer",
    "SecurityVocab",
    "WeightBuckets",
]
