"""LLM agents (P2): off-the-shelf vanilla strategy with injected chat models.

Build the model with :class:`LLMHandler` / :func:`create_llm`, then pass it to
:class:`VanillaLLMAgent`. Renders the env observation to a textual market brief,
asks the LLM for free-form reasoning plus a fenced JSON share-delta action, and
stores the rationale in ``last_thoughts``. Set ``last_parse_ok=False`` on parse
failure (treated as hold).

LEAKAGE NOTE: off-the-shelf LLMs may have memorized the future. Evaluate primarily on
data after the model's training cutoff. VanillaLLMAgent masks tickers as ASS1, ASS2, …
by default (``mask_symbols=True``). See SPEC §4.1.
"""

from __future__ import annotations

from alphaduel.agents.llm.factory import (
    DEFAULT_MODELS,
    DEFAULT_OPENAI_REASONING_EFFORT,
    LLMHandler,
    SUPPORTED_PROVIDERS,
    create_llm,
    resolve_llm,
)
from alphaduel.agents.llm.masking import SymbolMask, masked_symbol
from alphaduel.agents.llm.trajectory import (
    DEFAULT_DATASET_ROOT,
    LLMDatasetWriter,
    serialize_messages,
)
from alphaduel.agents.llm.vanilla import OffShelfLLMAgent, VanillaLLMAgent

__all__ = [
    "DEFAULT_DATASET_ROOT",
    "DEFAULT_MODELS",
    "DEFAULT_OPENAI_REASONING_EFFORT",
    "LLMDatasetWriter",
    "LLMHandler",
    "OffShelfLLMAgent",
    "SUPPORTED_PROVIDERS",
    "SymbolMask",
    "VanillaLLMAgent",
    "create_llm",
    "masked_symbol",
    "resolve_llm",
    "serialize_messages",
]
