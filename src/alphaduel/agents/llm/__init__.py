"""LLM agents (P2): off-the-shelf vanilla strategy with injected chat models.

Build the model with :class:`LLMHandler` / :func:`create_llm`, then pass it to
:class:`VanillaLLMAgent`. Renders the env observation to a textual market brief,
asks the LLM for free-form reasoning plus a fenced JSON share-delta action, and
stores the rationale in ``last_thoughts``. Set ``last_parse_ok=False`` on parse
failure (treated as hold).

LEAKAGE NOTE: off-the-shelf LLMs may have memorized the future. Evaluate primarily on
data after the model's training cutoff, and/or enable entity-masking. See SPEC §4.1.
"""

from __future__ import annotations

from alphaduel.agents.llm.factory import LLMHandler, create_llm, resolve_llm
from alphaduel.agents.llm.vanilla import OffShelfLLMAgent, VanillaLLMAgent

__all__ = [
    "LLMHandler",
    "OffShelfLLMAgent",
    "VanillaLLMAgent",
    "create_llm",
    "resolve_llm",
]
