"""LLM agents (P2/P3).

Renders the env observation to a textual market brief, asks an LLM for a structured
action (JSON) plus a rationale, and caches responses by (obs-hash, model, prompt).

LEAKAGE NOTE: off-the-shelf LLMs may have memorized the future. Evaluate primarily on
data after the model's training cutoff, and/or enable entity-masking. See SPEC §4.1.
"""

from __future__ import annotations

import numpy as np

from alphaduel.agents.base import Agent


class OffShelfLLMAgent(Agent):
    name = "llm_offshelf"

    def __init__(self, model: str = "gpt-4o-mini", prompt_version: str = "v1", **params) -> None:
        self.model = model
        self.prompt_version = prompt_version
        self.params = params

    def act(self, observation: np.ndarray, info: dict) -> np.ndarray:
        raise NotImplementedError(
            "LLM agent is scheduled for Phase 2: render observation -> brief, call model, "
            "parse JSON action + thoughts (stored in `last_thoughts`), cache by obs-hash."
        )
