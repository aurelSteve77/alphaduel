"""Prompt loading and Jinja2 rendering.

Usage::

    from alphaduel.prompts import prompt_manager

    raw = prompt_manager.get("base_agent_system").content
    text = prompt_manager.get("base_agent_user").render(market_state=state_str)
"""

from __future__ import annotations

from alphaduel.prompts.manager import Prompt, PromptManager, prompt_manager

__all__ = ["Prompt", "PromptManager", "prompt_manager"]
