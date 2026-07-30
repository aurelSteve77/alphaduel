"""Parse free-form LLM responses into share-delta actions."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

# Prefer fenced ```json ... ``` blocks; fall back to a bare {"actions": ...} object.
_FENCED = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE)
_ACTIONS_OBJ = re.compile(
    r"\{\s*\"actions\"\s*:\s*\{.*?\}\s*\}",
    re.DOTALL | re.IGNORECASE,
)


@dataclass
class ParsedLLMAction:
    """Result of parsing one LLM response."""

    actions: dict[str, int] = field(default_factory=dict)
    rationale: str = ""
    parse_ok: bool = False
    raw: str = ""


def parse_llm_response(raw: str) -> ParsedLLMAction:
    """Extract ``{"actions": {TICKER: delta, ...}}`` from free-form model text.

    On any failure returns empty actions with ``parse_ok=False`` (do-nothing).
    ``rationale`` is always the full raw response.
    """
    text = (raw or "").strip()
    result = ParsedLLMAction(rationale=text, raw=text)
    if not text:
        return result

    payload = _extract_json_object(text)
    if payload is None:
        return result

    actions_raw = payload.get("actions")
    if not isinstance(actions_raw, dict):
        return result

    actions: dict[str, int] = {}
    try:
        for ticker, value in actions_raw.items():
            sym = str(ticker).strip().upper()
            if not sym:
                continue
            # Allow ints or whole-number floats; reject non-numeric junk.
            delta = int(value)
            actions[sym] = delta
    except (TypeError, ValueError):
        return result

    result.actions = actions
    result.parse_ok = True
    return result


def _extract_json_object(text: str) -> dict | None:
    candidates: list[str] = []
    for match in _FENCED.finditer(text):
        candidates.append(match.group(1))
    bare = _ACTIONS_OBJ.search(text)
    if bare:
        candidates.append(bare.group(0))

    for blob in candidates:
        try:
            obj = json.loads(blob)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and "actions" in obj:
            return obj
    return None
