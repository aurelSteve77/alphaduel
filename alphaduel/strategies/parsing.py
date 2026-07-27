"""Parse free-form LLM trading replies into :class:`TradingDecision`.

The model writes natural language, then ends with a fenced JSON action block::

    Apple looks strong on RSI, so I will add a small long.
    My action is
    ```json
    {"actions": {"AAPL": 3, "MSFT": 0}}
    ```
"""

from __future__ import annotations

import json
import re
from typing import Any

from alphaduel.logger import get_logger
from alphaduel.strategies.schemas import ShareOrder, TradingDecision

log = get_logger(__name__)

_JSON_FENCE = re.compile(
    r"```(?:json)?\s*(\{.*?\})\s*```",
    re.IGNORECASE | re.DOTALL,
)
_BARE_ACTIONS_OBJECT = re.compile(
    r"(\{\s*\"actions\"\s*:\s*\{.*?\}\s*\})",
    re.IGNORECASE | re.DOTALL,
)
_BARE_ACTIONS_LIST = re.compile(
    r"(\{\s*\"actions\"\s*:\s*\[.*?\]\s*\})",
    re.IGNORECASE | re.DOTALL,
)


def message_content(response: Any) -> str:
    """Normalize a LangChain message / string / dict into plain text."""
    if response is None:
        return ""
    if isinstance(response, str):
        return response
    content = getattr(response, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and "text" in block:
                parts.append(str(block["text"]))
            else:
                parts.append(str(block))
        return "".join(parts)
    if isinstance(response, dict) and "content" in response:
        return str(response["content"])
    return str(response)


def extract_actions_json(text: str) -> dict[str, Any] | None:
    """Pull the ``actions`` JSON object out of free-form model text."""
    candidates: list[str] = [m.group(1) for m in _JSON_FENCE.finditer(text)]
    if not candidates:
        for pattern in (_BARE_ACTIONS_OBJECT, _BARE_ACTIONS_LIST):
            match = pattern.search(text)
            if match:
                candidates.append(match.group(1))

    for raw in reversed(candidates):
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and "actions" in payload:
            return payload
    return None


def _orders_from_actions(actions: Any) -> list[ShareOrder]:
    orders: list[ShareOrder] = []
    if actions is None:
        return orders

    if isinstance(actions, dict):
        for ticker, delta in actions.items():
            orders.append(ShareOrder(ticker=str(ticker), delta=int(delta)))
        return orders

    if isinstance(actions, list):
        for item in actions:
            if isinstance(item, dict):
                ticker = item.get("ticker", item.get("symbol"))
                delta = item.get("delta", item.get("shares", item.get("qty")))
                if ticker is None or delta is None:
                    continue
                orders.append(ShareOrder(ticker=str(ticker), delta=int(delta)))
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                orders.append(ShareOrder(ticker=str(item[0]), delta=int(item[1])))
        return orders

    raise ValueError(f"Unsupported actions payload type: {type(actions)!r}")


def parse_freeform_response(text: str) -> TradingDecision:
    """Parse free-form text; rationale is the full reply, orders come from JSON."""
    raw = text.strip()
    payload = extract_actions_json(raw)
    if payload is None:
        log.warning("No actions JSON found in LLM reply; holding all positions")
        return TradingDecision(rationale=raw, orders=[])

    try:
        orders = _orders_from_actions(payload.get("actions"))
    except (TypeError, ValueError) as exc:
        log.warning("Failed to parse actions JSON (%s); holding all positions", exc)
        return TradingDecision(rationale=raw, orders=[])

    return TradingDecision(rationale=raw, orders=orders)
