"""Tests for free-form LLM reply parsing."""

from __future__ import annotations

from alphaduel.strategies.parsing import extract_actions_json, parse_freeform_response


def test_parse_fenced_dict_actions():
    text = """
Apple looks washed out on RSI, so I will buy a few shares.
My action is
```json
{"actions": {"AAPL": 3, "MSFT": -1}}
```
"""
    decision = parse_freeform_response(text)
    assert decision.rationale.strip().startswith("Apple looks washed out")
    assert {o.ticker: o.delta for o in decision.orders} == {"AAPL": 3, "MSFT": -1}


def test_parse_list_of_objects():
    text = """Holding most names.
```json
{"actions": [{"ticker": "NVDA", "delta": 2}, {"ticker": "TSLA", "delta": -1}]}
```
"""
    decision = parse_freeform_response(text)
    assert {o.ticker: o.delta for o in decision.orders} == {"NVDA": 2, "TSLA": -1}


def test_parse_missing_json_holds():
    decision = parse_freeform_response("I like Apple but forgot the JSON.")
    assert decision.orders == []
    assert "forgot the JSON" in decision.rationale


def test_extract_prefers_last_fence():
    text = """
First draft
```json
{"actions": {"AAPL": 1}}
```
Final
```json
{"actions": {"AAPL": 5}}
```
"""
    payload = extract_actions_json(text)
    assert payload == {"actions": {"AAPL": 5}}
