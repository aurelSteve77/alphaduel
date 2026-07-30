"""Tests for PromptManager and the vanilla LLM agent (mocked LLM, no Ollama)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from alphaduel.agents.llm.parse import parse_llm_response
from alphaduel.agents.llm.state_format import (
    format_market_state,
    share_deltas_to_target_weights,
)
from alphaduel.agents.llm.vanilla import VanillaLLMAgent
from alphaduel.agents.registry import build_agent
from alphaduel.config.schema import AgentConfig
from alphaduel.prompts import prompt_manager
from alphaduel.prompts.manager import PromptManager


def test_prompt_manager_get_content_and_render():
    system = prompt_manager.get("base_agent_system")
    assert "actions" in system.content
    assert "Example" in system.content

    user = prompt_manager.get("base_agent_user")
    rendered = user.render(market_state="hello state", symbols=["AAPL", "MSFT"])
    assert "hello state" in rendered
    assert "AAPL, MSFT" in rendered


def test_prompt_manager_missing_key(tmp_path):
    mgr = PromptManager(tmp_path)
    with pytest.raises(KeyError):
        mgr.get("does_not_exist")


def test_parse_fenced_actions_ok():
    raw = """
    AAPL looks strong; adding a little.
    ```json
    {"actions": {"AAPL": 2, "MSFT": -1}}
    ```
    """
    parsed = parse_llm_response(raw)
    assert parsed.parse_ok is True
    assert parsed.actions == {"AAPL": 2, "MSFT": -1}
    assert "AAPL looks strong" in parsed.rationale


def test_parse_empty_actions_ok():
    raw = 'No edge.\n```json\n{"actions": {}}\n```'
    parsed = parse_llm_response(raw)
    assert parsed.parse_ok is True
    assert parsed.actions == {}


def test_parse_failure_sets_flag():
    parsed = parse_llm_response("I will buy some AAPL tomorrow maybe.")
    assert parsed.parse_ok is False
    assert parsed.actions == {}
    assert parsed.rationale.startswith("I will buy")


def test_share_deltas_to_weights_hold_and_buy():
    symbols = ["AAPL", "MSFT"]
    shares = np.array([10.0, 0.0])
    prices = np.array([100.0, 200.0])
    equity = 10_000.0
    # buy 5 AAPL -> 15*100/10000 = 0.15
    w = share_deltas_to_target_weights(
        {"AAPL": 5}, symbols=symbols, shares=shares, prices=prices, equity=equity
    )
    assert w.shape == (2,)
    assert abs(w[0] - 0.15) < 1e-6
    assert abs(w[1] - 0.0) < 1e-6


def test_format_market_state_contains_positions():
    obs = np.zeros(4, dtype=np.float32)
    info = {
        "timestamp": "2020-01-02",
        "equity": 100_000.0,
        "cash": 50_000.0,
        "shares": np.array([10, 0]),
        "prices": np.array([100.0, 200.0]),
        "symbols": ["AAPL", "MSFT"],
        "weights": np.array([0.01, 0.0]),
        "feature_names": ["ret_1", "mom_5"],
        "asset_features": np.array([[0.01, 0.02], [0.0, -0.01]]),
    }
    text = format_market_state(obs, info)
    assert "AAPL" in text
    assert "MSFT" in text
    assert "=== POSITIONS ===" in text
    assert "ret_1" in text


class _FakeLLM:
    def __init__(self, text: str) -> None:
        self.text = text

    def invoke(self, messages):
        return SimpleNamespace(content=self.text)


def test_vanilla_agent_parses_and_returns_weights():
    reply = (
        "Adding AAPL.\n"
        '```json\n{"actions": {"AAPL": 2}}\n```'
    )
    agent = VanillaLLMAgent(
        llm=_FakeLLM(reply),
        symbols=["AAPL", "MSFT"],
        use_memory=False,
    )
    obs = np.zeros(8, dtype=np.float32)
    info = {
        "equity": 10_000.0,
        "cash": 8_000.0,
        "shares": np.array([10.0, 0.0]),
        "prices": np.array([100.0, 200.0]),
        "symbols": ["AAPL", "MSFT"],
    }
    action = agent.act(obs, info)
    assert action.shape == (2,)
    assert agent.last_parse_ok is True
    assert agent.last_actions == {"AAPL": 2}
    assert agent.last_thoughts is not None
    # 12 shares * 100 / 10000 = 0.12
    assert abs(float(action[0]) - 0.12) < 1e-6


def test_vanilla_agent_parse_fail_is_hold():
    agent = VanillaLLMAgent(
        llm=_FakeLLM("I feel like buying but no json here"),
        symbols=["AAPL"],
    )
    info = {
        "equity": 10_000.0,
        "cash": 5_000.0,
        "shares": 50,
        "price": 100.0,
        "symbols": ["AAPL"],
    }
    action = agent.act(np.zeros(4, dtype=np.float32), info)
    assert agent.last_parse_ok is False
    assert agent.last_actions == {}
    # hold current weight 50*100/10000 = 0.5
    assert abs(float(action[0]) - 0.5) < 1e-6


def test_registry_builds_llm_agent():
    agent = build_agent(
        AgentConfig(
            name="llm_vanilla",
            kind="llm",
            params={"llm": _FakeLLM('```json\n{"actions": {}}\n```'), "symbols": ["AAPL"]},
        )
    )
    assert isinstance(agent, VanillaLLMAgent)
    action = agent.act(
        np.zeros(2, dtype=np.float32),
        {"equity": 1_000.0, "cash": 1_000.0, "shares": 0, "price": 10.0, "symbols": ["AAPL"]},
    )
    assert action.shape == (1,)
    assert agent.last_parse_ok is True


def test_resolve_llm_from_nested_config(monkeypatch):
    from alphaduel.agents.llm import factory as factory_mod

    created: dict = {}

    def _fake_create(name, *, provider="ollama", temperature=0.0, **kwargs):
        created.update(name=name, provider=provider, temperature=temperature, **kwargs)
        return _FakeLLM("ok")

    monkeypatch.setattr(factory_mod, "create_llm", _fake_create)
    llm, agent_params = factory_mod.resolve_llm(
        {
            "llm": {"model": "qwen3.5:2b", "provider": "ollama", "temperature": 0.1},
            "use_memory": True,
        }
    )
    assert created["name"] == "qwen3.5:2b"
    assert created["temperature"] == 0.1
    assert agent_params == {"use_memory": True}
    assert isinstance(llm, _FakeLLM)


def test_vanilla_agent_requires_llm():
    with pytest.raises(TypeError, match="llm"):
        VanillaLLMAgent(llm=None)  # type: ignore[arg-type]


def test_vanilla_agent_memory_keeps_prior_turns():
    class _RecordingLLM:
        def __init__(self) -> None:
            self.calls: list[int] = []

        def invoke(self, messages):
            self.calls.append(len(messages))
            return SimpleNamespace(content='```json\n{"actions": {}}\n```')

    llm = _RecordingLLM()
    agent = VanillaLLMAgent(llm=llm, symbols=["AAPL"], use_memory=True, max_memory_turns=4)
    info = {
        "equity": 1_000.0,
        "cash": 1_000.0,
        "shares": 0,
        "price": 10.0,
        "symbols": ["AAPL"],
    }
    obs = np.zeros(2, dtype=np.float32)
    agent.act(obs, info)
    agent.act(obs, info)
    # Second call should see more messages than the first (prior human/ai retained).
    assert llm.calls[1] > llm.calls[0]
    agent.reset()
    assert agent._memory == []
