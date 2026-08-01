"""Tests for PromptManager and the vanilla LLM agent (mocked LLM, no Ollama)."""

from __future__ import annotations

import json
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
    assert "ASS1" in system.content

    user = prompt_manager.get("base_agent_user")
    rendered = user.render(market_state="hello state", symbols=["ASS1", "ASS2"])
    assert "hello state" in rendered
    assert "ASS1, ASS2" in rendered


def test_symbol_mask_roundtrip():
    from alphaduel.agents.llm.masking import SymbolMask

    mask = SymbolMask.from_symbols(["AAPL", "MSFT", "GOOGL"])
    assert mask.masked == ("ASS1", "ASS2", "ASS3")
    assert mask.mask_ticker("aapl") == "ASS1"
    assert mask.unmask_ticker("ass2") == "MSFT"
    assert mask.unmask_actions({"ASS1": 2, "AAPL": 9, "ASS3": -1}) == {
        "AAPL": 2,
        "GOOGL": -1,
    }


def test_prompt_manager_missing_key(tmp_path):
    mgr = PromptManager(tmp_path)
    with pytest.raises(KeyError):
        mgr.get("does_not_exist")


def test_parse_fenced_actions_ok():
    raw = """
    ASS1 looks strong; adding a little.
    ```json
    {"actions": {"ASS1": 2, "ASS2": -1}}
    ```
    """
    parsed = parse_llm_response(raw)
    assert parsed.parse_ok is True
    assert parsed.actions == {"ASS1": 2, "ASS2": -1}
    assert "ASS1 looks strong" in parsed.rationale


def test_parse_empty_actions_ok():
    raw = 'No edge.\n```json\n{"actions": {}}\n```'
    parsed = parse_llm_response(raw)
    assert parsed.parse_ok is True
    assert parsed.actions == {}


def test_parse_failure_sets_flag():
    parsed = parse_llm_response("I will buy some ASS1 tomorrow maybe.")
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
    assert '{"actions": {"AAPL": 2, "MSFT": -1}}' in text


class _FakeLLM:
    def __init__(self, text: str) -> None:
        self.text = text
        self.last_messages = None

    def invoke(self, messages):
        self.last_messages = messages
        return SimpleNamespace(content=self.text)


def test_vanilla_agent_masks_symbols_and_unmasks_actions():
    reply = (
        "Adding ASS1.\n"
        '```json\n{"actions": {"ASS1": 2}}\n```'
    )
    llm = _FakeLLM(reply)
    agent = VanillaLLMAgent(
        llm=llm,
        symbols=["AAPL", "MSFT"],
        use_memory=False,
        mask_symbols=True,
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
    # Dashboard sees real tickers; LLM saw ASS*.
    assert agent.last_actions == {"AAPL": 2}
    assert agent.last_masked_actions == {"ASS1": 2}
    assert agent.last_trace is not None
    assert agent.last_trace["sft_messages"][0]["role"] == "system"
    assert agent.last_trace["sft_messages"][2]["role"] == "assistant"
    assert "ASS1" in agent.last_trace["market_state"]
    assert "AAPL" not in agent.last_trace["market_state"]
    prompt_blob = " ".join(str(getattr(m, "content", m)) for m in (llm.last_messages or []))
    assert "ASS1" in prompt_blob
    assert "ASS2" in prompt_blob
    assert "AAPL" not in prompt_blob
    assert "MSFT" not in prompt_blob
    # 12 shares * 100 / 10000 = 0.12
    assert abs(float(action[0]) - 0.12) < 1e-6


def test_llm_dataset_writer_writes_episode_and_sft(tmp_path):
    from alphaduel.agents.llm.trajectory import LLMDatasetWriter

    writer = LLMDatasetWriter(
        root=tmp_path / "run",
        run_id="eval_test",
        agent_label="LLM · openai · gpt",
        agent_name="llm_vanilla",
        agent_params={"llm": {"provider": "openai", "model": "gpt-5.4-mini"}},
        mode="multi_asset",
        symbols=["AAPL", "MSFT"],
        seed=1,
    )
    steps = [
        {
            "step": 1,
            "parse_ok": True,
            "sft_messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "user"},
                {"role": "assistant", "content": '```json\n{"actions": {"ASS1": 1}}\n```'},
            ],
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "user"},
                {"role": "assistant", "content": "asst"},
            ],
            "actions_masked": {"ASS1": 1},
            "actions_real": {"AAPL": 1},
            "market_state": "=== MARKET STATE ===",
            "symbol_map": {"ASS1": "AAPL", "ASS2": "MSFT"},
        },
        {
            "step": 2,
            "parse_ok": False,
            "sft_messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "user2"},
                {"role": "assistant", "content": "no json"},
            ],
            "actions_masked": {},
            "actions_real": {},
        },
    ]
    path = writer.write_episode(episode_index=0, episode_seed=1, steps=steps)
    assert path.exists()
    ep = json.loads(path.read_text())
    assert ep["n_steps"] == 2
    assert ep["steps"][0]["actions_real"] == {"AAPL": 1}
    sft_lines = (tmp_path / "run" / "llm_openai_gpt" / "sft.jsonl").read_text().strip().splitlines()
    assert len(sft_lines) == 1  # parse failures skipped
    row = json.loads(sft_lines[0])
    assert row["messages"][2]["role"] == "assistant"
    assert row["actions_masked"] == {"ASS1": 1}


def test_vanilla_agent_ignores_real_ticker_when_masked():
    # Model leaks a real ticker — must be dropped under masking.
    reply = '```json\n{"actions": {"AAPL": 5, "ASS2": 1}}\n```'
    agent = VanillaLLMAgent(
        llm=_FakeLLM(reply),
        symbols=["AAPL", "MSFT"],
        mask_symbols=True,
    )
    info = {
        "equity": 10_000.0,
        "cash": 8_000.0,
        "shares": np.array([0.0, 0.0]),
        "prices": np.array([100.0, 200.0]),
        "symbols": ["AAPL", "MSFT"],
    }
    action = agent.act(np.zeros(4, dtype=np.float32), info)
    assert agent.last_actions == {"MSFT": 1}
    assert "AAPL" not in agent.last_actions
    assert abs(float(action[1]) - 0.02) < 1e-6  # 1 * 200 / 10000


def test_vanilla_agent_can_disable_masking():
    reply = '```json\n{"actions": {"AAPL": 2}}\n```'
    llm = _FakeLLM(reply)
    agent = VanillaLLMAgent(
        llm=llm,
        symbols=["AAPL", "MSFT"],
        mask_symbols=False,
    )
    info = {
        "equity": 10_000.0,
        "cash": 8_000.0,
        "shares": np.array([10.0, 0.0]),
        "prices": np.array([100.0, 200.0]),
        "symbols": ["AAPL", "MSFT"],
    }
    agent.act(np.zeros(8, dtype=np.float32), info)
    assert agent.last_actions == {"AAPL": 2}
    prompt_blob = " ".join(str(getattr(m, "content", m)) for m in (llm.last_messages or []))
    assert "AAPL" in prompt_blob


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
            params={
                "llm": _FakeLLM('```json\n{"actions": {}}\n```'),
                "symbols": ["AAPL"],
            },
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


def test_factory_routes_replicate(monkeypatch):
    from alphaduel.agents.llm import factory as factory_mod

    captured: dict = {}

    def _fake_replicate(name, *, temperature, **kwargs):
        captured.update(name=name, temperature=temperature, **kwargs)
        return _FakeLLM("replicate")

    monkeypatch.setattr(factory_mod.LLMHandler, "_create_replicate", staticmethod(_fake_replicate))
    llm = factory_mod.LLMHandler.create(
        "meta/meta-llama-3-8b-instruct",
        provider="replicate",
        temperature=0.2,
    )
    assert isinstance(llm, _FakeLLM)
    assert captured["name"] == "meta/meta-llama-3-8b-instruct"
    assert captured["temperature"] == 0.2


def test_factory_routes_anthropic(monkeypatch):
    from alphaduel.agents.llm import factory as factory_mod

    captured: dict = {}

    def _fake_anthropic(name, *, temperature, **kwargs):
        captured.update(name=name, temperature=temperature, **kwargs)
        return _FakeLLM("anthropic")

    monkeypatch.setattr(
        factory_mod.LLMHandler, "_create_anthropic", staticmethod(_fake_anthropic)
    )
    llm = factory_mod.LLMHandler.create(
        "claude-haiku-4-5-20251001",
        provider="anthropic",
        temperature=0.0,
        api_key="sk-ant-test",
    )
    assert isinstance(llm, _FakeLLM)
    assert captured["name"] == "claude-haiku-4-5-20251001"
    assert captured["api_key"] == "sk-ant-test"


def test_anthropic_chat_instantiation(monkeypatch):
    from alphaduel.agents.llm import factory as factory_mod

    captured: dict = {}

    class _FakeChatAnthropic:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    import sys
    from types import ModuleType

    mod = ModuleType("langchain_anthropic")
    mod.ChatAnthropic = _FakeChatAnthropic
    monkeypatch.setitem(sys.modules, "langchain_anthropic", mod)

    factory_mod.LLMHandler.create(
        "claude-haiku-4-5-20251001",
        provider="anthropic",
        temperature=0.1,
        api_key="sk-ant-test",
        max_tokens=1024,
    )
    assert captured["model"] == "claude-haiku-4-5-20251001"
    assert captured["temperature"] == 0.1
    assert captured["api_key"] == "sk-ant-test"
    assert captured["max_tokens"] == 1024


def test_factory_routes_openai(monkeypatch):
    from alphaduel.agents.llm import factory as factory_mod

    captured: dict = {}

    def _fake_openai(name, *, temperature, **kwargs):
        captured.update(name=name, temperature=temperature, **kwargs)
        return _FakeLLM("openai")

    monkeypatch.setattr(factory_mod.LLMHandler, "_create_openai", staticmethod(_fake_openai))
    llm = factory_mod.LLMHandler.create(
        "gpt-5.4-mini",
        provider="openai",
        temperature=0.0,
        api_key="sk-test",
    )
    assert isinstance(llm, _FakeLLM)
    assert captured["name"] == "gpt-5.4-mini"
    assert captured["api_key"] == "sk-test"


def test_openai_defaults_reasoning_effort_low(monkeypatch):
    from alphaduel.agents.llm import factory as factory_mod

    captured: dict = {}

    class _FakeChatOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    import sys
    from types import ModuleType

    mod = ModuleType("langchain_openai")
    mod.ChatOpenAI = _FakeChatOpenAI
    monkeypatch.setitem(sys.modules, "langchain_openai", mod)

    factory_mod.LLMHandler.create("gpt-5.4-mini", provider="openai", temperature=0.0)
    assert captured.get("reasoning_effort") == "low"

    captured.clear()
    factory_mod.LLMHandler.create(
        "gpt-5.4-mini", provider="openai", temperature=0.0, reasoning_effort="high"
    )
    assert captured.get("reasoning_effort") == "high"


def test_factory_rejects_unknown_provider():
    from alphaduel.agents.llm.factory import LLMHandler

    with pytest.raises(ValueError, match="Unknown LLM provider"):
        LLMHandler.create("x", provider="nope")


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
