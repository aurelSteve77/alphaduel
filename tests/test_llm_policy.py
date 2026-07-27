"""Tests for the LLM trading policy (LLM calls are mocked)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alphaduel.environments import TradingEnv, TradingEnvConfig
from alphaduel.prompt_manager import PromptManager
from alphaduel.strategies import LLMPolicy, LLMPolicyConfig, ShareOrder, TradingDecision
from alphaduel.utils.singleton import Singleton


class _FakeLLM:
    """Returns free-form text the way ChatOllama would (``.content``)."""

    def __init__(self, text: str) -> None:
        self.text = text
        self.calls: list[list[dict[str, str]]] = []

    def invoke(self, messages: list[dict[str, str]]) -> object:
        self.calls.append(messages)

        class _Msg:
            def __init__(self, content: str) -> None:
                self.content = content

        return _Msg(self.text)


def _reply(actions_json: str, prose: str = "Taking a trade today.") -> str:
    return f"{prose}\nMy action is\n```json\n{actions_json}\n```\n"


@pytest.fixture
def synthetic_prices() -> pd.DataFrame:
    idx = pd.bdate_range("2024-01-02", periods=60)
    rng = np.random.default_rng(1)
    return pd.DataFrame(
        {
            "AAPL": 100 * np.cumprod(1 + rng.normal(0, 0.01, len(idx))),
            "MSFT": 200 * np.cumprod(1 + rng.normal(0, 0.01, len(idx))),
        },
        index=idx,
    )


@pytest.fixture
def env(synthetic_prices) -> TradingEnv:
    cfg = TradingEnvConfig(
        start="2024-01-02",
        end="2024-03-29",
        tickers=["AAPL", "MSFT"],
        initial_cash=10_000.0,
        n_days=5,
        max_shares=4,
        features=["close", "pct_change_1d"],
    )
    return TradingEnv(config=cfg, prices=synthetic_prices)


def test_decision_to_action_clips_and_ignores_unknown():
    decision = TradingDecision(
        orders=[
            ShareOrder(ticker="aapl", delta=99),
            ShareOrder(ticker="FAKE", delta=3),
            ShareOrder(ticker="MSFT", delta=-2),
        ],
        rationale="test",
    )
    action = decision.to_action(["AAPL", "MSFT"], max_shares=5)
    assert action.tolist() == [5, -2]


def test_act_with_mocked_llm(env: TradingEnv):
    Singleton._instances.pop(PromptManager, None)
    llm = _FakeLLM(_reply('{"actions": {"AAPL": 2, "MSFT": 0}}', prose="buy apple"))
    policy = LLMPolicy.from_env(
        env,
        config=LLMPolicyConfig(model="dummy"),
        llm=llm,
    )
    obs, info = env.reset(seed=0)
    action = policy.act(obs, info)
    assert action.dtype == np.int32
    assert action.tolist() == [2, 0]
    assert policy.last_decision is not None
    assert "buy apple" in policy.last_decision.rationale
    assert policy.last_raw is not None
    assert len(llm.calls) == 1
    roles = [m["role"] for m in llm.calls[0]]
    assert roles == ["system", "user"]
    assert "```json" in llm.calls[0][0]["content"]
    hist_roles = [m["role"] for m in policy.history]
    assert hist_roles == ["user", "ai"]
    assert "buy apple" in policy.history[1]["content"]
    Singleton._instances.pop(PromptManager, None)


def test_history_appended_on_subsequent_acts(env: TradingEnv):
    Singleton._instances.pop(PromptManager, None)
    llm = _FakeLLM(_reply('{"actions": {"AAPL": 1}}', prose="step"))
    policy = LLMPolicy.from_env(
        env,
        config=LLMPolicyConfig(model="dummy", max_history=5),
        llm=llm,
    )
    obs, info = env.reset(seed=0)
    action = policy.act(obs, info)
    obs, _, _, _, info = env.step(action)
    policy.act(obs, info)

    assert len(llm.calls) == 2
    second_roles = [m["role"] for m in llm.calls[1]]
    assert second_roles == ["system", "user", "ai", "user"]
    assert llm.calls[1][2]["role"] == "ai"
    assert "AAPL" in llm.calls[1][2]["content"]
    Singleton._instances.pop(PromptManager, None)


def test_history_trimmed_and_reset(env: TradingEnv):
    Singleton._instances.pop(PromptManager, None)
    llm = _FakeLLM(_reply('{"actions": {"MSFT": 1}}', prose="x"))
    policy = LLMPolicy.from_env(
        env,
        config=LLMPolicyConfig(model="dummy", max_history=1),
        llm=llm,
    )
    obs, info = env.reset(seed=0)
    for _ in range(3):
        action = policy.act(obs, info)
        obs, _, terminated, _, info = env.step(action)
        if terminated:
            break
    assert len(policy.history) == 2
    assert [m["role"] for m in policy.history] == ["user", "ai"]

    policy.reset()
    assert policy.history == []
    assert policy.last_decision is None
    Singleton._instances.pop(PromptManager, None)


def test_step_loop_with_policy(env: TradingEnv):
    Singleton._instances.pop(PromptManager, None)
    llm = _FakeLLM(_reply('{"actions": {"MSFT": 1}}', prose="hold bias"))
    policy = LLMPolicy.from_env(env, llm=llm)
    obs, info = env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(policy.act(obs, info))
    assert env.portfolio.holdings[1] == 1
    assert isinstance(reward, float)
    assert not truncated
    Singleton._instances.pop(PromptManager, None)
