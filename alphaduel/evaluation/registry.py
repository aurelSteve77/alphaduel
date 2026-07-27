"""Build a :class:`~alphaduel.strategies.base.Strategy` from a run config."""

from __future__ import annotations

from typing import Any, Callable

from alphaduel.environments.trading_env import TradingEnv
from alphaduel.evaluation.config import StrategyRunConfig
from alphaduel.logger import get_logger
from alphaduel.strategies.baselines import BuyAndHoldStrategy, HoldStrategy, RandomStrategy
from alphaduel.strategies.llm_policy import LLMPolicy, LLMPolicyConfig

log = get_logger(__name__)

StrategyFactory = Callable[[TradingEnv, StrategyRunConfig], Any]


def _build_llm(env: TradingEnv, run: StrategyRunConfig) -> LLMPolicy:
    policy_cfg = LLMPolicyConfig.from_project_yaml(**run.policy)
    return LLMPolicy.from_env(env, config=policy_cfg)


def _build_random(env: TradingEnv, run: StrategyRunConfig) -> RandomStrategy:
    seed = run.policy.get("seed", run.eval.seed)
    return RandomStrategy(env.n_assets, env.config.max_shares, seed=seed)


def _build_hold(env: TradingEnv, run: StrategyRunConfig) -> HoldStrategy:
    return HoldStrategy(env.n_assets)


def _build_buy_and_hold(env: TradingEnv, run: StrategyRunConfig) -> BuyAndHoldStrategy:
    return BuyAndHoldStrategy(env.n_assets, env.config.max_shares)


STRATEGY_REGISTRY: dict[str, StrategyFactory] = {
    "llm_policy": _build_llm,
    "random": _build_random,
    "hold": _build_hold,
    "buy_and_hold": _build_buy_and_hold,
}


def build_strategy(env: TradingEnv, run: StrategyRunConfig) -> Any:
    """Instantiate the strategy named by ``run.kind``."""
    key = run.kind.strip().lower()
    if key not in STRATEGY_REGISTRY:
        known = ", ".join(sorted(STRATEGY_REGISTRY))
        raise KeyError(f"Unknown strategy kind {run.kind!r}. Known: {known}")
    log.info("Building strategy kind=%s name=%s", key, run.name)
    return STRATEGY_REGISTRY[key](env, run)
