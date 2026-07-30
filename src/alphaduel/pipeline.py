"""End-to-end orchestration tying data -> features -> env -> agents -> evaluation.

Kept deliberately small and readable; this is the "glue" the CLI calls.
"""

from __future__ import annotations

import numpy as np

from alphaduel.agents.registry import build_agent
from alphaduel.config.schema import ExperimentConfig, Secrets
from alphaduel.data.macro import FredMacroSource
from alphaduel.data.prices import YahooPriceSource
from alphaduel.data.storage import ParquetCache
from alphaduel.envs.alphaduel_gym import AlphaDuelGym
from alphaduel.envs.multi_asset_gym import MultiAssetGym
from alphaduel.evaluation.backtest import run_episode
from alphaduel.evaluation.metrics import compute_metrics
from alphaduel.evaluation.report import summarize
from alphaduel.features.store import FeatureStore, MarketPanel, MultiAssetPanel
from alphaduel.log import get_logger

log = get_logger(__name__)


def download_data(config: ExperimentConfig, secrets: Secrets) -> None:
    """Fetch and cache all data sources declared in the config."""
    cache = ParquetCache(secrets.data_dir)
    YahooPriceSource(cache).fetch(config.data.symbols, config.data.start, config.data.end)
    if config.data.macro.provider == "fred":
        FredMacroSource(cache, secrets.fred_api_key, config.data.macro.series).fetch(
            config.data.symbols, config.data.start, config.data.end
        )


def build_panel(config: ExperimentConfig, secrets: Secrets) -> MarketPanel | MultiAssetPanel:
    cache = ParquetCache(secrets.data_dir)
    prices = YahooPriceSource(cache).fetch(
        config.data.symbols, config.data.start, config.data.end
    )
    macro = None
    if config.features.macro and config.data.macro.provider == "fred":
        macro = FredMacroSource(cache, secrets.fred_api_key, config.data.macro.series).fetch(
            config.data.symbols, config.data.start, config.data.end
        )
    store = FeatureStore(config.features)
    if config.env.kind == "multi_asset":
        return store.build_multi_asset_panel(prices, config.data.symbols, macro=macro)
    return store.build_panel(prices, symbol=config.data.symbols[0], macro=macro)


def _make_env(config: ExperimentConfig, panel: MarketPanel | MultiAssetPanel):
    if config.env.kind == "multi_asset":
        return MultiAssetGym(
            panel,
            config.env,
            include_portfolio_state=config.features.include_portfolio_state,
            seed=config.seed,
        )
    return AlphaDuelGym(
        panel,
        config.env,
        include_portfolio_state=config.features.include_portfolio_state,
        seed=config.seed,
    )


def evaluate_agents(
    config: ExperimentConfig, secrets: Secrets, panel: MarketPanel | MultiAssetPanel
) -> dict[str, dict]:
    """Run every configured agent over ``n_episodes`` and return aggregated metrics."""
    rng = np.random.default_rng(config.seed)
    n_trials = len(config.agents)
    results: dict[str, dict] = {}

    for agent_cfg in config.agents:
        env = _make_env(config, panel)
        agent = build_agent(agent_cfg)
        if agent_cfg.kind in ("rl", "generative"):
            log.info("Training agent '%s' (%s) ...", agent.name, agent_cfg.kind)
            agent.train(env)

        log.info(
            "Evaluating agent '%s' over %d episodes ...", agent.name, config.evaluation.n_episodes
        )
        per_episode: dict[str, list[float]] = {}
        for _ in range(config.evaluation.n_episodes):
            seed = int(rng.integers(0, 2**31 - 1))
            traj = run_episode(env, agent, seed=seed)
            metrics = compute_metrics(
                np.asarray(traj.equity),
                n_transactions=traj.n_transactions,
                total_reward=traj.total_reward,
                rf=config.evaluation.risk_free_rate,
                n_trials=n_trials,
            )
            for key, value in metrics.items():
                per_episode.setdefault(key, []).append(value)

        results[agent.name] = summarize(
            per_episode, n_samples=config.evaluation.bootstrap_samples
        )
    return results
