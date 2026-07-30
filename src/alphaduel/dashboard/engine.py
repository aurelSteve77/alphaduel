"""Dashboard engine: build a mock env, run agents, and collect rich per-step records.

The single public entry point :func:`run_benchmark` takes only primitives so it can be
cached by Streamlit. It returns plain arrays/dicts ready for the UI components.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from alphaduel.agents.registry import build_agent
from alphaduel.config.schema import AgentConfig, CostConfig, EnvConfig, RewardConfig
from alphaduel.dashboard import mock_data
from alphaduel.envs.alphaduel_gym import AlphaDuelGym
from alphaduel.envs.multi_asset_gym import MultiAssetGym
from alphaduel.evaluation.metrics import compute_metrics

SINGLE_AGENTS = ["buy_and_hold", "momentum", "mean_reversion", "volatility_target", "random"]
MULTI_AGENTS = ["equal_weight", "inverse_volatility", "random_weights", "genportfolio"]
_NEEDS_SEED = {"random", "random_weights"}
_GENERATIVE = {"genportfolio"}


def agents_for(mode: str) -> list[str]:
    return MULTI_AGENTS if mode == "multi_asset" else SINGLE_AGENTS


def default_params() -> dict:
    """Default experiment configuration shared by every dashboard page."""
    return {
        "mode": "multi_asset",
        "agent_names": tuple(MULTI_AGENTS),
        "n_assets": 4,
        "n_steps": 500,
        "episode_length": 120,
        "n_episodes": 20,
        "seed": 7,
        "initial_cash": 100_000.0,
        "commission_bps": 1.0,
        "half_spread_bps": 2.0,
        "reward_kind": "log_return",
        "drift": 0.0004,
        "vol": 0.012,
    }


@dataclass
class AgentRecord:
    name: str
    equity: np.ndarray
    rewards: np.ndarray
    exposure: np.ndarray            # total invested fraction per step (0..1)
    weights: np.ndarray             # (T, n_assets) target weights
    fills: np.ndarray
    costs: np.ndarray
    thoughts: list[str | None]
    metrics: dict[str, float]       # mean over episodes


@dataclass
class BenchmarkResult:
    mode: str
    symbols: list[str]
    timestamps: list
    prices: np.ndarray              # (T, n_assets) replay-episode prices
    agents: dict[str, AgentRecord] = field(default_factory=dict)


def _build_env(mode: str, panel, env_cfg: EnvConfig, seed: int):
    if mode == "multi_asset":
        return MultiAssetGym(panel, env_cfg, seed=seed)
    return AlphaDuelGym(panel, env_cfg, seed=seed)


def _kind_of(name: str) -> str:
    return "generative" if name in _GENERATIVE else "baseline"


def _make_agent(name: str, seed: int):
    params = {"seed": seed} if name in _NEEDS_SEED else {}
    return build_agent(AgentConfig(name=name, kind=_kind_of(name), params=params))


def build_named_agent(name: str, params: dict | None = None):
    """Build one configured agent (used by the live-run page)."""
    return build_agent(AgentConfig(name=name, kind=_kind_of(name), params=params or {}))


def _panel_for(params: dict):
    if params["mode"] == "multi_asset":
        panel = mock_data.make_multi_panel(
            params["n_steps"], params["n_assets"], params["drift"], params["vol"], params["seed"]
        )
        return panel, panel.symbols
    panel = mock_data.make_single_panel(
        params["n_steps"], params["drift"], params["vol"], params["seed"]
    )
    return panel, ["ASSET"]


def _env_config(params: dict) -> EnvConfig:
    return EnvConfig(
        kind=params["mode"],
        initial_cash=params["initial_cash"],
        episode_length=params["episode_length"],
        costs=CostConfig(
            commission_bps=params["commission_bps"], half_spread_bps=params["half_spread_bps"]
        ),
        reward=RewardConfig(kind=params["reward_kind"]),
    )


def make_live_session(params: dict, agent_name: str, agent_params: dict):
    """Build (env, agent, n_assets, symbols) ready for :func:`stream_episode`."""
    panel, symbols = _panel_for(params)
    env = _build_env(params["mode"], panel, _env_config(params), params["seed"])
    agent = build_named_agent(agent_name, agent_params)
    if agent_name in _GENERATIVE:
        agent.train(env)
    return env, agent, len(symbols), symbols


def _weights_from_info(info: dict, n_assets: int) -> np.ndarray:
    if "weights" in info:  # multi-asset env
        return np.asarray(info["weights"], dtype=np.float64)
    price = float(info["price"])
    shares = float(info["shares"])
    equity = float(info["equity"]) or 1.0
    return np.array([shares * price / equity])


def _prices_from_info(info: dict) -> np.ndarray:
    return np.asarray(info["prices"] if "prices" in info else [info["price"]], dtype=np.float64)


def stream_episode(env, agent, n_assets: int, seed: int | None = None):
    """Yield cumulative episode state after each step (drives the live page).

    Each yield shares the same growing lists; Streamlit reads them synchronously, so the
    live page can plot the running equity/allocation without copying every step.
    """
    obs, info = env.reset(seed=seed)
    agent.reset()
    equity = [float(info["equity"])]
    timestamps = [info["timestamp"]]
    prices = [_prices_from_info(info)]
    weights: list[np.ndarray] = []
    exposure: list[float] = []
    rewards: list[float] = []
    fills: list[int] = []
    costs: list[float] = []
    thoughts: list[str | None] = []

    def _state(done: bool, step: int) -> dict:
        return {
            "step": step,
            "done": done,
            "equity": equity,
            "timestamps": timestamps,
            "prices": prices,
            "weights_hist": weights,
            "exposure": exposure,
            "rewards": rewards,
            "fills": fills,
            "costs": costs,
            "thoughts": thoughts,
            "current_weights": weights[-1] if weights else _weights_from_info(info, n_assets),
            "cash": float(info["cash"]),
            "thought": thoughts[-1] if thoughts else None,
        }

    yield _state(False, 0)

    done = False
    step = 0
    while not done:
        action = agent.act(obs, info)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        step += 1
        equity.append(float(info["equity"]))
        timestamps.append(info["timestamp"])
        prices.append(_prices_from_info(info))
        weights.append(_weights_from_info(info, n_assets))
        exposure.append(1.0 - float(info["cash"]) / (float(info["equity"]) or 1.0))
        rewards.append(float(reward))
        fills.append(int(info["fill_shares"]))
        costs.append(float(info["fill_cost"]))
        thoughts.append(getattr(agent, "last_thoughts", None))
        yield _state(done, step)


def _run_episode(env, agent, n_assets: int, seed: int) -> dict:
    last: dict = {}
    for step_state in stream_episode(env, agent, n_assets, seed=seed):
        last = step_state
    return {
        "timestamps": last["timestamps"],
        "equity": np.asarray(last["equity"]),
        "rewards": np.asarray(last["rewards"]),
        "weights": np.vstack(last["weights_hist"]),
        "exposure": np.asarray(last["exposure"]),
        "fills": np.asarray(last["fills"], dtype=int),
        "costs": np.asarray(last["costs"]),
        "thoughts": last["thoughts"],
        "prices": np.vstack(last["prices"]),
    }


def run_benchmark(
    mode: str,
    agent_names: tuple[str, ...],
    n_assets: int = 4,
    n_steps: int = 500,
    episode_length: int = 120,
    n_episodes: int = 20,
    seed: int = 7,
    initial_cash: float = 100_000.0,
    commission_bps: float = 1.0,
    half_spread_bps: float = 2.0,
    reward_kind: str = "log_return",
    drift: float = 0.0004,
    vol: float = 0.012,
) -> BenchmarkResult:
    """Run every named agent over a mock market and return per-step + aggregate results."""
    params = {
        "mode": mode, "n_assets": n_assets, "n_steps": n_steps,
        "episode_length": episode_length, "seed": seed, "initial_cash": initial_cash,
        "commission_bps": commission_bps, "half_spread_bps": half_spread_bps,
        "reward_kind": reward_kind, "drift": drift, "vol": vol,
    }
    panel, symbols = _panel_for(params)
    n = len(symbols)
    env = _build_env(mode, panel, _env_config(params), seed)

    result = BenchmarkResult(mode=mode, symbols=symbols, timestamps=[], prices=np.empty(0))
    for name in agent_names:
        agent = _make_agent(name, seed)
        if name in _GENERATIVE:
            agent.train(env)

        episodes = [_run_episode(env, agent, n, seed + i) for i in range(n_episodes)]
        replay = episodes[0]
        metrics = _aggregate_metrics(episodes, n_trials=len(agent_names))

        result.agents[name] = AgentRecord(
            name=name,
            equity=replay["equity"],
            rewards=replay["rewards"],
            exposure=replay["exposure"],
            weights=replay["weights"],
            fills=replay["fills"],
            costs=replay["costs"],
            thoughts=replay["thoughts"],
            metrics=metrics,
        )
        if not result.timestamps:
            result.timestamps = replay["timestamps"]
            result.prices = replay["prices"]
    return result


def _aggregate_metrics(episodes: list[dict], n_trials: int) -> dict[str, float]:
    per_episode: dict[str, list[float]] = {}
    for ep in episodes:
        n_tx = int((ep["fills"] != 0).sum())
        m = compute_metrics(
            ep["equity"], n_transactions=n_tx, total_reward=float(ep["rewards"].sum()),
            n_trials=n_trials,
        )
        for k, v in m.items():
            per_episode.setdefault(k, []).append(v)
    return {k: float(np.mean(v)) for k, v in per_episode.items()}
