"""Dashboard engine: build a real-data env, run agents, and collect rich per-step records.

The single public entry point :func:`run_benchmark` takes only primitives so it can be
cached by Streamlit. It returns plain arrays/dicts ready for the UI components.
"""

from __future__ import annotations

import threading
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from alphaduel.agents.registry import build_agent
from alphaduel.config.schema import AgentConfig, CostConfig, EnvConfig, RewardConfig
from alphaduel.dashboard import mock_data, real_data
from alphaduel.envs.alphaduel_gym import AlphaDuelGym
from alphaduel.envs.multi_asset_gym import MultiAssetGym
from alphaduel.evaluation.metrics import compute_metrics

# Cap concurrent contestant workers (LLM calls are I/O-bound; keep Ollama/API load sane).
EVAL_MAX_WORKERS = 3

SINGLE_AGENTS = [
    "buy_and_hold",
    "momentum",
    "mean_reversion",
    "volatility_target",
    "random",
    "llm_vanilla",
]
MULTI_AGENTS = [
    "equal_weight",
    "inverse_volatility",
    "random_weights",
    "genportfolio",
    "llm_vanilla",
]
LLM_AGENTS = frozenset({"llm_vanilla"})
_NEEDS_SEED = {"random", "random_weights"}
_GENERATIVE = {"genportfolio"}

_DEFAULT_LLM = {
    "llm_provider": "ollama",
    "llm_model": "qwen3.5:2b",
    "llm_temperature": 0.0,
    "llm_reasoning": False,
    "llm_reasoning_effort": "low",
    "llm_use_memory": False,
    "llm_max_memory_turns": 8,
    "llm_system_prompt_key": "base_agent_system",
    "llm_user_prompt_key": "base_agent_user",
    "llm_mask_symbols": True,
}


def agents_for(mode: str) -> list[str]:
    return MULTI_AGENTS if mode == "multi_asset" else SINGLE_AGENTS


def baseline_agents(mode: str) -> list[str]:
    """Agents safe to run without an LLM backend (used as dashboard defaults)."""
    return [a for a in agents_for(mode) if a not in LLM_AGENTS]


def default_symbols(mode: str, n_assets: int = 4) -> tuple[str, ...]:
    """Default tickers for a dashboard mode (from YAML universe catalogs)."""
    catalog = real_data.symbol_catalog()
    if not catalog:
        return ("AAPL", "MSFT", "GOOGL", "AMZN")[: max(1, n_assets)]
    if mode == "single_asset":
        return (catalog[0],)
    n = max(2, min(int(n_assets), len(catalog)))
    return tuple(catalog[:n])


def default_params() -> dict:
    """Default experiment configuration shared by every dashboard page."""
    symbols = default_symbols("multi_asset", 4)
    start, end = real_data.default_date_range("multi_asset")
    return {
        "mode": "multi_asset",
        "agent_names": tuple(baseline_agents("multi_asset")),
        "symbols": symbols,
        "n_assets": len(symbols),
        "start_date": start,
        "end_date": end,
        "n_steps": 500,
        "episode_length": 120,
        "n_episodes": 20,
        "seed": 7,
        "initial_cash": 100_000.0,
        "commission_bps": 1.0,
        "half_spread_bps": 2.0,
        "reward_kind": "log_return",
        "use_mock": False,
        **_DEFAULT_LLM,
    }


def default_llm_params() -> dict:
    return dict(_DEFAULT_LLM)


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
    parse_oks: list[bool | None] = field(default_factory=list)
    llm_actions: list[dict | None] = field(default_factory=list)
    episode_metrics: list[dict[str, float]] = field(default_factory=list)


@dataclass
class BenchmarkResult:
    mode: str
    symbols: list[str]
    timestamps: list
    prices: np.ndarray              # (T, n_assets) replay-episode prices
    agents: dict[str, AgentRecord] = field(default_factory=dict)
    dataset_dir: str | None = None  # LLM SFT traces root when saved during evaluate


def _build_env(mode: str, panel, env_cfg: EnvConfig, seed: int):
    if mode == "multi_asset":
        return MultiAssetGym(panel, env_cfg, seed=seed)
    return AlphaDuelGym(panel, env_cfg, seed=seed)


def _kind_of(name: str) -> str:
    if name in LLM_AGENTS:
        return "llm"
    if name in _GENERATIVE:
        return "generative"
    return "baseline"


def _llm_agent_params(params: dict) -> dict:
    """Map dashboard LLM settings → VanillaLLMAgent / resolve_llm params."""
    return {
        "llm": {
            "provider": params.get("llm_provider", "ollama"),
            "model": params.get("llm_model", "qwen3.5:2b"),
            "temperature": float(params.get("llm_temperature", 0.0)),
            "reasoning": bool(params.get("llm_reasoning", False)),
            "reasoning_effort": params.get("llm_reasoning_effort", "low"),
        },
        "use_memory": bool(params.get("llm_use_memory", False)),
        "max_memory_turns": int(params.get("llm_max_memory_turns", 8)),
        "system_prompt_key": params.get("llm_system_prompt_key", "base_agent_system"),
        "user_prompt_key": params.get("llm_user_prompt_key", "base_agent_user"),
        "mask_symbols": bool(params.get("llm_mask_symbols", True)),
    }


def _make_agent(name: str, seed: int, dash_params: dict | None = None):
    dash_params = dash_params or {}
    if name in LLM_AGENTS:
        return build_agent(
            AgentConfig(name=name, kind="llm", params=_llm_agent_params(dash_params))
        )
    params = {"seed": seed} if name in _NEEDS_SEED else {}
    return build_agent(AgentConfig(name=name, kind=_kind_of(name), params=params))


def build_named_agent(name: str, params: dict | None = None):
    """Build one configured agent (used by the live-run page)."""
    return build_agent(AgentConfig(name=name, kind=_kind_of(name), params=params or {}))


def get_panel(params: dict):
    """Build the market panel for the current dashboard params (real or mock)."""
    return _panel_for(params)


def _panel_for(params: dict):
    symbols = params.get("symbols")
    if isinstance(symbols, str):
        symbols = (symbols,)
    elif symbols is not None:
        symbols = tuple(str(s).strip().upper() for s in symbols if str(s).strip())
        if not symbols:
            symbols = None

    if params.get("use_mock"):
        drift = float(params.get("drift", 0.0004))
        vol = float(params.get("vol", 0.012))
        seed = int(params.get("seed", 7))
        mock_steps = int(params["n_steps"] or 500)
        if params["mode"] == "multi_asset":
            panel = mock_data.make_multi_panel(
                mock_steps,
                params.get("n_assets", 4),
                drift,
                vol,
                seed,
                symbols=symbols,
            )
            return panel, panel.symbols
        panel = mock_data.make_single_panel(
            mock_steps,
            drift,
            vol,
            seed,
            symbol=(symbols[0] if symbols else "ASSET"),
        )
        return panel, list(panel.symbols)
    return real_data.load_panel(
        params["mode"],
        params.get("n_assets", 4),
        params["n_steps"],
        symbols=symbols,
        start=params.get("start_date"),
        end=params.get("end_date"),
    )


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
    # Live page may already nest llm={...}; otherwise fall back to dashboard LLM settings.
    if agent_name in LLM_AGENTS and "llm" not in agent_params:
        agent_params = {**_llm_agent_params(params), **agent_params}
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
    parse_oks: list[bool | None] = []
    llm_actions: list[dict | None] = []
    llm_traces: list[dict | None] = []

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
            "parse_oks": parse_oks,
            "llm_actions": llm_actions,
            "llm_traces": llm_traces,
            "current_weights": weights[-1] if weights else _weights_from_info(info, n_assets),
            "cash": float(info["cash"]),
            "thought": thoughts[-1] if thoughts else None,
            "parse_ok": parse_oks[-1] if parse_oks else None,
            "actions": llm_actions[-1] if llm_actions else None,
        }

    yield _state(False, 0)

    done = False
    step = 0
    while not done:
        action = agent.act(obs, info)
        trace = getattr(agent, "last_trace", None)
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
        parse_oks.append(getattr(agent, "last_parse_ok", None))
        raw_actions = getattr(agent, "last_actions", None)
        llm_actions.append(dict(raw_actions) if raw_actions else None)
        if isinstance(trace, dict):
            step_trace = dict(trace)
            step_trace["step"] = step
            step_trace["timestamp"] = info.get("timestamp")
            step_trace["reward"] = float(reward)
            step_trace["fill_shares"] = int(info.get("fill_shares", 0))
            step_trace["fill_cost"] = float(info.get("fill_cost", 0.0))
            step_trace["equity_after"] = float(info.get("equity", 0.0))
            step_trace["cash_after"] = float(info.get("cash", 0.0))
            step_trace["weights_after"] = _weights_from_info(info, n_assets).tolist()
            llm_traces.append(step_trace)
        else:
            llm_traces.append(None)
        yield _state(done, step)


def _run_episode(env, agent, n_assets: int, seed: int, on_step=None) -> dict:
    last: dict = {}
    for step_state in stream_episode(env, agent, n_assets, seed=seed):
        last = step_state
        if on_step is not None:
            on_step(int(step_state["step"]), bool(step_state["done"]))
    return {
        "timestamps": last["timestamps"],
        "equity": np.asarray(last["equity"]),
        "rewards": np.asarray(last["rewards"]),
        "weights": np.vstack(last["weights_hist"]) if last["weights_hist"] else np.zeros((0, n_assets)),
        "exposure": np.asarray(last["exposure"]),
        "fills": np.asarray(last["fills"], dtype=int),
        "costs": np.asarray(last["costs"]),
        "thoughts": last["thoughts"],
        "parse_oks": last["parse_oks"],
        "llm_actions": last["llm_actions"],
        "llm_traces": last.get("llm_traces") or [],
        "prices": np.vstack(last["prices"]),
    }


def run_benchmark(
    mode: str,
    agent_names: tuple[str, ...],
    n_assets: int = 4,
    n_steps: int | None = 500,
    episode_length: int = 120,
    n_episodes: int = 20,
    seed: int = 7,
    initial_cash: float = 100_000.0,
    commission_bps: float = 1.0,
    half_spread_bps: float = 2.0,
    reward_kind: str = "log_return",
    use_mock: bool = False,
    drift: float = 0.0004,
    vol: float = 0.012,
    llm_provider: str = "ollama",
    llm_model: str = "qwen3.5:2b",
    llm_temperature: float = 0.0,
    llm_reasoning: bool = False,
    llm_reasoning_effort: str = "low",
    llm_use_memory: bool = False,
    llm_max_memory_turns: int = 8,
    llm_system_prompt_key: str = "base_agent_system",
    llm_user_prompt_key: str = "base_agent_user",
    llm_mask_symbols: bool = True,
    symbols: tuple[str, ...] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> BenchmarkResult:
    """Run every named agent over a real (or mock) market and return per-step + aggregate results."""
    params = {
        "mode": mode, "n_assets": n_assets, "n_steps": n_steps,
        "episode_length": episode_length, "seed": seed, "initial_cash": initial_cash,
        "commission_bps": commission_bps, "half_spread_bps": half_spread_bps,
        "reward_kind": reward_kind, "use_mock": use_mock, "drift": drift, "vol": vol,
        "llm_provider": llm_provider, "llm_model": llm_model,
        "llm_temperature": llm_temperature, "llm_reasoning": llm_reasoning,
        "llm_reasoning_effort": llm_reasoning_effort,
        "llm_use_memory": llm_use_memory, "llm_max_memory_turns": llm_max_memory_turns,
        "llm_system_prompt_key": llm_system_prompt_key,
        "llm_user_prompt_key": llm_user_prompt_key,
        "llm_mask_symbols": llm_mask_symbols,
        "symbols": symbols,
        "start_date": start_date,
        "end_date": end_date,
    }
    panel, symbols = _panel_for(params)
    n = len(symbols)
    env = _build_env(mode, panel, _env_config(params), seed)

    result = BenchmarkResult(mode=mode, symbols=symbols, timestamps=[], prices=np.empty(0))
    for name in agent_names:
        agent = _make_agent(name, seed, params)
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
            parse_oks=replay["parse_oks"],
            llm_actions=replay["llm_actions"],
        )
        if not result.timestamps:
            result.timestamps = replay["timestamps"]
            result.prices = replay["prices"]
    return result


def _per_episode_metrics(episodes: list[dict], n_trials: int) -> list[dict[str, float]]:
    out: list[dict[str, float]] = []
    for ep in episodes:
        n_tx = int((ep["fills"] != 0).sum())
        out.append(
            compute_metrics(
                ep["equity"],
                n_transactions=n_tx,
                total_reward=float(ep["rewards"].sum()),
                n_trials=n_trials,
            )
        )
    return out


def _aggregate_metrics(episodes: list[dict], n_trials: int) -> dict[str, float]:
    per_episode = _per_episode_metrics(episodes, n_trials)
    if not per_episode:
        return {}
    keys = per_episode[0].keys()
    return {k: float(np.mean([m[k] for m in per_episode])) for k in keys}


def default_contestant_label(agent_name: str, params: dict | None = None) -> str:
    """Human-readable label for a contestant (esp. LLM provider/model)."""
    params = params or {}
    if agent_name in LLM_AGENTS:
        llm = params.get("llm") or {}
        provider = llm.get("provider", "ollama")
        model = llm.get("model", "qwen3.5:2b")
        return f"LLM · {provider} · {model}"
    return agent_name


def _make_agent_from_spec(agent_name: str, seed: int, agent_params: dict | None = None):
    """Build an agent from an evaluation contestant spec."""
    agent_params = dict(agent_params or {})
    if agent_name in LLM_AGENTS:
        # Accept either nested live-style params or flat dashboard keys.
        if "llm" not in agent_params:
            agent_params = _llm_agent_params(agent_params)
        return build_agent(AgentConfig(name=agent_name, kind="llm", params=agent_params))
    if agent_name in _NEEDS_SEED and "seed" not in agent_params:
        agent_params = {**agent_params, "seed": seed}
    return build_agent(
        AgentConfig(name=agent_name, kind=_kind_of(agent_name), params=agent_params)
    )


@dataclass
class AgentProgress:
    """Live status for one contestant during :func:`run_evaluation`."""

    label: str
    status: str = "queued"  # queued | training | running | done | error
    episode: int = 0  # 1-based while running
    n_episodes: int = 0
    step: int = 0
    episode_length: int = 0
    detail: str = ""

    def row(self) -> dict:
        return asdict(self)


class EvalProgressBoard:
    """Thread-safe per-agent progress for parallel evaluation."""

    def __init__(self, labels: list[str], n_episodes: int, episode_length: int):
        self._lock = threading.Lock()
        self._order = list(labels)
        self._agents = {
            label: AgentProgress(
                label=label, n_episodes=n_episodes, episode_length=episode_length
            )
            for label in labels
        }

    def update(self, label: str, **fields) -> None:
        with self._lock:
            agent = self._agents[label]
            for key, value in fields.items():
                setattr(agent, key, value)

    def snapshot(self) -> list[dict]:
        with self._lock:
            return [self._agents[label].row() for label in self._order]

    def fraction(self) -> float:
        """Approximate completion over all agents × episodes × steps."""
        with self._lock:
            total = 0.0
            done = 0.0
            for agent in self._agents.values():
                length = max(int(agent.episode_length), 1)
                units = float(agent.n_episodes * length)
                total += units
                if agent.status in {"done", "error"}:
                    done += units
                elif agent.status == "running":
                    ep_done = max(int(agent.episode) - 1, 0)
                    done += ep_done * length + min(int(agent.step), length)
                elif agent.status == "training":
                    done += 0.0
            return min(done / total, 1.0) if total else 1.0

    def summary_message(self) -> str:
        with self._lock:
            running = [a for a in self._agents.values() if a.status == "running"]
            done = sum(1 for a in self._agents.values() if a.status == "done")
            n = len(self._agents)
            if not running:
                return f"{done}/{n} contestants finished"
            bits = []
            for a in running[:3]:
                bits.append(
                    f"{a.label}: ep {a.episode}/{a.n_episodes} step {a.step}/{a.episode_length}"
                )
            extra = f" (+{len(running) - 3} more)" if len(running) > 3 else ""
            return f"{done}/{n} done · " + " · ".join(bits) + extra


def _evaluate_contestant(
    *,
    label: str,
    agent_name: str,
    agent_params: dict,
    mode: str,
    panel,
    env_cfg: EnvConfig,
    n_assets: int,
    n_episodes: int,
    episode_length: int,
    seed: int,
    n_trials: int,
    board: EvalProgressBoard | None = None,
    dataset_root: Path | None = None,
    run_id: str | None = None,
    symbols: list[str] | None = None,
) -> tuple[str, AgentRecord, list, np.ndarray]:
    """Run one contestant on a private env (safe for thread-pool workers)."""
    from alphaduel.agents.llm.trajectory import LLMDatasetWriter

    try:
        if board is not None:
            board.update(label, status="training" if agent_name in _GENERATIVE else "running",
                         episode=0, step=0, detail="building env")

        env = _build_env(mode, panel, env_cfg, seed)
        agent = _make_agent_from_spec(agent_name, seed, agent_params)
        if agent_name in _GENERATIVE:
            if board is not None:
                board.update(label, status="training", detail="training GenPortfolio")
            agent.train(env)

        writer = None
        if (
            dataset_root is not None
            and run_id is not None
            and agent_name in LLM_AGENTS
        ):
            writer = LLMDatasetWriter(
                root=Path(dataset_root),
                run_id=run_id,
                agent_label=label,
                agent_name=agent_name,
                agent_params=agent_params,
                mode=mode,
                symbols=list(symbols or []),
                seed=seed,
            )

        episodes: list[dict] = []
        for j in range(n_episodes):
            if board is not None:
                board.update(
                    label,
                    status="running",
                    episode=j + 1,
                    step=0,
                    detail=f"episode {j + 1}/{n_episodes}",
                )

            def _on_step(step: int, _done: bool, *, _ep=j + 1) -> None:
                if board is not None:
                    board.update(
                        label,
                        status="running",
                        episode=_ep,
                        step=step,
                        detail=f"episode {_ep}/{n_episodes} · step {step}/{episode_length}",
                    )

            ep = _run_episode(env, agent, n_assets, seed + j, on_step=_on_step)
            episodes.append(ep)

            if writer is not None:
                traces = [t for t in (ep.get("llm_traces") or []) if isinstance(t, dict)]
                n_tx = int((ep["fills"] != 0).sum())
                ep_metrics = compute_metrics(
                    ep["equity"],
                    n_transactions=n_tx,
                    total_reward=float(ep["rewards"].sum()),
                    n_trials=n_trials,
                )
                writer.write_episode(
                    episode_index=j,
                    episode_seed=seed + j,
                    steps=traces,
                    episode_metrics=ep_metrics,
                )

        replay = episodes[0]
        episode_metrics = _per_episode_metrics(episodes, n_trials=n_trials)
        metrics = {
            k: float(np.mean([m[k] for m in episode_metrics])) for k in episode_metrics[0]
        }
        record = AgentRecord(
            name=label,
            equity=replay["equity"],
            rewards=replay["rewards"],
            exposure=replay["exposure"],
            weights=replay["weights"],
            fills=replay["fills"],
            costs=replay["costs"],
            thoughts=replay["thoughts"],
            metrics=metrics,
            parse_oks=replay["parse_oks"],
            llm_actions=replay["llm_actions"],
            episode_metrics=episode_metrics,
        )
        if board is not None:
            board.update(
                label,
                status="done",
                episode=n_episodes,
                step=episode_length,
                detail="done",
            )
        return label, record, replay["timestamps"], replay["prices"]
    except Exception as exc:
        if board is not None:
            board.update(label, status="error", detail=str(exc))
        raise


def run_evaluation(
    mode: str,
    contestants: list[dict] | tuple[dict, ...],
    n_assets: int = 4,
    n_steps: int | None = 500,
    episode_length: int = 120,
    n_episodes: int = 20,
    seed: int = 7,
    initial_cash: float = 100_000.0,
    commission_bps: float = 1.0,
    half_spread_bps: float = 2.0,
    reward_kind: str = "log_return",
    use_mock: bool = False,
    drift: float = 0.0004,
    vol: float = 0.012,
    progress=None,
    max_workers: int = EVAL_MAX_WORKERS,
    save_llm_dataset: bool = True,
    dataset_root: str | Path | None = None,
    symbols: list[str] | tuple[str, ...] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> BenchmarkResult:
    """Compare independently configured agents (including multiple LLM Vanilla instances).

    Each contestant is a dict::

        {"label": "LLM · openai · gpt-5.4-mini", "agent": "llm_vanilla", "params": {...}}

    ``params`` are passed to :func:`build_agent` (nested ``llm={...}`` for LLM agents).
    Labels must be unique; they become keys in :class:`BenchmarkResult.agents`.

    Contestants run in a thread pool (default ``max_workers=3``, hard-capped) so several
    LLM agents can call providers concurrently. Each worker builds its own env.

    When ``save_llm_dataset`` is True, every LLM contestant writes per-episode JSON traces
    and an ``sft.jsonl`` under ``dataset_root/<run_id>/`` (default ``datasets/llm_sft``).

    ``progress(fraction, message, agents=...)`` is called from the main thread. ``agents`` is
    a list of per-contestant status dicts (episode, step, status).
    """
    from alphaduel.agents.llm.trajectory import (
        DEFAULT_DATASET_ROOT,
        new_run_id,
        write_run_manifest,
    )

    if not contestants:
        raise ValueError("run_evaluation requires at least one contestant")

    params = {
        "mode": mode,
        "n_assets": n_assets,
        "n_steps": n_steps,
        "episode_length": episode_length,
        "seed": seed,
        "initial_cash": initial_cash,
        "commission_bps": commission_bps,
        "half_spread_bps": half_spread_bps,
        "reward_kind": reward_kind,
        "use_mock": use_mock,
        "drift": drift,
        "vol": vol,
        "symbols": symbols,
        "start_date": start_date,
        "end_date": end_date,
    }
    panel, symbols = _panel_for(params)
    n = len(symbols)
    env_cfg = _env_config(params)
    n_trials = len(contestants)
    workers = max(1, min(int(max_workers), EVAL_MAX_WORKERS, n_trials))

    specs: list[tuple[str, str, dict]] = []
    seen: set[str] = set()
    for spec in contestants:
        label = str(
            spec.get("label") or default_contestant_label(spec["agent"], spec.get("params"))
        )
        if label in seen:
            raise ValueError(f"Duplicate contestant label: {label!r}")
        seen.add(label)
        specs.append((label, str(spec["agent"]), dict(spec.get("params") or {})))

    board = EvalProgressBoard(
        [label for label, _, _ in specs],
        n_episodes=n_episodes,
        episode_length=episode_length,
    )

    run_id = None
    run_dataset_dir = None
    has_llm = any(name in LLM_AGENTS for _, name, _ in specs)
    if save_llm_dataset and has_llm:
        run_id = new_run_id()
        base = Path(dataset_root) if dataset_root else DEFAULT_DATASET_ROOT
        run_dataset_dir = base / run_id
        write_run_manifest(
            run_dataset_dir,
            run_id=run_id,
            mode=mode,
            symbols=symbols,
            seed=seed,
            n_episodes=n_episodes,
            episode_length=episode_length,
            contestants=[
                {"label": label, "agent": agent_name, "params": agent_params}
                for label, agent_name, agent_params in specs
            ],
        )

    def _emit(message: str | None = None) -> None:
        if progress is None:
            return
        progress(
            board.fraction(),
            message or board.summary_message(),
            agents=board.snapshot(),
        )

    result = BenchmarkResult(
        mode=mode,
        symbols=symbols,
        timestamps=[],
        prices=np.empty(0),
        dataset_dir=str(run_dataset_dir) if run_dataset_dir is not None else None,
    )
    _emit(f"Starting {n_trials} contestants ({workers} workers)…")

    finished: dict[str, tuple[AgentRecord, list, np.ndarray]] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                _evaluate_contestant,
                label=label,
                agent_name=agent_name,
                agent_params=agent_params,
                mode=mode,
                panel=panel,
                env_cfg=env_cfg,
                n_assets=n,
                n_episodes=n_episodes,
                episode_length=episode_length,
                seed=seed,
                n_trials=n_trials,
                board=board,
                dataset_root=run_dataset_dir,
                run_id=run_id,
                symbols=symbols,
            ): label
            for label, agent_name, agent_params in specs
        }
        pending = set(futures)
        while pending:
            completed, pending = wait(
                pending, timeout=0.25, return_when=FIRST_COMPLETED
            )
            for fut in completed:
                label, record, timestamps, prices = fut.result()
                finished[label] = (record, timestamps, prices)
            _emit()

    for label, _, _ in specs:
        record, timestamps, prices = finished[label]
        result.agents[label] = record
        if not result.timestamps:
            result.timestamps = timestamps
            result.prices = prices

    _emit("Done")
    return result
