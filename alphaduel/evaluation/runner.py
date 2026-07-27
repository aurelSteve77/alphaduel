"""Run a strategy through :class:`~alphaduel.environments.TradingEnv`."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from alphaduel.environments.config import TradingEnvConfig
from alphaduel.environments.trading_env import TradingEnv
from alphaduel.evaluation.config import StrategyRunConfig
from alphaduel.evaluation.metrics import EvalResult, TradeRecord, compute_metrics
from alphaduel.evaluation.registry import build_strategy
from alphaduel.logger import get_logger

log = get_logger(__name__)

StepCallback = Callable[["StepEvent", EvalResult], None]


@dataclass
class StepEvent:
    """One trading decision after ``env.step`` has been applied."""

    step: int
    decision_date: str
    mark_date: str
    action: list[int]
    executed: list[int]
    trades: list[TradeRecord]
    reward: float
    pnl: float
    fees: float
    portfolio_value: float
    cash: float
    holdings: list[float]
    rationale: str
    done: bool


def build_env(
    run: StrategyRunConfig,
    *,
    prices: pd.DataFrame | None = None,
    news: pd.DataFrame | None = None,
) -> TradingEnv:
    """Construct the trading env from project defaults + strategy ``env`` overrides."""
    env_cfg = TradingEnvConfig.from_project_yaml(**run.env)
    return TradingEnv(config=env_cfg, prices=prices, news=news)


def _expand_trades(
    *,
    date: str,
    tickers: list[str],
    executed: np.ndarray,
    prices: np.ndarray,
    fee_rate: float,
) -> list[TradeRecord]:
    trades: list[TradeRecord] = []
    for i, qty in enumerate(np.asarray(executed, dtype=np.int64).tolist()):
        if qty == 0:
            continue
        price = float(prices[i])
        notional = abs(qty) * price
        fee = notional * fee_rate
        trades.append(
            TradeRecord(
                date=date,
                ticker=tickers[i],
                shares=int(qty),
                price=price,
                fee=float(fee),
                side="buy" if qty > 0 else "sell",
            )
        )
    return trades


def _strategy_rationale(strategy: Any) -> str:
    """Best-effort rationale text from LLM (or other) strategies."""
    decision = getattr(strategy, "last_decision", None)
    if decision is not None:
        rationale = getattr(decision, "rationale", "") or ""
        if rationale.strip():
            return str(rationale).strip()
    raw = getattr(strategy, "last_raw", None)
    if raw:
        return str(raw).strip()
    return ""


def _seed_result(
    *,
    name: str,
    kind: str,
    seed: int,
    tickers: list[str],
    info: dict[str, Any],
) -> EvalResult:
    result = EvalResult(name=name, kind=kind, seed=seed, tickers=tickers)
    result.dates.append(str(info["date"]))
    result.portfolio_values.append(float(info["portfolio_value"]))
    result.cash.append(float(info["cash"]))
    result.holdings.append(np.asarray(info["holdings"], dtype=np.float64).tolist())
    result.rewards.append(0.0)
    result.pnls.append(0.0)
    result.fees.append(0.0)
    return result


def _finalize(result: EvalResult, initial_cash: float) -> EvalResult:
    result.metrics = compute_metrics(
        initial_cash=initial_cash,
        portfolio_values=result.portfolio_values,
        rewards=result.rewards[1:],
        trades=result.trades,
    )
    return result


def iter_episode(
    strategy: Any,
    env: TradingEnv,
    *,
    seed: int = 77,
    name: str = "run",
    kind: str = "unknown",
) -> Iterator[tuple[StepEvent, EvalResult]]:
    """Yield ``(StepEvent, EvalResult)`` after every trading day (live-friendly)."""
    if hasattr(strategy, "reset"):
        strategy.reset()

    obs, info = env.reset(seed=seed)
    tickers = list(info.get("tickers", env.tickers))
    result = _seed_result(name=name, kind=kind, seed=seed, tickers=tickers, info=info)

    done = False
    step = 0
    while not done:
        action = np.asarray(strategy.act(obs, info), dtype=np.int32).reshape(-1)
        rationale = _strategy_rationale(strategy)
        decision_date = str(info["date"])
        prices_t = env.price_array[env._t].copy()

        obs, reward, terminated, truncated, info = env.step(action)
        done = bool(terminated or truncated)
        step += 1

        executed = np.asarray(info.get("executed", np.zeros(len(tickers))), dtype=np.int64)
        day_trades = _expand_trades(
            date=decision_date,
            tickers=tickers,
            executed=executed,
            prices=prices_t,
            fee_rate=env.config.transaction_fee,
        )
        result.trades.extend(day_trades)
        result.actions.append(action.astype(int).tolist())
        result.rationales.append(rationale)
        result.decision_dates.append(decision_date)

        result.dates.append(str(info["date"]))
        result.portfolio_values.append(float(info["portfolio_value"]))
        result.cash.append(float(info["cash"]))
        result.holdings.append(np.asarray(info["holdings"], dtype=np.float64).tolist())
        result.rewards.append(float(reward))
        result.pnls.append(float(info.get("pnl", 0.0)))
        result.fees.append(float(info.get("fees_paid", 0.0)))

        event = StepEvent(
            step=step,
            decision_date=decision_date,
            mark_date=str(info["date"]),
            action=action.astype(int).tolist(),
            executed=executed.astype(int).tolist(),
            trades=day_trades,
            reward=float(reward),
            pnl=float(info.get("pnl", 0.0)),
            fees=float(info.get("fees_paid", 0.0)),
            portfolio_value=float(info["portfolio_value"]),
            cash=float(info["cash"]),
            holdings=np.asarray(info["holdings"], dtype=np.float64).tolist(),
            rationale=rationale,
            done=done,
        )
        if done:
            _finalize(result, env.config.initial_cash)
            log.info(
                "Episode done name=%s steps=%d final_pnl=%.2f fees=%.2f tx=%d",
                name,
                result.metrics.n_steps if result.metrics else step,
                result.metrics.final_pnl if result.metrics else 0.0,
                result.metrics.total_fees if result.metrics else 0.0,
                result.metrics.n_transactions if result.metrics else 0,
            )
        yield event, result


def run_episode(
    strategy: Any,
    env: TradingEnv,
    *,
    seed: int = 77,
    name: str = "run",
    kind: str = "unknown",
    on_step: StepCallback | None = None,
) -> EvalResult:
    """Roll out ``strategy`` until the episode ends; collect a full trajectory."""
    result: EvalResult | None = None
    for event, result in iter_episode(
        strategy, env, seed=seed, name=name, kind=kind
    ):
        if on_step is not None:
            on_step(event, result)
    assert result is not None
    if result.metrics is None:
        _finalize(result, env.config.initial_cash)
    return result


def evaluate(
    run: StrategyRunConfig,
    *,
    prices: pd.DataFrame | None = None,
    news: pd.DataFrame | None = None,
    strategy: Any | None = None,
    on_step: StepCallback | None = None,
) -> EvalResult:
    """Build env + strategy from ``run`` (unless injected) and evaluate."""
    env = build_env(run, prices=prices, news=news)
    agent = strategy if strategy is not None else build_strategy(env, run)
    return run_episode(
        agent,
        env,
        seed=run.eval.seed,
        name=run.name,
        kind=run.kind,
        on_step=on_step,
    )
