"""Run a strategy through :class:`~alphaduel.environments.TradingEnv`."""

from __future__ import annotations

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
        # Attribute fee proportionally when info only has the day total —
        # here we recompute per-leg from the known fee rate.
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


def run_episode(
    strategy: Any,
    env: TradingEnv,
    *,
    seed: int = 77,
    name: str = "run",
    kind: str = "unknown",
) -> EvalResult:
    """Roll out ``strategy`` until the episode ends; collect a full trajectory."""
    if hasattr(strategy, "reset"):
        strategy.reset()

    obs, info = env.reset(seed=seed)
    tickers = list(info.get("tickers", env.tickers))
    result = EvalResult(
        name=name,
        kind=kind,
        seed=seed,
        tickers=tickers,
    )
    # Record the pre-trade starting state.
    result.dates.append(str(info["date"]))
    result.portfolio_values.append(float(info["portfolio_value"]))
    result.cash.append(float(info["cash"]))
    result.holdings.append(np.asarray(info["holdings"], dtype=np.float64).tolist())
    result.rewards.append(0.0)
    result.pnls.append(0.0)
    result.fees.append(0.0)

    done = False
    while not done:
        action = strategy.act(obs, info)
        # Prices at decision time (before step advances the clock).
        decision_date = str(info["date"])
        prices_t = env.price_array[env._t].copy()

        obs, reward, terminated, truncated, info = env.step(action)
        done = bool(terminated or truncated)

        executed = np.asarray(info.get("executed", np.zeros(len(tickers))), dtype=np.int64)
        day_trades = _expand_trades(
            date=decision_date,
            tickers=tickers,
            executed=executed,
            prices=prices_t,
            fee_rate=env.config.transaction_fee,
        )
        result.trades.extend(day_trades)

        result.dates.append(str(info["date"]))
        result.portfolio_values.append(float(info["portfolio_value"]))
        result.cash.append(float(info["cash"]))
        result.holdings.append(np.asarray(info["holdings"], dtype=np.float64).tolist())
        result.rewards.append(float(reward))
        result.pnls.append(float(info.get("pnl", 0.0)))
        result.fees.append(float(info.get("fees_paid", 0.0)))

    result.metrics = compute_metrics(
        initial_cash=env.config.initial_cash,
        portfolio_values=result.portfolio_values,
        rewards=result.rewards[1:],  # skip the synthetic reset row
        trades=result.trades,
    )
    log.info(
        "Episode done name=%s steps=%d final_pnl=%.2f fees=%.2f tx=%d",
        name,
        result.metrics.n_steps,
        result.metrics.final_pnl,
        result.metrics.total_fees,
        result.metrics.n_transactions,
    )
    return result


def evaluate(
    run: StrategyRunConfig,
    *,
    prices: pd.DataFrame | None = None,
    news: pd.DataFrame | None = None,
    strategy: Any | None = None,
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
    )
