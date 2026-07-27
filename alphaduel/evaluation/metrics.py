"""Episode trajectory and summary metrics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class TradeRecord:
    date: str
    ticker: str
    shares: int
    price: float
    fee: float
    side: str  # buy | sell


@dataclass
class EvalMetrics:
    initial_cash: float
    final_value: float
    final_pnl: float
    total_return: float
    n_steps: int
    n_transactions: int
    total_fees: float
    mean_reward: float
    sum_reward: float
    max_drawdown: float
    sharpe: float | None
    hit_rate: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "initial_cash": self.initial_cash,
            "final_value": self.final_value,
            "final_pnl": self.final_pnl,
            "total_return": self.total_return,
            "n_steps": self.n_steps,
            "n_transactions": self.n_transactions,
            "total_fees": self.total_fees,
            "mean_reward": self.mean_reward,
            "sum_reward": self.sum_reward,
            "max_drawdown": self.max_drawdown,
            "sharpe": self.sharpe,
            "hit_rate": self.hit_rate,
        }


@dataclass
class EvalResult:
    """Full evaluation artefact for one strategy run."""

    name: str
    kind: str
    seed: int
    dates: list[str] = field(default_factory=list)
    portfolio_values: list[float] = field(default_factory=list)
    cash: list[float] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    pnls: list[float] = field(default_factory=list)
    fees: list[float] = field(default_factory=list)
    holdings: list[list[float]] = field(default_factory=list)
    tickers: list[str] = field(default_factory=list)
    trades: list[TradeRecord] = field(default_factory=list)
    metrics: EvalMetrics | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    def equity_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "date": self.dates,
                "portfolio_value": self.portfolio_values,
                "cash": self.cash,
                "reward": self.rewards,
                "pnl": self.pnls,
                "fees": self.fees,
            }
        )

    def trades_frame(self) -> pd.DataFrame:
        if not self.trades:
            return pd.DataFrame(
                columns=["date", "ticker", "shares", "price", "fee", "side"]
            )
        return pd.DataFrame([t.__dict__ for t in self.trades])


def max_drawdown(values: list[float] | np.ndarray) -> float:
    """Peak-to-trough drawdown as a positive fraction of the peak."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return 0.0
    peaks = np.maximum.accumulate(arr)
    with np.errstate(divide="ignore", invalid="ignore"):
        dd = np.where(peaks > 0, (peaks - arr) / peaks, 0.0)
    return float(np.nanmax(dd)) if dd.size else 0.0


def sharpe_ratio(rewards: list[float] | np.ndarray, *, periods_per_year: int = 252) -> float | None:
    """Annualised Sharpe of per-step rewards (proxy for daily returns)."""
    arr = np.asarray(rewards, dtype=np.float64)
    if arr.size < 2:
        return None
    std = float(arr.std(ddof=1))
    if std < 1e-12:
        return None
    return float(np.sqrt(periods_per_year) * arr.mean() / std)


def compute_metrics(
    *,
    initial_cash: float,
    portfolio_values: list[float],
    rewards: list[float],
    trades: list[TradeRecord],
) -> EvalMetrics:
    final_value = float(portfolio_values[-1]) if portfolio_values else float(initial_cash)
    final_pnl = final_value - float(initial_cash)
    total_return = final_pnl / float(initial_cash) if initial_cash else 0.0
    rewards_arr = np.asarray(rewards, dtype=np.float64)
    n_tx = sum(1 for t in trades if t.shares != 0)
    total_fees = float(sum(t.fee for t in trades))
    hit_rate = float((rewards_arr > 0).mean()) if rewards_arr.size else 0.0
    return EvalMetrics(
        initial_cash=float(initial_cash),
        final_value=final_value,
        final_pnl=final_pnl,
        total_return=total_return,
        n_steps=len(rewards),
        n_transactions=n_tx,
        total_fees=total_fees,
        mean_reward=float(rewards_arr.mean()) if rewards_arr.size else 0.0,
        sum_reward=float(rewards_arr.sum()) if rewards_arr.size else 0.0,
        max_drawdown=max_drawdown(portfolio_values),
        sharpe=sharpe_ratio(rewards),
        hit_rate=hit_rate,
    )
