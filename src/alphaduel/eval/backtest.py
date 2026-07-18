"""Deterministic backtesting on the shared simulator.

``run_backtest`` drives a :class:`PortfolioSimulator` with a policy over a fixed
``[start_idx, end_idx)`` index range. ``walk_forward`` chains out-of-sample
evaluation over expanding folds and exposes a ``fit`` hook for future *learned*
agents (baselines pass ``fit=None``).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from ..env.portfolio import PortfolioSimulator
from .metrics import compute_metrics


@dataclass
class BacktestResult:
    name: str
    step_returns: np.ndarray
    equity_curve: np.ndarray  # [n_steps + 1], starts at budget
    turnovers: np.ndarray
    weights: np.ndarray  # [n_steps, K] target weights applied each step
    decision_index: np.ndarray  # [n_steps] time index at each decision
    value_index: np.ndarray  # [n_steps] time index the value is realized at
    metrics: dict


def run_backtest(
    policy,
    market,
    start_idx: int,
    end_idx: int,
    *,
    budget: float = 100_000.0,
    cost_bps: float = 5.0,
    max_weight: float = 0.30,
    rebalance_every: int = 1,
) -> BacktestResult:
    sim = PortfolioSimulator(
        market.prices,
        budget=budget,
        cost_bps=cost_bps,
        max_weight=max_weight,
        rebalance_every=rebalance_every,
        horizon=None,
    )
    sim.reset(start_idx)
    policy.reset()

    returns: list[float] = []
    equity: list[float] = [budget]
    turnovers: list[float] = []
    weights: list[np.ndarray] = []
    decision_index: list[int] = []
    value_index: list[int] = []

    while sim._t < end_idx - 1:
        t0 = sim._t
        target = np.asarray(policy.act(market, t0, sim.state()), dtype=float)
        delta = target - sim.weights
        result = sim.step(delta)
        returns.append(result.reward)
        equity.append(result.portfolio_value)
        turnovers.append(result.turnover)
        weights.append(sim.weights.copy())
        decision_index.append(t0)
        value_index.append(sim._t)
        if result.terminated:
            break

    metrics = compute_metrics(
        returns, periods_per_year=252.0 / rebalance_every, turnovers=turnovers
    )
    return BacktestResult(
        name=getattr(policy, "name", "policy"),
        step_returns=np.asarray(returns),
        equity_curve=np.asarray(equity),
        turnovers=np.asarray(turnovers),
        weights=np.asarray(weights) if weights else np.empty((0, market.n_assets)),
        decision_index=np.asarray(decision_index, dtype=int),
        value_index=np.asarray(value_index, dtype=int),
        metrics=metrics,
    )


def walk_forward(
    make_policy: Callable[[], object],
    market,
    *,
    train_end: str,
    val_end: str,
    n_folds: int = 4,
    fit: Callable[[object, object, int, int], None] | None = None,
    **backtest_kwargs,
) -> list[BacktestResult]:
    """Expanding-window walk-forward over the out-of-sample (test) region.

    The test span after ``val_end`` is divided into ``n_folds`` consecutive folds.
    For each fold we (optionally) ``fit`` a fresh policy on everything strictly
    before the fold, then evaluate on the fold — never using future data.
    """
    splits = market.split_indices(train_end, val_end)
    test_lo, test_hi = splits["test"]
    edges = np.linspace(test_lo, test_hi, n_folds + 1, dtype=int)

    results: list[BacktestResult] = []
    for i in range(n_folds):
        fold_lo, fold_hi = int(edges[i]), int(edges[i + 1])
        if fold_hi - fold_lo < 2:
            continue
        policy = make_policy()
        if fit is not None:
            fit(policy, market, market.valid_from, fold_lo)  # train on data before the fold
        res = run_backtest(policy, market, fold_lo, fold_hi, **backtest_kwargs)
        res.name = f"{res.name}/fold{i}"
        results.append(res)
    return results
