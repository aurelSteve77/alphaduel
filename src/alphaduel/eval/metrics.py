"""Performance metrics computed from a series of per-step returns.

``periods_per_year`` is ``252 / rebalance_every`` so annualization matches the
decision cadence.
"""

from __future__ import annotations

import numpy as np


def compute_metrics(
    step_returns,
    *,
    periods_per_year: float,
    turnovers=None,
) -> dict[str, float]:
    r = np.asarray(step_returns, dtype=float)
    if r.size == 0:
        return {}

    equity = np.cumprod(1.0 + r)
    total_return = float(equity[-1] - 1.0)

    n_years = r.size / periods_per_year
    cagr = float(equity[-1] ** (1.0 / n_years) - 1.0) if n_years > 0 else float("nan")

    vol = float(r.std(ddof=1) * np.sqrt(periods_per_year)) if r.size > 1 else float("nan")
    mean_ann = float(r.mean() * periods_per_year)
    sharpe = mean_ann / vol if vol and vol > 0 else float("nan")

    downside = r[r < 0]
    dvol = (
        float(downside.std(ddof=1) * np.sqrt(periods_per_year))
        if downside.size > 1
        else float("nan")
    )
    sortino = mean_ann / dvol if dvol and dvol > 0 else float("nan")

    peak = np.maximum.accumulate(equity)
    max_drawdown = float((equity / peak - 1.0).min())

    metrics = {
        "total_return": total_return,
        "cagr": cagr,
        "volatility": vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_drawdown,
        "hit_rate": float((r > 0).mean()),
        "n_steps": int(r.size),
    }
    if turnovers is not None and len(turnovers) > 0:
        metrics["avg_turnover"] = float(np.mean(turnovers))
    return metrics
