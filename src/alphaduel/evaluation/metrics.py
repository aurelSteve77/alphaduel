"""Performance metrics (pure functions of an equity curve / return series).

All annualization assumes daily bars (252 trading days). Metrics are risk-adjusted and
include a deflated Sharpe ratio to account for multiple-testing over a config sweep.
"""

from __future__ import annotations

import numpy as np

_TRADING_DAYS = 252


def _returns(equity: np.ndarray) -> np.ndarray:
    equity = np.asarray(equity, dtype=np.float64)
    return equity[1:] / equity[:-1] - 1.0


def sharpe(returns: np.ndarray, rf: float = 0.0) -> float:
    excess = returns - rf / _TRADING_DAYS
    sd = excess.std(ddof=1)
    return float(np.sqrt(_TRADING_DAYS) * excess.mean() / sd) if sd > 1e-12 else 0.0


def sortino(returns: np.ndarray, rf: float = 0.0) -> float:
    excess = returns - rf / _TRADING_DAYS
    downside = excess[excess < 0]
    dd = downside.std(ddof=1) if downside.size > 1 else 0.0
    return float(np.sqrt(_TRADING_DAYS) * excess.mean() / dd) if dd > 1e-12 else 0.0


def max_drawdown(equity: np.ndarray) -> float:
    equity = np.asarray(equity, dtype=np.float64)
    peak = np.maximum.accumulate(equity)
    return float(((peak - equity) / peak).max()) if equity.size else 0.0


def deflated_sharpe(observed_sharpe: float, n_trials: int, n_obs: int) -> float:
    """Deflated Sharpe ratio (Bailey & López de Prado), simplified.

    Penalizes the observed Sharpe for the number of configurations tried. Returns a
    haircut Sharpe; a value near/below 0 means the result is likely a multiple-testing
    artifact.
    """
    if n_trials < 1 or n_obs < 2:
        return observed_sharpe
    # Expected max of ``n_trials`` standard normals (approx) as the benchmark hurdle.
    euler = 0.5772156649
    z = (1 - euler) * _norm_ppf(1 - 1.0 / n_trials) + euler * _norm_ppf(
        1 - 1.0 / (n_trials * np.e)
    )
    hurdle = z / np.sqrt(n_obs)
    return float(observed_sharpe - hurdle)


def _norm_ppf(p: float) -> float:
    # Acklam's rational approximation to the inverse normal CDF (no scipy dependency).
    a = [-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
         1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00]
    b = [-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
         6.680131188771972e01, -1.328068155288572e01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
         -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
         3.754408661907416e00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = np.sqrt(-2 * np.log(p))
        num = (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5])
        den = ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
        return num / den
    if p > phigh:
        q = np.sqrt(-2 * np.log(1 - p))
        num = (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5])
        den = ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
        return -num / den
    q = p - 0.5
    r = q * q
    num = (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q
    den = (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)
    return num / den


def compute_metrics(
    equity: np.ndarray,
    n_transactions: int,
    total_reward: float,
    rf: float = 0.0,
    n_trials: int = 1,
) -> dict[str, float]:
    equity = np.asarray(equity, dtype=np.float64)
    rets = _returns(equity)
    total_return = float(equity[-1] / equity[0] - 1.0) if equity.size else 0.0
    n = max(rets.size, 1)
    ann_return = float((1 + total_return) ** (_TRADING_DAYS / n) - 1.0)
    sr = sharpe(rets, rf)
    mdd = max_drawdown(equity)
    return {
        "total_return": total_return,
        "annualized_return": ann_return,
        "volatility": float(rets.std(ddof=1) * np.sqrt(_TRADING_DAYS)) if rets.size > 1 else 0.0,
        "sharpe": sr,
        "sortino": sortino(rets, rf),
        "max_drawdown": mdd,
        "calmar": float(ann_return / mdd) if mdd > 1e-9 else 0.0,
        "deflated_sharpe": deflated_sharpe(sr, n_trials, rets.size),
        "n_transactions": float(n_transactions),
        "total_reward": float(total_reward),
    }
