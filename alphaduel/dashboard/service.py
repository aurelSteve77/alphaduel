"""Dashboard helpers: build run configs and drive evaluation from the UI."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from alphaduel.configuration import Configuration
from alphaduel.evaluation.config import StrategyRunConfig
from alphaduel.evaluation.metrics import EvalResult
from alphaduel.evaluation.registry import STRATEGY_REGISTRY, build_strategy
from alphaduel.evaluation.runner import (
    StepCallback,
    build_env,
    evaluate,
    iter_episode,
)
from alphaduel.logger import get_logger

log = get_logger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_STRATEGIES_DIR = _PROJECT_ROOT / "configs" / "strategies"


def list_strategy_presets() -> list[str]:
    """Stem names of YAML files under ``configs/strategies/``."""
    if not _STRATEGIES_DIR.is_dir():
        return []
    return sorted(p.stem for p in _STRATEGIES_DIR.glob("*.yaml"))


def list_strategy_kinds() -> list[str]:
    return sorted(STRATEGY_REGISTRY)


def default_tickers() -> list[str]:
    cfg = Configuration()
    tickers = cfg.get("env.tickers") or cfg.get("data.tickers") or []
    return [str(t) for t in tickers]


def load_prices(tickers: Sequence[str], start: str, end: str) -> pd.DataFrame:
    """Load (or download) the adjusted-close panel for the UI run."""
    from alphaduel.data.download import download_ohlcv

    return download_ohlcv(list(tickers), start, end)


def build_run_config(
    *,
    name: str,
    kind: str,
    tickers: Sequence[str],
    start: str,
    end: str,
    initial_cash: float,
    transaction_fee: float,
    n_days: int,
    max_shares: int,
    include_news: bool = False,
    allow_short: bool = False,
    seed: int = 77,
    policy: dict[str, Any] | None = None,
    features: Sequence[str] | None = None,
) -> StrategyRunConfig:
    env: dict[str, Any] = {
        "start": start,
        "end": end,
        "tickers": list(tickers),
        "initial_cash": float(initial_cash),
        "transaction_fee": float(transaction_fee),
        "n_days": int(n_days),
        "max_shares": int(max_shares),
        "include_news": bool(include_news),
        "allow_short": bool(allow_short),
    }
    if features:
        env["features"] = list(features)
    return StrategyRunConfig(
        name=name,
        kind=kind,
        env=env,
        policy=dict(policy or {}),
        eval={"seed": int(seed), "output_dir": f"reports/eval/{name}", "save_plots": False},
    )


def run_from_preset(
    preset: str,
    *,
    overrides: dict[str, Any] | None = None,
    prices: pd.DataFrame | None = None,
    on_step: StepCallback | None = None,
) -> EvalResult:
    run = StrategyRunConfig.from_yaml(preset, **(overrides or {}))
    return evaluate(run, prices=prices, on_step=on_step)


def run_config(
    run: StrategyRunConfig,
    *,
    prices: pd.DataFrame | None = None,
    on_step: StepCallback | None = None,
) -> EvalResult:
    if prices is None:
        tickers = run.env.get("tickers") or default_tickers()
        start = str(run.env.get("start") or Configuration().get("env.start", "2020-01-01"))
        end = str(run.env.get("end") or Configuration().get("env.end", "2025-12-31"))
        prices = load_prices(tickers, start, end)
    return evaluate(run, prices=prices, on_step=on_step)


def iter_live(
    run: StrategyRunConfig,
    *,
    prices: pd.DataFrame | None = None,
):
    """Yield live step events for the Streamlit Live Run page."""
    if prices is None:
        tickers = run.env.get("tickers") or default_tickers()
        start = str(run.env.get("start") or Configuration().get("env.start", "2020-01-01"))
        end = str(run.env.get("end") or Configuration().get("env.end", "2025-12-31"))
        prices = load_prices(tickers, start, end)
    env = build_env(run, prices=prices)
    strategy = build_strategy(env, run)
    yield from iter_episode(
        strategy,
        env,
        seed=run.eval.seed,
        name=run.name,
        kind=run.kind,
    )


def metrics_comparison_table(results: Sequence[EvalResult]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for result in results:
        m = result.metrics
        if m is None:
            continue
        rows.append(
            {
                "strategy": result.name,
                "kind": result.kind,
                "final_value": m.final_value,
                "final_pnl": m.final_pnl,
                "total_return": m.total_return,
                "max_drawdown": m.max_drawdown,
                "sharpe": m.sharpe,
                "hit_rate": m.hit_rate,
                "n_transactions": m.n_transactions,
                "total_fees": m.total_fees,
                "n_steps": m.n_steps,
            }
        )
    return pd.DataFrame(rows)
