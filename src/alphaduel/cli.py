"""Command-line entry points.

``alphaduel-baselines`` runs the rule-based baselines out-of-sample and prints a
metrics table. It uses live data when the ``data`` extra and a network are
available, and transparently falls back to synthetic data otherwise.
"""

from __future__ import annotations

import argparse

from .baselines.policies import BuyAndHold, EqualWeight, Momentum
from .config import Config, load_config
from .data.dataset import MarketData
from .eval.backtest import run_backtest

_COLUMNS = [
    "total_return",
    "cagr",
    "volatility",
    "sharpe",
    "max_drawdown",
    "hit_rate",
    "avg_turnover",
]


def _load_market(cfg: Config, synthetic: bool) -> MarketData:
    n_assets = len(cfg.data.tickers)
    # Enough business days to span the configured date range (~10 years).
    n_days = 2600
    if synthetic:
        return MarketData.from_synthetic(n_days=n_days, n_assets=n_assets, seed=cfg.seed)
    try:
        return MarketData.from_config(cfg)
    except Exception as exc:  # noqa: BLE001 - offline-friendly fallback
        print(f"[warn] live data unavailable ({exc}); using synthetic data.\n")
        return MarketData.from_synthetic(n_days=n_days, n_assets=n_assets, seed=cfg.seed)


def _print_table(rows: list[tuple[str, dict]]) -> None:
    header = "policy".ljust(16) + "".join(c.rjust(14) for c in _COLUMNS)
    print(header)
    print("-" * len(header))
    for name, metrics in rows:
        line = name.ljust(16) + "".join(
            f"{metrics.get(c, float('nan')):14.3f}" for c in _COLUMNS
        )
        print(line)


def baselines_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run alphaduel rule-based baselines OOS.")
    parser.add_argument("--config", default=None, help="Path to a YAML config file.")
    parser.add_argument("--synthetic", action="store_true", help="Force synthetic data.")
    args = parser.parse_args(argv)

    cfg = load_config(args.config) if args.config else Config()
    market = _load_market(cfg, args.synthetic)

    lo, hi = market.split_indices(cfg.data.train_end, cfg.data.val_end)["test"]
    print(
        f"Out-of-sample backtest: {market.dates[lo].date()} -> "
        f"{market.dates[hi - 1].date()} ({hi - lo} steps, {market.n_assets} assets)\n"
    )

    policies = [
        EqualWeight(cfg.env.max_weight),
        BuyAndHold(cfg.env.max_weight),
        Momentum(max_weight=cfg.env.max_weight),
    ]
    rows = []
    for policy in policies:
        res = run_backtest(
            policy,
            market,
            lo,
            hi,
            budget=cfg.env.budget,
            cost_bps=cfg.env.cost_bps,
            max_weight=cfg.env.max_weight,
            rebalance_every=cfg.env.rebalance_every,
        )
        rows.append((policy.name, res.metrics))

    _print_table(rows)


if __name__ == "__main__":
    baselines_main()
