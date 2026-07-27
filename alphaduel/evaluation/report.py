"""Numerical summary + matplotlib charts for an :class:`EvalResult`."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from alphaduel.evaluation.metrics import EvalResult
from alphaduel.logger import get_logger

log = get_logger(__name__)


def format_metrics_table(result: EvalResult) -> str:
    """Human-readable metrics block for the terminal."""
    m = result.metrics
    if m is None:
        return f"[{result.name}] no metrics"
    sharpe = f"{m.sharpe:.3f}" if m.sharpe is not None else "n/a"
    lines = [
        f"=== Evaluation: {result.name} ({result.kind}) ===",
        f"seed                 : {result.seed}",
        f"steps                : {m.n_steps}",
        f"initial cash         : {m.initial_cash:,.2f}",
        f"final portfolio value: {m.final_value:,.2f}",
        f"final PnL            : {m.final_pnl:,.2f}",
        f"total return         : {m.total_return:.2%}",
        f"max drawdown         : {m.max_drawdown:.2%}",
        f"Sharpe (ann.)        : {sharpe}",
        f"mean reward          : {m.mean_reward:.6f}",
        f"sum reward           : {m.sum_reward:.6f}",
        f"hit rate             : {m.hit_rate:.2%}",
        f"transactions         : {m.n_transactions}",
        f"total fees           : {m.total_fees:,.2f}",
    ]
    return "\n".join(lines)


def print_report(result: EvalResult) -> None:
    print(format_metrics_table(result))
    trades = result.trades_frame()
    if not trades.empty:
        print("\n--- Trades (head) ---")
        print(trades.head(20).to_string(index=False))
        if len(trades) > 20:
            print(f"... ({len(trades) - 20} more)")


def save_artifacts(result: EvalResult, output_dir: Path | str) -> dict[str, Path]:
    """Persist metrics JSON + equity/trades CSVs. Returns written paths."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    metrics_path = out / "metrics.json"
    payload: dict[str, Any] = {
        "name": result.name,
        "kind": result.kind,
        "seed": result.seed,
        "tickers": result.tickers,
        "metrics": result.metrics.as_dict() if result.metrics else None,
    }
    metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    paths["metrics"] = metrics_path

    equity_path = out / "equity.csv"
    result.equity_frame().to_csv(equity_path, index=False)
    paths["equity"] = equity_path

    trades_path = out / "trades.csv"
    result.trades_frame().to_csv(trades_path, index=False)
    paths["trades"] = trades_path

    log.info("Wrote eval artifacts to %s", out)
    return paths


def plot_evaluation(
    result: EvalResult,
    *,
    output_dir: Path | str | None = None,
    show: bool = False,
    save: bool = True,
) -> list[Path]:
    """Plot portfolio value, reward, cash and cumulative fees."""
    import matplotlib

    if not show:
        matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    if not result.dates:
        log.warning("No trajectory to plot for %s", result.name)
        return []

    dates = result.dates
    # Align series: rewards/pnls/fees after reset row still length-matched.
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    fig.suptitle(f"{result.name} ({result.kind})", fontsize=13)

    axes[0].plot(dates, result.portfolio_values, color="#1f4e79", lw=1.8)
    axes[0].set_ylabel("Portfolio value")
    axes[0].grid(True, alpha=0.3)
    if result.metrics is not None:
        axes[0].axhline(
            result.metrics.initial_cash,
            color="#888888",
            ls="--",
            lw=1,
            label="initial cash",
        )
        axes[0].legend(loc="best", fontsize=8)

    axes[1].plot(dates, result.rewards, color="#2a9d8f", lw=1.2, label="reward")
    axes[1].plot(
        dates,
        _cumulative(result.rewards),
        color="#e76f51",
        lw=1.4,
        label="cumulative reward",
    )
    axes[1].set_ylabel("Reward")
    axes[1].legend(loc="best", fontsize=8)
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(dates, result.cash, color="#264653", lw=1.4, label="cash")
    axes[2].plot(
        dates,
        _cumulative(result.fees),
        color="#e9c46a",
        lw=1.4,
        label="cumulative fees",
    )
    axes[2].set_ylabel("Cash / fees")
    axes[2].set_xlabel("Date")
    axes[2].legend(loc="best", fontsize=8)
    axes[2].grid(True, alpha=0.3)

    _sparse_xticks(axes[2], dates)
    fig.autofmt_xdate()
    fig.tight_layout()

    written: list[Path] = []
    if save and output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        path = out / "evaluation.png"
        fig.savefig(path, dpi=140, bbox_inches="tight")
        written.append(path)
        log.info("Wrote plot %s", path)

    if show:
        plt.show()
    else:
        plt.close(fig)
    return written


def _cumulative(values: list[float]) -> list[float]:
    total = 0.0
    out: list[float] = []
    for v in values:
        total += float(v)
        out.append(total)
    return out


def _sparse_xticks(ax: Any, dates: list[str], max_ticks: int = 8) -> None:
    if not dates:
        return
    step = max(1, len(dates) // max_ticks)
    idxs = list(range(0, len(dates), step))
    if idxs[-1] != len(dates) - 1:
        idxs.append(len(dates) - 1)
    ax.set_xticks(idxs)
    ax.set_xticklabels([dates[i] for i in idxs], rotation=30, ha="right")


def render_evaluation(
    result: EvalResult,
    *,
    output_dir: Path | str,
    save_plots: bool = True,
    show_plots: bool = False,
    save_trajectory: bool = True,
) -> Path:
    """Print metrics, optionally save artefacts + plots. Returns output dir."""
    out = Path(output_dir)
    print_report(result)
    if save_trajectory:
        save_artifacts(result, out)
    if save_plots or show_plots:
        plot_evaluation(result, output_dir=out, show=show_plots, save=save_plots)
    return out
