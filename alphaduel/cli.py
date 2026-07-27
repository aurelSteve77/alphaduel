"""Command-line interface for alphaduel.

A single ``alphaduel`` entry point dispatches to subcommands, e.g.::

    alphaduel download --tickers AAPL MSFT

Adding a new task is a two-step change: write ``_add_<task>_args`` and
``_run_<task>`` functions, then register them in :data:`SUBCOMMANDS`.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from datetime import UTC, datetime

from alphaduel.configuration import Configuration
from alphaduel.logger import get_logger

log = get_logger(__name__)


# --------------------------------------------------------------------------- #
# download
# --------------------------------------------------------------------------- #
def _add_download_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=None,
        help="Ticker symbols (default: data.tickers from project.yaml).",
    )
    parser.add_argument(
        "--start",
        default=None,
        help="Start date YYYY-MM-DD (default: data.start from project.yaml).",
    )
    parser.add_argument(
        "--end",
        default=None,
        help="End date YYYY-MM-DD (default: data.end from project.yaml).",
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--market-only",
        action="store_true",
        help="Download only market (OHLCV) data.",
    )
    source.add_argument(
        "--news-only",
        action="store_true",
        help="Download only news / filings data.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Path to write the merged parquet (implies building the merged dataset).",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not persist the merged dataset to the processed cache.",
    )


def _run_download(args: argparse.Namespace) -> int:
    cfg = Configuration()

    tickers = args.tickers or cfg.get("data.tickers", [])
    start = args.start or cfg.get("data.start", "2000-01-01")
    end = args.end or cfg.get("data.end", datetime.now(tz=UTC).strftime("%Y-%m-%d"))

    if not tickers:
        log.error("No tickers given and data.tickers is empty in project.yaml.")
        return 1

    log.info("Download requested: %s | %s -> %s", tickers, start, end)

    if args.market_only:
        from alphaduel.data.download import download_ohlcv

        prices = download_ohlcv(tickers, start, end)
        log.info("Market panel: %d rows x %d tickers", len(prices), prices.shape[1])
        return 0

    if args.news_only:
        from alphaduel.data.download_news import download_news

        news = download_news(tickers, start, end)
        log.info("News panel: %d items", len(news))
        return 0

    from alphaduel.data.dataset import build_dataset

    save = not args.no_save or args.out is not None
    merged = build_dataset(tickers, start, end, save=save, out_path=args.out)
    log.info("Merged dataset: %d rows x %d columns", len(merged), merged.shape[1])
    return 0


# --------------------------------------------------------------------------- #
# eval
# --------------------------------------------------------------------------- #
def _add_eval_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--strategy",
        "-s",
        required=True,
        help=(
            "Strategy YAML path or bare name under configs/strategies/ "
            "(e.g. buy_and_hold or configs/strategies/llm_policy.yaml)."
        ),
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Override eval.output_dir from the strategy YAML.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override eval.seed.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display matplotlib charts interactively.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip writing/showing plots.",
    )


def _run_eval(args: argparse.Namespace) -> int:
    from alphaduel.evaluation import StrategyRunConfig, evaluate, render_evaluation

    run = StrategyRunConfig.from_yaml(args.strategy)

    # Apply CLI overrides onto the loaded eval settings.
    eval_updates: dict[str, object] = {}
    if args.out is not None:
        eval_updates["output_dir"] = args.out
    if args.seed is not None:
        eval_updates["seed"] = args.seed
    if args.show:
        eval_updates["show_plots"] = True
    if args.no_plots:
        eval_updates["save_plots"] = False
        eval_updates["show_plots"] = False
    if eval_updates:
        run = run.model_copy(update={"eval": run.eval.model_copy(update=eval_updates)})

    log.info("Evaluating strategy=%s kind=%s", run.name, run.kind)
    result = evaluate(run)
    render_evaluation(
        result,
        output_dir=run.eval.output_dir,
        save_plots=run.eval.save_plots,
        show_plots=run.eval.show_plots,
        save_trajectory=run.eval.save_trajectory,
    )
    return 0


# --------------------------------------------------------------------------- #
# subcommand registry
# --------------------------------------------------------------------------- #
Subcommand = dict[str, object]

SUBCOMMANDS: dict[str, Subcommand] = {
    "download": {
        "help": "Download market and/or news data and optionally merge them.",
        "add_args": _add_download_args,
        "run": _run_download,
    },
    "eval": {
        "help": "Evaluate a strategy YAML (metrics + equity/reward/fee plots).",
        "add_args": _add_eval_args,
        "run": _run_eval,
    },
}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="alphaduel", description="alphaduel command-line interface.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name, spec in SUBCOMMANDS.items():
        sub = subparsers.add_parser(name, help=str(spec["help"]))
        add_args = spec["add_args"]
        assert callable(add_args)
        add_args(sub)
        sub.set_defaults(_run=spec["run"])
    return parser


def main(argv: list[str] | None = None) -> int:
    """Top-level ``alphaduel`` entry point: parse and dispatch a subcommand."""
    args = _build_parser().parse_args(argv)
    run: Callable[[argparse.Namespace], int] = args._run
    return run(args)


def download_main(argv: list[str] | None = None) -> int:
    """Backwards-compatible direct entry point for the download subcommand."""
    parser = argparse.ArgumentParser(prog="alphaduel-download")
    _add_download_args(parser)
    return _run_download(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
