"""AlphaDuel command-line interface."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from alphaduel.config.loader import load_experiment_config, resolve_config_hash
from alphaduel.config.schema import ExperimentConfig, Secrets
from alphaduel.log import get_logger, setup_logging
from alphaduel.pipeline import build_panel, download_data, evaluate_agents
from alphaduel.tracking import log_metrics, mlflow_run

app = typer.Typer(add_completion=False, help="AlphaDuel: benchmark RL vs LLM trading agents.")
console = Console()
log = get_logger(__name__)

_CONFIG_OPT = typer.Option(..., "--config", "-c", help="Path to an experiment YAML.")


def _load(config: Path) -> ExperimentConfig:
    """Load a config and configure logging from its ``logging`` section."""
    cfg = load_experiment_config(config)
    lg = cfg.logging
    setup_logging(
        level=lg.level, log_dir=lg.log_dir, log_file=lg.log_file, use_rich=lg.rich_console
    )
    return cfg


@app.command()
def download(config: Path = _CONFIG_OPT) -> None:
    """Fetch and cache all data sources for an experiment."""
    cfg = _load(config)
    secrets = Secrets()
    log.info("Downloading data for %s ...", cfg.data.symbols)
    download_data(cfg, secrets)
    log.info("Done. Cached under %s", secrets.data_dir)


@app.command()
def run(config: Path = _CONFIG_OPT) -> None:
    """Run and evaluate all configured agents over the data."""
    _run_and_report(config)


@app.command()
def evaluate(config: Path = _CONFIG_OPT) -> None:
    """Alias for ``run``: evaluate agents and log metrics to MLflow."""
    _run_and_report(config)


@app.command()
def dashboard() -> None:
    """Launch the Streamlit dashboard (requires the `dashboard` extra)."""
    import subprocess

    app_path = Path(__file__).resolve().parent / "dashboard" / "app.py"
    # Run from the project root so Streamlit finds .streamlit/config.toml (CWD-relative).
    project_root = Path(__file__).resolve().parents[2]
    subprocess.run(["streamlit", "run", str(app_path)], check=False, cwd=project_root)


def _run_and_report(config: Path) -> None:
    cfg = _load(config)
    secrets = Secrets()
    config_hash = resolve_config_hash(cfg)
    log.info("Experiment %s (%s)", cfg.name, config_hash)
    console.print(f"[bold]Experiment[/bold] {cfg.name} [dim]({config_hash})[/dim]")

    panel = build_panel(cfg, secrets)
    log.info("Panel built: %s steps, %s features.", panel.n_steps, panel.n_features)
    console.print(f"Panel: {panel.n_steps} steps, {panel.n_features} features.")

    with mlflow_run(cfg, secrets, config_hash):
        results = evaluate_agents(cfg, secrets, panel)
        for agent_name, summary in results.items():
            flat = {f"{agent_name}.{m}": v["mean"] for m, v in summary.items()}
            log_metrics(flat)

    log.info("Evaluation complete for %d agents.", len(results))
    _print_results(results)


def _print_results(results: dict[str, dict]) -> None:
    table = Table(title="AlphaDuel results (mean [95% CI])")
    table.add_column("agent", style="bold")
    for col in ("total_return", "sharpe", "max_drawdown", "n_transactions"):
        table.add_column(col, justify="right")
    for agent_name, summary in results.items():
        row = [agent_name]
        for col in ("total_return", "sharpe", "max_drawdown", "n_transactions"):
            s = summary.get(col, {})
            mean, lo, hi = s.get("mean", 0), s.get("ci_low", 0), s.get("ci_high", 0)
            row.append(f"{mean:.3f} [{lo:.3f},{hi:.3f}]")
        table.add_row(*row)
    console.print(table)


if __name__ == "__main__":
    app()
