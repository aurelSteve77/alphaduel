"""AlphaDuel command-line interface."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from alphaduel.config.loader import load_experiment_config, resolve_config_hash
from alphaduel.config.schema import Secrets
from alphaduel.pipeline import build_panel, download_data, evaluate_agents
from alphaduel.tracking import log_metrics, mlflow_run

app = typer.Typer(add_completion=False, help="AlphaDuel: benchmark RL vs LLM trading agents.")
console = Console()

_CONFIG_OPT = typer.Option(..., "--config", "-c", help="Path to an experiment YAML.")


@app.command()
def download(config: Path = _CONFIG_OPT) -> None:
    """Fetch and cache all data sources for an experiment."""
    cfg = load_experiment_config(config)
    secrets = Secrets()
    console.print(f"[bold]Downloading data[/bold] for {cfg.data.symbols} ...")
    download_data(cfg, secrets)
    console.print("[green]Done.[/green] Cached under", secrets.data_dir)


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

    app_path = Path(__file__).resolve().parents[2] / "dashboard" / "app.py"
    subprocess.run(["streamlit", "run", str(app_path)], check=False)


def _run_and_report(config: Path) -> None:
    cfg = load_experiment_config(config)
    secrets = Secrets()
    config_hash = resolve_config_hash(cfg)
    console.print(f"[bold]Experiment[/bold] {cfg.name} [dim]({config_hash})[/dim]")

    panel = build_panel(cfg, secrets)
    console.print(f"Panel: {panel.n_steps} steps, {panel.n_features} features.")

    with mlflow_run(cfg, secrets, config_hash):
        results = evaluate_agents(cfg, secrets, panel)
        for agent_name, summary in results.items():
            flat = {f"{agent_name}.{m}": v["mean"] for m, v in summary.items()}
            log_metrics(flat)

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
