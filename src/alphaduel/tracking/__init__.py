"""MLflow tracking helpers (self-hosted / local file store by default)."""

from __future__ import annotations

from contextlib import contextmanager

from alphaduel.config.schema import ExperimentConfig, Secrets


@contextmanager
def mlflow_run(config: ExperimentConfig, secrets: Secrets, config_hash: str):
    """Context manager yielding an active MLflow run (or a no-op if disabled/unavailable)."""
    if not config.tracking.enabled:
        yield None
        return

    try:
        import mlflow
    except ImportError:  # pragma: no cover
        yield None
        return

    mlflow.set_tracking_uri(secrets.mlflow_tracking_uri)
    mlflow.set_experiment(secrets.mlflow_experiment_name)
    run_name = config.tracking.run_name or f"{config.name}-{config_hash}"
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tags({**config.tracking.tags, "config_hash": config_hash})
        mlflow.log_params(_flatten(config.model_dump(mode="json")))
        yield run


def log_metrics(metrics: dict[str, float], prefix: str = "") -> None:
    try:
        import mlflow
    except ImportError:  # pragma: no cover
        return
    for key, value in metrics.items():
        mlflow.log_metric(f"{prefix}{key}", value)


def _flatten(d: dict, parent: str = "", sep: str = ".") -> dict[str, str]:
    out: dict[str, str] = {}
    for k, v in d.items():
        key = f"{parent}{sep}{k}" if parent else k
        if isinstance(v, dict):
            out.update(_flatten(v, key, sep))
        else:
            out[key] = str(v)
    return out
