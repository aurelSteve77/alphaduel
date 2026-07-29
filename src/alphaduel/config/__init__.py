"""Configuration: Pydantic schema + YAML/env loader."""

from alphaduel.config.loader import load_experiment_config, resolve_config_hash
from alphaduel.config.schema import (
    AgentConfig,
    DataConfig,
    EnvConfig,
    ExperimentConfig,
    FeatureConfig,
    Secrets,
    TrackingConfig,
)

__all__ = [
    "AgentConfig",
    "DataConfig",
    "EnvConfig",
    "ExperimentConfig",
    "FeatureConfig",
    "Secrets",
    "TrackingConfig",
    "load_experiment_config",
    "resolve_config_hash",
]
