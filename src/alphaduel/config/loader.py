"""Load and compose YAML configs into a validated ``ExperimentConfig``.

Supports a lightweight ``includes:`` mechanism so experiment files can compose
reusable fragments from ``configs/{data,env,features,agent}/`` without pulling in a
heavier framework. Deep-merges included fragments, then validates with Pydantic.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from alphaduel.config.schema import ExperimentConfig


def _deep_merge(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, Mapping):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config {path} must be a mapping at top level.")
    return data


def _resolve_includes(raw: dict[str, Any], root: Path) -> dict[str, Any]:
    """Resolve an optional ``includes: [rel/path.yaml, ...]`` list.

    Later includes and the file's own keys override earlier includes.
    """
    includes = raw.pop("includes", []) or []
    merged: dict[str, Any] = {}
    for rel in includes:
        fragment_path = (root / rel).resolve()
        fragment = _resolve_includes(_load_yaml(fragment_path), fragment_path.parent)
        merged = _deep_merge(merged, fragment)
    return _deep_merge(merged, raw)


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    """Load, compose (``includes``), and validate an experiment config."""
    path = Path(path).resolve()
    raw = _resolve_includes(_load_yaml(path), path.parent)
    return ExperimentConfig.model_validate(raw)


def resolve_config_hash(config: ExperimentConfig) -> str:
    """Deterministic short hash of the resolved config (for reproducibility/tracking)."""
    payload = json.dumps(config.model_dump(mode="json"), sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
