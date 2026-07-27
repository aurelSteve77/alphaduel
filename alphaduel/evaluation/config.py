"""YAML configuration for a single strategy evaluation run."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_STRATEGIES_DIR = _PROJECT_ROOT / "configs" / "strategies"


class EvalSettings(BaseModel):
    """How to run and report an evaluation."""

    seed: int = Field(77, description="Episode RNG seed passed to env.reset")
    output_dir: str = Field(
        "reports/eval",
        description="Directory for metrics JSON, equity CSV and plot PNGs",
    )
    save_plots: bool = Field(True, description="Write PNG charts under output_dir")
    show_plots: bool = Field(False, description="Display charts interactively")
    save_trajectory: bool = Field(True, description="Write equity/trades CSVs")

    model_config = {"extra": "forbid"}


class StrategyRunConfig(BaseModel):
    """One ``configs/strategies/*.yaml`` file.

    ``env`` and ``policy`` are shallow overrides on top of ``project.yaml``.
    """

    name: str = Field(..., description="Run label used in reports")
    kind: str = Field(
        ...,
        description="Strategy registry key: llm_policy | random | hold | buy_and_hold",
    )
    env: dict[str, Any] = Field(
        default_factory=dict,
        description="Overrides for TradingEnvConfig.from_project_yaml",
    )
    policy: dict[str, Any] = Field(
        default_factory=dict,
        description="Strategy-specific knobs (e.g. LLMPolicyConfig fields)",
    )
    eval: EvalSettings = Field(default_factory=EvalSettings)

    model_config = {"extra": "forbid"}

    @classmethod
    def from_yaml(cls, path: Path | str, **overrides: Any) -> StrategyRunConfig:
        """Load a strategy YAML and optionally override top-level fields."""
        path = Path(path)
        if not path.is_file():
            # Allow bare names like ``llm_policy`` → configs/strategies/llm_policy.yaml
            candidate = _DEFAULT_STRATEGIES_DIR / f"{path.stem}.yaml"
            if candidate.is_file():
                path = candidate
            else:
                raise FileNotFoundError(f"Strategy config not found: {path}")

        with path.open(encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        if not isinstance(raw, dict):
            raise ValueError(f"Strategy config must be a mapping: {path}")

        if "name" not in raw:
            raw["name"] = path.stem
        raw.update({k: v for k, v in overrides.items() if v is not None})
        return cls.model_validate(raw)

    @property
    def path_slug(self) -> str:
        return self.name.replace(" ", "_").lower()
