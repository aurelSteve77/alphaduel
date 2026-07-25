"""Configuration models for the Gymnasium trading environment."""

from __future__ import annotations

from typing import Sequence

from pydantic import BaseModel, Field, field_validator, model_validator

from alphaduel.data.features import DEFAULT_FEATURES


class TradingEnvConfig(BaseModel):
    """ knobs for :class:`~alphaduel.environments.trading_env.TradingEnv`.

    Parameters map 1:1 onto the environment constructor so the same object can
    be loaded from YAML or built in code.
    """

    start: str = Field(..., description="Inclusive episode start date YYYY-MM-DD")
    end: str = Field(..., description="Inclusive episode end date YYYY-MM-DD")
    tickers: list[str] = Field(..., min_length=1, description="Tradable universe")
    initial_cash: float = Field(100_000.0, gt=0, description="Starting cash balance")
    transaction_fee: float = Field(
        0.001,
        ge=0.0,
        description="Proportional fee rate charged on notional traded",
    )
    n_days: int | None = Field(
        None,
        gt=0,
        description="Max trading days per episode (None = use full date window)",
    )
    features: list[str] = Field(
        default_factory=lambda: list(DEFAULT_FEATURES),
        description="Feature registry names included in the observation",
    )
    include_news: bool = Field(
        False,
        description="Attach daily headlines/summaries in info and news_count feature",
    )
    max_shares: int = Field(
        10,
        ge=1,
        description="Max shares that can be bought or sold per ticker per day",
    )
    allow_short: bool = Field(False, description="Allow selling shares not held")
    reward_scale: float = Field(
        1.0,
        gt=0,
        description="Divide daily P&L by initial_cash * reward_scale",
    )

    model_config = {"extra": "forbid"}

    @field_validator("tickers")
    @classmethod
    def _normalize_tickers(cls, tickers: Sequence[str]) -> list[str]:
        cleaned = [t.strip().upper() for t in tickers if str(t).strip()]
        if not cleaned:
            raise ValueError("tickers must be non-empty")
        if len(set(cleaned)) != len(cleaned):
            raise ValueError("tickers must be unique")
        return cleaned

    @field_validator("features")
    @classmethod
    def _non_empty_features(cls, features: Sequence[str]) -> list[str]:
        if not features:
            raise ValueError("features must be non-empty")
        return list(features)

    @model_validator(mode="after")
    def _dates_ordered(self) -> TradingEnvConfig:
        if self.start > self.end:
            raise ValueError(f"start ({self.start}) must be <= end ({self.end})")
        return self

    @classmethod
    def from_project_yaml(cls, **overrides: object) -> TradingEnvConfig:
        """Build a config from ``configs/project.yaml`` with optional overrides."""
        from alphaduel.configuration import Configuration

        cfg = Configuration()
        payload = {
            "start": cfg.get("env.start") or cfg.get("data.start", "2000-01-01"),
            "end": cfg.get("env.end") or cfg.get("data.end", "2026-07-30"),
            "tickers": cfg.get("env.tickers") or cfg.get("data.tickers", []),
            "initial_cash": cfg.get("env.initial_cash", 100_000.0),
            "transaction_fee": cfg.get("env.transaction_fee", 0.001),
            "n_days": cfg.get("env.n_days"),
            "features": cfg.get("env.features", list(DEFAULT_FEATURES)),
            "include_news": cfg.get("env.include_news", False),
            "max_shares": cfg.get("env.max_shares", 10),
            "allow_short": cfg.get("env.allow_short", False),
            "reward_scale": cfg.get("env.reward_scale", 1.0),
        }
        payload.update(overrides)
        return cls.model_validate(payload)
