"""Pydantic models describing a fully-resolved AlphaDuel experiment.

The schema is intentionally explicit: a single ``ExperimentConfig`` is the source of
truth for a run. Its JSON dump + hash is logged to MLflow for reproducibility.
"""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Secrets(BaseSettings):
    """Secrets loaded from the environment / ``.env`` (never committed)."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    fred_api_key: str | None = Field(default=None, alias="FRED_API_KEY")
    polygon_api_key: str | None = Field(default=None, alias="POLYGON_API_KEY")
    tiingo_api_key: str | None = Field(default=None, alias="TIINGO_API_KEY")
    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    hf_token: str | None = Field(default=None, alias="HF_TOKEN")
    mlflow_tracking_uri: str = Field(default="file:./mlruns", alias="MLFLOW_TRACKING_URI")
    mlflow_experiment_name: str = Field(default="alphaduel", alias="MLFLOW_EXPERIMENT_NAME")
    data_dir: str = Field(default="./data_cache", alias="ALPHADUEL_DATA_DIR")


# --------------------------------------------------------------------------- data


class PricesConfig(BaseModel):
    provider: Literal["yfinance", "polygon", "tiingo"] = "yfinance"
    interval: Literal["1d"] = "1d"


class MacroConfig(BaseModel):
    provider: Literal["fred"] = "fred"
    # FRED series ids exposed to the feature store, e.g. VIX, CPI, DGS10.
    series: list[str] = Field(default_factory=lambda: ["VIXCLS", "DGS10", "DGS2", "CPIAUCSL"])


class NewsConfig(BaseModel):
    provider: Literal["gdelt", "none"] = "none"  # enabled in P2


class DataConfig(BaseModel):
    symbols: list[str] = Field(default_factory=lambda: ["AAPL"])
    start: date
    end: date
    prices: PricesConfig = PricesConfig()
    macro: MacroConfig = MacroConfig()
    news: NewsConfig = NewsConfig()

    @model_validator(mode="after")
    def _check_dates(self) -> DataConfig:
        if self.end <= self.start:
            raise ValueError("data.end must be after data.start")
        return self


# ----------------------------------------------------------------------- features


class FeatureConfig(BaseModel):
    """Ablation switchboard: which feature groups the observation contains."""

    technical: bool = True
    macro: bool = False
    text: bool = False  # P2

    # Per-group parameters (kept generic; consumed by the feature registry).
    technical_windows: list[int] = Field(default_factory=lambda: [5, 10, 20])
    rsi_period: int = 14
    include_portfolio_state: bool = True


# ---------------------------------------------------------------------------- env


class CostConfig(BaseModel):
    commission_flat: float = 0.0
    commission_per_share: float = 0.005
    commission_bps: float = 1.0  # bps of notional
    half_spread_bps: float = 2.0
    slippage_impact_coef: float = 0.1  # linear impact vs fraction of ADV
    use_sqrt_impact: bool = False


class RewardConfig(BaseModel):
    kind: Literal["log_return", "differential_sharpe", "terminal_pnl"] = "log_return"
    drawdown_penalty: float = 0.0
    turnover_penalty: float = 0.0
    dsr_eta: float = 0.01  # differential Sharpe EMA rate


class EnvConfig(BaseModel):
    kind: Literal["single_asset", "multi_asset"] = "single_asset"
    initial_cash: float = 100_000.0
    episode_length: int = 60  # trading days
    execution_lag: int = 1  # decide on t, execute at t+1 open (leakage guard)
    action_space: Literal["weights", "discrete_shares"] = "weights"
    allow_short: bool = False
    costs: CostConfig = CostConfig()
    reward: RewardConfig = RewardConfig()
    random_start: bool = True


# -------------------------------------------------------------------------- agent


class AgentConfig(BaseModel):
    name: str = "buy_and_hold"
    kind: Literal["baseline", "rl", "llm", "generative"] = "baseline"
    # Free-form, agent-specific params (SB3 kwargs, LLM model id, prompt version...).
    params: dict = Field(default_factory=dict)


# ----------------------------------------------------------------------- eval/track


class SplitConfig(BaseModel):
    scheme: Literal["walk_forward", "holdout"] = "walk_forward"
    train_days: int = 504
    val_days: int = 126
    test_days: int = 126
    embargo_days: int = 5  # purge/embargo boundary
    n_folds: int = 3


class EvalConfig(BaseModel):
    n_episodes: int = 20
    bootstrap_samples: int = 1000
    risk_free_rate: float = 0.0
    splits: SplitConfig = SplitConfig()


class TrackingConfig(BaseModel):
    enabled: bool = True
    run_name: str | None = None
    tags: dict[str, str] = Field(default_factory=dict)


class LoggingConfig(BaseModel):
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_dir: str | None = "logs"  # None disables file logging
    log_file: str = "alphaduel.log"
    rich_console: bool = True


# ----------------------------------------------------------------------- top level


class ExperimentConfig(BaseModel):
    name: str = "p0_mvp"
    seed: int = 42
    data: DataConfig
    features: FeatureConfig = FeatureConfig()
    env: EnvConfig = EnvConfig()
    agents: list[AgentConfig] = Field(default_factory=lambda: [AgentConfig()])
    evaluation: EvalConfig = EvalConfig()
    tracking: TrackingConfig = TrackingConfig()
    logging: LoggingConfig = LoggingConfig()
