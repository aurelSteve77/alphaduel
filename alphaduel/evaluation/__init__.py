"""Evaluation harness: YAML strategy configs → episode metrics + plots."""

from alphaduel.evaluation.config import EvalSettings, StrategyRunConfig
from alphaduel.evaluation.metrics import EvalMetrics, EvalResult, TradeRecord
from alphaduel.evaluation.registry import STRATEGY_REGISTRY, build_strategy
from alphaduel.evaluation.report import format_metrics_table, render_evaluation
from alphaduel.evaluation.runner import build_env, evaluate, run_episode

__all__ = [
    "STRATEGY_REGISTRY",
    "EvalMetrics",
    "EvalResult",
    "EvalSettings",
    "StrategyRunConfig",
    "TradeRecord",
    "build_env",
    "build_strategy",
    "evaluate",
    "format_metrics_table",
    "render_evaluation",
    "run_episode",
]
