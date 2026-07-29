"""Evaluation: backtest runner, metrics, splits, reporting."""

from alphaduel.evaluation.backtest import Trajectory, run_episode
from alphaduel.evaluation.metrics import compute_metrics
from alphaduel.evaluation.splits import walk_forward_splits

__all__ = ["Trajectory", "compute_metrics", "run_episode", "walk_forward_splits"]
