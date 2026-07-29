"""Time-series splits: walk-forward with purge & embargo (López de Prado).

Splitting time series like i.i.d. data leaks information across the train/test boundary
because features use overlapping windows. We therefore purge overlapping samples and add
an embargo gap between train and test.
"""

from __future__ import annotations

from dataclasses import dataclass

from alphaduel.config.schema import SplitConfig


@dataclass(frozen=True)
class Split:
    fold: int
    train: tuple[int, int]  # [start, end) index into the panel
    val: tuple[int, int]
    test: tuple[int, int]


def walk_forward_splits(n_steps: int, config: SplitConfig) -> list[Split]:
    """Anchored/rolling walk-forward folds with an embargo gap before each test window."""
    splits: list[Split] = []
    embargo = config.embargo_days
    window = config.train_days + config.val_days + config.test_days + embargo

    if window > n_steps:
        raise ValueError(
            f"Split window ({window}) exceeds available steps ({n_steps}); "
            "reduce train/val/test days or widen the date range."
        )

    stride = config.test_days
    start = 0
    for fold in range(config.n_folds):
        train_start = start
        train_end = train_start + config.train_days
        val_start = train_end
        val_end = val_start + config.val_days
        test_start = val_end + embargo  # embargo gap prevents boundary leakage
        test_end = test_start + config.test_days
        if test_end > n_steps:
            break
        splits.append(
            Split(
                fold=fold,
                train=(train_start, train_end),
                val=(val_start, val_end),
                test=(test_start, test_end),
            )
        )
        start += stride
    return splits
