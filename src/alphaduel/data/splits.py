"""Chronological train/val/test splits with no-lookahead assertions."""

from __future__ import annotations

import pandas as pd


def split_indices(
    dates: pd.DatetimeIndex,
    train_end: str,
    val_end: str,
    valid_from: int = 0,
) -> dict[str, tuple[int, int]]:
    """Return contiguous, chronologically ordered ``[lo, hi)`` index ranges.

    ``valid_from`` skips the feature warmup period at the start of ``train``.
    The returned splits are validated to be contiguous and non-overlapping so
    that no test/val information can leak into training.
    """
    train_end_ts = pd.Timestamp(train_end)
    val_end_ts = pd.Timestamp(val_end)
    n = len(dates)

    train_hi = int((dates <= train_end_ts).sum())
    val_hi = int((dates <= val_end_ts).sum())

    splits = {
        "train": (valid_from, train_hi),
        "val": (train_hi, val_hi),
        "test": (val_hi, n),
    }
    _assert_ordered_contiguous(splits, n, valid_from)
    return splits


def _assert_ordered_contiguous(
    splits: dict[str, tuple[int, int]], n: int, valid_from: int
) -> None:
    prev_hi = valid_from
    for name in ("train", "val", "test"):
        lo, hi = splits[name]
        if lo != prev_hi:
            raise ValueError(f"split '{name}' is not contiguous with the previous split")
        if not (lo <= hi <= n):
            raise ValueError(f"split '{name}' has invalid bounds ({lo}, {hi}) for n={n}")
        prev_hi = hi
    if prev_hi != n:
        raise ValueError("splits do not cover the full series")
