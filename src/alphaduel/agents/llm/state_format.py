"""Render env observation + info into a structured textual market brief."""

from __future__ import annotations

from typing import Any

import numpy as np


def format_market_state(
    observation: np.ndarray,
    info: dict[str, Any],
    *,
    symbols: list[str] | None = None,
) -> str:
    """Build a human-readable, structured state string for the LLM."""
    symbols = _resolve_symbols(info, symbols)
    prices = _prices(info, len(symbols))
    shares = _shares(info, len(symbols))
    weights = _weights(info, shares, prices)
    feature_names = list(info.get("feature_names") or [])
    asset_features = info.get("asset_features")

    lines: list[str] = [
        "=== MARKET STATE ===",
        f"timestamp: {_fmt_ts(info.get('timestamp'))}",
        f"equity: {_money(info.get('equity'))}",
        f"cash: {_money(info.get('cash'))}",
        f"n_assets: {len(symbols)}",
        "",
        "=== POSITIONS ===",
        f"{'TICKER':<10} {'SHARES':>10} {'PRICE':>12} {'WEIGHT':>10}",
    ]
    for i, sym in enumerate(symbols):
        lines.append(
            f"{sym:<10} {int(shares[i]):>10d} {_money(prices[i]):>12} "
            f"{float(weights[i]):>9.2%}"
        )

    lines.extend(["", "=== FEATURES ==="])
    if asset_features is not None:
        feats = np.asarray(asset_features, dtype=np.float64)
        if feats.ndim == 1:
            feats = feats.reshape(1, -1)
        names = feature_names or [f"f{j}" for j in range(feats.shape[1])]
        header = f"{'TICKER':<10} " + " ".join(f"{n:>10}" for n in names)
        lines.append(header)
        for i, sym in enumerate(symbols):
            row = " ".join(f"{float(feats[i, j]):>10.4f}" for j in range(feats.shape[1]))
            lines.append(f"{sym:<10} {row}")
    else:
        obs = np.asarray(observation, dtype=np.float64).ravel()
        lines.append("observation_vector (flat):")
        # Keep the brief readable: show up to 24 values.
        preview = obs[:24]
        lines.append("  " + ", ".join(f"{v:.4f}" for v in preview))
        if obs.size > 24:
            lines.append(f"  ... ({obs.size - 24} more values omitted)")

    example_keys = symbols[:2] if len(symbols) >= 2 else (symbols + ["ASS2"])[:2]
    example_json = (
        '{"actions": {'
        + ", ".join(f'"{sym}": {delta}' for sym, delta in zip(example_keys, (2, -1)))
        + "}}"
    )
    lines.extend(
        [
            "",
            "=== ACTION PROTOCOL ===",
            "Decide share deltas for tickers you want to trade.",
            "Positive = buy shares, negative = sell shares, omit = hold (no trade).",
            "Use only ticker labels listed in this state.",
            "End your reply with a fenced JSON block:",
            "```json",
            example_json,
            "```",
        ]
    )
    return "\n".join(lines)


def share_deltas_to_target_weights(
    actions: dict[str, int],
    *,
    symbols: list[str],
    shares: np.ndarray,
    prices: np.ndarray,
    equity: float,
    allow_short: bool = False,
) -> np.ndarray:
    """Map sparse share-delta actions → env target weight vector in [0, 1]^N."""
    n = len(symbols)
    new_shares = np.asarray(shares, dtype=np.float64).ravel()[:n].copy()
    if new_shares.size < n:
        new_shares = np.pad(new_shares, (0, n - new_shares.size))
    px = np.asarray(prices, dtype=np.float64).ravel()[:n]
    if px.size < n:
        px = np.pad(px, (0, n - px.size), constant_values=1.0)

    sym_index = {s.upper(): i for i, s in enumerate(symbols)}
    for ticker, delta in actions.items():
        i = sym_index.get(str(ticker).upper())
        if i is None:
            continue
        new_shares[i] += int(delta)

    if not allow_short:
        new_shares = np.maximum(new_shares, 0.0)

    if equity <= 0:
        return np.zeros(n, dtype=np.float32)

    weights = (new_shares * px) / float(equity)
    weights = np.clip(weights, 0.0, 1.0)
    total = float(weights.sum())
    if total > 1.0:
        weights = weights / total
    return weights.astype(np.float32)


def _resolve_symbols(info: dict, symbols: list[str] | None) -> list[str]:
    if symbols:
        return list(symbols)
    if "symbols" in info:
        return [str(s) for s in info["symbols"]]
    return ["ASSET"]


def _prices(info: dict, n: int) -> np.ndarray:
    if "prices" in info:
        return np.asarray(info["prices"], dtype=np.float64).ravel()[:n]
    if "price" in info:
        return np.full(n, float(info["price"]), dtype=np.float64)
    return np.ones(n, dtype=np.float64)


def _shares(info: dict, n: int) -> np.ndarray:
    raw = info.get("shares", 0)
    arr = np.asarray(raw, dtype=np.float64).ravel()
    if arr.size == 0:
        return np.zeros(n, dtype=np.float64)
    if arr.size == 1 and n > 1:
        out = np.zeros(n, dtype=np.float64)
        out[0] = arr[0]
        return out
    if arr.size < n:
        return np.pad(arr, (0, n - arr.size))
    return arr[:n]


def _weights(info: dict, shares: np.ndarray, prices: np.ndarray) -> np.ndarray:
    if "weights" in info:
        w = np.asarray(info["weights"], dtype=np.float64).ravel()
        if w.size >= shares.size:
            return w[: shares.size]
    equity = float(info.get("equity") or 0.0)
    if equity <= 0:
        return np.zeros_like(shares)
    return (shares * prices) / equity


def _money(value: Any) -> str:
    try:
        return f"{float(value):,.2f}"
    except (TypeError, ValueError):
        return "n/a"


def _fmt_ts(value: Any) -> str:
    if value is None:
        return "n/a"
    return str(value)
