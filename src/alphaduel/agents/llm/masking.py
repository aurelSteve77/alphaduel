"""Anonymize tickers shown to LLM agents (ASS1, ASS2, …) to limit pretraining leakage.

Real symbols stay in the env / dashboard. Only the textual market brief and action
keys the model sees/emits are masked; actions are unmasked before execution.
"""

from __future__ import annotations

from dataclasses import dataclass


def masked_symbol(index: int) -> str:
    """Return the anonymous label for asset position ``index`` (0-based → ASS1)."""
    if index < 0:
        raise ValueError(f"asset index must be >= 0, got {index}")
    return f"ASS{index + 1}"


@dataclass(frozen=True)
class SymbolMask:
    """Bidirectional map between real tickers and anonymous ASS* labels.

    Ordering follows the env symbol list (index 0 → ASS1), so positions stay aligned
    with feature rows and weight vectors.
    """

    real: tuple[str, ...]
    masked: tuple[str, ...]
    _real_to_masked: dict[str, str]
    _masked_to_real: dict[str, str]

    @classmethod
    def from_symbols(cls, symbols: list[str] | tuple[str, ...]) -> SymbolMask:
        real = tuple(str(s) for s in symbols)
        if not real:
            raise ValueError("SymbolMask requires at least one symbol")
        masked = tuple(masked_symbol(i) for i in range(len(real)))
        real_to_masked = {r.upper(): m for r, m in zip(real, masked)}
        masked_to_real = {m.upper(): r for r, m in zip(real, masked)}
        # If a real ticker somehow equals an ASS* label, prefer positional ASS mapping
        # for model I/O; real→masked still uses the assigned ASS label.
        return cls(
            real=real,
            masked=masked,
            _real_to_masked=real_to_masked,
            _masked_to_real=masked_to_real,
        )

    def mask_ticker(self, ticker: str) -> str | None:
        return self._real_to_masked.get(str(ticker).upper())

    def unmask_ticker(self, ticker: str) -> str | None:
        return self._masked_to_real.get(str(ticker).upper())

    def unmask_actions(self, actions: dict[str, int]) -> dict[str, int]:
        """Map ASS* action keys → real tickers. Unknown keys are dropped."""
        out: dict[str, int] = {}
        for key, delta in actions.items():
            real = self.unmask_ticker(str(key))
            if real is None:
                continue
            out[real] = int(delta)
        return out

    def mask_actions(self, actions: dict[str, int]) -> dict[str, int]:
        """Map real ticker action keys → ASS* (for tests / display)."""
        out: dict[str, int] = {}
        for key, delta in actions.items():
            masked = self.mask_ticker(str(key))
            if masked is None:
                continue
            out[masked] = int(delta)
        return out
