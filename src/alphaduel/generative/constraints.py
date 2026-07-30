"""Mandate constraints for constrained decoding (SPEC §15.5).

At each generation step the policy's logits are masked so only rule-compliant securities can
be selected. Because each security is a single token, mandate rules map directly to
token-level masks — the model cannot emit a non-compliant book by construction.

This is a functional skeleton covering the common rules; richer rules (dedup, category
consistency, row pinning) plug in the same way.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class MandateConstraints:
    max_position_weight: float = 1.0
    sector_caps: dict[str, float] = field(default_factory=dict)
    #: Optional per-symbol sector labels used for sector-cap enforcement.
    sectors: dict[str, str] = field(default_factory=dict)
    allow_short: bool = False

    def eligible_securities(
        self, symbols: list[str], current_weights: np.ndarray
    ) -> np.ndarray:
        """Boolean mask (len == n_symbols) of securities still allowed to receive weight."""
        mask = np.ones(len(symbols), dtype=bool)

        # Position-size cap.
        mask &= current_weights < self.max_position_weight

        # Sector caps.
        if self.sector_caps and self.sectors:
            sector_w: dict[str, float] = {}
            for i, sym in enumerate(symbols):
                sec = self.sectors.get(sym)
                if sec is not None:
                    sector_w[sec] = sector_w.get(sec, 0.0) + current_weights[i]
            for i, sym in enumerate(symbols):
                sec = self.sectors.get(sym)
                if sec in self.sector_caps and sector_w.get(sec, 0.0) >= self.sector_caps[sec]:
                    mask[i] = False

        return mask

    @staticmethod
    def apply_to_logits(logits: np.ndarray, eligible: np.ndarray) -> np.ndarray:
        """Set logits of ineligible securities to -inf (returns a masked copy)."""
        out = logits.astype(np.float64).copy()
        out[~eligible] = -np.inf
        return out
