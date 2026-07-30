"""Domain-specific tokenizer for generative portfolio construction (SPEC §15.3).

Mirrors GenPage's custom tokenization: each security and sleeve is a single token, and
continuous quant signals are bucketized into discrete tokens. A book (the "response") is a
sequence of ``[Sleeve] [Security] [WeightBucket]`` triples.

LEAKAGE NOTE: ``QuantBucketizer`` bin edges MUST be fit on the training window only. Fitting
edges on the full series leaks the future distribution into past observations. See SPEC §4.1.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Structural / segment tokens. Kept explicit so business rules map to token-level masks.
SPECIAL_TOKENS: list[str] = [
    "[PAD]", "[BOS]", "[EOS]", "[SEP]",
    "[PRICES]", "[FACTORS]", "[MACRO]", "[HOLDINGS]",
    "[SLEEVE]", "[BUY]", "[SELL]", "[HOLD]",
    "[SECURITY_FALLBACK]", "[SLEEVE_FALLBACK]",
]


@dataclass
class SecurityVocab:
    """Maps security symbols to token ids, with a fallback token for cold-start names."""

    symbols: list[str]
    fallback: str = "[SECURITY_FALLBACK]"

    def __post_init__(self) -> None:
        self._sym_to_id = {s: i for i, s in enumerate(self.symbols)}
        self.fallback_id = len(self.symbols)

    def encode(self, symbol: str) -> int:
        return self._sym_to_id.get(symbol, self.fallback_id)

    def decode(self, token_id: int) -> str:
        if token_id == self.fallback_id:
            return self.fallback
        return self.symbols[token_id]

    @property
    def size(self) -> int:
        return len(self.symbols) + 1  # + fallback


@dataclass
class QuantBucketizer:
    """Quantile bucketization of continuous features. Fit on TRAIN data only."""

    n_buckets: int = 16
    edges_: list[np.ndarray] = field(default_factory=list)

    def fit(self, features: np.ndarray) -> QuantBucketizer:
        """Compute per-feature quantile edges. ``features`` is ``(..., F)``."""
        flat = features.reshape(-1, features.shape[-1])
        qs = np.linspace(0.0, 1.0, self.n_buckets + 1)[1:-1]
        self.edges_ = [np.nanquantile(flat[:, f], qs) for f in range(flat.shape[-1])]
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        """Return integer bucket ids in ``[0, n_buckets)`` with the same leading shape."""
        if not self.edges_:
            raise RuntimeError("QuantBucketizer must be fit (on train data) before transform.")
        out = np.zeros(values.shape, dtype=np.int64)
        for f in range(values.shape[-1]):
            out[..., f] = np.digitize(values[..., f], self.edges_[f])
        return out

    @property
    def size(self) -> int:
        return self.n_buckets


@dataclass
class WeightBuckets:
    """Discretize a target weight in [0, 1] into K buckets (bucket -> representative weight)."""

    n_buckets: int = 10

    def encode(self, weight: float) -> int:
        return int(np.clip(round(weight * (self.n_buckets - 1)), 0, self.n_buckets - 1))

    def decode(self, token_id: int) -> float:
        return float(token_id) / (self.n_buckets - 1)

    @property
    def size(self) -> int:
        return self.n_buckets


class PortfolioTokenizer:
    """Ties the sub-tokenizers together and builds context / book token sequences.

    The vocabulary is laid out in fixed blocks so business rules and decoding map cleanly to
    id ranges: [specials][securities(+fallback)][feature-buckets][weight-buckets].
    """

    def __init__(
        self,
        symbols: list[str],
        n_feature_buckets: int = 16,
        n_weight_buckets: int = 10,
    ) -> None:
        self.vocab = SecurityVocab(symbols)
        self.bucketizer = QuantBucketizer(n_feature_buckets)
        self.weights = WeightBuckets(n_weight_buckets)

        self.special_base = 0
        self.security_base = len(SPECIAL_TOKENS)
        self.feature_base = self.security_base + self.vocab.size
        self.weight_base = self.feature_base + self.bucketizer.size
        self.vocab_size = self.weight_base + self.weights.size
        self._special_to_id = {t: i for i, t in enumerate(SPECIAL_TOKENS)}

    # --- fitting -----------------------------------------------------------

    def fit_buckets(self, train_features: np.ndarray) -> None:
        """Fit feature bucket edges on TRAIN features only (leakage guard)."""
        self.bucketizer.fit(train_features)

    # --- id helpers --------------------------------------------------------

    def special_id(self, token: str) -> int:
        return self.special_base + self._special_to_id[token]

    def security_id(self, symbol: str) -> int:
        return self.security_base + self.vocab.encode(symbol)

    def weight_id(self, weight: float) -> int:
        return self.weight_base + self.weights.encode(weight)

    # --- sequences ---------------------------------------------------------

    def build_context(self, asset_features: np.ndarray) -> list[int]:
        """Serialize one timestep's per-security features into context tokens.

        ``asset_features`` is ``(N, F)``. Layout: [BOS][FACTORS] then, per security,
        [Security_ID] followed by its bucketized feature tokens.
        """
        buckets = self.bucketizer.transform(asset_features)  # (N, F)
        seq = [self.special_id("[BOS]"), self.special_id("[FACTORS]")]
        for i, symbol in enumerate(self.vocab.symbols):
            seq.append(self.security_id(symbol))
            seq.extend(self.feature_base + int(b) for b in buckets[i])
        seq.append(self.special_id("[SEP]"))
        return seq

    def encode_book(self, weights: np.ndarray) -> list[int]:
        """Encode a target weight vector (aligned to ``vocab.symbols``) as a book sequence."""
        seq: list[int] = []
        for i, symbol in enumerate(self.vocab.symbols):
            if weights[i] <= 0:
                continue
            seq.extend(
                [self.security_id(symbol), self.weight_id(float(weights[i]))]
            )
        seq.append(self.special_id("[EOS]"))
        return seq

    def decode_book(self, token_ids: list[int]) -> np.ndarray:
        """Decode a book sequence back into a weight vector aligned to ``vocab.symbols``."""
        weights = np.zeros(len(self.vocab.symbols), dtype=np.float64)
        pending_sec: int | None = None
        for tid in token_ids:
            if self.security_base <= tid < self.feature_base:
                pending_sec = tid - self.security_base
            elif self.weight_base <= tid < self.vocab_size and pending_sec is not None:
                if pending_sec < len(self.vocab.symbols):
                    weights[pending_sec] = self.weights.decode(tid - self.weight_base)
                pending_sec = None
        total = weights.sum()
        return weights / total if total > 1.0 else weights
