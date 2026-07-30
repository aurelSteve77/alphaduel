"""GenPortfolio: autoregressive book-construction agent (SPEC §15).

Wires the full generative path — tokenize market context -> transformer -> constrained
greedy decode of ``[Security][WeightBucket]`` pairs -> target weight vector — into the
shared ``Agent`` interface, so it is benchmarked identically to the baselines / RL arms.

Scaffold behavior:
- ``use_model=False`` (default): acts with a **constrained equal-weight prior**, so the P5
  experiment runs end-to-end today without torch or a trained model.
- ``use_model=True``: builds the transformer (needs the ``generative`` extra) and decodes
  with it. The pretrain -> WBC -> Dr. GRPO training recipe lives as stubs in
  ``generative/model.py`` and is the P5 implementation work.
"""

from __future__ import annotations

import numpy as np

from alphaduel.agents.base import Agent
from alphaduel.generative.constraints import MandateConstraints
from alphaduel.generative.tokenizer import PortfolioTokenizer


class GenPortfolioAgent(Agent):
    name = "genportfolio"

    def __init__(
        self,
        n_feature_buckets: int = 16,
        n_weight_buckets: int = 10,
        max_position_weight: float = 0.2,
        use_model: bool = False,
        model_kwargs: dict | None = None,
        **params,
    ) -> None:
        self.n_feature_buckets = n_feature_buckets
        self.n_weight_buckets = n_weight_buckets
        self.max_position_weight = max_position_weight
        self.use_model = use_model
        self.model_kwargs = model_kwargs or {}
        self.tokenizer: PortfolioTokenizer | None = None
        self.constraints: MandateConstraints | None = None
        self.model = None

    # ------------------------------------------------------------------ setup

    def _build_tokenizer(self, symbols: list[str]) -> None:
        self.tokenizer = PortfolioTokenizer(
            symbols, self.n_feature_buckets, self.n_weight_buckets
        )
        self.constraints = MandateConstraints(max_position_weight=self.max_position_weight)

    def train(self, env) -> None:
        """Build the tokenizer, fit bucket edges on the train window, and (optionally) the model."""
        panel = env.panel  # MultiAssetPanel
        self._build_tokenizer(list(panel.symbols))
        train_end = max(1, int(0.6 * panel.n_steps))  # TODO: use evaluation splits (leakage)
        self.tokenizer.fit_buckets(panel.features[:train_end])

        if self.use_model:
            try:
                from alphaduel.generative.model import GenPortfolioModel

                self.model = GenPortfolioModel(self.tokenizer.vocab_size, **self.model_kwargs)
                # Pretrain / WBC / RL are P5 deliverables (see generative/model.py stubs).
            except ImportError:
                self.model = None  # falls back to the constrained prior

    def reset(self) -> None:
        pass

    # ------------------------------------------------------------------- act

    def act(self, observation: np.ndarray, info: dict) -> np.ndarray:
        symbols = list(info["symbols"])
        if self.tokenizer is None:
            self._build_tokenizer(symbols)
            self.tokenizer.fit_buckets(info["asset_features"])  # degenerate fallback fit

        current_w = np.asarray(info["weights"], dtype=np.float64)
        eligible = self.constraints.eligible_securities(symbols, current_w)

        if self.use_model and self.model is not None:
            weights = self._decode_with_model(info, symbols)
            self.last_thoughts = "generative transformer decode (constrained)"
        else:
            weights = self._equal_weight_prior(eligible)
            self.last_thoughts = "untrained prior: constrained equal-weight (model disabled)"
        return weights.astype(np.float32)

    # --------------------------------------------------------------- helpers

    def _equal_weight_prior(self, eligible: np.ndarray) -> np.ndarray:
        weights = np.zeros(len(eligible), dtype=np.float64)
        n = int(eligible.sum())
        if n > 0:
            weights[eligible] = min(self.max_position_weight, 1.0 / n)
        return weights

    def _decode_with_model(self, info: dict, symbols: list[str]) -> np.ndarray:
        """Autoregressive constrained greedy decode of a book into a weight vector."""
        tok = self.tokenizer
        seq = list(tok.build_context(info["asset_features"]))
        weights = np.zeros(len(symbols), dtype=np.float64)
        current = np.zeros(len(symbols), dtype=np.float64)

        for _ in range(len(symbols)):
            logits = self.model.next_token_logits(seq)
            sec_logits = logits[tok.security_base : tok.security_base + len(symbols)]
            eligible = self.constraints.eligible_securities(symbols, current)
            if not eligible.any():
                break
            masked = MandateConstraints.apply_to_logits(sec_logits, eligible)
            sec = int(np.argmax(masked))

            w_logits = logits[tok.weight_base : tok.vocab_size]
            weight = tok.weights.decode(int(np.argmax(w_logits)))
            weight = min(weight, self.max_position_weight, max(0.0, 1.0 - current.sum()))
            if weight <= 0:
                break

            current[sec] += weight
            weights[sec] += weight
            seq.append(tok.security_id(symbols[sec]))
            seq.append(tok.weight_id(weight))
            if current.sum() >= 1.0:
                break
        return weights
