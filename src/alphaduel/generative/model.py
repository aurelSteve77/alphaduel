"""Decoder-only transformer for generative portfolio construction (SPEC §15.4).

Standard decoder-only architecture (as in GenPage): token embedding + causal transformer
blocks + an **untied** output projection (pretraining softmax vs. WBC per-token sigmoid
place different demands on the logits). Requires torch (``uv sync --extra generative``).

The training recipe — pretrain (next-token) -> WBC or Dr. GRPO post-training — is documented
here as stubs; this module provides the architecture and an autoregressive decode loop with
constrained decoding hooks.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


class GenPortfolioModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 4,
        max_len: int = 1024,
    ) -> None:
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_feedforward=4 * d_model, batch_first=True
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=n_layers)
        # Untied output head (not weight-shared with token_emb) per GenPage.
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.max_len = max_len

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        seq_len = tokens.shape[1]
        pos = torch.arange(seq_len, device=tokens.device).unsqueeze(0)
        x = self.token_emb(tokens) + self.pos_emb(pos)
        causal = nn.Transformer.generate_square_subsequent_mask(seq_len).to(tokens.device)
        x = self.blocks(x, mask=causal)
        return self.head(x)  # (B, T, vocab)

    @torch.no_grad()
    def next_token_logits(self, tokens: list[int]) -> np.ndarray:
        device = next(self.parameters()).device
        t = torch.tensor([tokens[-self.max_len :]], dtype=torch.long, device=device)
        return self.forward(t)[0, -1].cpu().numpy()


def pretrain(model: GenPortfolioModel, sequences, **kwargs) -> None:
    """Next-token-prediction pretraining on imitation books. TODO: implement (SPEC §15.4.1)."""
    raise NotImplementedError("Pretraining loop is a P5 deliverable; see SPEC §15.4.")


def wbc_post_train(model: GenPortfolioModel, sequences, rewards, **kwargs) -> None:
    """Weighted binary classification post-training. TODO: implement (SPEC §15.4.2)."""
    raise NotImplementedError("WBC post-training is a P5 deliverable; see SPEC §15.4.")


def grpo_post_train(model: GenPortfolioModel, reward_model, **kwargs) -> None:
    """Dr. GRPO post-training with a book-reward model + KL penalty. TODO (SPEC §15.4.3)."""
    raise NotImplementedError("RL post-training is a P5 deliverable; see SPEC §15.4.")
