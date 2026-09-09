"""Canonical autoregressive protein sequence policy.

Training uses teacher forcing with a causal mask. Inference samples one residue
at a time from the same policy, so the model used for DPO is also the model used
for generation.
"""
from __future__ import annotations

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class AutoregressiveSequencePolicy(nn.Module):
    """Causal Transformer policy for amino-acid sequences."""

    def __init__(self, context_dim: int, vocab_size: int = 20, layers: int = 2, heads: int = 4):
        super().__init__()
        if context_dim % heads:
            raise ValueError("context_dim must be divisible by heads")
        if vocab_size < 1 or layers < 1:
            raise ValueError("vocab_size and layers must be positive")
        self.context = nn.Linear(context_dim, context_dim)
        block = nn.TransformerEncoderLayer(
            d_model=context_dim,
            nhead=heads,
            dim_feedforward=context_dim * 4,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerEncoder(block, layers)
        self.token_embedding = nn.Embedding(vocab_size + 1, context_dim)
        self.head = nn.Linear(context_dim, vocab_size)
        self.vocab_size = vocab_size

    def _causal_logits(self, context: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 2:
            raise ValueError("tokens must have shape (B,L)")
        if context.ndim == 2:
            context = context.unsqueeze(1).expand(-1, tokens.shape[1], -1)
        if context.ndim != 3 or context.shape[:2] != tokens.shape:
            raise ValueError("context must have shape (B,L,D) matching tokens")
        decoder_input = torch.cat(
            [torch.full_like(tokens[:, :1], self.vocab_size), tokens[:, :-1]], dim=1
        )
        hidden = self.context(context) + self.token_embedding(decoder_input)
        L = hidden.shape[1]
        causal = torch.triu(
            torch.ones(L, L, device=hidden.device, dtype=torch.bool), diagonal=1
        )
        return self.head(self.decoder(hidden, mask=causal))

    def forward(self, context: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        """Teacher-forced logits: logit i depends only on tokens < i."""
        return self._causal_logits(context, tokens)

    def token_logprobs(self, context: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        """Per-token log p(y_i | context, y_<i)."""
        return F.log_softmax(self._causal_logits(context, tokens), dim=-1).gather(
            -1, tokens.unsqueeze(-1)
        ).squeeze(-1)

    def logprob(self, context: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        return self.token_logprobs(context, tokens).sum(dim=-1)

    @torch.no_grad()
    def generate(
        self,
        context: torch.Tensor,
        length: int,
        temperature: float = 1.0,
        generator: Optional[torch.Generator] = None,
        fixed_tokens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Causal left-to-right sampling from the same policy used for DPO."""
        if length < 1:
            raise ValueError("length must be >= 1")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if context.ndim == 2:
            context = context.unsqueeze(1).expand(-1, length, -1)
        if context.ndim != 3 or context.shape[1] != length:
            raise ValueError("context must have shape (B,length,D)")
        B = context.shape[0]
        if fixed_tokens is not None and fixed_tokens.shape != (B, length):
            raise ValueError("fixed_tokens must have shape (B,length)")

        tokens = torch.empty(B, 0, dtype=torch.long, device=context.device)
        for i in range(length):
            # _causal_logits expects a target slot at the current position. The
            # dummy value is never consumed by the current-position prediction;
            # it only lets the decoder construct BOS + previously generated tokens.
            dummy = torch.full((B, 1), 0, dtype=torch.long, device=context.device)
            decoder_tokens = torch.cat([tokens, dummy], dim=1)
            logits = self._causal_logits(context[:, : i + 1], decoder_tokens)[:, -1]
            if fixed_tokens is not None:
                fixed = fixed_tokens[:, i].to(device=context.device, dtype=torch.long)
                forced = fixed >= 0
                if forced.any():
                    if fixed[forced].min() < 0 or fixed[forced].max() >= self.vocab_size:
                        raise ValueError("fixed token IDs are outside the policy vocabulary")
                    logits = logits.clone()
                    logits[forced] = -torch.inf
                    logits[forced, fixed[forced]] = 0.0
            probs = F.softmax(logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, 1, generator=generator).squeeze(-1)
            tokens = torch.cat([tokens, next_token[:, None]], dim=1)
        return tokens


__all__ = ["AutoregressiveSequencePolicy"]
