"""Canonical structure-aware autoregressive protein sequence policy.

Teacher-forced training uses a causal, context-derived pair bias. Inference uses
the same weights with incremental per-layer key/value caches, so past residues
are not repeatedly projected or run through feed-forward blocks. Structural
context comes from CHIMERA's caller-supplied residue states, not another model.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class AutoregressiveSequencePolicy(nn.Module):
    """Causal Transformer policy with structure-biased attention and KV caching."""

    def __init__(
        self, context_dim: int, vocab_size: int = 20, layers: int = 10, heads: int = 8
    ):
        super().__init__()
        if heads < 1 or context_dim % heads:
            raise ValueError("context_dim must be divisible by positive heads")
        if vocab_size < 1 or layers < 1:
            raise ValueError("vocab_size and layers must be positive")
        self.context_dim = context_dim
        self.heads = heads
        self.head_dim = context_dim // heads
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

    def _expand_context(self, context: torch.Tensor, length: int) -> torch.Tensor:
        if context.ndim == 2:
            context = context.unsqueeze(1).expand(-1, length, -1)
        if context.ndim != 3 or context.shape[1:] != (length, self.context_dim):
            raise ValueError("context must have shape (B,L,D) or (B,D) matching tokens")
        return context

    def _pair_bias(self, projected_context: torch.Tensor) -> torch.Tensor:
        """Head-specific residue compatibility bias, shaped (B,H,L,L).

        Context channels are partitioned by attention head, so each head can
        learn a different compatibility geometry without adding checkpoint
        parameters. The bounded cosine bias perturbs, rather than replaces,
        the query-key attention scores.
        """
        batch, length, _ = projected_context.shape
        split = projected_context.reshape(batch, length, self.heads, self.head_dim)
        split = F.normalize(split, p=2, dim=-1, eps=1e-6)
        return 0.25 * torch.einsum("bihd,bjhd->bhij", split, split)

    def _run_full_layer(
        self,
        layer: nn.TransformerEncoderLayer,
        hidden: torch.Tensor,
        pair_bias: torch.Tensor,
    ) -> torch.Tensor:
        batch, length, width = hidden.shape
        attention = layer.self_attn
        normalized = layer.norm1(hidden)
        qkv = F.linear(normalized, attention.in_proj_weight, attention.in_proj_bias)
        query, key, value = qkv.chunk(3, dim=-1)

        def split_heads(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.view(batch, length, self.heads, self.head_dim).transpose(1, 2)

        query, key, value = map(split_heads, (query, key, value))
        causal = torch.zeros(length, length, dtype=hidden.dtype, device=hidden.device)
        causal = causal.masked_fill(
            torch.triu(
                torch.ones(length, length, dtype=torch.bool, device=hidden.device),
                diagonal=1,
            ),
            float("-inf"),
        )
        attention_bias = pair_bias + causal.view(1, 1, length, length)
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_bias,
            dropout_p=attention.dropout if layer.training else 0.0,
        )
        attended = attended.transpose(1, 2).contiguous().view(batch, length, width)
        hidden = hidden + layer.dropout1(attention.out_proj(attended))
        feed_forward = layer.linear2(
            layer.dropout(layer.activation(layer.linear1(layer.norm2(hidden))))
        )
        return hidden + layer.dropout2(feed_forward)

    def _causal_logits(self, context: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 2 or tokens.dtype not in (torch.int32, torch.int64):
            raise ValueError("tokens must be an integer tensor with shape (B,L)")
        batch, length = tokens.shape
        if length < 1:
            raise ValueError("tokens must contain at least one residue")
        if torch.any((tokens < 0) | (tokens >= self.vocab_size)):
            raise ValueError("tokens contain amino-acid IDs outside the policy vocabulary")
        context = self._expand_context(context, length)
        if context.shape[0] != batch:
            raise ValueError("context and tokens must have the same batch size")
        projected_context = self.context(context)
        decoder_input = torch.cat(
            [torch.full_like(tokens[:, :1], self.vocab_size), tokens[:, :-1]], dim=1
        )
        hidden = projected_context + self.token_embedding(decoder_input)
        pair_bias = self._pair_bias(projected_context)
        for layer in self.decoder.layers:
            hidden = self._run_full_layer(layer, hidden, pair_bias)
        if self.decoder.norm is not None:
            hidden = self.decoder.norm(hidden)
        return self.head(hidden)

    def forward(self, context: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        """Teacher-forced logits; residue i only sees sequence tokens < i."""
        return self._causal_logits(context, tokens)

    def token_logprobs(self, context: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        """Per-token log p(y_i | context, y_<i)."""
        return F.log_softmax(self._causal_logits(context, tokens), dim=-1).gather(
            -1, tokens.unsqueeze(-1)
        ).squeeze(-1)

    def logprob(self, context: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        return self.token_logprobs(context, tokens).sum(dim=-1)

    def _incremental_step(
        self,
        projected_context: torch.Tensor,
        previous_token: torch.Tensor,
        position: int,
        caches: list,
    ) -> tuple[torch.Tensor, list]:
        """Advance one position, caching each layer's projected keys and values."""
        batch = projected_context.shape[0]
        context_at_position = projected_context[:, position : position + 1]
        hidden = context_at_position + self.token_embedding(previous_token.reshape(batch, 1))
        context_query = context_at_position.reshape(
            batch, 1, self.heads, self.head_dim
        )
        context_query = F.normalize(context_query, p=2, dim=-1, eps=1e-6)
        context_keys = projected_context[:, : position + 1].reshape(
            batch, position + 1, self.heads, self.head_dim
        )
        context_keys = F.normalize(context_keys, p=2, dim=-1, eps=1e-6)
        pair_bias = 0.25 * torch.einsum(
            "bqhd,bkhd->bhqk", context_query, context_keys
        )
        updated_caches = []

        for layer, cache in zip(self.decoder.layers, caches):
            attention = layer.self_attn
            normalized = layer.norm1(hidden)
            qkv = F.linear(normalized, attention.in_proj_weight, attention.in_proj_bias)
            query, key, value = qkv.chunk(3, dim=-1)

            def split_step(tensor: torch.Tensor) -> torch.Tensor:
                return tensor.view(batch, 1, self.heads, self.head_dim).transpose(1, 2)

            query, key, value = map(split_step, (query, key, value))
            if cache is not None:
                key = torch.cat([cache[0], key], dim=2)
                value = torch.cat([cache[1], value], dim=2)
            updated_caches.append((key, value))
            attended = F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=pair_bias,
                dropout_p=attention.dropout if layer.training else 0.0,
            )
            attended = attended.transpose(1, 2).contiguous().view(batch, 1, self.context_dim)
            hidden = hidden + layer.dropout1(attention.out_proj(attended))
            feed_forward = layer.linear2(
                layer.dropout(layer.activation(layer.linear1(layer.norm2(hidden))))
            )
            hidden = hidden + layer.dropout2(feed_forward)

        if self.decoder.norm is not None:
            hidden = self.decoder.norm(hidden)
        return self.head(hidden[:, 0]), updated_caches

    @torch.no_grad()
    def generate(
        self,
        context: torch.Tensor,
        length: int,
        temperature: float = 1.0,
        generator: Optional[torch.Generator] = None,
        fixed_tokens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Sample left-to-right using incremental KV caches and hard constraints."""
        if length < 1:
            raise ValueError("length must be >= 1")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        context = self._expand_context(context, length)
        batch = context.shape[0]
        if fixed_tokens is not None:
            if fixed_tokens.shape != (batch, length):
                raise ValueError("fixed_tokens must have shape (B,length)")
            if fixed_tokens.dtype not in (torch.int32, torch.int64):
                raise ValueError("fixed_tokens must be an integer tensor")
            fixed_tokens = fixed_tokens.to(device=context.device, dtype=torch.long)
            invalid = (fixed_tokens < -1) | (fixed_tokens >= self.vocab_size)
            if torch.any(invalid):
                raise ValueError("fixed_tokens must be -1 or valid amino-acid IDs")

        projected_context = self.context(context)
        caches = [None for _ in self.decoder.layers]
        bos = torch.full((batch,), self.vocab_size, dtype=torch.long, device=context.device)
        sampled = []
        for position in range(length):
            previous = bos if position == 0 else sampled[-1]
            logits, caches = self._incremental_step(
                projected_context, previous, position, caches
            )
            if fixed_tokens is not None:
                forced = fixed_tokens[:, position] >= 0
                if forced.any():
                    forced_ids = fixed_tokens[forced, position]
                    logits = logits.clone()
                    logits[forced] = float("-inf")
                    logits[forced, forced_ids] = 0.0
            probabilities = F.softmax(logits / temperature, dim=-1)
            sampled.append(torch.multinomial(probabilities, 1, generator=generator).squeeze(-1))
        return torch.stack(sampled, dim=1)


__all__ = ["AutoregressiveSequencePolicy"]
