"""Masked CHIMERA implementation of the AlphaFold-2 EvoFormer core.

This module implements the coupled MSA/pair representation operations in
PyTorch. It is an architectural implementation, not a pretrained AlphaFold
model or a checkpoint-compatible port.
"""

from __future__ import annotations

import math
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Mapping, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def _masked_softmax(logits: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    output_dtype = logits.dtype
    mask = mask.to(torch.bool)
    logits = logits.float().masked_fill(~mask, torch.finfo(torch.float32).min)
    weights = torch.softmax(logits, dim=dim) * mask.to(logits.dtype)
    weights = weights / weights.sum(dim=dim, keepdim=True).clamp_min(1e-9)
    return weights.to(dtype=output_dtype)


class _Float32LayerNorm(nn.LayerNorm):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(
            value.float(),
            self.normalized_shape,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps,
        ).to(value.dtype)


class _SharedDropout(nn.Module):
    """Drop activations with a mask shared over one semantic axis."""

    def __init__(self, probability: float, shared_axis: int):
        super().__init__()
        if not 0.0 <= probability < 1.0:
            raise ValueError("dropout probability must be in [0, 1)")
        self.probability = probability
        self.shared_axis = shared_axis

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if not self.training or self.probability == 0.0:
            return value
        shape = list(value.shape)
        shape[self.shared_axis] = 1
        keep = 1.0 - self.probability
        mask = value.new_empty(shape).bernoulli_(keep).div_(keep)
        return value * mask


def _initialize_residual_projection(module: nn.Linear, depth: int) -> None:
    nn.init.xavier_uniform_(module.weight, gain=depth ** -0.5)
    if module.bias is not None:
        nn.init.zeros_(module.bias)


def _initialize_linear(module: nn.Linear) -> None:
    nn.init.xavier_uniform_(module.weight)
    if module.bias is not None:
        nn.init.zeros_(module.bias)


def _initialize_gate(module: nn.Linear) -> None:
    nn.init.zeros_(module.weight)
    nn.init.ones_(module.bias)


def _linear_float32(module: nn.Linear, value: torch.Tensor) -> torch.Tensor:
    bias = module.bias.float() if module.bias is not None else None
    return F.linear(value, module.weight.float(), bias)


def _validate_mask(mask: Optional[torch.Tensor], shape: tuple[int, ...], device, name: str):
    if mask is None:
        return torch.ones(shape, dtype=torch.bool, device=device)
    if (
        not torch.is_tensor(mask)
        or mask.shape != shape
        or mask.dtype != torch.bool
        or mask.device != device
    ):
        raise ValueError(f"{name} must be bool with shape {shape} on the input device")
    return mask


def _chunk_ranges(length: int, chunk_size: Optional[int]):
    size = length if chunk_size is None else chunk_size
    for start in range(0, length, size):
        yield start, min(start + size, length)


class MSARowAttentionWithPairBias(nn.Module):
    """Gated residue attention in each MSA row, biased by the pair state."""

    def __init__(self, c_m: int, c_z: int, n_heads: int, dropout: float, depth: int):
        super().__init__()
        if c_m % n_heads:
            raise ValueError("MSA width must be divisible by MSA attention heads")
        self.n_heads = n_heads
        self.c_head = c_m // n_heads
        self.norm_m = _Float32LayerNorm(c_m)
        self.norm_z = _Float32LayerNorm(c_z)
        self.q = nn.Linear(c_m, c_m, bias=False)
        self.k = nn.Linear(c_m, c_m, bias=False)
        self.v = nn.Linear(c_m, c_m, bias=False)
        self.gate = nn.Linear(c_m, c_m)
        self.pair_bias = nn.Linear(c_z, n_heads, bias=False)
        self.output = nn.Linear(c_m, c_m)
        self.dropout = _SharedDropout(dropout, shared_axis=1)
        for projection in (self.q, self.k, self.v, self.pair_bias):
            _initialize_linear(projection)
        _initialize_gate(self.gate)
        _initialize_residual_projection(self.output, depth)

    def forward(
        self, msa: torch.Tensor, pair: torch.Tensor, msa_mask: torch.Tensor
    ) -> torch.Tensor:
        batch, n_seq, n_res, c_m = msa.shape
        normalized = self.norm_m(msa)
        pair_bias = self._project_pair_bias(pair)
        outputs = []
        for start, end in _chunk_ranges(n_seq, self.attention_chunk_size):
            rows = normalized[:, start:end]
            q, k, v = (
                projection(rows)
                .reshape(batch, end - start, n_res, self.n_heads, self.c_head)
                .permute(0, 1, 3, 2, 4)
                for projection in (self.q, self.k, self.v)
            )
            logits = torch.matmul(q, k.transpose(-1, -2)) * (self.c_head ** -0.5)
            logits = logits + pair_bias
            key_mask = msa_mask[:, start:end, None, None, :]
            weights = _masked_softmax(logits, key_mask, dim=-1)
            attended = torch.matmul(weights, v)
            gate = torch.sigmoid(self.gate(rows)).reshape(
                batch, end - start, n_res, self.n_heads, self.c_head
            ).permute(0, 1, 3, 2, 4)
            attended = (attended * gate).permute(0, 1, 3, 2, 4).reshape(
                batch, end - start, n_res, c_m
            )
            attended = self.output(attended)
            attended = attended * msa_mask[:, start:end, :, None].to(attended.dtype)
            outputs.append(attended)
        return self.dropout(torch.cat(outputs, dim=1))

    def _project_pair_bias(self, pair: torch.Tensor) -> torch.Tensor:
        """Return per-head pair bias with query/key axes [j,k]."""
        return self.pair_bias(self.norm_z(pair)).permute(0, 3, 1, 2)[:, None]

    attention_chunk_size: Optional[int] = 32


class MSAColumnAttention(nn.Module):
    """At every aligned residue, let valid MSA rows attend over sequences."""

    def __init__(self, c_m: int, n_heads: int, dropout: float, depth: int):
        super().__init__()
        if c_m % n_heads:
            raise ValueError("MSA width must be divisible by MSA attention heads")
        self.n_heads = n_heads
        self.c_head = c_m // n_heads
        self.norm = _Float32LayerNorm(c_m)
        self.q = nn.Linear(c_m, c_m, bias=False)
        self.k = nn.Linear(c_m, c_m, bias=False)
        self.v = nn.Linear(c_m, c_m, bias=False)
        self.gate = nn.Linear(c_m, c_m)
        self.output = nn.Linear(c_m, c_m)
        self.dropout = _SharedDropout(dropout, shared_axis=1)
        for projection in (self.q, self.k, self.v):
            _initialize_linear(projection)
        _initialize_gate(self.gate)
        _initialize_residual_projection(self.output, depth)

    def forward(self, msa: torch.Tensor, msa_mask: torch.Tensor) -> torch.Tensor:
        batch, n_seq, n_res, c_m = msa.shape
        normalized = self.norm(msa).permute(0, 2, 1, 3)
        valid = msa_mask.permute(0, 2, 1)
        output_chunks = []
        for start, end in _chunk_ranges(n_res, self.attention_chunk_size):
            rows = normalized[:, start:end]
            query, key, value = (
                projection(rows)
                .reshape(batch, end - start, n_seq, self.n_heads, self.c_head)
                .permute(0, 1, 3, 2, 4)
                for projection in (self.q, self.k, self.v)
            )
            logits = torch.matmul(query, key.transpose(-1, -2)) * (self.c_head ** -0.5)
            key_mask = valid[:, start:end, None, None, :]
            weights = _masked_softmax(logits, key_mask, dim=-1)
            attended = torch.matmul(weights, value)
            gate = torch.sigmoid(self.gate(rows)).reshape(
                batch, end - start, n_seq, self.n_heads, self.c_head
            ).permute(0, 1, 3, 2, 4)
            attended = (attended * gate).permute(0, 1, 3, 2, 4).reshape(
                batch, end - start, n_seq, c_m
            )
            attended = self.output(attended)
            attended = attended * valid[:, start:end, :, None].to(attended.dtype)
            output_chunks.append(attended)
        output = torch.cat(output_chunks, dim=1).permute(0, 2, 1, 3)
        return self.dropout(output)

    attention_chunk_size: Optional[int] = 32


class MSATransition(nn.Module):
    def __init__(self, c_m: int, depth: int, transition_factor: int = 4):
        super().__init__()
        if transition_factor <= 0:
            raise ValueError("transition_factor must be positive")
        self.norm = _Float32LayerNorm(c_m)
        self.input = nn.Linear(c_m, c_m * transition_factor)
        self.output = nn.Linear(c_m * transition_factor, c_m)
        _initialize_linear(self.input)
        _initialize_residual_projection(self.output, depth)

    def forward(self, msa: torch.Tensor) -> torch.Tensor:
        return self.output(F.relu(self.input(self.norm(msa))))


class OuterProductMean(nn.Module):
    """Chunked, validity-normalized MSA outer-product pair update."""

    def __init__(
        self,
        c_m: int,
        c_z: int,
        c_hidden: int,
        depth: int,
        chunk_size: Optional[int],
        epsilon: float = 1e-3,
    ):
        super().__init__()
        if epsilon < 0.0:
            raise ValueError("OPM epsilon must be non-negative")
        self.norm = _Float32LayerNorm(c_m)
        self.left = nn.Linear(c_m, c_hidden)
        self.right = nn.Linear(c_m, c_hidden)
        self.output = nn.Linear(c_hidden * c_hidden, c_z)
        self.chunk_size = chunk_size
        self.epsilon = epsilon
        for projection in (self.left, self.right):
            _initialize_linear(projection)
        _initialize_residual_projection(self.output, depth)

    def forward(self, msa: torch.Tensor, msa_mask: torch.Tensor) -> torch.Tensor:
        batch, _, n_res, _ = msa.shape
        autocast_context = (
            torch.autocast(device_type=msa.device.type, enabled=False)
            if torch.amp.autocast_mode.is_autocast_available(msa.device.type)
            else nullcontext()
        )
        with autocast_context:
            normalized = self.norm(msa.float())
            left = _linear_float32(self.left, normalized)
            right = _linear_float32(self.right, normalized)
            valid = msa_mask.to(torch.float32)
            left = left * valid[..., None]
            right = right * valid[..., None]
            counts = torch.einsum("bni,bnj->bij", valid, valid)
            rows = []
            for start, end in _chunk_ranges(n_res, self.chunk_size):
                outer = torch.einsum(
                    "bnih,bnje->bijhe",
                    left[:, :, start:end],
                    right,
                )
                outer = outer.reshape(batch, end - start, n_res, -1)
                projected = _linear_float32(self.output, outer)
                denominator = counts[:, start:end, :, None] + self.epsilon
                projected = projected / denominator.clamp_min(
                    torch.finfo(denominator.dtype).tiny
                )
                rows.append(projected.to(msa.dtype))
        return torch.cat(rows, dim=1) * (counts > 0)[..., None].to(msa.dtype)


class _TriangleMultiplication(nn.Module):
    def __init__(
        self, c_z: int, c_hidden: int, depth: int, orientation: str, dropout: float
    ):
        super().__init__()
        if orientation not in {"outgoing", "incoming"}:
            raise ValueError("triangle orientation must be outgoing or incoming")
        self.orientation = orientation
        self.norm = _Float32LayerNorm(c_z)
        self.left = nn.Linear(c_z, c_hidden)
        self.left_gate = nn.Linear(c_z, c_hidden)
        self.right = nn.Linear(c_z, c_hidden)
        self.right_gate = nn.Linear(c_z, c_hidden)
        self.output_norm = _Float32LayerNorm(c_hidden)
        self.output = nn.Linear(c_hidden, c_z)
        self.output_gate = nn.Linear(c_z, c_z)
        # OpenFold DropoutRowwise shares the mask over pair axis -3 (i).
        self.dropout = _SharedDropout(dropout, shared_axis=1)
        for projection in (self.left, self.right):
            _initialize_linear(projection)
        for gate in (self.left_gate, self.right_gate, self.output_gate):
            _initialize_gate(gate)
        _initialize_residual_projection(self.output, depth)

    def forward(self, pair: torch.Tensor, pair_mask: torch.Tensor) -> torch.Tensor:
        normalized = self.norm(pair)
        valid = pair_mask[..., None].to(pair.dtype)
        left = torch.sigmoid(self.left_gate(normalized)) * self.left(normalized) * valid
        right = torch.sigmoid(self.right_gate(normalized)) * self.right(normalized) * valid
        if self.orientation == "outgoing":
            product = torch.einsum("bikc,bjkc->bijc", left, right)
        else:
            product = torch.einsum("bkic,bkjc->bijc", left, right)
        update = self.output(self.output_norm(product))
        update = update * torch.sigmoid(self.output_gate(normalized))
        update = update * valid
        return self.dropout(update)


class TriangleMultiplicationOutgoing(_TriangleMultiplication):
    def __init__(self, c_z: int, c_hidden: int, depth: int, dropout: float):
        super().__init__(c_z, c_hidden, depth, "outgoing", dropout)


class TriangleMultiplicationIncoming(_TriangleMultiplication):
    def __init__(self, c_z: int, c_hidden: int, depth: int, dropout: float):
        super().__init__(c_z, c_hidden, depth, "incoming", dropout)


class _TriangleAttention(nn.Module):
    def __init__(
        self,
        c_z: int,
        n_heads: int,
        depth: int,
        dropout: float,
        orientation: str,
    ):
        super().__init__()
        if c_z % n_heads:
            raise ValueError("pair width must be divisible by pair attention heads")
        if orientation not in {"starting", "ending"}:
            raise ValueError("triangle attention orientation must be starting or ending")
        self.orientation = orientation
        self.n_heads = n_heads
        self.c_head = c_z // n_heads
        self.norm = _Float32LayerNorm(c_z)
        self.q = nn.Linear(c_z, c_z, bias=False)
        self.k = nn.Linear(c_z, c_z, bias=False)
        self.v = nn.Linear(c_z, c_z, bias=False)
        self.bias = nn.Linear(c_z, n_heads, bias=False)
        self.gate = nn.Linear(c_z, c_z)
        self.output = nn.Linear(c_z, c_z)
        # Apply rowwise dropout only after all row chunks have been combined.
        self.dropout = _SharedDropout(dropout, shared_axis=1)
        for projection in (self.q, self.k, self.v, self.bias):
            _initialize_linear(projection)
        _initialize_gate(self.gate)
        _initialize_residual_projection(self.output, depth)

    def forward(self, pair: torch.Tensor, pair_mask: torch.Tensor) -> torch.Tensor:
        if self.orientation == "ending":
            pair = pair.transpose(1, 2)
            pair_mask = pair_mask.transpose(1, 2)
        normalized = self.norm(pair)
        batch, n_res, _, c_z = pair.shape
        outputs = []
        for start, end in _chunk_ranges(n_res, self.attention_chunk_size):
            query_nodes = normalized[:, start:end]
            q, k, v = (
                projection(query_nodes)
                .reshape(batch, end - start, n_res, self.n_heads, self.c_head)
                .permute(0, 1, 3, 2, 4)
                for projection in (self.q, self.k, self.v)
            )
            logits = self._attention_logits(q, k, query_nodes)
            query_mask = pair_mask[:, start:end, None, :, None]
            key_mask = pair_mask[:, start:end, None, None, :]
            weights = _masked_softmax(logits, query_mask & key_mask, dim=-1)
            attended = torch.matmul(weights, v)
            gate = torch.sigmoid(self.gate(query_nodes)).reshape(
                batch, end - start, n_res, self.n_heads, self.c_head
            ).permute(0, 1, 3, 2, 4)
            attended = (attended * gate).permute(0, 1, 3, 2, 4).reshape(
                batch, end - start, n_res, c_z
            )
            attended = self.output(attended)
            attended = attended * pair_mask[:, start:end, :, None].to(attended.dtype)
            outputs.append(attended)
        output = torch.cat(outputs, dim=1)
        if self.orientation == "ending":
            output = output.transpose(1, 2)
        return self.dropout(output)

    def _project_key_bias(self, pair_rows: torch.Tensor) -> torch.Tensor:
        """Project Z[i,k] to per-head bias indexed on attention key k."""
        return self.bias(pair_rows).permute(0, 1, 3, 2).unsqueeze(-2)

    def _attention_logits(
        self, query: torch.Tensor, key: torch.Tensor, pair_rows: torch.Tensor
    ) -> torch.Tensor:
        logits = torch.matmul(query, key.transpose(-1, -2)) * (self.c_head ** -0.5)
        return logits + self._project_key_bias(pair_rows)

    attention_chunk_size: Optional[int] = 32


class TriangleAttentionStartingNode(_TriangleAttention):
    def __init__(self, c_z: int, n_heads: int, depth: int, dropout: float):
        super().__init__(c_z, n_heads, depth, dropout, "starting")


class TriangleAttentionEndingNode(_TriangleAttention):
    def __init__(self, c_z: int, n_heads: int, depth: int, dropout: float):
        super().__init__(c_z, n_heads, depth, dropout, "ending")


class PairTransition(nn.Module):
    def __init__(self, c_z: int, depth: int, transition_factor: int = 4):
        super().__init__()
        if transition_factor <= 0:
            raise ValueError("transition_factor must be positive")
        self.norm = _Float32LayerNorm(c_z)
        self.input = nn.Linear(c_z, c_z * transition_factor)
        self.output = nn.Linear(c_z * transition_factor, c_z)
        _initialize_linear(self.input)
        _initialize_residual_projection(self.output, depth)

    def forward(self, pair: torch.Tensor) -> torch.Tensor:
        return self.output(F.relu(self.input(self.norm(pair))))


@dataclass(frozen=True)
class EvoformerOutput:
    msa_repr: torch.Tensor
    pair_repr: torch.Tensor
    single_repr: torch.Tensor
    diagnostics: Mapping[str, Any]


class EvoFormerBlock(nn.Module):
    """One full coupled EvoFormer block in canonical operation order."""

    def __init__(
        self,
        c_m: int,
        c_z: int,
        n_heads_msa: int,
        n_heads_pair: int,
        c_hidden_opm: int,
        c_hidden_triangle: int,
        depth: int,
        dropout_msa_row: float,
        dropout_msa_column: float,
        dropout_triangle: float,
        opm_chunk_size: Optional[int],
        transition_factor: int,
        opm_epsilon: float,
    ):
        super().__init__()
        self.msa_row_attention = MSARowAttentionWithPairBias(
            c_m, c_z, n_heads_msa, dropout_msa_row, depth
        )
        self.msa_column_attention = MSAColumnAttention(
            c_m, n_heads_msa, dropout_msa_column, depth
        )
        self.msa_transition = MSATransition(c_m, depth, transition_factor)
        self.outer_product_mean = OuterProductMean(
            c_m, c_z, c_hidden_opm, depth, opm_chunk_size, opm_epsilon
        )
        self.triangle_multiplication_outgoing = TriangleMultiplicationOutgoing(
            c_z, c_hidden_triangle, depth, dropout_triangle
        )
        self.triangle_multiplication_incoming = TriangleMultiplicationIncoming(
            c_z, c_hidden_triangle, depth, dropout_triangle
        )
        self.triangle_attention_starting = TriangleAttentionStartingNode(
            c_z, n_heads_pair, depth, dropout_triangle
        )
        self.triangle_attention_ending = TriangleAttentionEndingNode(
            c_z, n_heads_pair, depth, dropout_triangle
        )
        self.pair_transition = PairTransition(c_z, depth, transition_factor)

    def forward(
        self,
        msa: torch.Tensor,
        pair: torch.Tensor,
        msa_mask: torch.Tensor,
        pair_mask: torch.Tensor,
        collect_diagnostics: bool = False,
    ):
        msa_valid = msa_mask[..., None].to(msa.dtype)
        pair_valid = pair_mask[..., None].to(pair.dtype)
        diagnostics: dict[str, float] = {}

        def residual(name: str, value: torch.Tensor, update: torch.Tensor, mask: torch.Tensor):
            result = (value + update) * mask
            if collect_diagnostics:
                diagnostics[name] = float(update.detach().float().norm().cpu())
            return result

        update = self.msa_row_attention(msa, pair, msa_mask)
        msa = residual("msa_row_attention_update_norm", msa, update, msa_valid)
        update = self.msa_column_attention(msa, msa_mask)
        msa = residual("msa_column_attention_update_norm", msa, update, msa_valid)
        update = self.msa_transition(msa)
        msa = residual("msa_transition_update_norm", msa, update, msa_valid)

        update = self.outer_product_mean(msa, msa_mask)
        pair = residual("outer_product_mean_update_norm", pair, update, pair_valid)
        update = self.triangle_multiplication_outgoing(pair, pair_mask)
        pair = residual("triangle_multiplication_outgoing_update_norm", pair, update, pair_valid)
        update = self.triangle_multiplication_incoming(pair, pair_mask)
        pair = residual("triangle_multiplication_incoming_update_norm", pair, update, pair_valid)
        update = self.triangle_attention_starting(pair, pair_mask)
        pair = residual("triangle_attention_starting_update_norm", pair, update, pair_valid)
        update = self.triangle_attention_ending(pair, pair_mask)
        pair = residual("triangle_attention_ending_update_norm", pair, update, pair_valid)
        update = self.pair_transition(pair)
        pair = residual("pair_transition_update_norm", pair, update, pair_valid)
        return msa, pair, diagnostics


class EvoFormerStack(nn.Module):
    """A depth-faithful stack; ``n_blocks`` controls instantiated blocks."""

    def __init__(
        self,
        d_msa: int = 256,
        d_pair: int = 128,
        d_single: int = 256,
        n_blocks: int = 48,
        n_heads_msa: int = 8,
        n_heads_pair: int = 4,
        c_hidden_opm: int = 32,
        c_hidden_triangle: int = 128,
        dropout_msa_row: float = 0.15,
        dropout_msa_column: float = 0.0,
        dropout_triangle: float = 0.25,
        opm_chunk_size: Optional[int] = 16,
        attention_chunk_size: Optional[int] = 32,
        gradient_checkpointing: bool = False,
        max_relative_position: int = 32,
        transition_factor: int = 4,
        opm_epsilon: float = 1e-3,
    ):
        super().__init__()
        if min(d_msa, d_pair, d_single, n_blocks, n_heads_msa, n_heads_pair) <= 0:
            raise ValueError("EvoFormer dimensions, heads, and depth must be positive")
        if d_msa % n_heads_msa or d_pair % n_heads_pair:
            raise ValueError("MSA/pair widths must be divisible by their attention heads")
        if min(c_hidden_opm, c_hidden_triangle) <= 0:
            raise ValueError("EvoFormer hidden widths and chunk sizes must be positive")
        if opm_chunk_size is not None and opm_chunk_size <= 0:
            raise ValueError("opm_chunk_size must be positive or None")
        if attention_chunk_size is not None and attention_chunk_size <= 0:
            raise ValueError("attention_chunk_size must be positive or None")
        if transition_factor <= 0:
            raise ValueError("transition_factor must be positive")
        if max_relative_position < 1:
            raise ValueError("max_relative_position must be positive")
        self.d_msa = d_msa
        self.d_pair = d_pair
        self.d_single = d_single
        self.n_blocks = n_blocks
        self.gradient_checkpointing = gradient_checkpointing
        self.padding_token = 22
        self.mask_token = 23
        self.token_embed = nn.Embedding(24, d_msa, padding_idx=self.padding_token)
        self.relative_position_embed = nn.Embedding(2 * max_relative_position + 1, d_pair)
        self.max_relative_position = max_relative_position
        nn.init.normal_(self.token_embed.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.token_embed.weight[self.padding_token])
        nn.init.normal_(self.relative_position_embed.weight, mean=0.0, std=0.02)
        self.blocks = nn.ModuleList(
            EvoFormerBlock(
                d_msa,
                d_pair,
                n_heads_msa,
                n_heads_pair,
                c_hidden_opm,
                c_hidden_triangle,
                n_blocks,
                dropout_msa_row,
                dropout_msa_column,
                dropout_triangle,
                opm_chunk_size,
                transition_factor,
                opm_epsilon,
            )
            for _ in range(n_blocks)
        )
        for block in self.blocks:
            for attention in (
                block.msa_row_attention,
                block.msa_column_attention,
                block.triangle_attention_starting,
                block.triangle_attention_ending,
            ):
                attention.attention_chunk_size = attention_chunk_size
        self.single_projection = nn.Linear(d_msa, d_single)
        self.reconstruction_head = nn.Linear(d_msa, 22)
        self.pair_supervision_head = nn.Linear(d_pair, 1)
        _initialize_linear(self.single_projection)
        _initialize_linear(self.reconstruction_head)
        nn.init.xavier_uniform_(self.pair_supervision_head.weight, gain=0.1)
        nn.init.zeros_(self.pair_supervision_head.bias)
        self.configuration = {
            "architecture": "chimera_af2_evoformer_core_v2",
            "d_msa": d_msa,
            "d_pair": d_pair,
            "d_single": d_single,
            "n_blocks": n_blocks,
            "n_heads_msa": n_heads_msa,
            "n_heads_pair": n_heads_pair,
            "c_hidden_opm": c_hidden_opm,
            "c_hidden_triangle": c_hidden_triangle,
            "dropout_msa_row": dropout_msa_row,
            "dropout_msa_column": dropout_msa_column,
            "dropout_triangle": dropout_triangle,
            "opm_chunk_size": opm_chunk_size,
            "attention_chunk_size": attention_chunk_size,
            "gradient_checkpointing": gradient_checkpointing,
            "max_relative_position": max_relative_position,
            "transition_factor": transition_factor,
            "opm_epsilon": opm_epsilon,
        }

    def _pair_mask(self, residue_mask: torch.Tensor) -> torch.Tensor:
        return residue_mask[:, :, None] & residue_mask[:, None, :]

    def _relative_pair_features(
        self, residue_index: Optional[torch.Tensor], batch: int, n_res: int, device
    ) -> torch.Tensor:
        if residue_index is None:
            residue_index = torch.arange(n_res, device=device).expand(batch, -1)
        elif not torch.is_tensor(residue_index):
            raise ValueError("residue_index must be a tensor")
        elif residue_index.shape == (n_res,):
            residue_index = residue_index.to(device).expand(batch, -1)
        elif (
            not torch.is_tensor(residue_index)
            or residue_index.shape != (batch, n_res)
            or residue_index.device != device
        ):
            raise ValueError("residue_index must have shape (L,) or (B,L) on the input device")
        if residue_index.dtype not in (
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        ):
            raise ValueError("residue_index must have an integer dtype")
        relative = residue_index[:, :, None] - residue_index[:, None, :]
        relative = relative.clamp(-self.max_relative_position, self.max_relative_position)
        relative = relative + self.max_relative_position
        return self.relative_position_embed(relative.long())

    def forward_with_representations(
        self,
        msa_tokens: torch.Tensor,
        pair_features: torch.Tensor,
        msa_padding_mask: Optional[torch.Tensor] = None,
        residue_mask: Optional[torch.Tensor] = None,
        residue_index: Optional[torch.Tensor] = None,
        collect_diagnostics: bool = False,
    ) -> EvoformerOutput:
        if (
            not torch.is_tensor(msa_tokens)
            or msa_tokens.ndim != 3
            or msa_tokens.dtype not in (torch.int32, torch.int64)
        ):
            raise ValueError("msa_tokens must be an integer tensor with shape (B,N_seq,L)")
        batch, n_seq, n_res = msa_tokens.shape
        if n_seq < 1 or n_res < 1:
            raise ValueError("MSA sequence and residue dimensions must be non-empty")
        if torch.any((msa_tokens < 0) | (msa_tokens >= self.token_embed.num_embeddings)):
            raise ValueError("msa_tokens values must lie in the 0..23 vocabulary")
        if not torch.is_tensor(pair_features) or pair_features.shape != (
            batch, n_res, n_res, self.d_pair
        ):
            raise ValueError(
                f"pair_features must have shape {(batch, n_res, n_res, self.d_pair)}"
            )
        if not torch.is_floating_point(pair_features):
            raise ValueError("pair_features must have a floating-point dtype")
        if pair_features.device != msa_tokens.device:
            raise ValueError("MSA tokens and pair features must be on the same device")
        if not torch.isfinite(pair_features).all():
            raise ValueError("pair_features must be finite")
        padding = _validate_mask(
            msa_padding_mask, (batch, n_seq, n_res), msa_tokens.device, "msa_padding_mask"
        )
        if msa_padding_mask is None:
            padding = msa_tokens.eq(self.padding_token)
        valid = ~padding
        if residue_mask is not None:
            residue_mask = _validate_mask(
                residue_mask, (batch, n_res), msa_tokens.device, "residue_mask"
            )
            valid = valid & residue_mask[:, None, :]
        residue_valid = valid.any(dim=1)
        pair_mask = self._pair_mask(residue_valid)

        msa = self.token_embed(msa_tokens)
        msa = msa * valid[..., None].to(msa.dtype)
        pair = pair_features + self._relative_pair_features(residue_index, batch, n_res, msa_tokens.device)
        pair = pair * pair_mask[..., None].to(pair.dtype)
        block_diagnostics: list[dict[str, float]] = []
        for block in self.blocks:
            if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
                msa, pair = checkpoint(
                    lambda m, z, mm, pm, current_block=block: current_block(
                        m, z, mm, pm, False
                    )[:2],
                    msa,
                    pair,
                    valid,
                    pair_mask,
                    use_reentrant=False,
                )
            else:
                msa, pair, diagnostics = block(
                    msa, pair, valid, pair_mask, collect_diagnostics
                )
                if collect_diagnostics:
                    block_diagnostics.append(diagnostics)
        single = self.single_projection(msa[:, 0])
        single_valid = valid[:, 0]
        single = single * single_valid[..., None].to(single.dtype)
        diagnostics_result: dict[str, Any] = {}
        if collect_diagnostics:
            diagnostics_result = {
                "msa_activation_norm": float(msa.detach().float().norm().cpu()),
                "pair_activation_norm": float(pair.detach().float().norm().cpu()),
                "single_activation_norm": float(single.detach().float().norm().cpu()),
                "per_block_update_norms": block_diagnostics,
            }
        return EvoformerOutput(msa, pair, single, diagnostics_result)

    def forward(
        self,
        msa_tokens: torch.Tensor,
        pair_features: torch.Tensor,
        msa_padding_mask: Optional[torch.Tensor] = None,
        residue_mask: Optional[torch.Tensor] = None,
        residue_index: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        output = self.forward_with_representations(
            msa_tokens,
            pair_features,
            msa_padding_mask=msa_padding_mask,
            residue_mask=residue_mask,
            residue_index=residue_index,
        )
        return output.single_repr, output.pair_repr

    def encode_msa(
        self,
        msa_tokens: torch.Tensor,
        msa_padding_mask: Optional[torch.Tensor] = None,
        pair_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return the evolved MSA stream; zero pair input is for diagnostics only."""
        batch, _, n_res = msa_tokens.shape
        if pair_features is None:
            pair_features = self.token_embed.weight.new_zeros(
                batch, n_res, n_res, self.d_pair
            )
        return self.forward_with_representations(
            msa_tokens, pair_features, msa_padding_mask=msa_padding_mask
        ).msa_repr

    @staticmethod
    def _coevolution_target(
        msa_tokens: torch.Tensor, msa_padding_mask: torch.Tensor, chunk_size: int = 8
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute masked categorical mutual information targets from aligned rows."""
        valid = (~msa_padding_mask) & (msa_tokens < 22)
        vocab = 22
        one_hot = F.one_hot(msa_tokens.clamp(0, vocab - 1).long(), vocab).float()
        one_hot = one_hot * valid[..., None].float()
        valid_float = valid.float()
        batch, _, n_res, _ = one_hot.shape
        targets = []
        pair_counts = []
        for start in range(0, n_res, chunk_size):
            end = min(start + chunk_size, n_res)
            joint = torch.einsum(
                "bnia,bnjc->bijac", one_hot[:, :, start:end], one_hot
            )
            counts = torch.einsum(
                "bni,bnj->bij", valid_float[:, :, start:end], valid_float
            )
            joint = joint / counts[..., None, None].clamp_min(1.0)
            # Pair-specific marginals exclude sequences missing either residue.
            p_i_pair = torch.einsum(
                "bnia,bnj->bija",
                one_hot[:, :, start:end],
                valid_float,
            )
            p_j_pair = torch.einsum(
                "bni,bnjc->bijc",
                valid_float[:, :, start:end],
                one_hot,
            )
            p_i_pair = p_i_pair / counts[..., None].clamp_min(1.0)
            p_j_pair = p_j_pair / counts[..., None].clamp_min(1.0)
            independent = p_i_pair[..., :, None] * p_j_pair[..., None, :]
            log_ratio = torch.log(joint.clamp_min(1e-8)) - torch.log(
                independent.clamp_min(1e-8)
            )
            mi = torch.where(joint > 0, joint * log_ratio, 0.0).sum(dim=(-1, -2))
            targets.append((mi / math.log(vocab)).clamp(0.0, 1.0))
            pair_counts.append(counts)
        return torch.cat(targets, dim=1), torch.cat(pair_counts, dim=1)

    def representation_loss(
        self,
        msa_tokens: torch.Tensor,
        pair_features: torch.Tensor,
        msa_padding_mask: Optional[torch.Tensor] = None,
        residue_mask: Optional[torch.Tensor] = None,
        mask_probability: float = 0.15,
        pair_loss_weight: float = 1.0,
        generator: Optional[torch.Generator] = None,
        residue_index: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Combine masked-MSA recovery with an MSA-derived pairwise MI target."""
        if not 0.0 < mask_probability <= 1.0:
            raise ValueError("mask_probability must lie in (0, 1]")
        if pair_loss_weight < 0.0:
            raise ValueError("pair_loss_weight must be non-negative")
        batch, n_seq, n_res = msa_tokens.shape
        padding = msa_tokens.eq(self.padding_token) if msa_padding_mask is None else msa_padding_mask
        if (
            not torch.is_tensor(padding)
            or padding.shape != msa_tokens.shape
            or padding.dtype != torch.bool
            or padding.device != msa_tokens.device
        ):
            raise ValueError(
                "msa_padding_mask must be bool with the same shape and device as msa_tokens"
            )
        if residue_mask is not None:
            residue_mask = _validate_mask(
                residue_mask,
                (batch, n_res),
                msa_tokens.device,
                "residue_mask",
            )
            padding = padding | ~residue_mask[:, None, :]
        valid = ~padding
        if not valid.any():
            raise ValueError("MSA must contain at least one non-padding token")
        selected = (
            torch.rand(msa_tokens.shape, device=msa_tokens.device, generator=generator)
            < mask_probability
        ) & valid & (msa_tokens < 22)
        if not selected.any():
            candidates = torch.nonzero((valid & (msa_tokens < 22)).reshape(-1), as_tuple=False)
            if candidates.numel() == 0:
                raise ValueError("MSA must contain at least one amino-acid token")
            selected.reshape(-1)[candidates[0, 0]] = True
        corrupted = msa_tokens.masked_fill(selected, self.mask_token)
        output = self.forward_with_representations(
            corrupted,
            pair_features,
            msa_padding_mask=padding,
            residue_index=residue_index,
        )
        logits = self.reconstruction_head(output.msa_repr)
        reconstruction_loss = F.cross_entropy(logits[selected], msa_tokens[selected])
        mi_target, pair_counts = self._coevolution_target(msa_tokens, padding)
        pair_valid = pair_counts >= 2
        pair_valid &= ~torch.eye(n_res, dtype=torch.bool, device=msa_tokens.device)[None]
        predicted = self.pair_supervision_head(output.pair_repr).squeeze(-1)
        if pair_valid.any():
            pair_loss = F.smooth_l1_loss(predicted[pair_valid], mi_target[pair_valid])
        else:
            pair_loss = predicted.sum() * 0.0
        total = reconstruction_loss + pair_loss_weight * pair_loss
        return total, {
            "masked_msa_loss": reconstruction_loss.detach(),
            "pair_coevolution_loss": pair_loss.detach(),
        }

    def gradient_diagnostics(self) -> dict[str, float]:
        """Return opt-in gradient norms grouped by major EvoFormer operation."""
        groups = {
            "msa_row_attention": self.blocks[0].msa_row_attention,
            "msa_column_attention": self.blocks[0].msa_column_attention,
            "msa_transition": self.blocks[0].msa_transition,
            "outer_product_mean": self.blocks[0].outer_product_mean,
            "triangle_multiplication_outgoing": self.blocks[0].triangle_multiplication_outgoing,
            "triangle_multiplication_incoming": self.blocks[0].triangle_multiplication_incoming,
            "triangle_attention_starting": self.blocks[0].triangle_attention_starting,
            "triangle_attention_ending": self.blocks[0].triangle_attention_ending,
            "pair_transition": self.blocks[0].pair_transition,
        }
        return {
            name: sum(
                float(parameter.grad.detach().float().norm().cpu()) ** 2
                for parameter in module.parameters()
                if parameter.grad is not None
            ) ** 0.5
            for name, module in groups.items()
        }
