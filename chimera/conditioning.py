"""Substrate conditioning components shared by canonical and legacy models."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class SubstratePocketConditioner(nn.Module):
    """Condition pair features on substrate identity and optional coordinates."""

    def __init__(self, d_pair: int = 512, d_sub: int = 128):
        super().__init__()
        self.substrate_emb = nn.Embedding(30, d_sub)
        self.substrate_proj = nn.Linear(d_sub, d_pair)
        self.atom_encoder = nn.Sequential(
            nn.Linear(3 + 8, d_sub),
            nn.LayerNorm(d_sub),
            nn.GELU(),
            nn.Linear(d_sub, d_sub),
        )
        self.sub_cross_attn = nn.MultiheadAttention(
            embed_dim=d_pair,
            num_heads=4,
            kdim=d_sub,
            vdim=d_sub,
            batch_first=True,
        )
        self.sub_norm = nn.LayerNorm(d_pair)
        self.distance_gate = nn.Linear(1, d_pair)

    def forward(
        self,
        pair_repr: torch.Tensor,
        substrate_id: torch.Tensor,
        substrate_coords: Optional[torch.Tensor] = None,
        substrate_types: Optional[torch.Tensor] = None,
        residue_coords: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if pair_repr.ndim != 4 or pair_repr.shape[1] != pair_repr.shape[2]:
            raise ValueError("pair_repr must have shape (B,L,L,d_pair)")
        batch_size, length, _, channels = pair_repr.shape
        if substrate_id.ndim != 1 or substrate_id.shape[0] != batch_size:
            raise ValueError(f"substrate_id must have shape ({batch_size},)")
        if substrate_id.device != pair_repr.device:
            raise ValueError("substrate_id and pair_repr must be on the same device")
        if torch.any((substrate_id < 0) | (substrate_id >= self.substrate_emb.num_embeddings)):
            raise ValueError("substrate_id contains IDs outside the supported vocabulary")

        sub_condition = self.substrate_proj(self.substrate_emb(substrate_id))
        pair_repr = pair_repr + sub_condition[:, None, None, :]
        if substrate_coords is None and substrate_types is None:
            return pair_repr
        if substrate_coords is None or substrate_types is None:
            raise ValueError("substrate_coords and substrate_types must be provided together")
        if substrate_coords.ndim != 3 or substrate_coords.shape[0] != batch_size or substrate_coords.shape[-1] != 3:
            raise ValueError("substrate_coords must have shape (B,N,3)")
        if substrate_types.shape != (*substrate_coords.shape[:2], 8):
            raise ValueError("substrate_types must have shape (B,N,8)")
        if substrate_coords.device != pair_repr.device or substrate_types.device != pair_repr.device:
            raise ValueError("substrate tensors and pair_repr must be on the same device")
        atom_features = torch.cat(
            [substrate_coords.to(pair_repr.dtype), substrate_types.to(pair_repr.dtype)],
            dim=-1,
        )
        atom_repr = self.atom_encoder(atom_features)
        pair_flat = pair_repr.reshape(batch_size, length * length, channels)
        substrate_pair, _ = self.sub_cross_attn(pair_flat, atom_repr, atom_repr)
        substrate_pair = substrate_pair.reshape(batch_size, length, length, channels)
        if residue_coords is not None:
            if residue_coords.shape != (batch_size, length, 3):
                raise ValueError("residue_coords must have shape (B,L,3)")
            if residue_coords.device != pair_repr.device:
                raise ValueError("residue_coords and pair_repr must be on the same device")
            min_distance = torch.cdist(residue_coords, substrate_coords).amin(dim=-1)
            gate = torch.exp(-min_distance / 8.0)
            substrate_pair = substrate_pair * gate[:, :, None, None] * gate[:, None, :, None]
        return self.sub_norm(pair_repr + substrate_pair)


__all__ = ["SubstratePocketConditioner"]