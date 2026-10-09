"""Opt-in CHIMERA conditioning adapter for native ProteinMPNN probabilities.

This module deliberately sits outside the vendored reference model. Its zero-
initialized residual head begins as a no-op over the native 20-amino-acid
distribution, then can learn corrections from EvoFormer, geometric-residue, and
NRPS-hierarchy features without changing native ProteinMPNN parameter keys.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class ProteinMPNNConditioningAdapter(nn.Module):
    """Add trainable CHIMERA context while keeping ProteinMPNN independently testable.

    Inputs:
      native_log_probs: (B, L, 21), from ProteinMPNNReference.
      evolutionary_features: (B, L, d_evolutionary), from EvoFormer single states.
      geometric_features: (B, L, d_geometry), from CHIMERA's geometric residue trunk.
      nrps_logits: (B, L, 20), from MultiScaleNRPSDesigner.

    Returns normalized log probabilities over the 20 standard amino acids. The
    upstream X token is excluded and the remaining 20 probabilities are
    renormalized. The adapter's final layer is initialized to zero, so before
    training its output exactly equals the native distribution renormalized
    over the 20 standard amino acids.
    """

    def __init__(
        self,
        d_evolutionary: int = 256,
        d_geometry: int = 512,
        hidden_dim: int = 128,
        nrps_vocab_size: int = 20,
        output_vocab_size: int = 20,
    ) -> None:
        super().__init__()
        if min(d_evolutionary, d_geometry, hidden_dim, nrps_vocab_size, output_vocab_size) <= 0:
            raise ValueError("conditioning dimensions must be positive")
        if nrps_vocab_size != 20 or output_vocab_size != 20:
            raise ValueError("CHIMERA sequence conditioning currently requires the 20 standard amino acids")
        self.d_evolutionary = d_evolutionary
        self.d_geometry = d_geometry
        self.hidden_dim = hidden_dim
        self.nrps_vocab_size = nrps_vocab_size
        self.output_vocab_size = output_vocab_size

        self.evolutionary_projection = nn.Sequential(
            nn.LayerNorm(d_evolutionary),
            nn.Linear(d_evolutionary, hidden_dim),
            nn.GELU(),
        )
        self.geometry_projection = nn.Sequential(
            nn.LayerNorm(d_geometry),
            nn.Linear(d_geometry, hidden_dim),
            nn.GELU(),
        )
        self.nrps_projection = nn.Sequential(
            nn.LayerNorm(nrps_vocab_size),
            nn.Linear(nrps_vocab_size, hidden_dim),
            nn.GELU(),
        )
        self.residual_head = nn.Sequential(
            nn.LayerNorm(hidden_dim * 3),
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, output_vocab_size),
        )
        final = self.residual_head[-1]
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def forward(
        self,
        native_log_probs: torch.Tensor,
        evolutionary_features: torch.Tensor,
        geometric_features: torch.Tensor,
        nrps_logits: torch.Tensor,
        residue_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if native_log_probs.ndim != 3 or native_log_probs.shape[-1] != 21:
            raise ValueError("native_log_probs must have shape (B, L, 21)")
        batch, length = native_log_probs.shape[:2]
        expected = (batch, length)
        for name, value, width in (
            ("evolutionary_features", evolutionary_features, self.d_evolutionary),
            ("geometric_features", geometric_features, self.d_geometry),
            ("nrps_logits", nrps_logits, self.nrps_vocab_size),
        ):
            if value.ndim != 3 or value.shape[:2] != expected or value.shape[-1] != width:
                raise ValueError(f"{name} must have shape (B, L, {width})")
            if value.device != native_log_probs.device:
                raise ValueError(f"{name} and native_log_probs must be on the same device")
            if value.dtype != native_log_probs.dtype:
                raise ValueError(f"{name} and native_log_probs must have the same dtype")
        if not torch.isfinite(native_log_probs).all():
            raise ValueError("native_log_probs must be finite")
        if residue_mask is not None:
            if residue_mask.shape != expected or residue_mask.dtype != torch.bool:
                raise ValueError("residue_mask must be bool with shape (B, L)")
            if residue_mask.device != native_log_probs.device:
                raise ValueError("residue_mask and native_log_probs must be on the same device")

        native_20 = F.log_softmax(native_log_probs[..., :20], dim=-1)
        fused = torch.cat(
            (
                self.evolutionary_projection(evolutionary_features),
                self.geometry_projection(geometric_features),
                self.nrps_projection(nrps_logits),
            ),
            dim=-1,
        )
        correction = self.residual_head(fused)
        if residue_mask is not None:
            correction = correction * residue_mask.unsqueeze(-1).to(correction.dtype)
        return F.log_softmax(native_20 + correction, dim=-1)
