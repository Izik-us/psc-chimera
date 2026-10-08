"""Distributional confidence heads: pLDDT, aligned-error (PAE/pTM) and distogram."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from chimera.validation.metrics import tm_d0


class ConfidenceHeads(nn.Module):
    def __init__(
        self,
        d_single: int,
        d_pair: int,
        n_plddt_bins: int = 50,
        n_pae_bins: int = 64,
        pae_max: float = 32.0,
        n_dist_bins: int = 64,
        dist_min: float = 2.0,
        dist_max: float = 22.0,
    ):
        super().__init__()
        self.n_plddt_bins, self.n_pae_bins, self.n_dist_bins = n_plddt_bins, n_pae_bins, n_dist_bins
        self.pae_max, self.dist_min, self.dist_max = pae_max, dist_min, dist_max
        self.plddt = nn.Sequential(
            nn.LayerNorm(d_single), nn.Linear(d_single, d_single), nn.ReLU(), nn.Linear(d_single, n_plddt_bins)
        )
        self.pae = nn.Sequential(nn.LayerNorm(d_pair), nn.Linear(d_pair, n_pae_bins))
        self.distogram = nn.Sequential(nn.LayerNorm(d_pair), nn.Linear(d_pair, n_dist_bins))
        self.psi = nn.Sequential(nn.LayerNorm(d_single), nn.Linear(d_single, 2))

    # --- bin geometry -------------------------------------------------
    def plddt_centers(self, device, dtype):
        edges = torch.linspace(0, 100, self.n_plddt_bins + 1, device=device, dtype=dtype)
        return 0.5 * (edges[1:] + edges[:-1])

    def pae_centers(self, device, dtype):
        edges = torch.linspace(0, self.pae_max, self.n_pae_bins + 1, device=device, dtype=dtype)
        return 0.5 * (edges[1:] + edges[:-1])

    def dist_edges(self, device, dtype):
        return torch.linspace(self.dist_min, self.dist_max, self.n_dist_bins - 1, device=device, dtype=dtype)

    def forward(self, s: torch.Tensor, z: torch.Tensor) -> dict[str, torch.Tensor]:
        pae_logits = self.pae(z)
        dist_logits = self.distogram(z + z.transpose(1, 2))  # symmetrised
        return {
            "plddt_logits": self.plddt(s),
            "pae_logits": pae_logits,
            "distogram_logits": dist_logits,
            "psi_sincos": self.psi(s),
        }

    # --- expectations -------------------------------------------------
    def expected_plddt(self, logits):
        return (F.softmax(logits, -1) * self.plddt_centers(logits.device, logits.dtype)).sum(-1)

    def expected_pae(self, logits):
        return (F.softmax(logits, -1) * self.pae_centers(logits.device, logits.dtype)).sum(-1)

    def predicted_tm(self, pae_logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """pTM = max_i  (1/N) sum_j E_p[ 1 / (1 + (e_ij / d0(N))^2) ] (Xu & Zhang weighting)."""
        N = mask.sum(-1)
        d0 = tm_d0(N).to(pae_logits.device, pae_logits.dtype)  # (B,)
        centers = self.pae_centers(pae_logits.device, pae_logits.dtype)
        tm_w = 1.0 / (1.0 + (centers[None, :] / d0[:, None]) ** 2)  # (B, bins)
        per_pair = (F.softmax(pae_logits, -1) * tm_w[:, None, None, :]).sum(-1)  # (B, L, L)
        m = mask.to(per_pair.dtype)
        per_i = (per_pair * m.unsqueeze(1)).sum(-1) / m.sum(-1, keepdim=True).clamp_min(1)
        per_i = per_i.masked_fill(~mask, -1.0)
        return per_i.max(-1).values
