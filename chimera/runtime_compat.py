"""Merge-safe runtime compatibility implementations for public CHIMERAv2.

The historical ``chimera_v2.py`` remains import-compatible, while the public
package substitutes corrected components before CHIMERAv2 construction. This
keeps the repair isolated and prevents users from silently taking a stale
scientific path through the public API.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .multi_objective import MultiScaleNRPSDesigner as _LegacyDesigner
from .chimera_v2 import (
    FlowMatchingBackbone as _LegacyFlowBackbone,
    SubstratePocketConditioner as _LegacyConditioner,
)
from .schrodinger_bridge import SE3SchrodingerBridge


class MergeReadyMultiScaleNRPSDesigner(_LegacyDesigner):
    """Hierarchical designer with the corrected 28-D geometry edge contract."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        d_residue = kwargs.get("d_residue", 128)
        self.edge_proj = nn.Linear(28, d_residue)


class MergeReadySubstratePocketConditioner(_LegacyConditioner):
    """Substrate conditioner with direct residue-to-atom distance gating."""

    def forward(
        self,
        pair_repr: torch.Tensor,
        substrate_id: torch.Tensor,
        substrate_coords: torch.Tensor | None = None,
        substrate_types: torch.Tensor | None = None,
        residue_coords: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, L, _, d = pair_repr.shape
        if substrate_id.ndim != 1 or substrate_id.shape[0] != B:
            raise ValueError(f"substrate_id must have shape ({B},)")

        sub_cond = self.substrate_proj(self.substrate_emb(substrate_id))
        pair_repr = pair_repr + sub_cond[:, None, None, :]
        if substrate_coords is None or substrate_types is None:
            return pair_repr
        if substrate_coords.ndim != 3 or substrate_coords.shape[0] != B or substrate_coords.shape[-1] != 3:
            raise ValueError("substrate_coords must have shape (B, N_atoms, 3)")
        if substrate_types.shape[:2] != substrate_coords.shape[:2] or substrate_types.shape[-1] != 8:
            raise ValueError("substrate_types must have shape (B, N_atoms, 8)")

        atom_feat = torch.cat([substrate_coords, substrate_types.to(pair_repr.dtype)], dim=-1)
        atom_repr = self.atom_encoder(atom_feat)
        pair_flat = pair_repr.reshape(B, L * L, d)
        sub_pair, _ = self.sub_cross_attn(pair_flat, atom_repr, atom_repr)
        sub_pair = sub_pair.reshape(B, L, L, d)

        if residue_coords is not None:
            if residue_coords.shape != (B, L, 3):
                raise ValueError("residue_coords must have shape (B, L, 3)")
            min_dist = torch.cdist(residue_coords, substrate_coords).amin(dim=-1)
            residue_gate = torch.exp(-min_dist / 8.0)
            pair_gate = residue_gate[:, :, None, None] * residue_gate[:, None, :, None]
            sub_pair = sub_pair * pair_gate

        return self.sub_norm(pair_repr + sub_pair)


class MergeReadyFlowMatchingBackbone(_LegacyFlowBackbone):
    """CHIMERAv2 structure backbone using the canonical stochastic SB path."""

    def __init__(self, d_single: int = 256, d_pair: int = 256, n_blocks: int = 8):
        super().__init__(d_single, d_pair, n_blocks)
        self.sb_model = SE3SchrodingerBridge(
            self.flow_model.velocity_field,
            diffusion=0.05,
            sinkhorn_iters=50,
        )

    def sample(
        self,
        R0,
        t0,
        pair_cond,
        evol_single,
        n_steps=20,
        fixed_mask=None,
        substrate_coords=None,
        evol_conditioning_fn=None,
    ):
        evol_single = self.frozen_bridge(evol_single)
        return self.sb_model.sample(
            R0,
            t0,
            pair_cond,
            evol_single,
            n_steps=n_steps,
            fixed_mask=fixed_mask,
            substrate_coords=substrate_coords,
            evol_conditioning_fn=evol_conditioning_fn,
        )

    def loss(
        self,
        R0,
        t0,
        R1,
        t1,
        pair_cond,
        evol_single,
        fixed_mask=None,
        substrate_coords=None,
    ):
        evol_single = self.frozen_bridge(evol_single)
        return self.sb_model.bridge_loss(
            R0,
            t0,
            R1,
            t1,
            pair_cond,
            evol_single,
            fixed_mask=fixed_mask,
            substrate_coords=substrate_coords,
        )
