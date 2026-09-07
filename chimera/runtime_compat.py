"""Small compatibility implementations used by the public CHIMERAv2 entrypoint.

These classes keep the legacy large module source untouched while repairing two
runtime contracts that changed when geometric edge features were upgraded:

* MultiScaleNRPSDesigner now consumes the 28-dimensional ProteinMPNN geometry
  edge representation (16 RBF + 3 local-direction + 9 relative-rotation terms).
* SubstratePocketConditioner computes residue-to-atom distances directly,
  rather than measuring distances to the centroid and then taking a coordinate
  minimum.

The public package entrypoint installs these implementations before users can
construct CHIMERAv2. Direct imports from ``chimera.chimera_v2`` remain legacy
compatibility imports and should be avoided for production code.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .multi_objective import MultiScaleNRPSDesigner as _LegacyDesigner
from .chimera_v2 import SubstratePocketConditioner as _LegacyConditioner


class MergeReadyMultiScaleNRPSDesigner(_LegacyDesigner):
    """Legacy hierarchical designer with the corrected 28-D edge contract."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        d_residue = kwargs.get("d_residue", 128)
        self.edge_proj = nn.Linear(28, d_residue)


class MergeReadySubstratePocketConditioner(_LegacyConditioner):
    """Substrate conditioner with correct residue-to-atom distance gating."""

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
            # Correct Euclidean residue-to-each-atom distances: (B,L,N_atoms).
            min_dist = torch.cdist(residue_coords, substrate_coords).amin(dim=-1)
            residue_gate = torch.exp(-min_dist / 8.0)
            pair_gate = residue_gate[:, :, None, None] * residue_gate[:, None, :, None]
            sub_pair = sub_pair * pair_gate

        return self.sub_norm(pair_repr + sub_pair)
