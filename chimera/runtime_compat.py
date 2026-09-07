"""Merge-safe runtime compatibility implementations for public CHIMERAv2.

The historical ``chimera_v2.py`` remains import-compatible, while the public
package substitutes corrected components before CHIMERAv2 construction. This
keeps the repair isolated and prevents users from silently taking a stale
scientific path.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import flow_matching as _flow_matching
from .multi_objective import MultiScaleNRPSDesigner as _LegacyDesigner
from .chimera_v2 import (
    FlowMatchingBackbone as _LegacyFlowBackbone,
    SubstratePocketConditioner as _LegacyConditioner,
)
from .flow_matching import InvariantPointAttention as _LegacyIPA
from .schrodinger_bridge import SE3SchrodingerBridge


class MergeReadyInvariantPointAttention(_LegacyIPA):
    """IPA with translation-safe local vector aggregation.

    The historical implementation formed a translation-invariant relative
    vector and then subtracted the absolute query translation a second time.
    That reintroduced global-origin dependence. The corrected path rotates the
    already-relative vector directly into the query residue's local frame.
    """

    def forward(
        self,
        s: torch.Tensor,
        z: torch.Tensor,
        R: torch.Tensor,
        t: torch.Tensor,
        substrate_coords: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, L, _ = s.shape

        def split_heads(x, n):
            return x.view(B, L, n, -1).permute(0, 2, 1, 3)

        Q_s = split_heads(self.q_s(s), self.n_head)
        K_s = split_heads(self.k_s(s), self.n_head)
        V_s = split_heads(self.v_s(s), self.n_head)

        def transform_points(pts_local, R_frames, t_frames):
            pts = pts_local.view(B, L, -1, 3)
            R_exp = R_frames.unsqueeze(2).expand(-1, -1, pts.shape[2], -1, -1)
            t_exp = t_frames.unsqueeze(2).expand(-1, -1, pts.shape[2], -1)
            return torch.einsum("blnij,blnj->blni", R_exp, pts) + t_exp

        Q_p = transform_points(
            self.q_p(s).view(B, L, self.n_head * self.n_qk_pts, 3), R, t
        ).view(B, L, self.n_head, self.n_qk_pts, 3)
        K_p = transform_points(
            self.k_p(s).view(B, L, self.n_head * self.n_qk_pts, 3), R, t
        ).view(B, L, self.n_head, self.n_qk_pts, 3)
        V_p = transform_points(
            self.v_p(s).view(B, L, self.n_head * self.n_v_pts, 3), R, t
        ).view(B, L, self.n_head, self.n_v_pts, 3)

        scale = Q_s.shape[-1] ** -0.5
        attn_s = torch.einsum("bhid,bhjd->bhij", Q_s, K_s) * scale
        Q_p_h = Q_p.permute(0, 2, 1, 3, 4)
        K_p_h = K_p.permute(0, 2, 1, 3, 4)
        diff_p = Q_p_h.unsqueeze(3) - K_p_h.unsqueeze(2)
        attn_p = -(diff_p.norm(dim=-1) ** 2).sum(dim=-1)
        attn_z = self.pair_bias(z).permute(0, 3, 1, 2)

        if substrate_coords is not None:
            diff_sub = t.unsqueeze(2) - substrate_coords.unsqueeze(1)
            min_dist = diff_sub.norm(dim=-1).amin(dim=-1)
            prox_bias = torch.exp(-min_dist / 5.0)
            gate = self.substrate_gate(s).permute(0, 2, 1).unsqueeze(-1)
            attn_z = attn_z + gate * prox_bias.unsqueeze(1).unsqueeze(-1)

        w = F.softplus(self.gamma).view(1, self.n_head, 1, 1)
        attn = F.softmax(attn_s + w * attn_p + attn_z, dim=-1)
        out_s = torch.einsum("bhij,bhjd->bhid", attn, V_s)

        # ``relative`` is already translation invariant. Do not subtract the
        # absolute query translation again before applying R^T.
        relative = t.unsqueeze(1) - t.unsqueeze(2)
        out_p = torch.einsum("bhij,bijc->bhic", attn, relative)
        out_p_local = torch.einsum(
            "blij,bhlj->bhli", R.transpose(-1, -2), out_p
        )
        out_z = torch.einsum("bhij,bijc->bhic", attn, z)

        out_s_ = out_s.permute(0, 2, 1, 3).reshape(B, L, -1)
        out_p_ = out_p_local.permute(0, 2, 1, 3).reshape(B, L, -1)
        out_z_ = out_z.permute(0, 2, 1, 3).reshape(B, L, -1)
        return self.out(torch.cat([out_s_, out_p_, out_z_], dim=-1))


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


# Install these classes before CHIMERAv2 constructs any of its child modules.
_flow_matching.InvariantPointAttention = MergeReadyInvariantPointAttention
