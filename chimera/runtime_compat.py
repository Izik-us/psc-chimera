"""Merge-safe runtime compatibility implementations for public CHIMERAv2.

The historical modules remain import-compatible, while the public package
installs corrected scientific contracts before constructing CHIMERAv2.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import flow_matching as _flow_matching
from . import schrodinger_bridge as _schrodinger_bridge
from .multi_objective import MultiScaleNRPSDesigner as _LegacyDesigner
from .chimera_v2 import (
    FlowMatchingBackbone as _LegacyFlowBackbone,
    SubstratePocketConditioner as _LegacyConditioner,
)
from .schrodinger_bridge import SE3SchrodingerBridge


def merge_ready_so3_log(R: torch.Tensor) -> torch.Tensor:
    """Numerically stable principal SO(3) logarithm with signed pi branches."""
    if R.shape[-2:] != (3, 3):
        raise ValueError("R must end in (3,3)")
    trace = R.diagonal(dim1=-2, dim2=-1).sum(-1)
    theta = torch.acos(((trace - 1.0) * 0.5).clamp(-1.0, 1.0))
    skew = 0.5 * (R - R.transpose(-1, -2))
    vee = torch.stack([skew[..., 2, 1], skew[..., 0, 2], skew[..., 1, 0]], dim=-1)

    d0 = (1.0 + R[..., 0, 0] - R[..., 1, 1] - R[..., 2, 2]).clamp_min(0.0)
    d1 = (1.0 - R[..., 0, 0] + R[..., 1, 1] - R[..., 2, 2]).clamp_min(0.0)
    d2 = (1.0 - R[..., 0, 0] - R[..., 1, 1] + R[..., 2, 2]).clamp_min(0.0)
    q = torch.stack(
        [
            0.5 * torch.sqrt(d0) * torch.where(R[..., 2, 1] - R[..., 1, 2] >= 0, 1.0, -1.0),
            0.5 * torch.sqrt(d1) * torch.where(R[..., 0, 2] - R[..., 2, 0] >= 0, 1.0, -1.0),
            0.5 * torch.sqrt(d2) * torch.where(R[..., 1, 0] - R[..., 0, 1] >= 0, 1.0, -1.0),
        ],
        dim=-1,
    )
    sin_half = q.norm(dim=-1)
    quat_omega = q * (2.0 * theta / sin_half.clamp_min(1e-7)).unsqueeze(-1)
    return torch.where((theta < 1e-4).unsqueeze(-1), vee, quat_omega)


class MergeReadyInvariantPointAttention(_flow_matching.InvariantPointAttention):
    """IPA with translation-safe local vector aggregation."""

    def forward(self, s, z, R, t, substrate_coords: Optional[torch.Tensor] = None):
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

        Q_p = transform_points(self.q_p(s).view(B, L, self.n_head * self.n_qk_pts, 3), R, t).view(B, L, self.n_head, self.n_qk_pts, 3)
        K_p = transform_points(self.k_p(s).view(B, L, self.n_head * self.n_qk_pts, 3), R, t).view(B, L, self.n_head, self.n_qk_pts, 3)
        V_p = transform_points(self.v_p(s).view(B, L, self.n_head * self.n_v_pts, 3), R, t).view(B, L, self.n_head, self.n_v_pts, 3)

        attn_s = torch.einsum("bhid,bhjd->bhij", Q_s, K_s) * (Q_s.shape[-1] ** -0.5)
        diff_p = Q_p.permute(0, 2, 1, 3, 4).unsqueeze(3) - K_p.permute(0, 2, 1, 3, 4).unsqueeze(2)
        attn_p = -(diff_p.norm(dim=-1) ** 2).sum(dim=-1)
        attn_z = self.pair_bias(z).permute(0, 3, 1, 2)

        if substrate_coords is not None:
            min_dist = (t.unsqueeze(2) - substrate_coords.unsqueeze(1)).norm(dim=-1).amin(dim=-1)
            gate = self.substrate_gate(s).permute(0, 2, 1).unsqueeze(-1)
            attn_z = attn_z + gate * torch.exp(-min_dist / 5.0).unsqueeze(1).unsqueeze(-1)

        attn = F.softmax(attn_s + F.softplus(self.gamma).view(1, self.n_head, 1, 1) * attn_p + attn_z, dim=-1)
        out_s = torch.einsum("bhij,bhjd->bhid", attn, V_s)
        relative = t.unsqueeze(1) - t.unsqueeze(2)
        out_p = torch.einsum("bhij,bijc->bhic", attn, relative)
        out_p_local = torch.einsum("blij,bhlj->bhli", R.transpose(-1, -2), out_p)
        out_z = torch.einsum("bhij,bijc->bhic", attn, z)
        out = torch.cat(
            [
                out_s.permute(0, 2, 1, 3).reshape(B, L, -1),
                out_p_local.permute(0, 2, 1, 3).reshape(B, L, -1),
                out_z.permute(0, 2, 1, 3).reshape(B, L, -1),
            ],
            dim=-1,
        )
        return self.out(out)


class MergeReadyMultiScaleNRPSDesigner(_LegacyDesigner):
    """Hierarchical designer with the corrected 28-D geometry edge contract."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        d_residue = kwargs.get("d_residue", 128)
        self.edge_proj = nn.Linear(28, d_residue)


class MergeReadySubstratePocketConditioner(_LegacyConditioner):
    """Substrate conditioner with direct residue-to-atom distance gating."""

    def forward(self, pair_repr, substrate_id, substrate_coords=None, substrate_types=None, residue_coords=None):
        B, L, _, d = pair_repr.shape
        if substrate_id.ndim != 1 or substrate_id.shape[0] != B:
            raise ValueError(f"substrate_id must have shape ({B},)")
        pair_repr = pair_repr + self.substrate_proj(self.substrate_emb(substrate_id))[:, None, None, :]
        if substrate_coords is None or substrate_types is None:
            return pair_repr
        if substrate_coords.ndim != 3 or substrate_coords.shape[0] != B or substrate_coords.shape[-1] != 3:
            raise ValueError("substrate_coords must have shape (B, N_atoms, 3)")
        if substrate_types.shape[:2] != substrate_coords.shape[:2] or substrate_types.shape[-1] != 8:
            raise ValueError("substrate_types must have shape (B, N_atoms, 8)")
        atom_repr = self.atom_encoder(torch.cat([substrate_coords, substrate_types.to(pair_repr.dtype)], dim=-1))
        sub_pair, _ = self.sub_cross_attn(pair_repr.reshape(B, L * L, d), atom_repr, atom_repr)
        sub_pair = sub_pair.reshape(B, L, L, d)
        if residue_coords is not None:
            if residue_coords.shape != (B, L, 3):
                raise ValueError("residue_coords must have shape (B, L, 3)")
            min_dist = torch.cdist(residue_coords, substrate_coords).amin(dim=-1)
            gate = torch.exp(-min_dist / 8.0)
            sub_pair = sub_pair * gate[:, :, None, None] * gate[:, None, :, None]
        return self.sub_norm(pair_repr + sub_pair)


class MergeReadyFlowMatchingBackbone(_LegacyFlowBackbone):
    """CHIMERAv2 structure backbone using the canonical stochastic SB path."""

    def __init__(self, d_single=256, d_pair=256, n_blocks=8):
        super().__init__(d_single, d_pair, n_blocks)
        self.sb_model = SE3SchrodingerBridge(self.flow_model.velocity_field, diffusion=0.05, sinkhorn_iters=50)

    def sample(self, R0, t0, pair_cond, evol_single, n_steps=20, fixed_mask=None, substrate_coords=None, evol_conditioning_fn=None):
        evol_single = self.frozen_bridge(evol_single)
        return self.sb_model.sample(R0, t0, pair_cond, evol_single, n_steps=n_steps, fixed_mask=fixed_mask, substrate_coords=substrate_coords, evol_conditioning_fn=evol_conditioning_fn)

    def loss(self, R0, t0, R1, t1, pair_cond, evol_single, fixed_mask=None, substrate_coords=None):
        evol_single = self.frozen_bridge(evol_single)
        return self.sb_model.bridge_loss(R0, t0, R1, t1, pair_cond, evol_single, fixed_mask=fixed_mask, substrate_coords=substrate_coords)


# Install corrected implementations before CHIMERAv2 constructs child modules.
_flow_matching.InvariantPointAttention = MergeReadyInvariantPointAttention
_flow_matching.so3_log = merge_ready_so3_log
_schrodinger_bridge.so3_log = merge_ready_so3_log
