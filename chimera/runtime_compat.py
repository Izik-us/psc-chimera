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
from .chimera_v2 import FlowMatchingBackbone as _LegacyFlowBackbone, SubstratePocketConditioner as _LegacyConditioner
from .dpo import DPOBatch, DPOTrainer
from .schrodinger_bridge import SE3SchrodingerBridge

AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"


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
    q = torch.stack([
        0.5 * torch.sqrt(d0) * torch.where(R[..., 2, 1] - R[..., 1, 2] >= 0, 1.0, -1.0),
        0.5 * torch.sqrt(d1) * torch.where(R[..., 0, 2] - R[..., 2, 0] >= 0, 1.0, -1.0),
        0.5 * torch.sqrt(d2) * torch.where(R[..., 1, 0] - R[..., 0, 1] >= 0, 1.0, -1.0),
    ], dim=-1)
    sin_half = q.norm(dim=-1)
    # q_vec = sin(theta/2) * axis, so theta/sin(theta/2) recovers the
    # principal axis-angle vector. The previous factor of 2 doubled omega.
    quat_omega = q * (theta / sin_half.clamp_min(1e-7)).unsqueeze(-1)
    return torch.where((theta < 1e-4).unsqueeze(-1), vee, quat_omega)


class MergeReadyInvariantPointAttention(_flow_matching.InvariantPointAttention):
    """IPA with translation-safe local vector aggregation."""

    def forward(self, s, z, R, t, substrate_coords: Optional[torch.Tensor] = None):
        B, L, _ = s.shape
        def split_heads(x, n):
            return x.view(B, L, n, -1).permute(0, 2, 1, 3)
        Q_s, K_s, V_s = split_heads(self.q_s(s), self.n_head), split_heads(self.k_s(s), self.n_head), split_heads(self.v_s(s), self.n_head)
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
        return self.out(torch.cat([out_s.permute(0, 2, 1, 3).reshape(B, L, -1), out_p_local.permute(0, 2, 1, 3).reshape(B, L, -1), out_z.permute(0, 2, 1, 3).reshape(B, L, -1)], dim=-1))


class MergeReadyMultiScaleNRPSDesigner(_LegacyDesigner):
    """Hierarchical designer with the corrected 28-D geometry edge contract."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        d_residue = self.edge_proj.out_features
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
            gate = torch.exp(-torch.cdist(residue_coords, substrate_coords).amin(dim=-1) / 8.0)
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


def merge_ready_update_from_proteus(self, survivors, failures, msa, pair_features, n_dpo_steps=50, learning_rate=1e-5, best_context_batch_index=0):
    """Run canonical DPO on the actual autoregressive sequence policy."""
    if not survivors or not failures:
        raise ValueError("Need at least one survivor and one failure for DPO")
    if learning_rate <= 0 or n_dpo_steps < 1:
        raise ValueError("learning_rate must be positive and n_dpo_steps must be >= 1")
    if msa.ndim != 3 or pair_features.ndim != 4:
        raise ValueError("msa must be (B,N_seq,L) and pair_features must be (B,L,L,C)")
    if msa.shape[0] != pair_features.shape[0] or msa.shape[2] != pair_features.shape[1] or pair_features.shape[1] != pair_features.shape[2]:
        raise ValueError("MSA and pair feature batch/length dimensions do not match")
    if not 0 <= best_context_batch_index < msa.shape[0]:
        raise ValueError("best_context_batch_index is outside the MSA batch")
    if self._reference_model is None:
        self.init_dpo_reference()
    with torch.no_grad():
        single_repr, _ = self.evoformer(msa, pair_features)
        context = single_repr[best_context_batch_index:best_context_batch_index + 1]
    def encode(sequences):
        length = len(sequences[0])
        if length == 0 or any(len(seq) != length for seq in sequences):
            raise ValueError("all DPO sequences must have the same non-zero length")
        invalid = sorted({aa for seq in sequences for aa in seq if aa not in AA_ORDER})
        if invalid:
            raise ValueError(f"invalid amino-acid symbols in PROTEUS data: {invalid}")
        return torch.tensor([[AA_ORDER.index(aa) for aa in seq] for seq in sequences], device=context.device, dtype=torch.long)
    n_pairs = max(len(survivors), len(failures))
    chosen = encode([survivors[i % len(survivors)] for i in range(n_pairs)])
    rejected = encode([failures[i % len(failures)] for i in range(n_pairs)])
    context_batch = context.expand(n_pairs, -1, -1).contiguous()
    mask = torch.ones_like(chosen, dtype=torch.bool)
    batch = DPOBatch(context_batch, chosen, rejected, mask, mask)
    policy, reference = self.sequence_policy, self._reference_model.sequence_policy
    optimizer = torch.optim.AdamW(policy.parameters(), lr=learning_rate)
    trainer = DPOTrainer(beta=0.1)
    metrics = {}
    for _ in range(n_dpo_steps):
        metrics = trainer.step(optimizer, policy, reference, batch)
    return metrics


_flow_matching.InvariantPointAttention = MergeReadyInvariantPointAttention
_flow_matching.so3_log = merge_ready_so3_log
_schrodinger_bridge.so3_log = merge_ready_so3_log
