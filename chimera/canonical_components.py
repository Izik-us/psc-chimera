"""Canonical CHIMERAv2 component implementations."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import flow_matching as _flow_matching
from .multi_objective import MultiScaleNRPSDesigner as _LegacyDesigner
from .chimera_v2 import FlowMatchingBackbone as _LegacyFlowBackbone, SubstratePocketConditioner as _LegacyConditioner
from .dpo import DPOBatch, DPOTrainer
from .bayesian import BayesianUncertaintyEstimator
from .schrodinger_bridge import SE3SchrodingerBridge

AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"


def so3_log(R: torch.Tensor) -> torch.Tensor:
    if R.shape[-2:] != (3, 3):
        raise ValueError("R must end in (3,3)")
    trace = R.diagonal(dim1=-2, dim2=-1).sum(-1)
    theta = torch.acos(((trace - 1.0) * 0.5).clamp(-1.0, 1.0))
    skew = 0.5 * (R - R.transpose(-1, -2))
    vee = torch.stack([skew[..., 2, 1], skew[..., 0, 2], skew[..., 1, 0]], dim=-1)
    regular = vee * (theta / torch.sin(theta).clamp_min(1e-7)).unsqueeze(-1)
    axis = (((torch.diagonal(R, dim1=-2, dim2=-1) + 1.0) * 0.5).clamp_min(0.0)).sqrt()
    signs = torch.where(vee >= 0.0, torch.ones_like(vee), -torch.ones_like(vee))
    signed_axis = axis * signs
    signed_axis = signed_axis / signed_axis.norm(dim=-1, keepdim=True).clamp_min(1e-7)
    result = torch.where((theta > torch.pi - 1e-4).unsqueeze(-1), theta.unsqueeze(-1) * signed_axis, regular)
    return torch.where((theta < 1e-4).unsqueeze(-1), vee, result)


class InvariantPointAttention(_flow_matching.InvariantPointAttention):
    """SE(3)-invariant IPA using local-frame query/key/value points."""

    def forward(self, s, z, R, t, substrate_coords: Optional[torch.Tensor] = None):
        B, L, _ = s.shape

        def split_heads(x):
            return x.view(B, L, self.n_head, -1).permute(0, 2, 1, 3)

        Q_s, K_s, V_s = split_heads(self.q_s(s)), split_heads(self.k_s(s)), split_heads(self.v_s(s))
        Q_local = self.q_p(s).view(B, L, self.n_head, self.n_qk_pts, 3)
        K_local = self.k_p(s).view(B, L, self.n_head, self.n_qk_pts, 3)
        V_local = self.v_p(s).view(B, L, self.n_head, self.n_v_pts, 3)

        Q_global = torch.einsum("blij,bhlpj->bhlpi", R, Q_local.permute(0, 2, 1, 3, 4)) + t[:, None, :, None, :]
        K_global = torch.einsum("blij,bhlpj->bhlpi", R, K_local.permute(0, 2, 1, 3, 4)) + t[:, None, :, None, :]
        V_global = torch.einsum("blij,bhlpj->bhlpi", R, V_local.permute(0, 2, 1, 3, 4)) + t[:, None, :, None, :]

        attn_s = torch.einsum("bhid,bhjd->bhij", Q_s, K_s) * (Q_s.shape[-1] ** -0.5)
        diff = Q_global.unsqueeze(3) - K_global.unsqueeze(2)
        attn_p = -(diff.square().sum(-1)).sum(-1)
        attn_z = self.pair_bias(z).permute(0, 3, 1, 2)

        if substrate_coords is not None:
            if substrate_coords.ndim != 3 or substrate_coords.shape[0] != B or substrate_coords.shape[-1] != 3:
                raise ValueError("substrate_coords must have shape (B,K,3)")
            min_dist = torch.cdist(t, substrate_coords).amin(dim=-1)
            gate = self.substrate_gate(s).permute(0, 2, 1).unsqueeze(-1)
            attn_z = attn_z + gate * torch.exp(-min_dist / 5.0).unsqueeze(1).unsqueeze(-1)

        weights = F.softplus(self.gamma).view(1, self.n_head, 1, 1)
        attn = F.softmax(attn_s + weights * attn_p + attn_z, dim=-1)
        out_s = torch.einsum("bhij,bhjd->bhid", attn, V_s)
        out_v_global = torch.einsum("bhij,bhjpc->bhlpc", attn, V_global)
        # Average the configured value points into one equivariant vector per
        # head. Mapping it back with R_i^T removes the arbitrary global frame.
        out_v_local = torch.einsum("blji,bhljc->bhlic", R, out_v_global.mean(dim=3)).squeeze(3)
        out_z = torch.einsum("bhij,bijc->bhic", attn, z)
        return self.out(torch.cat([
            out_s.permute(0, 2, 1, 3).reshape(B, L, -1),
            out_v_local.permute(0, 2, 1, 3).reshape(B, L, -1),
            out_z.permute(0, 2, 1, 3).reshape(B, L, -1),
        ], dim=-1))


class MultiScaleNRPSDesigner(_LegacyDesigner):
    """Hierarchical designer consuming the canonical 28-D edge representation."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.edge_proj = nn.Linear(28, self.edge_proj.out_features)


class SubstratePocketConditioner(_LegacyConditioner):
    """Substrate conditioner with explicit residue/pocket geometry gating."""
    def forward(self, pair_repr, substrate_id, substrate_coords=None, substrate_types=None, residue_coords=None):
        B, L, _, d = pair_repr.shape
        if substrate_id.shape != (B,):
            raise ValueError(f"substrate_id must have shape ({B},)")
        pair_repr = pair_repr + self.substrate_proj(self.substrate_emb(substrate_id))[:, None, None, :]
        if substrate_coords is None or substrate_types is None:
            return pair_repr
        if substrate_coords.ndim != 3 or substrate_coords.shape[0] != B or substrate_coords.shape[-1] != 3:
            raise ValueError("substrate_coords must have shape (B,N,3)")
        if substrate_types.shape[:2] != substrate_coords.shape[:2] or substrate_types.shape[-1] != 8:
            raise ValueError("substrate_types must have shape (B,N,8)")
        atom_repr = self.atom_encoder(torch.cat([substrate_coords, substrate_types.to(pair_repr.dtype)], dim=-1))
        enriched, _ = self.sub_cross_attn(pair_repr.reshape(B, L * L, d), atom_repr, atom_repr)
        enriched = enriched.reshape(B, L, L, d)
        if residue_coords is not None:
            if residue_coords.shape != (B, L, 3):
                raise ValueError("residue_coords must have shape (B,L,3)")
            gate = torch.exp(-torch.cdist(residue_coords, substrate_coords).amin(dim=-1) / 8.0)
            enriched = enriched * gate[:, :, None, None] * gate[:, None, :, None]
        return self.sub_norm(pair_repr + enriched)


class FlowMatchingBackbone(_LegacyFlowBackbone):
    """Structure backbone whose instantiated SB velocity field uses canonical IPA."""
    def __init__(self, d_single=256, d_pair=256, n_blocks=8):
        super().__init__(d_single, d_pair, n_blocks)
        for block in self.flow_model.velocity_field.ipa_blocks:
            block.ipa = InvariantPointAttention(d_single, d_pair, block.ipa.n_head, block.ipa.n_qk_pts, block.ipa.n_v_pts)
        self.sb_model = SE3SchrodingerBridge(self.flow_model.velocity_field, diffusion=0.05, sinkhorn_iters=50)

    def sample(self, R0, t0, pair_cond, evol_single, n_steps=20, fixed_mask=None, substrate_coords=None, evol_conditioning_fn=None):
        evol_single = self.frozen_bridge(evol_single)
        return self.sb_model.sample(R0, t0, pair_cond, evol_single, n_steps=n_steps, fixed_mask=fixed_mask, substrate_coords=substrate_coords, evol_conditioning_fn=evol_conditioning_fn)

    def loss(self, R0, t0, R1, t1, pair_cond, evol_single, fixed_mask=None, substrate_coords=None):
        evol_single = self.frozen_bridge(evol_single)
        return self.sb_model.bridge_loss(R0, t0, R1, t1, pair_cond, evol_single, fixed_mask=fixed_mask, substrate_coords=substrate_coords)


def update_from_proteus(self, survivors, failures, msa, pair_features, n_dpo_steps=50, learning_rate=1e-5, best_context_batch_index=0):
    if not survivors or not failures:
        raise ValueError("Need at least one survivor and one failure for DPO")
    if learning_rate <= 0 or n_dpo_steps < 1:
        raise ValueError("learning_rate must be positive and n_dpo_steps must be >= 1")
    if msa.ndim != 3 or pair_features.ndim != 4 or msa.shape[0] != pair_features.shape[0] or msa.shape[2] != pair_features.shape[1] or pair_features.shape[1] != pair_features.shape[2]:
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
        if length == 0 or any(len(s) != length for s in sequences):
            raise ValueError("all DPO sequences must have the same non-zero length")
        invalid = sorted({aa for s in sequences for aa in s if aa not in AA_ORDER})
        if invalid:
            raise ValueError(f"invalid amino-acid symbols in PROTEUS data: {invalid}")
        return torch.tensor([[AA_ORDER.index(aa) for aa in s] for s in sequences], device=context.device, dtype=torch.long)

    n_pairs = max(len(survivors), len(failures))
    chosen = encode([survivors[i % len(survivors)] for i in range(n_pairs)])
    rejected = encode([failures[i % len(failures)] for i in range(n_pairs)])
    context = context.expand(n_pairs, -1, -1).contiguous()
    mask = torch.ones_like(chosen, dtype=torch.bool)
    batch = DPOBatch(context, chosen, rejected, mask, mask)
    optimizer = torch.optim.AdamW(self.sequence_policy.parameters(), lr=learning_rate)
    trainer = DPOTrainer(beta=0.1)
    metrics = {}
    for _ in range(n_dpo_steps):
        metrics = trainer.step(optimizer, self.sequence_policy, self._reference_model.sequence_policy, batch)
    return metrics


def set_best_observed(self, value: Optional[float]) -> None:
    if value is not None and not 0.0 <= float(value) <= 1.0:
        raise ValueError("best_observed must be in normalized [0,1] utility space")
    self.best_observed = None if value is None else float(value)


def expected_improvement(self, mean, std, best_observed):
    return BayesianUncertaintyEstimator.expected_improvement(mean, std, getattr(self, "best_observed", best_observed))


__all__ = ["so3_log", "InvariantPointAttention", "MultiScaleNRPSDesigner", "SubstratePocketConditioner", "FlowMatchingBackbone", "update_from_proteus", "set_best_observed", "expected_improvement"]
