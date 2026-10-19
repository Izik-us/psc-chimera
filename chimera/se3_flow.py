"""Canonical SE(3) drift network and Schrodinger-bridge backbone."""

from __future__ import annotations

import math
from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .schrodinger_bridge import SE3SchrodingerBridge

def _resolve_ipa_heads(d_single: int, n_head: Optional[int]) -> int:
    if d_single <= 0:
        raise ValueError("d_single must be positive")
    if n_head is not None:
        if n_head <= 0 or d_single % n_head != 0:
            raise ValueError(
                f"d_single ({d_single}) must be divisible by n_head ({n_head})"
            )
        return n_head
    return next(heads for heads in range(min(12, d_single), 0, -1) if d_single % heads == 0)


class InvariantPointAttention(nn.Module):
    """
    Local geometric attention using scalar, pair, and residue-frame features.

    CHIMERA's IPA variant follows the AlphaFold-style scalar/pair/point
    attention decomposition while retaining CHIMERA-specific substrate
    conditioning. Point values are aggregated in global coordinates and
    transformed back into the query residue's local frame before projection.
    """

    def __init__(
        self,
        d_single: int = 256,
        d_pair: int = 256,
        n_head: Optional[int] = None,
        n_qk_pts: int = 4,
        n_v_pts: int = 8,
        inf: float = 1e9,
    ):
        super().__init__()
        n_head = _resolve_ipa_heads(d_single, n_head)
        if n_qk_pts <= 0 or n_v_pts <= 0:
            raise ValueError("n_qk_pts and n_v_pts must be positive")
        self.n_head = n_head
        self.n_qk_pts = n_qk_pts
        self.n_v_pts = n_v_pts
        self.inf = float(inf)
        d_head = d_single // n_head

        # Scalar attention projections
        self.q_s = nn.Linear(d_single, n_head * d_head, bias=False)
        self.k_s = nn.Linear(d_single, n_head * d_head, bias=False)
        self.v_s = nn.Linear(d_single, n_head * d_head, bias=False)

        # 3D point projections (equivariant)
        self.q_p = nn.Linear(d_single, n_head * n_qk_pts * 3, bias=False)
        self.k_p = nn.Linear(d_single, n_head * n_qk_pts * 3, bias=False)
        self.v_p = nn.Linear(d_single, n_head * n_v_pts * 3, bias=False)

        # Pair bias from the projected evolutionary pair representation.
        self.pair_bias = nn.Linear(d_pair, n_head, bias=False)

        # Learnable per-head weight for 3D vs scalar attention
        self.gamma = nn.Parameter(torch.ones(n_head))

        # Variance control for the squared point-distance logits.
        self.scale_scalar = d_head ** -0.5
        self.scale_point = 1.0 / (2.0 * 3.0 * n_qk_pts)

        # Each head contributes scalar values, 3 coordinates per value point,
        # one invariant norm per value point, and pair values.
        d_out = n_head * (d_head + (n_v_pts * 3) + n_v_pts + d_pair)
        self.out = nn.Linear(d_out, d_single)

        # When pocket coordinates are supplied, bias attention toward nearby residues.
        self.substrate_gate = nn.Linear(d_single, n_head)

    def forward(
        self,
        s: torch.Tensor,
        z: torch.Tensor,
        R: torch.Tensor,
        t: torch.Tensor,
        substrate_coords: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, L, _ = s.shape

        if z.shape[:3] != (B, L, L):
            raise ValueError("z must have shape (B, L, L, d_pair)")
        if R.shape != (B, L, 3, 3) or t.shape != (B, L, 3):
            raise ValueError("R must be (B,L,3,3) and t must be (B,L,3)")

        def split_heads(x: torch.Tensor, n: int) -> torch.Tensor:
            return x.view(B, L, n, -1).permute(0, 2, 1, 3)

        Q_s = split_heads(self.q_s(s), self.n_head)
        K_s = split_heads(self.k_s(s), self.n_head)
        V_s = split_heads(self.v_s(s), self.n_head)

        def transform_points(pts_local, R_frames, t_frames, n_pts):
            # pts_local is (B,L,H,n_pts,3) in residue-local coordinates.
            pts = pts_local.view(B, L, self.n_head, n_pts, 3)
            # x_global = R x_local + t. Return (B,H,L,n_pts,3).
            pts_global = torch.einsum(
                "blij,bhlvj->bhlvi",
                R_frames,
                pts.permute(0, 2, 1, 3, 4),
            )
            return pts_global + t_frames.unsqueeze(1).unsqueeze(3)

        Q_p = transform_points(self.q_p(s), R, t, self.n_qk_pts)
        K_p = transform_points(self.k_p(s), R, t, self.n_qk_pts)
        V_p = transform_points(self.v_p(s), R, t, self.n_v_pts)

        # Scalar attention logits.
        attn_s = torch.einsum("bhid,bhjd->bhij", Q_s, K_s) * self.scale_scalar

        # Point attention logits: sum squared distances across coordinates and
        # query/key point slots, with variance control before softmax.
        diff_p = Q_p.unsqueeze(3) - K_p.unsqueeze(2)
        attn_p = -0.5 * diff_p.square().sum(dim=-1).sum(dim=-1) * self.scale_point

        # Pair bias.
        attn_z = self.pair_bias(z).permute(0, 3, 1, 2)

        # Substrate proximity gate.
        if substrate_coords is not None:
            if substrate_coords.ndim != 3 or substrate_coords.shape[0] != B or substrate_coords.shape[-1] != 3:
                raise ValueError("substrate_coords must have shape (B,K,3)")
            diff_sub = t.unsqueeze(2) - substrate_coords.unsqueeze(1)
            min_dist = diff_sub.norm(dim=-1).min(dim=-1).values
            prox_bias = torch.exp(-min_dist / 5.0)
            gate = self.substrate_gate(s).permute(0, 2, 1).unsqueeze(-1)
            attn_z = attn_z + gate * prox_bias.unsqueeze(1).unsqueeze(-1)

        gamma_soft = F.softplus(self.gamma).view(1, self.n_head, 1, 1)
        logits = attn_s + gamma_soft * attn_p + attn_z
        attn = F.softmax(logits, dim=-1)

        # Scalar value aggregation.
        out_s = torch.einsum("bhij,bhjd->bhid", attn, V_s)

        # Correct IPA point-value aggregation: V_p is mixed over the key
        # residue axis in global coordinates, rather than replacing V_p with
        # raw C-alpha displacement vectors.
        out_p_global = torch.einsum("bhij,bhjvc->bhivc", attn, V_p)

        # Convert the aggregated global points into the query residue's local
        # frame: x_local = R^T (x_global - t).
        out_p_shifted = out_p_global - t.unsqueeze(1).unsqueeze(3)
        out_p_local = torch.einsum("blji,bhlvj->bhlvi", R, out_p_shifted)

        # Point norms are invariant to global rigid transforms and retain
        # radial information associated with each learned value point.
        out_p_norm = out_p_local.norm(dim=-1)

        # Pair value aggregation remains CHIMERA's explicit pair-to-single
        # output pathway.
        out_z = torch.einsum("bhij,bijc->bhic", attn, z)

        out_s_ = out_s.permute(0, 2, 1, 3).reshape(B, L, -1)
        out_p_ = out_p_local.permute(0, 2, 1, 3, 4).reshape(B, L, -1)
        out_p_norm_ = out_p_norm.permute(0, 2, 1, 3).reshape(B, L, -1)
        out_z_ = out_z.permute(0, 2, 1, 3).reshape(B, L, -1)
        out = torch.cat([out_s_, out_p_, out_p_norm_, out_z_], dim=-1)
        return self.out(out)


# ── Canonical SE(3) drift field ───────────────────────────────────────────────


class VelocityField(nn.Module):
    """
    Predicts the velocity field v_θ(x_t, t) for SE(3) flow matching.

    At each time t ∈ [0,1] and current state (R_t, t_t), predicts the velocity:
      v_R: d(R_t)/dt  — velocity in so(3) tangent space (an axis-angle vector)
      v_t: d(t_t)/dt  — velocity in R³

    Architecture: Stack of IPA blocks conditioned on:
      - Current backbone state (R_t, t_t)
      - Pair features (co-evolutionary context)
      - Time embedding (sinusoidal + learned projection)
      - Optional substrate pocket coordinates
      - Source backbone features
    """

    def __init__(
        self,
        d_single: int = 768,
        d_pair: int = 512,
        n_blocks: int = 28,
        n_head: Optional[int] = None,
        ipa_class=InvariantPointAttention,
    ):
        super().__init__()
        self.d_single = d_single
        n_head = _resolve_ipa_heads(d_single, n_head)

        # Time embedding: sinusoidal → learned projection
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(d_single),
            nn.Linear(d_single, d_single * 4),
            nn.SiLU(),
            nn.Linear(d_single * 4, d_single),
        )

        # Backbone state encoder: encodes current (R_t, t_t) → node features
        # Features: Cα position + orientation encoded as 9 rotation matrix entries
        self.backbone_encoder = nn.Sequential(
            nn.Linear(4, d_single),  # invariant local geometry statistics
            nn.LayerNorm(d_single),
            nn.SiLU(),
            nn.Linear(d_single, d_single),
        )

        # Source backbone encoder (bridge variant — knows x0)
        self.source_encoder = nn.Sequential(
            nn.Linear(4, d_single // 2),
            nn.SiLU(),
            nn.Linear(d_single // 2, d_single),
        )

        # IPA stack
        self.ipa_blocks = nn.ModuleList(
            [IPABlock(d_single, d_pair, n_head, ipa_class) for _ in range(n_blocks)]
        )

        # Velocity output heads
        self.v_rot_head = nn.Sequential(
            nn.LayerNorm(d_single),
            nn.Linear(d_single, d_single // 2),
            nn.SiLU(),
            nn.Linear(d_single // 2, 3),  # axis-angle velocity in so(3)
        )
        self.v_trans_head = nn.Sequential(
            nn.LayerNorm(d_single),
            nn.Linear(d_single, d_single // 2),
            nn.SiLU(),
            nn.Linear(d_single // 2, 3),  # translation velocity in R³
        )

    def forward(
        self,
        R_t: torch.Tensor,  # (B, L, 3, 3) current rotations
        t_t: torch.Tensor,  # (B, L, 3)    current translations
        t_flow: torch.Tensor,  # (B,) flow time ∈ [0, 1]
        pair_cond: torch.Tensor,  # (B, L, L, d_pair) pair conditioning
        evol_single: torch.Tensor,  # (B, L, d_single) projected evolutionary single representation
        R0: Optional[torch.Tensor] = None,  # (B, L, 3, 3) source backbone (bridge)
        t0: Optional[torch.Tensor] = None,  # (B, L, 3)    source translations
        substrate_coords: Optional[torch.Tensor] = None,  # (B, K, 3)
        evol_conditioning_fn: Optional[Callable] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, L, _, _ = R_t.shape

        # Scalar geometry is translation/rotation invariant. Translation
        # velocity is reconstructed from a local-frame vector below.
        rel = t_t.unsqueeze(2) - t_t.unsqueeze(1)
        distances = rel.norm(dim=-1)
        local_distance = distances.mean(dim=-1, keepdim=True)
        local_scale = distances.std(dim=-1, keepdim=True)
        frame_trace = R_t.transpose(-1, -2) @ R_t
        frame_invariant = frame_trace.diagonal(dim1=-2, dim2=-1).mean(-1, keepdim=True)
        backbone_feat = torch.cat([local_distance, local_scale, frame_invariant, frame_invariant * 0], dim=-1)
        node_feat = self.backbone_encoder(backbone_feat)

        # Source backbone features (bridge: tells network where it came from)
        if R0 is not None and t0 is not None:
            source_rel = t0.unsqueeze(2) - t0.unsqueeze(1)
            source_dist = source_rel.norm(dim=-1)
            source_mean = source_dist.mean(dim=-1, keepdim=True)
            source_std = source_dist.std(dim=-1, keepdim=True)
            source_frame = R0.transpose(-1, -2) @ R0
            source_trace = source_frame.diagonal(dim1=-2, dim2=-1).mean(-1, keepdim=True)
            source_feat = torch.cat([source_mean, source_std, source_trace, source_trace * 0], dim=-1)
            node_feat = node_feat + self.source_encoder(source_feat)

        # Add evolutionary context
        node_feat = node_feat + evol_single
        if evol_conditioning_fn is not None:
            node_feat = node_feat + evol_conditioning_fn(node_feat, t_flow)

        # Add time conditioning (broadcast across sequence length)
        time_emb = self.time_mlp(t_flow)  # (B, d_single)
        node_feat = node_feat + time_emb.unsqueeze(1)

        # IPA refinement
        for block in self.ipa_blocks:
            node_feat = block(node_feat, pair_cond, R_t, t_t, substrate_coords)

        # Predict velocities
        v_rot = self.v_rot_head(node_feat)  # (B, L, 3)
        v_trans_local = self.v_trans_head(node_feat)  # (B, L, 3)
        v_trans = torch.einsum("blij,blj->bli", R_t, v_trans_local)

        return v_rot, v_trans


class IPABlock(nn.Module):
    """One IPA block: IPA → FFN with layer norms."""

    def __init__(self, d_single: int, d_pair: int, n_head: int, ipa_class=InvariantPointAttention):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_single)
        self.ipa = ipa_class(d_single, d_pair, n_head)
        self.norm2 = nn.LayerNorm(d_single)
        self.ffn = nn.Sequential(
            nn.Linear(d_single, d_single * 4),
            nn.SiLU(),
            nn.Linear(d_single * 4, d_single),
        )

    def forward(self, s, z, R, t, substrate_coords=None):
        s = s + self.ipa(self.norm1(s), z, R, t, substrate_coords)
        s = s + self.ffn(self.norm2(s))
        return s


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal time embedding for the flow time t ∈ [0,1]."""

    def __init__(self, d: int):
        super().__init__()
        self.d = d
        self.proj = nn.Linear(d, d)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (B,) scalar times
        half = self.d // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        args = t.unsqueeze(-1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return self.proj(emb)


# ── Canonical stochastic SE(3) backbone transport ───────────────────────────
class FlowMatchingBackbone(nn.Module):
    """Canonical SE(3) flow backbone composed from flow and bridge primitives."""

    def __init__(
        self,
        d_single: int = 768,
        d_pair: int = 512,
        n_blocks: int = 28,
        n_head: Optional[int] = None,
        diffusion: float = 0.05,
    ) -> None:
        super().__init__()
        self.d_single = d_single
        self.d_pair = d_pair
        self.n_blocks = n_blocks
        self.flow_model = nn.Module()
        self.flow_model.add_module(
            "velocity_field",
            VelocityField(
                d_single=d_single,
                d_pair=d_pair,
                n_blocks=n_blocks,
                n_head=n_head,
                ipa_class=InvariantPointAttention,
            ),
        )
        self.frozen_bridge = nn.Identity()
        self.sb_model = SE3SchrodingerBridge(
            self.flow_model.velocity_field,
            diffusion=diffusion,
            sinkhorn_iters=50,
        )

    def sample(
        self,
        R0: torch.Tensor,
        t0: torch.Tensor,
        pair_cond: torch.Tensor,
        evol_single: torch.Tensor,
        n_steps: int = 20,
        fixed_mask: Optional[torch.Tensor] = None,
        substrate_coords: Optional[torch.Tensor] = None,
        evol_conditioning_fn: Optional[Callable] = None,
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.sb_model.sample(
            R0,
            t0,
            pair_cond,
            self.frozen_bridge(evol_single),
            n_steps=n_steps,
            fixed_mask=fixed_mask,
            substrate_coords=substrate_coords,
            evol_conditioning_fn=evol_conditioning_fn,
            generator=generator,
        )

    def loss(
        self,
        R0: torch.Tensor,
        t0: torch.Tensor,
        R1: torch.Tensor,
        t1: torch.Tensor,
        pair_cond: torch.Tensor,
        evol_single: torch.Tensor,
        fixed_mask: Optional[torch.Tensor] = None,
        substrate_coords: Optional[torch.Tensor] = None,
        evol_conditioning_fn: Optional[Callable] = None,
    ) -> torch.Tensor:
        return self.sb_model.bridge_loss(
            R0,
            t0,
            R1,
            t1,
            pair_cond,
            self.frozen_bridge(evol_single),
            fixed_mask=fixed_mask,
            substrate_coords=substrate_coords,
            evol_conditioning_fn=evol_conditioning_fn,
        )
