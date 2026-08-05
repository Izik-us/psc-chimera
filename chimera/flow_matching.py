"""
CHIMERA v2 — SE(3) Conditional Optimal Transport Flow Matching
==============================================================

Replaces the DDPM-based SE3Denoiser with conditional OT-Flow Matching.

Why this matters for PSC NRPS design:
  DDPM requires 200 denoising steps at inference — too slow for iterative
  PROTEUS active learning. OT-Flow Matching requires 10-20 function
  evaluations with equal or better sample quality.

Core idea (Lipman et al. 2022 / Yim et al. 2023 FrameDiff):
  Instead of a stochastic SDE, learn a deterministic ODE whose vector field
  transports probability mass from a source distribution to the target.

  For protein backbone generation:
    Source p0: Bacterial NRPS frames (not pure noise — this is the BRIDGE variant)
    Target p1: Real mammalian-functional A-domain structures
    Interpolation: x_t = (1-t)*x0 + t*x1  (optimal transport = straight paths)
    Velocity field: v*(x_t, t) = x1 - x0  (constant along path!)

  Training: minimize E[||v_θ(x_t, t) - (x1-x0)||²]
  Inference: integrate dx/dt = v_θ(x, t) with RK4 (20 steps)

Diffusion Bridge variant:
  When x0 is a specific bacterial NRPS backbone (not random noise), the flow
  learns to transform that backbone toward a mammalian-functional design while
  preserving catalytically essential geometry.

  This is a Schrödinger Bridge: minimum-energy transport between two known
  distributions with boundary constraints (fixed catalytic residues).

References:
  Lipman et al. 2022 — Flow Matching for Generative Modeling (ICLR 2023)
  Yim et al. 2023 — SE(3) Diffusion Model with Application to Protein Backbone Generation
  Liu et al. 2023 — I²SB: Image-to-Image Schrödinger Bridge (ICML 2023)
  Bose et al. 2023 — SE(3)-Stochastic Flow Matching for Protein Backbone Generation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Callable
from einops import rearrange, repeat

# ── SO(3) Operations ─────────────────────────────────────────────────────────


def hat(v: torch.Tensor) -> torch.Tensor:
    """Skew-symmetric matrix from 3-vector (the 'hat' operator). (..., 3) → (..., 3, 3)"""
    x, y, z = v[..., 0], v[..., 1], v[..., 2]
    O = torch.zeros_like(x)
    return torch.stack(
        [
            O,
            -z,
            y,
            z,
            O,
            -x,
            -y,
            x,
            O,
        ],
        dim=-1,
    ).reshape(*v.shape[:-1], 3, 3)


def so3_exp(omega: torch.Tensor) -> torch.Tensor:
    """
    Rodrigues' exponential map: so(3) → SO(3).
    omega: (..., 3) axis-angle → R: (..., 3, 3)
    """
    theta = omega.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    axis = omega / theta
    K = hat(axis)
    I = torch.eye(3, device=omega.device, dtype=omega.dtype)
    c = torch.cos(theta).unsqueeze(-1)
    s = torch.sin(theta).unsqueeze(-1)
    return I + s * K + (1 - c) * torch.einsum("...ij,...jk->...ik", K, K)


def so3_log(R: torch.Tensor) -> torch.Tensor:
    """
    SO(3) logarithmic map: SO(3) → so(3).
    R: (..., 3, 3) → omega: (..., 3)
    """
    trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    cos_a = ((trace - 1) / 2).clamp(-1 + 1e-6, 1 - 1e-6)
    angle = torch.acos(cos_a)
    denom = (2 * torch.sin(angle) + 1e-8).unsqueeze(-1).unsqueeze(-1)
    skew = (R - R.transpose(-1, -2)) / denom
    return torch.stack(
        [skew[..., 2, 1], skew[..., 0, 2], skew[..., 1, 0]], dim=-1
    ) * angle.unsqueeze(-1)


def so3_geodesic_interp(R0: torch.Tensor, R1: torch.Tensor, t: float) -> torch.Tensor:
    """
    Geodesic interpolation on SO(3) at fraction t ∈ [0,1].
    SLERP: R_t = R0 · exp(t · log(R0^T · R1))
    """
    delta = so3_log(torch.einsum("...ij,...kj->...ik", R0, R1))  # R0^T R1
    return torch.einsum("...ij,...jk->...ik", R0, so3_exp(t * delta))


def se3_interp(
    R0: torch.Tensor,
    t0: torch.Tensor,  # source frames
    R1: torch.Tensor,
    t1: torch.Tensor,  # target frames
    t: float,  # interpolation time ∈ [0,1]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Linear interpolation on SE(3) = SO(3) × R³."""
    Rt = so3_geodesic_interp(R0, R1, t)
    tt = (1 - t) * t0 + t * t1
    return Rt, tt


def so3_velocity(R_t: torch.Tensor, R0: torch.Tensor, R1: torch.Tensor) -> torch.Tensor:
    """
    Constant velocity field for SO(3) OT-Flow Matching.
    This is the target for the velocity network: d/dt[interp(R0,R1,t)] at R_t.
    """
    # In the tangent space at R_t: v* = log_{R_t}(R1) - log_{R_t}(R0)
    # Simplified for OT: v* = log(R0^T R1) in Lie algebra coords
    delta = so3_log(torch.einsum("...ij,...kj->...ik", R0, R1))
    return delta  # constant along the geodesic


# ── Invariant Point Attention (IPA) ──────────────────────────────────────────


class InvariantPointAttention(nn.Module):
    """
    SE(3)-equivariant attention mechanism (Jumper et al. AF2).
    Each residue attends to others using both scalar features and
    3D point coordinates — guaranteed equivariant to global rotations.

    Enhanced for CHIMERA v2:
      - Pair bias from EvoFormer pair_repr (co-evolutionary conditioning)
      - Additional point queries from substrate binding pocket coordinates
      - Substrate-adaptive attention weights
    """

    def __init__(
        self,
        d_single: int = 256,
        d_pair: int = 256,  # after PairProjection: 256 dim
        n_head: int = 12,
        n_qk_pts: int = 4,  # number of 3D point queries/keys per head
        n_v_pts: int = 8,  # number of 3D point values per head
        inf: float = 1e9,
    ):
        super().__init__()
        self.n_head = n_head
        self.n_qk_pts = n_qk_pts
        self.n_v_pts = n_v_pts
        d_head = d_single // n_head

        # Scalar attention projections
        self.q_s = nn.Linear(d_single, n_head * d_head, bias=False)
        self.k_s = nn.Linear(d_single, n_head * d_head, bias=False)
        self.v_s = nn.Linear(d_single, n_head * d_head, bias=False)

        # 3D point projections (equivariant)
        self.q_p = nn.Linear(d_single, n_head * n_qk_pts * 3, bias=False)
        self.k_p = nn.Linear(d_single, n_head * n_qk_pts * 3, bias=False)
        self.v_p = nn.Linear(d_single, n_head * n_v_pts * 3, bias=False)

        # Pair bias (from EvoFormer projected pair_repr)
        self.pair_bias = nn.Linear(d_pair, n_head, bias=False)

        # Learnable per-head weight for 3D vs scalar attention
        self.gamma = nn.Parameter(torch.ones(n_head))

        # Output projection
        d_out = n_head * (d_head + n_v_pts * 3 + d_pair)
        self.out = nn.Linear(d_out, d_single)

        # Substrate pocket conditioning (NEW in v2)
        # When substrate binding pocket coords are known, weight attention
        # toward residues near the pocket
        self.substrate_gate = nn.Linear(d_single, n_head)

    def forward(
        self,
        s: torch.Tensor,  # (B, L, d_single) single repr
        z: torch.Tensor,  # (B, L, L, d_pair) pair repr (projected EvoFormer)
        R: torch.Tensor,  # (B, L, 3, 3) rotation frames
        t: torch.Tensor,  # (B, L, 3)     translation (Cα positions)
        substrate_coords: Optional[torch.Tensor] = None,  # (B, K, 3) pocket atoms
    ) -> torch.Tensor:
        B, L, _ = s.shape

        # ── Scalar Q, K, V ───────────────────────────────────────────────
        def split_heads(x, n):
            return x.view(B, L, n, -1).permute(0, 2, 1, 3)  # (B, H, L, d)

        Q_s = split_heads(self.q_s(s), self.n_head)  # (B, H, L, d_head)
        K_s = split_heads(self.k_s(s), self.n_head)
        V_s = split_heads(self.v_s(s), self.n_head)

        # ── 3D Point Q, K, V — local to each residue frame ───────────────
        def transform_points(pts_local, R_frames, t_frames):
            # pts_local: (B, L, H*n_pts, 3) in local frame
            # Returns:   (B, L, H*n_pts, 3) in global frame
            pts = pts_local.view(B, L, -1, 3)
            R_exp = R_frames.unsqueeze(2).expand(-1, -1, pts.shape[2], -1, -1)
            t_exp = t_frames.unsqueeze(2).expand(-1, -1, pts.shape[2], -1)
            return torch.einsum("blnij,blnj->blni", R_exp, pts) + t_exp

        Q_p = transform_points(
            self.q_p(s).view(B, L, self.n_head * self.n_qk_pts, 3), R, t
        ).view(
            B, L, self.n_head, self.n_qk_pts, 3
        )  # (B, L, H, n_qk, 3)

        K_p = transform_points(
            self.k_p(s).view(B, L, self.n_head * self.n_qk_pts, 3), R, t
        ).view(B, L, self.n_head, self.n_qk_pts, 3)

        V_p = transform_points(
            self.v_p(s).view(B, L, self.n_head * self.n_v_pts, 3), R, t
        ).view(B, L, self.n_head, self.n_v_pts, 3)

        # ── Attention logits ──────────────────────────────────────────────
        # Scalar: (B, H, L, L)
        scale = Q_s.shape[-1] ** -0.5
        attn_s = torch.einsum("bhid,bhjd->bhij", Q_s, K_s) * scale

        # Point: sum of squared distances, summed over query points
        # (B, H, L, L, n_qk, 3) → (B, H, L, L)
        diff_p = Q_p.unsqueeze(3) - K_p.unsqueeze(2)  # (B, L, H, n_qk, 3) diff
        # Rearrange for (B, H, L, L, n_qk)
        diff_p = diff_p.permute(0, 2, 1, 3, 4, 5)  # (B,H,L,L,n_qk,3)
        attn_p = -(diff_p.norm(dim=-1) ** 2).sum(dim=-1)  # (B,H,L,L)

        # Pair bias
        attn_z = self.pair_bias(z).permute(0, 3, 1, 2)  # (B,H,L,L)

        # Substrate proximity gate (NEW)
        if substrate_coords is not None:
            # Bias attention toward substrate pocket residues
            diff_sub = t.unsqueeze(2) - substrate_coords.unsqueeze(1)  # (B,L,K,3)
            min_dist = (
                diff_sub.norm(dim=-1).min(dim=-1).values
            )  # (B,L) min dist to pocket
            prox_bias = torch.exp(-min_dist / 5.0)  # (B,L) decay ~5Å
            gate = self.substrate_gate(s).permute(0, 2, 1).unsqueeze(-1)  # (B,H,L,1)
            attn_z = attn_z + gate * prox_bias.unsqueeze(1).unsqueeze(-1)

        # Weight balance between scalar, point, pair terms
        w = F.softplus(self.gamma)  # (H,)
        w = w.view(1, self.n_head, 1, 1)
        logits = attn_s + w * attn_p + attn_z
        attn = F.softmax(logits, dim=-1)  # (B, H, L, L)

        # ── Aggregate ─────────────────────────────────────────────────────
        # Scalar output
        out_s = torch.einsum("bhij,bhjd->bhid", attn, V_s)  # (B,H,L,d_head)

        # Point output (global frame)
        out_p = torch.einsum("bhij,bhjnd->bhind", attn, V_p.permute(0, 2, 1, 3, 4)).sum(
            dim=-2
        )  # (B,H,L,3)
        # Transform back to local frame of each residue
        out_p_local = torch.einsum(
            "blji,bhlj->bhli",
            R,  # R^T = R.transpose(-1,-2) = R^-1 for SO(3)
            out_p - t.unsqueeze(1).expand_as(out_p),
        )  # (B, H, L, 3)

        # Pair output (weighted sum of pair features)
        out_z = torch.einsum("bhij,bijc->bhic", attn, z)  # (B,H,L,d_pair)

        # Concatenate and project
        out_s_ = out_s.permute(0, 2, 1, 3).reshape(B, L, -1)
        out_p_ = out_p_local.permute(0, 2, 1, 3).reshape(B, L, -1)
        out_z_ = out_z.permute(0, 2, 1, 3).reshape(B, L, -1)
        out = torch.cat([out_s_, out_p_, out_z_], dim=-1)
        return self.out(out)  # (B, L, d_single)


# ── Velocity Field Network (the heart of flow matching) ──────────────────────


class VelocityField(nn.Module):
    """
    Predicts the velocity field v_θ(x_t, t) for SE(3) flow matching.

    At each time t ∈ [0,1] and current state (R_t, t_t), predicts the velocity:
      v_R: d(R_t)/dt  — velocity in so(3) tangent space (an axis-angle vector)
      v_t: d(t_t)/dt  — velocity in R³

    Architecture: Stack of IPA blocks conditioned on:
      - Current backbone state (R_t, t_t)
      - EvoFormer pair_repr (co-evolutionary context)
      - Time embedding (sinusoidal + learned projection)
      - Substrate pocket coordinates (NEW: determines direction of change)
      - Source backbone features (NEW: bridge variant knows where it started)
    """

    def __init__(
        self,
        d_single: int = 256,
        d_pair: int = 256,
        n_blocks: int = 8,
        n_head: int = 12,
    ):
        super().__init__()
        self.d_single = d_single

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
            nn.Linear(12, d_single),  # 3 (position) + 9 (rotation flattened)
            nn.LayerNorm(d_single),
            nn.SiLU(),
            nn.Linear(d_single, d_single),
        )

        # Source backbone encoder (bridge variant — knows x0)
        self.source_encoder = nn.Sequential(
            nn.Linear(12, d_single // 2),
            nn.SiLU(),
            nn.Linear(d_single // 2, d_single),
        )

        # IPA stack
        self.ipa_blocks = nn.ModuleList(
            [IPABlock(d_single, d_pair, n_head) for _ in range(n_blocks)]
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
        pair_cond: torch.Tensor,  # (B, L, L, 256) EvoFormer conditioning
        evol_single: torch.Tensor,  # (B, L, 256) EvoFormer single repr
        R0: Optional[torch.Tensor] = None,  # (B, L, 3, 3) source backbone (bridge)
        t0: Optional[torch.Tensor] = None,  # (B, L, 3)    source translations
        substrate_coords: Optional[torch.Tensor] = None,  # (B, K, 3)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, L, _, _ = R_t.shape

        # Current backbone state → node features
        backbone_feat = torch.cat(
            [
                t_t,  # (B, L, 3) positions
                R_t.reshape(B, L, 9),  # (B, L, 9) rotation matrix
            ],
            dim=-1,
        )
        node_feat = self.backbone_encoder(backbone_feat)

        # Source backbone features (bridge: tells network where it came from)
        if R0 is not None and t0 is not None:
            source_feat = torch.cat([t0, R0.reshape(B, L, 9)], dim=-1)
            node_feat = node_feat + self.source_encoder(source_feat)

        # Add evolutionary context
        node_feat = node_feat + evol_single

        # Add time conditioning (broadcast across sequence length)
        time_emb = self.time_mlp(t_flow)  # (B, d_single)
        node_feat = node_feat + time_emb.unsqueeze(1)

        # IPA refinement
        for block in self.ipa_blocks:
            node_feat = block(node_feat, pair_cond, R_t, t_t, substrate_coords)

        # Predict velocities
        v_rot = self.v_rot_head(node_feat)  # (B, L, 3)
        v_trans = self.v_trans_head(node_feat)  # (B, L, 3)

        return v_rot, v_trans


class IPABlock(nn.Module):
    """One IPA block: IPA → FFN with layer norms."""

    def __init__(self, d_single: int, d_pair: int, n_head: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_single)
        self.ipa = InvariantPointAttention(d_single, d_pair, n_head)
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


# ── SE(3) Optimal Transport Flow Matching ────────────────────────────────────


class SE3FlowMatching(nn.Module):
    """
    Full SE(3) Conditional OT-Flow Matching Model.

    Training:
      1. Sample (R0, t0) from source (bacterial NRPS or random SO(3) × R3)
      2. Sample (R1, t1) from target (real mammalian-functional structure)
      3. Sample time t ~ U[0, 1]
      4. Interpolate: (R_t, t_t) = geodesic_interp((R0,t0), (R1,t1), t)
      5. Compute target velocity: v* = (R1-R0, t1-t0) in Lie algebra
      6. Minimize: ||v_θ(R_t, t_t, t) - v*||²

    Inference:
      1. Start from source distribution (R0, t0)
      2. Integrate ODE: d(R,t)/dt = v_θ(R, t, time)
      3. Use 4th-order Runge-Kutta, 20 steps (vs 200 for DDPM)

    Bridge Variant (PSC-specific):
      Source = specific bacterial NRPS backbone
      Target = desired mammalian-functional design
      Constraints = fixed catalytic residue positions (not moved by flow)
    """

    def __init__(
        self,
        d_single: int = 256,
        d_pair: int = 256,
        n_blocks: int = 8,
        n_head: int = 12,
    ):
        super().__init__()
        self.velocity_field = VelocityField(d_single, d_pair, n_blocks, n_head)

    def get_interpolation(
        self,
        R0: torch.Tensor,
        t0: torch.Tensor,
        R1: torch.Tensor,
        t1: torch.Tensor,
        t: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Linear interpolation on SE(3)."""
        return se3_interp(R0, t0, R1, t1, t)

    def flow_matching_loss(
        self,
        R0: torch.Tensor,  # (B, L, 3, 3) source rotations
        t0: torch.Tensor,  # (B, L, 3)    source translations
        R1: torch.Tensor,  # (B, L, 3, 3) target rotations
        t1: torch.Tensor,  # (B, L, 3)    target translations
        pair_cond: torch.Tensor,  # (B, L, L, 256)
        evol_single: torch.Tensor,  # (B, L, 256)
        fixed_mask: Optional[
            torch.Tensor
        ] = None,  # (B, L) don't move catalytic residues
        substrate_coords: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Conditional flow matching loss.
        Only updates residues not in fixed_mask (catalytic residues are fixed).
        """
        B = R0.shape[0]
        device = R0.device

        # Sample random time for each batch element
        t_flow = torch.rand(B, device=device)

        # Interpolate along the flow path
        Rt = torch.stack(
            [so3_geodesic_interp(R0[i], R1[i], t_flow[i].item()) for i in range(B)]
        )
        tt = t_flow.view(B, 1, 1) * t1 + (1 - t_flow.view(B, 1, 1)) * t0

        # Predict velocity
        v_rot_pred, v_trans_pred = self.velocity_field(
            R_t=Rt,
            t_t=tt,
            t_flow=t_flow,
            pair_cond=pair_cond,
            evol_single=evol_single,
            R0=R0,
            t0=t0,
            substrate_coords=substrate_coords,
        )

        # Target velocity (constant along OT path)
        v_rot_target = so3_log(torch.einsum("...ij,...kj->...ik", R0, R1))  # (B,L,3)
        v_trans_target = t1 - t0  # (B, L, 3)

        # Compute loss
        rot_loss = F.mse_loss(v_rot_pred, v_rot_target, reduction="none").sum(-1)
        trans_loss = F.mse_loss(v_trans_pred, v_trans_target, reduction="none").sum(-1)
        loss = rot_loss + trans_loss  # (B, L)

        # Zero out loss on fixed positions (catalytic residues)
        if fixed_mask is not None:
            loss = loss * (~fixed_mask).float()

        return loss.mean()

    @torch.no_grad()
    def sample(
        self,
        R0: torch.Tensor,  # (B, L, 3, 3) source backbone
        t0: torch.Tensor,  # (B, L, 3)    source positions
        pair_cond: torch.Tensor,
        evol_single: torch.Tensor,
        n_steps: int = 20,  # 20 steps — 10x faster than DDPM
        fixed_mask: Optional[torch.Tensor] = None,
        substrate_coords: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate backbone by integrating the ODE with RK4.
        Much faster than DDPM: 20 NFE vs 200.
        """
        B = R0.shape[0]
        device = R0.device
        dt = 1.0 / n_steps

        R_curr, t_curr = R0.clone(), t0.clone()

        for step in range(n_steps):
            t_now = torch.full((B,), step * dt, device=device)

            # RK4 integration on SE(3)
            k1_r, k1_t = self.velocity_field(
                R_curr, t_curr, t_now, pair_cond, evol_single, R0, t0, substrate_coords
            )

            R_mid1 = torch.stack(
                [
                    so3_geodesic_interp(
                        R_curr[i], so3_exp(k1_r[i] * dt / 2) @ R_curr[i], 0.5
                    )
                    for i in range(B)
                ]
            )
            t_mid1 = t_curr + k1_t * dt / 2

            k2_r, k2_t = self.velocity_field(
                R_mid1,
                t_mid1,
                t_now + dt / 2,
                pair_cond,
                evol_single,
                R0,
                t0,
                substrate_coords,
            )

            R_mid2 = torch.stack(
                [
                    so3_geodesic_interp(
                        R_curr[i], so3_exp(k2_r[i] * dt / 2) @ R_curr[i], 0.5
                    )
                    for i in range(B)
                ]
            )
            t_mid2 = t_curr + k2_t * dt / 2

            k3_r, k3_t = self.velocity_field(
                R_mid2,
                t_mid2,
                t_now + dt / 2,
                pair_cond,
                evol_single,
                R0,
                t0,
                substrate_coords,
            )

            R_end = torch.stack(
                [
                    so3_geodesic_interp(
                        R_curr[i], so3_exp(k3_r[i] * dt) @ R_curr[i], 1.0
                    )
                    for i in range(B)
                ]
            )
            t_end = t_curr + k3_t * dt

            k4_r, k4_t = self.velocity_field(
                R_end,
                t_end,
                t_now + dt,
                pair_cond,
                evol_single,
                R0,
                t0,
                substrate_coords,
            )

            # RK4 update
            v_r = (k1_r + 2 * k2_r + 2 * k3_r + k4_r) / 6
            v_t = (k1_t + 2 * k2_t + 2 * k3_t + k4_t) / 6

            # Update rotations via exponential map
            R_new = torch.stack([so3_exp(v_r[i] * dt) @ R_curr[i] for i in range(B)])
            t_new = t_curr + v_t * dt

            # Clamp fixed positions to source (catalytic residues don't move)
            if fixed_mask is not None:
                R_new = torch.where(fixed_mask.view(B, -1, 1, 1), R0, R_new)
                t_new = torch.where(fixed_mask.view(B, -1, 1), t0, t_new)

            R_curr, t_curr = R_new, t_new

        return R_curr, t_curr
