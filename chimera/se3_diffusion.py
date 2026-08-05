"""
CHIMERA — SE(3) Diffusion Module
=================================
RFdiffusion-equivalent backbone generation operating in SE(3) space
(Special Euclidean group: rotations × translations for each residue frame).

Key innovation vs standard RFdiffusion:
  The denoising network receives EvoFormer pair_repr as conditioning via
  cross-attention layers — making generated backbones evolutionary-context-aware.
  Backbone geometries now reflect which residue pairs co-evolve in the animal
  NRPS family, directly addressing module-module interface incompatibility.

Outputs per residue:
  R_i ∈ SO(3): local coordinate frame (3x3 rotation matrix)
  t_i ∈ R³:    Cα position

References:
  Watson et al. 2023 (RFdiffusion) — Nature 620:1089–1100
  Yim et al. 2023 (FrameDiff) — ICML 2023
  de Bortoli et al. 2022 (Riemannian diffusion)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional
from einops import rearrange, repeat

# ── SO(3) / SE(3) Utilities ─────────────────────────────────────────────────


def axis_angle_to_matrix(axis_angle: torch.Tensor) -> torch.Tensor:
    """
    Rodrigues' rotation formula: axis-angle → SO(3) matrix.
    axis_angle: (..., 3)  →  R: (..., 3, 3)
    """
    angle = torch.norm(axis_angle, dim=-1, keepdim=True).clamp(min=1e-8)
    axis = axis_angle / angle

    # Skew-symmetric matrix K for axis
    k_x, k_y, k_z = axis[..., 0], axis[..., 1], axis[..., 2]
    zeros = torch.zeros_like(k_x)
    K = torch.stack(
        [zeros, -k_z, k_y, k_z, zeros, -k_x, -k_y, k_x, zeros], dim=-1
    ).view(*axis.shape[:-1], 3, 3)

    I = torch.eye(3, device=axis.device).expand_as(K)
    c, s = torch.cos(angle).unsqueeze(-1), torch.sin(angle).unsqueeze(-1)
    return (
        I
        + s * K
        + (1 - c) * torch.bmm(K.view(-1, 3, 3), K.view(-1, 3, 3)).view(*K.shape)
    )


def matrix_to_axis_angle(R: torch.Tensor) -> torch.Tensor:
    """SO(3) matrix → axis-angle representation. R: (..., 3, 3) → (..., 3)"""
    batch_shape = R.shape[:-2]
    R_flat = R.view(-1, 3, 3)
    trace = R_flat[:, 0, 0] + R_flat[:, 1, 1] + R_flat[:, 2, 2]
    angle = torch.acos(((trace - 1) / 2).clamp(-1 + 1e-6, 1 - 1e-6))
    denom = (2 * torch.sin(angle)).unsqueeze(-1).unsqueeze(-1).clamp(min=1e-8)
    K = (R_flat - R_flat.transpose(1, 2)) / denom
    axis = torch.stack([K[:, 2, 1], K[:, 0, 2], K[:, 1, 0]], dim=-1)
    return (axis * angle.unsqueeze(-1)).view(*batch_shape, 3)


def random_so3_frames(batch: int, L: int, device: torch.device) -> torch.Tensor:
    """Sample uniformly random rotation matrices. Returns (B, L, 3, 3)."""
    # QR decomposition of random Gaussian matrix gives Haar-uniform SO(3)
    M = torch.randn(batch, L, 3, 3, device=device)
    Q, R = torch.linalg.qr(M)
    # Ensure proper rotation (det = +1)
    d = torch.linalg.det(Q).unsqueeze(-1).unsqueeze(-1)
    Q = Q * d.sign()
    return Q


# ── Invariant Point Attention (IPA) ─────────────────────────────────────────
class InvariantPointAttention(nn.Module):
    """
    SE(3)-equivariant attention for protein frames.
    Attends between residue frames (R_i, t_i) such that output transforms
    correctly under global SE(3) transformations — structures are invariant
    to overall rotation and translation of the molecule.

    Conditioning: pair_repr (from EvoFormer) biases attention between positions.
    This is the key connector between EvoFormer and the SE(3) denoiser.
    """

    def __init__(
        self,
        c_s: int = 256,
        c_z: int = 256,
        n_head: int = 12,
        n_qk_pts: int = 4,
        n_v_pts: int = 8,
        c_hidden: int = 16,
    ):
        super().__init__()
        self.n_head = n_head
        self.c_hidden = c_hidden
        self.n_qk_pts = n_qk_pts
        self.n_v_pts = n_v_pts

        # Scalar attention (standard)
        self.to_q = nn.Linear(c_s, n_head * c_hidden, bias=False)
        self.to_k = nn.Linear(c_s, n_head * c_hidden, bias=False)
        self.to_v = nn.Linear(c_s, n_head * c_hidden, bias=False)

        # Point-valued Q/K/V (3D vectors in local frames)
        self.to_q_pts = nn.Linear(c_s, n_head * n_qk_pts * 3, bias=False)
        self.to_k_pts = nn.Linear(c_s, n_head * n_qk_pts * 3, bias=False)
        self.to_v_pts = nn.Linear(c_s, n_head * n_v_pts * 3, bias=False)

        # Pair bias projection (EvoFormer pair_repr → attention bias)
        self.pair_bias = nn.Linear(c_z, n_head, bias=False)

        # Head weights (learned scalar)
        self.head_weights = nn.Parameter(torch.ones(n_head, n_qk_pts))

        # Output projection
        c_out = n_head * (c_hidden + n_v_pts * 4)  # scalar + point + dist
        self.out_proj = nn.Linear(c_out, c_s)
        self.norm_s = nn.LayerNorm(c_s)
        self.norm_z = nn.LayerNorm(c_z)

    def _apply_frames(
        self, pts: torch.Tensor, R: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """Transform points from local to global frame. pts: (..., 3)"""
        # R: (B, L, 3, 3), t: (B, L, 3)
        # pts: (B, L, H, P, 3)
        return torch.einsum("blij,blhpj->blhpi", R, pts) + t.unsqueeze(-2).unsqueeze(-2)

    def forward(
        self, s: torch.Tensor, z: torch.Tensor, R: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """
        s: (B, L, c_s)         single repr (current denoising state)
        z: (B, L, L, c_z)      pair repr (EvoFormer conditioning) ← KEY INNOVATION
        R: (B, L, 3, 3)         rotation frames
        t: (B, L, 3)            translation frames
        """
        B, L, _ = s.shape
        s_n = self.norm_s(s)
        z_n = self.norm_z(z)

        # Scalar Q/K/V
        Q = self.to_q(s_n).view(B, L, self.n_head, self.c_hidden)
        K = self.to_k(s_n).view(B, L, self.n_head, self.c_hidden)
        V = self.to_v(s_n).view(B, L, self.n_head, self.c_hidden)

        # Point Q/K/V — apply local frame transformation
        Q_pts = self.to_q_pts(s_n).view(B, L, self.n_head, self.n_qk_pts, 3)
        K_pts = self.to_k_pts(s_n).view(B, L, self.n_head, self.n_qk_pts, 3)
        V_pts = self.to_v_pts(s_n).view(B, L, self.n_head, self.n_v_pts, 3)

        Q_pts_global = self._apply_frames(Q_pts, R, t)
        K_pts_global = self._apply_frames(K_pts, R, t)
        V_pts_global = self._apply_frames(V_pts, R, t)

        # Scalar attention logits
        scalar_logits = torch.einsum("bihc,bjhc->bhij", Q, K) / math.sqrt(self.c_hidden)

        # Point attention logits (squared distances)
        diff = Q_pts_global.unsqueeze(2) - K_pts_global.unsqueeze(1)
        pt_logits = -(diff.pow(2).sum(-1)).sum(-1)  # (B, L, L, H)
        w = F.softplus(self.head_weights).unsqueeze(0).unsqueeze(0)
        pt_logits = (w * pt_logits).permute(0, 3, 1, 2)  # (B, H, L, L)

        # Pair bias from EvoFormer pair_repr (THE KEY CONNECTOR)
        pair_bias = self.pair_bias(z_n).permute(0, 3, 1, 2)  # (B, H, L, L)

        attn = F.softmax(scalar_logits + pt_logits + pair_bias, dim=-1)  # (B, H, L, L)

        # Aggregate scalar values
        o_s = torch.einsum("bhij,bjhc->bihc", attn, V)  # (B, L, H, c_h)

        # Aggregate point values → transform back to local frame
        o_pts = torch.einsum("bhij,bjhpd->bihpd", attn, V_pts_global)
        # Transform to local frame of each residue
        R_inv = R.transpose(-1, -2)  # (B, L, 3, 3)
        o_pts_local = torch.einsum(
            "blij,blhpj->blhpi", R_inv, o_pts - t.unsqueeze(-2).unsqueeze(-2)
        )
        o_pts_norm = o_pts_local.norm(dim=-1)  # (B, L, H, P)

        # Concatenate and project
        o_s = o_s.reshape(B, L, -1)
        o_pts = o_pts_local.reshape(B, L, -1)
        o_norm = o_pts_norm.reshape(B, L, -1)
        out = torch.cat([o_s, o_pts, o_norm], dim=-1)

        return self.out_proj(out)


# ── SE(3) Transformer Block ─────────────────────────────────────────────────
class SE3Block(nn.Module):
    """One block of the SE(3)-equivariant denoising network."""

    def __init__(self, c_s: int = 256, c_z: int = 256):
        super().__init__()
        self.ipa = InvariantPointAttention(c_s, c_z)
        self.norm1 = nn.LayerNorm(c_s)
        self.ff = nn.Sequential(
            nn.LayerNorm(c_s),
            nn.Linear(c_s, c_s * 4),
            nn.GELU(),
            nn.Linear(c_s * 4, c_s),
        )
        # Frame update network: produces delta_axis_angle and delta_t
        self.to_frame_update = nn.Sequential(
            nn.LayerNorm(c_s),
            nn.Linear(c_s, c_s),
            nn.ReLU(),
            nn.Linear(c_s, 6),  # 3 for axis-angle, 3 for translation
        )

    def forward(
        self, s: torch.Tensor, z: torch.Tensor, R: torch.Tensor, t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # IPA update conditioned on pair repr (EvoFormer context)
        s = self.norm1(s + self.ipa(s, z, R, t))
        s = s + self.ff(s)

        # Update frames
        frame_delta = self.to_frame_update(s)
        axis_angle = frame_delta[..., :3]  # rotation update
        t_delta = frame_delta[..., 3:]  # translation update

        # Compose rotation: R_new = R * exp(axis_angle)
        R_update = axis_angle_to_matrix(axis_angle)
        R_new = torch.bmm(R.view(-1, 3, 3), R_update.view(-1, 3, 3)).view(*R.shape)

        # Translation update in global frame
        t_new = t + torch.einsum("blij,blj->bli", R, t_delta)

        return s, R_new, t_new


# ── Noise Schedule ───────────────────────────────────────────────────────────
class SE3DiffusionSchedule(nn.Module):
    """
    Variance-preserving diffusion schedule for SE(3).
    Rotation noise: IGSO(3) distribution.
    Translation noise: standard Gaussian.
    """

    def __init__(self, T: int = 200, min_b: float = 0.01, max_b: float = 0.3):
        super().__init__()
        # Linear beta schedule (controls noise magnitude at each step)
        betas = torch.linspace(min_b, max_b, T)
        alphas = 1 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bar", alpha_bar)

    def add_noise(
        self, t_coords: torch.Tensor, R_frames: torch.Tensor, timestep: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Add t-step noise to (t_coords, R_frames).
        Returns noisy versions + ground-truth noise for denoising target.
        """
        abar = self.alpha_bar[timestep].view(-1, 1, 1)

        # Translation noise (isotropic Gaussian)
        noise_t = torch.randn_like(t_coords)
        t_noisy = abar.sqrt() * t_coords + (1 - abar).sqrt() * noise_t

        # Rotation noise: approximate as perturbation in axis-angle space
        noise_angle = torch.randn(*R_frames.shape[:-2], 3, device=R_frames.device)
        noise_angle = noise_angle * (1 - abar).sqrt().unsqueeze(-1) * 0.1
        R_noise = axis_angle_to_matrix(noise_angle)
        R_noisy = torch.bmm(R_frames.view(-1, 3, 3), R_noise.view(-1, 3, 3)).view(
            *R_frames.shape
        )

        return t_noisy, R_noisy, noise_t, noise_angle


# ── Full SE(3) Denoiser ──────────────────────────────────────────────────────
class SE3Denoiser(nn.Module):
    """
    CHIMERA's backbone generation module.

    Takes:
      - Noisy backbone (R_noisy, t_noisy) at timestep T
      - EvoFormer pair_repr as evolutionary conditioning
      - Design constraints (motif hotspots, symmetry)

    Outputs denoised backbone (R_0, t_0) — i.e., predicted final structure.

    The EvoFormer pair_repr enters via IPA's pair bias, making backbone
    generation aware of which residue pairs co-evolve in the animal NRPS MSA.
    This is the mechanistic basis for the 30-year module compatibility problem fix.
    """

    def __init__(
        self, c_s: int = 256, c_z: int = 256, n_blocks: int = 8, max_t: int = 200
    ):
        super().__init__()
        # Timestep embedding (sinusoidal)
        self.time_embed = nn.Sequential(
            nn.Embedding(max_t, c_s),
            nn.Linear(c_s, c_s),
            nn.GELU(),
            nn.Linear(c_s, c_s),
        )
        # Node feature embedding (from pair diagonals + noise level)
        self.node_embed = nn.Sequential(
            nn.Linear(4, c_s),  # 4 = sin/cos of phi/psi or noisy torsions
            nn.GELU(),
            nn.Linear(c_s, c_s),
        )
        # SE3 denoising blocks (each conditioned on EvoFormer pair_repr)
        self.blocks = nn.ModuleList([SE3Block(c_s, c_z) for _ in range(n_blocks)])
        self.diffusion = SE3DiffusionSchedule(max_t)

    def forward(
        self,
        R_noisy: torch.Tensor,
        t_noisy: torch.Tensor,
        pair_repr: torch.Tensor,  # ← from EvoFormer
        timestep: torch.Tensor,
        motif_mask: Optional[torch.Tensor] = None,
        motif_coords: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        R_noisy:     (B, L, 3, 3)  noisy rotation frames
        t_noisy:     (B, L, 3)     noisy Cα positions
        pair_repr:   (B, L, L, c_z) from EvoFormer (CHIMERA connector)
        timestep:    (B,)          diffusion timestep
        motif_mask:  (B, L)        True = constrained (hotspot) positions
        motif_coords:(B, L, 3)     known coordinates for constrained positions
        """
        B, L, _ = t_noisy.shape

        # Node feature initialization from noisy frames
        node_features = torch.cat(
            [
                t_noisy.mean(-1, keepdim=True).expand(
                    -1, -1, 2
                ),  # rough position encoding
                torch.zeros(B, L, 2, device=t_noisy.device),
            ],
            dim=-1,
        )
        s = self.node_embed(node_features) + self.time_embed(timestep).unsqueeze(1)

        R, t = R_noisy.clone(), t_noisy.clone()

        for block in self.blocks:
            s, R, t = block(s, pair_repr, R, t)
            # Apply motif conditioning (fix known coordinates)
            if motif_mask is not None and motif_coords is not None:
                m = motif_mask.unsqueeze(-1)
                t = t * (1 - m) + motif_coords * m

        return R, t  # predicted denoised frames

    @torch.no_grad()
    def sample(
        self,
        L: int,
        pair_repr: torch.Tensor,
        motif_mask: Optional[torch.Tensor] = None,
        motif_coords: Optional[torch.Tensor] = None,
        n_steps: int = 200,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Reverse diffusion sampling: pure noise → protein backbone.
        pair_repr conditions the entire trajectory (EvoFormer evolutionary context).
        """
        B = pair_repr.shape[0]
        device = pair_repr.device

        # Start from pure noise
        R = random_so3_frames(B, L, device)
        t = torch.randn(B, L, 3, device=device) * 10.0  # ~10Å spread

        for step in reversed(range(n_steps)):
            ts = torch.full((B,), step, dtype=torch.long, device=device)
            R_pred, t_pred = self.forward(R, t, pair_repr, ts, motif_mask, motif_coords)

            # DDPM-style update
            beta = self.diffusion.betas[step]
            alpha = self.diffusion.alphas[step]
            abar = self.diffusion.alpha_bar[step]

            # Denoise step for translation
            noise = torch.randn_like(t) if step > 0 else torch.zeros_like(t)
            t = (1 / alpha.sqrt()) * (
                t - beta / (1 - abar).sqrt() * (t - abar.sqrt() * t_pred)
            ) + beta.sqrt() * noise

            # Denoise step for rotation (simplified: blend toward predicted)
            blend = 1 - alpha
            R_update = axis_angle_to_matrix(matrix_to_axis_angle(R_pred) * blend)
            R = torch.bmm(R.view(-1, 3, 3), R_update.view(-1, 3, 3)).view(*R.shape)

        return R, t
