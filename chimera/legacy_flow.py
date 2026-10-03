"""Historical deterministic OT-flow API retained for compatibility."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, Optional, Tuple

from .lie import relative_rotation, so3_exp, so3_log
from .se3_flow import InvariantPointAttention, VelocityField, _resolve_ipa_heads

def so3_geodesic_interp(R0: torch.Tensor, R1: torch.Tensor, t: float) -> torch.Tensor:
    """
    Geodesic interpolation on SO(3) at fraction t ∈ [0,1].
    SLERP: R_t = R0 · exp(t · log(R0^T · R1))
    """
    delta = relative_rotation(R0, R1)
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
    delta = relative_rotation(R0, R1)
    return delta  # constant along the geodesic


# ── Invariant Point Attention (IPA) ──────────────────────────────────────────



class SE3FlowMatching(nn.Module):
    """Historical SE(3) velocity-field API with deterministic OT/RK4 methods.

    Canonical structural generation uses ``se3_flow.FlowMatchingBackbone``.
    This class remains only for callers of the historical flow API.
    """

    def __init__(
        self,
        d_single: int = 256,
        d_pair: int = 256,
        n_blocks: int = 8,
        n_head: Optional[int] = None,
        ipa_class=InvariantPointAttention,
    ):
        super().__init__()
        n_head = _resolve_ipa_heads(d_single, n_head)
        self.velocity_field = VelocityField(d_single, d_pair, n_blocks, n_head, ipa_class)

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
        use_schrodinger_bridge: bool = False,
        sb_sigma: float = 0.1,  # Noise level for Schrödinger Bridge
    ) -> torch.Tensor:
        """
        Conditional flow matching loss with optional Schrödinger Bridge SDE.

        Modes:
          1. Optimal Transport Flow Matching (CFM, Lipman et al. 2022)
             - Deterministic flow: straight paths from source to target
             - Velocity = constant along path
             - Training loss: MSE between predicted and target velocity

          2. Schrödinger Bridge SDE (Bose et al. 2023, Liu et al. 2023)
             - Stochastic flow with noise injection during training
             - Minimum-energy transport respecting boundary constraints
             - For PSC: bacterial backbone → mammalian design with fixed catalytic sites
             - Velocity target includes drift correction for noise
             - Formula: L = E[||v_θ - v*||²] where v* = (x1-x0)/(1-t) + drift_correction

        Only updates residues not in fixed_mask (catalytic residues are fixed).
        """
        B = R0.shape[0]
        device = R0.device

        # Sample random time for each batch element
        t_flow = torch.rand(B, device=device)
        t_flow_view = t_flow.view(B, 1, 1)

        if use_schrodinger_bridge:
            # Schrödinger Bridge: add stochastic noise to interpolation path
            # This creates "minimum-energy" paths that respect boundary conditions

            # Sample noise for stochastic interpolation (per residue)
            noise_omega = torch.randn(B, R0.shape[1], 3, device=device) * sb_sigma  # (B, L, 3)
            noise_trans = torch.randn_like(t0) * sb_sigma  # (B, L, 3)

            # Stochastic interpolation: perturb the straight-line path with per-residue noise
            # x_t = (1-t)*x0 + t*x1 + σ*noise
            Rt_base = torch.stack(
                [so3_geodesic_interp(R0[i], R1[i], t_flow[i].item()) for i in range(B)]
            )  # (B, L, 3, 3)

            # Add noise perturbation via matrix exponential for each residue
            # (B, L, 3) axis-angle → (B, L, 3, 3) rotation → apply to base interpolation
            noise_rot_exp = torch.stack(
                [so3_exp(noise_omega[i]) @ Rt_base[i] for i in range(B)]
            )  # (B, L, 3, 3)
            Rt = noise_rot_exp
            tt = t_flow_view * t1 + (1 - t_flow_view) * t0 + noise_trans

            # Drift correction for SDE: v* = (x1 - x0) / (1 - t)
            # This accounts for the minimum-energy path under noise
            v_rot_target = so3_log(torch.einsum("...ij,...kj->...ik", R0, R1))  # (B,L,3)
            v_trans_target = t1 - t0  # (B, L, 3)

            # Apply time scaling (Brownian bridge correction)
            v_rot_target = v_rot_target / (1 - t_flow_view.clamp(min=0.01))
            v_trans_target = v_trans_target / (1 - t_flow_view.clamp(min=0.01))
        else:
            # Standard CFM: deterministic OT paths
            # Interpolate along the flow path
            Rt = torch.stack(
                [so3_geodesic_interp(R0[i], R1[i], t_flow[i].item()) for i in range(B)]
            )
            tt = t_flow_view * t1 + (1 - t_flow_view) * t0

            # Target velocity (constant along OT path)
            v_rot_target = so3_log(torch.einsum("...ij,...kj->...ik", R0, R1))  # (B,L,3)
            v_trans_target = t1 - t0  # (B, L, 3)

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

        # Compute loss
        rot_loss = F.mse_loss(v_rot_pred, v_rot_target, reduction="none").sum(-1)
        trans_loss = F.mse_loss(v_trans_pred, v_trans_target, reduction="none").sum(-1)
        loss = rot_loss + trans_loss  # (B, L)

        # Zero out loss on fixed positions (catalytic residues)
        if fixed_mask is not None:
            loss = loss * (~fixed_mask).float()

        return loss.mean()

    def sample(
        self,
        R0: torch.Tensor,  # (B, L, 3, 3) source backbone
        t0: torch.Tensor,  # (B, L, 3)    source positions
        pair_cond: torch.Tensor,
        evol_single: torch.Tensor,
        n_steps: int = 20,  # 20 steps — 10x faster than DDPM
        fixed_mask: Optional[torch.Tensor] = None,
        substrate_coords: Optional[torch.Tensor] = None,
        evol_conditioning_fn: Optional[Callable] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate backbone by integrating the ODE with RK4.
        Much faster than DDPM: 20 NFE vs 200.
        """
        B = R0.shape[0]
        device = R0.device
        dt = 1.0 / n_steps

        R_curr, t_curr = R0.clone(), t0.clone()

        # Helper to clamp fixed residues at intermediate states
        def clamp_fixed(R, t):
            if fixed_mask is None:
                return R, t
            mask_R = fixed_mask.view(B, -1, 1, 1)
            mask_t = fixed_mask.view(B, -1, 1)
            R = torch.where(mask_R, R0, R)
            t = torch.where(mask_t, t0, t)
            return R, t

        for step in range(n_steps):
            t_now = torch.full((B,), step * dt, device=device)

            # RK4 integration on SE(3)
            k1_r, k1_t = self.velocity_field(
                R_curr, t_curr, t_now, pair_cond, evol_single, R0, t0, substrate_coords, evol_conditioning_fn
            )

            # First RK4 stage: use right-multiplication for SO(3) consistency with training convention
            R_mid1 = torch.stack(
                [
                    so3_geodesic_interp(
                        R_curr[i], R_curr[i] @ so3_exp(k1_r[i] * dt / 2), 0.5
                    )
                    for i in range(B)
                ]
            )
            t_mid1 = t_curr + k1_t * dt / 2
            R_mid1, t_mid1 = clamp_fixed(R_mid1, t_mid1)

            k2_r, k2_t = self.velocity_field(
                R_mid1,
                t_mid1,
                t_now + dt / 2,
                pair_cond,
                evol_single,
                R0,
                t0,
                substrate_coords,
                evol_conditioning_fn,
            )

            # Second RK4 stage
            R_mid2 = torch.stack(
                [
                    so3_geodesic_interp(
                        R_curr[i], R_curr[i] @ so3_exp(k2_r[i] * dt / 2), 0.5
                    )
                    for i in range(B)
                ]
            )
            t_mid2 = t_curr + k2_t * dt / 2
            R_mid2, t_mid2 = clamp_fixed(R_mid2, t_mid2)

            k3_r, k3_t = self.velocity_field(
                R_mid2,
                t_mid2,
                t_now + dt / 2,
                pair_cond,
                evol_single,
                R0,
                t0,
                substrate_coords,
                evol_conditioning_fn,
            )

            # Third RK4 stage
            R_end = torch.stack(
                [
                    so3_geodesic_interp(
                        R_curr[i], R_curr[i] @ so3_exp(k3_r[i] * dt), 1.0
                    )
                    for i in range(B)
                ]
            )
            t_end = t_curr + k3_t * dt
            R_end, t_end = clamp_fixed(R_end, t_end)

            k4_r, k4_t = self.velocity_field(
                R_end,
                t_end,
                t_now + dt,
                pair_cond,
                evol_single,
                R0,
                t0,
                substrate_coords,
                evol_conditioning_fn,
            )

            # RK4 update: combine velocity estimates
            v_r = (k1_r + 2 * k2_r + 2 * k3_r + k4_r) / 6
            v_t = (k1_t + 2 * k2_t + 2 * k3_t + k4_t) / 6

            # Update rotations via exponential map using right-multiplication for consistency
            R_new = torch.stack([R_curr[i] @ so3_exp(v_r[i] * dt) for i in range(B)])
            t_new = t_curr + v_t * dt

            # Final clamp: ensure fixed positions are never modified (deliberate redundancy)
            R_new, t_new = clamp_fixed(R_new, t_new)

            R_curr, t_curr = R_new, t_new

        return R_curr, t_curr

