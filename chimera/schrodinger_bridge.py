"""Schrödinger bridge utilities for conditional SE(3) backbone transport.

This module implements a Brownian Schrödinger-bridge approximation rather than
calling noisy OT interpolation a Schrödinger bridge.  The endpoint coupling is
estimated with entropic Sinkhorn scaling and the bridge drift is trained from
Brownian-bridge conditionals.

For a reference SDE dX_t = sqrt(2D) dW_t, a bridge conditioned on endpoints
x_0, x_1 has conditional marginal
    X_t | x_0,x_1 ~ N((1-t)x_0+t x_1, 2D t(1-t) I)
and conditional drift
    b^*(x,t|x_1) = (x_1-x)/(1-t).
The learned marginal SB drift is the conditional expectation of this quantity
over the entropically optimal endpoint coupling.

Rotations are represented in the Lie algebra using
omega = Log(R_0^T R), then mapped back with Exp.  This avoids treating 3x3
rotation matrices as Euclidean vectors and preserves SO(3) membership.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .flow_matching import so3_exp, so3_log


class SchrodingerBridge(nn.Module):
    """Brownian Schrödinger bridge in Euclidean product coordinates."""

    def __init__(self, diffusion: float = 0.05, sinkhorn_iters: int = 50,
                 sinkhorn_reg: Optional[float] = None):
        super().__init__()
        if diffusion <= 0:
            raise ValueError("diffusion must be positive")
        if sinkhorn_iters < 1:
            raise ValueError("sinkhorn_iters must be positive")
        self.diffusion = float(diffusion)
        self.sinkhorn_iters = int(sinkhorn_iters)
        self.sinkhorn_reg = float(sinkhorn_reg or (2.0 * diffusion))

    def endpoint_cost(
        self,
        omega0: torch.Tensor,
        x0: torch.Tensor,
        omega1: torch.Tensor,
        x1: torch.Tensor,
        rotation_weight: float = 1.0,
    ) -> torch.Tensor:
        """Return pairwise squared endpoint cost, shape (N,M)."""
        if omega0.ndim != 2 or omega1.ndim != 2 or x0.ndim != 2 or x1.ndim != 2:
            raise ValueError("endpoint coordinates must be rank-2 tensors")
        rot = torch.cdist(omega0, omega1).square() * rotation_weight
        trans = torch.cdist(x0, x1).square()
        return rot + trans

    def sinkhorn_coupling(
        self,
        cost: torch.Tensor,
        source_weights: Optional[torch.Tensor] = None,
        target_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute an entropic OT endpoint coupling in log space.

        The Gibbs kernel is K_ij = exp(-C_ij / eps). Uniform marginals are used
        by default.  The returned matrix is normalized to sum to one.
        """
        if cost.ndim != 2:
            raise ValueError("cost must have shape (N,M)")
        n, m = cost.shape
        dtype = cost.dtype
        device = cost.device
        a = source_weights if source_weights is not None else torch.full((n,), 1.0 / n, device=device, dtype=dtype)
        b = target_weights if target_weights is not None else torch.full((m,), 1.0 / m, device=device, dtype=dtype)
        if a.shape != (n,) or b.shape != (m,):
            raise ValueError("marginals have incompatible shapes")
        a = a / a.sum().clamp_min(torch.finfo(dtype).eps)
        b = b / b.sum().clamp_min(torch.finfo(dtype).eps)

        log_a = a.clamp_min(torch.finfo(dtype).tiny).log()
        log_b = b.clamp_min(torch.finfo(dtype).tiny).log()
        log_k = -cost / self.sinkhorn_reg
        log_u = torch.zeros_like(log_a)
        log_v = torch.zeros_like(log_b)
        for _ in range(self.sinkhorn_iters):
            log_u = log_a - torch.logsumexp(log_k + log_v.unsqueeze(0), dim=1)
            log_v = log_b - torch.logsumexp(log_k + log_u.unsqueeze(1), dim=0)
        coupling = torch.exp(log_u.unsqueeze(1) + log_k + log_v.unsqueeze(0))
        return coupling / coupling.sum().clamp_min(torch.finfo(dtype).eps)

    def sample_bridge(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        t: torch.Tensor,
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample Brownian bridge states and their exact conditional drift.

        x0/x1: (B,D), t: (B,), returns x_t and b*(x_t,t|x1).
        """
        if x0.shape != x1.shape or x0.ndim != 2:
            raise ValueError("x0 and x1 must have the same shape (B,D)")
        if t.shape != (x0.shape[0],):
            raise ValueError("t must have shape (B,)")
        eps = torch.randn(x0.shape, device=x0.device, dtype=x0.dtype, generator=generator)
        var = 2.0 * self.diffusion * t * (1.0 - t)
        xt = (1.0 - t[:, None]) * x0 + t[:, None] * x1 + var.clamp_min(0).sqrt()[:, None] * eps
        denom = (1.0 - t).clamp_min(1e-4)[:, None]
        drift = (x1 - xt) / denom
        return xt, drift

    def loss(self, pred_drift: torch.Tensor, target_drift: torch.Tensor,
             valid_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Bridge drift regression loss."""
        per_dim = F.mse_loss(pred_drift, target_drift, reduction="none")
        if valid_mask is not None:
            while valid_mask.ndim < per_dim.ndim:
                valid_mask = valid_mask.unsqueeze(-1)
            per_dim = per_dim * valid_mask.to(per_dim.dtype)
            denom = valid_mask.to(per_dim.dtype).sum().clamp_min(1.0)
            return per_dim.sum() / denom
        return per_dim.mean()


class SE3SchrodingerBridge(nn.Module):
    """SE(3) Brownian Schrödinger bridge around a supplied drift network.

    The drift network must accept ``(R, t, time, *conditioning)`` and return
    angular and translational local/global drifts.  Training uses an entropic
    endpoint coupling and Brownian bridge conditionals.  Sampling uses
    Euler-Maruyama, because the SB solution is stochastic, not an RK4 ODE.
    """

    def __init__(self, drift_model: nn.Module, diffusion: float = 0.05,
                 sinkhorn_iters: int = 50):
        super().__init__()
        self.drift_model = drift_model
        self.bridge = SchrodingerBridge(diffusion, sinkhorn_iters)

    @staticmethod
    def _relative_rotation(R0: torch.Tensor, R1: torch.Tensor) -> torch.Tensor:
        rel = R0.transpose(-1, -2) @ R1
        return so3_log(rel)

    def interpolate(
        self,
        R0: torch.Tensor,
        t0: torch.Tensor,
        R1: torch.Tensor,
        t1: torch.Tensor,
        time: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample an SE(3) Brownian bridge and return its exact conditional drifts."""
        B = R0.shape[0]
        delta_rot = self._relative_rotation(R0, R1)
        rot0 = torch.zeros_like(delta_rot)
        rot_t, rot_drift = self.bridge.sample_bridge(
            rot0.reshape(B, -1), delta_rot.reshape(B, -1), time
        )
        rot_t = rot_t.reshape_as(delta_rot)
        rot_drift = rot_drift.reshape_as(delta_rot)
        R_t = R0 @ so3_exp(rot_t)

        trans_t, trans_drift = self.bridge.sample_bridge(
            t0.reshape(B, -1), t1.reshape(B, -1), time
        )
        trans_t = trans_t.reshape_as(t0)
        trans_drift = trans_drift.reshape_as(t0)
        return R_t, trans_t, rot_drift, trans_drift

    def bridge_loss(
        self,
        R0: torch.Tensor,
        t0: torch.Tensor,
        R1: torch.Tensor,
        t1: torch.Tensor,
        pair_cond: torch.Tensor,
        evol_single: torch.Tensor,
        fixed_mask: Optional[torch.Tensor] = None,
        substrate_coords: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Train the drift field against Brownian-bridge conditional drifts."""
        B = R0.shape[0]
        time = torch.rand(B, device=R0.device, dtype=R0.dtype).clamp_(1e-4, 1.0 - 1e-4)
        R_t, t_t, target_r, target_t = self.interpolate(R0, t0, R1, t1, time)
        pred_r, pred_t = self.drift_model(
            R_t, t_t, time, pair_cond, evol_single,
            R0, t0, substrate_coords, None,
        )
        if fixed_mask is not None:
            valid = (~fixed_mask).to(pred_r.dtype)
            rot_err = (pred_r - target_r).square().sum(-1) * valid
            trans_err = (pred_t - target_t).square().sum(-1) * valid
            return (rot_err + trans_err).sum() / valid.sum().clamp_min(1.0)
        return (pred_r - target_r).square().mean() + (pred_t - target_t).square().mean()

    @torch.no_grad()
    def sample(
        self,
        R0: torch.Tensor,
        t0: torch.Tensor,
        pair_cond: torch.Tensor,
        evol_single: torch.Tensor,
        n_steps: int = 50,
        fixed_mask: Optional[torch.Tensor] = None,
        substrate_coords: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample the learned forward SB with Euler-Maruyama."""
        if n_steps < 2:
            raise ValueError("n_steps must be at least 2")
        R, x = R0.clone(), t0.clone()
        B = R.shape[0]
        dt = 1.0 / n_steps
        noise_scale = (2.0 * self.bridge.diffusion * dt) ** 0.5
        for step in range(n_steps):
            tau = torch.full((B,), step * dt, device=R.device, dtype=x.dtype)
            vr, vt = self.drift_model(
                R, x, tau, pair_cond, evol_single, R0, t0, substrate_coords, None,
            )
            if step < n_steps - 1:
                vr = vr + noise_scale * torch.randn_like(vr)
                vt = vt + noise_scale * torch.randn_like(vt)
            R = R @ so3_exp(vr * dt)
            x = x + vt * dt
            if fixed_mask is not None:
                mR = fixed_mask[..., None, None]
                mx = fixed_mask[..., None]
                R = torch.where(mR, R0, R)
                x = torch.where(mx, t0, x)
        return R, x
