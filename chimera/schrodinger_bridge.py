"""Schrödinger bridge utilities for conditional SE(3) backbone transport.

This module implements a Brownian Schrödinger-bridge approximation in local
SE(3) coordinates. The endpoint coupling is obtained with entropic Sinkhorn
scaling, bridge training samples endpoint pairs from that coupling, and
sampling uses Euler-Maruyama because the SB solution is stochastic.

For the reference SDE dX_t = sqrt(2D) dW_t, the Euclidean bridge conditional is
    X_t | x_0,x_1 ~ N((1-t)x_0+t x_1, 2D t(1-t) I)
with conditional drift
    b*(x,t|x_1) = (x_1-x)/(1-t).
The learned marginal SB drift is the conditional expectation of this drift
under the entropically optimal endpoint coupling.

Rotations are represented in a local Lie-algebra chart,
    omega = Log(R_0^T R),
and mapped back with Exp. This is a local SE(3) approximation to Brownian
motion on the group, not a claim of an exact closed-form heat-kernel bridge on
SO(3). It is therefore explicit about the geometric approximation used.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .flow_matching import so3_exp, so3_log


class SchrodingerBridge(nn.Module):
    """Entropic Brownian Schrödinger bridge in Euclidean coordinates."""

    def __init__(
        self,
        diffusion: float = 0.05,
        sinkhorn_iters: int = 50,
        sinkhorn_reg: Optional[float] = None,
    ):
        super().__init__()
        if diffusion <= 0:
            raise ValueError("diffusion must be positive")
        if sinkhorn_iters < 1:
            raise ValueError("sinkhorn_iters must be positive")
        self.diffusion = float(diffusion)
        self.sinkhorn_iters = int(sinkhorn_iters)
        self.sinkhorn_reg = float(sinkhorn_reg if sinkhorn_reg is not None else 2.0 * diffusion)
        if self.sinkhorn_reg <= 0:
            raise ValueError("sinkhorn_reg must be positive")

    def endpoint_cost(
        self,
        omega0: torch.Tensor,
        x0: torch.Tensor,
        omega1: torch.Tensor,
        x1: torch.Tensor,
        rotation_weight: float = 1.0,
    ) -> torch.Tensor:
        """Return pairwise squared endpoint cost with shape ``(N,M)``."""
        if any(x.ndim != 2 for x in (omega0, x0, omega1, x1)):
            raise ValueError("endpoint coordinates must be rank-2 tensors")
        if omega0.shape[0] != x0.shape[0] or omega1.shape[0] != x1.shape[0]:
            raise ValueError("rotation and translation endpoint batches must agree")
        if rotation_weight < 0:
            raise ValueError("rotation_weight must be non-negative")
        return rotation_weight * torch.cdist(omega0, omega1).square() + torch.cdist(x0, x1).square()

    def sinkhorn_coupling(
        self,
        cost: torch.Tensor,
        source_weights: Optional[torch.Tensor] = None,
        target_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute an entropic OT endpoint coupling in log space."""
        if cost.ndim != 2 or cost.numel() == 0:
            raise ValueError("cost must be a non-empty rank-2 tensor")
        n, m = cost.shape
        dtype, device = cost.dtype, cost.device
        eps = torch.finfo(dtype).eps
        a = (
            torch.full((n,), 1.0 / n, device=device, dtype=dtype)
            if source_weights is None
            else source_weights.to(device=device, dtype=dtype)
        )
        b = (
            torch.full((m,), 1.0 / m, device=device, dtype=dtype)
            if target_weights is None
            else target_weights.to(device=device, dtype=dtype)
        )
        if a.shape != (n,) or b.shape != (m,):
            raise ValueError("marginals have incompatible shapes")
        if torch.any(a < 0) or torch.any(b < 0) or a.sum() <= 0 or b.sum() <= 0:
            raise ValueError("marginals must be non-negative with positive mass")
        a = a / a.sum().clamp_min(eps)
        b = b / b.sum().clamp_min(eps)

        log_a = a.clamp_min(torch.finfo(dtype).tiny).log()
        log_b = b.clamp_min(torch.finfo(dtype).tiny).log()
        log_k = -cost / self.sinkhorn_reg
        log_u = torch.zeros_like(log_a)
        log_v = torch.zeros_like(log_b)
        for _ in range(self.sinkhorn_iters):
            log_u = log_a - torch.logsumexp(log_k + log_v.unsqueeze(0), dim=1)
            log_v = log_b - torch.logsumexp(log_k + log_u.unsqueeze(1), dim=0)
        coupling = torch.exp(log_u[:, None] + log_k + log_v[None, :])
        return coupling / coupling.sum().clamp_min(eps)

    def sample_bridge(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        t: torch.Tensor,
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample Brownian bridge states and exact conditional drift."""
        if x0.shape != x1.shape or x0.ndim != 2:
            raise ValueError("x0 and x1 must have the same shape (B,D)")
        if t.shape != (x0.shape[0],):
            raise ValueError("t must have shape (B,)")
        if torch.any((t <= 0) | (t >= 1)):
            raise ValueError("bridge times must lie strictly inside (0,1)")
        eps = torch.randn(x0.shape, device=x0.device, dtype=x0.dtype, generator=generator)
        var = 2.0 * self.diffusion * t * (1.0 - t)
        xt = (1.0 - t[:, None]) * x0 + t[:, None] * x1 + var.sqrt()[:, None] * eps
        drift = (x1 - xt) / (1.0 - t)[:, None]
        return xt, drift

    def loss(
        self,
        pred_drift: torch.Tensor,
        target_drift: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Bridge drift regression loss."""
        if pred_drift.shape != target_drift.shape:
            raise ValueError("pred_drift and target_drift must have identical shapes")
        per_dim = F.mse_loss(pred_drift, target_drift, reduction="none")
        if valid_mask is not None:
            while valid_mask.ndim < per_dim.ndim:
                valid_mask = valid_mask.unsqueeze(-1)
            mask = valid_mask.to(per_dim.dtype)
            return (per_dim * mask).sum() / mask.sum().clamp_min(1.0)
        return per_dim.mean()


class SE3SchrodingerBridge(nn.Module):
    """SE(3) Schrödinger bridge in a local Lie-algebra coordinate chart.

    ``bridge_loss`` estimates the marginal SB drift by first computing an
    entropic endpoint coupling between the source and target batches, then
    sampling target endpoints from each source row of that coupling. This is
    essential: independently pairing source[i] with target[i] is not the SB
    coupling unless the data happen to be already optimally coupled.
    """

    def __init__(self, drift_model: nn.Module, diffusion: float = 0.05, sinkhorn_iters: int = 50):
        super().__init__()
        self.drift_model = drift_model
        self.bridge = SchrodingerBridge(diffusion, sinkhorn_iters)

    @staticmethod
    def _relative_rotation(R0: torch.Tensor, R1: torch.Tensor) -> torch.Tensor:
        return so3_log(R0.transpose(-1, -2) @ R1)

    def _coupled_targets(
        self,
        R0: torch.Tensor,
        t0: torch.Tensor,
        R1: torch.Tensor,
        t1: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample one target endpoint per source from the Sinkhorn coupling."""
        B = R0.shape[0]
        rot0 = torch.zeros(B, R0.shape[1] * 3, device=R0.device, dtype=R0.dtype)
        rot1 = self._relative_rotation(R0, R1).reshape(B, -1)
        cost = self.bridge.endpoint_cost(rot0, t0.reshape(B, -1), rot1, t1.reshape(B, -1))
        coupling = self.bridge.sinkhorn_coupling(cost).to(dtype=torch.float32)
        # One draw from each conditional row p(j|i). This is an unbiased
        # Monte-Carlo estimate of the coupling expectation in the drift target.
        row_probs = coupling / coupling.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(coupling.dtype).eps)
        target_idx = torch.multinomial(row_probs, num_samples=1).squeeze(-1)
        return R1[target_idx], t1[target_idx]

    def interpolate(
        self,
        R0: torch.Tensor,
        t0: torch.Tensor,
        R1: torch.Tensor,
        t1: torch.Tensor,
        time: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample an SE(3) Brownian bridge for a specified endpoint pairing."""
        B = R0.shape[0]
        delta_rot = self._relative_rotation(R0, R1)
        rot_t, rot_drift = self.bridge.sample_bridge(
            torch.zeros_like(delta_rot).reshape(B, -1),
            delta_rot.reshape(B, -1),
            time,
        )
        rot_t = rot_t.reshape_as(delta_rot)
        rot_drift = rot_drift.reshape_as(delta_rot)
        R_t = R0 @ so3_exp(rot_t)
        trans_t, trans_drift = self.bridge.sample_bridge(
            t0.reshape(B, -1), t1.reshape(B, -1), time
        )
        return R_t, trans_t.reshape_as(t0), rot_drift.reshape_as(t0), trans_drift.reshape_as(t0)

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
        """Train the marginal SB drift against coupled Brownian-bridge targets."""
        if R0.shape != R1.shape or t0.shape != t1.shape:
            raise ValueError("source and target SE(3) tensors must have matching shapes")
        if R0.ndim != 4 or t0.ndim != 3 or R0.shape[-2:] != (3, 3) or t0.shape[-1] != 3:
            raise ValueError("expected R=(B,L,3,3) and t=(B,L,3)")
        B = R0.shape[0]
        target_R, target_t = self._coupled_targets(R0, t0, R1, t1)
        time = torch.rand(B, device=R0.device, dtype=t0.dtype).clamp_(1e-4, 1.0 - 1e-4)
        R_t, t_t, target_r, target_t_drift = self.interpolate(R0, t0, target_R, target_t, time)
        pred_r, pred_t = self.drift_model(
            R_t, t_t, time, pair_cond, evol_single, R0, t0, substrate_coords, None
        )
        valid = None if fixed_mask is None else (~fixed_mask).to(pred_r.dtype)
        rot_err = (pred_r - target_r).square().sum(-1)
        trans_err = (pred_t - target_t_drift).square().sum(-1)
        if valid is not None:
            return ((rot_err + trans_err) * valid).sum() / valid.sum().clamp_min(1.0)
        return (rot_err + trans_err).mean()

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
        """Sample the learned SB with Euler-Maruyama."""
        if n_steps < 2:
            raise ValueError("n_steps must be at least 2")
        R, x = R0.clone(), t0.clone()
        B = R.shape[0]
        dt = 1.0 / n_steps
        noise_scale = (2.0 * self.bridge.diffusion * dt) ** 0.5
        for step in range(n_steps):
            tau = torch.full((B,), min(step * dt, 1.0 - 1e-4), device=R.device, dtype=x.dtype)
            vr, vt = self.drift_model(
                R, x, tau, pair_cond, evol_single, R0, t0, substrate_coords, None
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
