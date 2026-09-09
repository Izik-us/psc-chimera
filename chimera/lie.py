"""Numerically stable SO(3) Lie-group operations used by canonical transport."""

from __future__ import annotations

import math

import torch


def hat(v: torch.Tensor) -> torch.Tensor:
    if v.shape[-1] != 3:
        raise ValueError("v must have final dimension 3")
    x, y, z = v.unbind(-1)
    zero = torch.zeros_like(x)
    return torch.stack(
        (zero, -z, y, z, zero, -x, -y, x, zero), dim=-1
    ).reshape(*v.shape[:-1], 3, 3)


def so3_exp(omega: torch.Tensor) -> torch.Tensor:
    """Rodrigues exponential map with a stable small-angle branch."""
    if omega.shape[-1] != 3:
        raise ValueError("omega must have final dimension 3")
    theta2 = (omega * omega).sum(-1, keepdim=True)
    theta = theta2.sqrt()
    K = hat(omega)
    I = torch.eye(3, device=omega.device, dtype=omega.dtype)
    while I.ndim < K.ndim:
        I = I.unsqueeze(0)
    theta2_safe = theta2.clamp_min(torch.finfo(omega.dtype).eps)
    a = torch.where(
        theta2 < 1e-8,
        1.0 - theta2 / 6.0 + theta2 * theta2 / 120.0,
        torch.sin(theta) / theta.clamp_min(torch.finfo(omega.dtype).eps),
    )
    b = torch.where(
        theta2 < 1e-8,
        0.5 - theta2 / 24.0 + theta2 * theta2 / 720.0,
        (1.0 - torch.cos(theta)) / theta2_safe,
    )
    return I + a.unsqueeze(-1) * K + b.unsqueeze(-1) * (K @ K)


def so3_log(R: torch.Tensor) -> torch.Tensor:
    """Principal logarithm on SO(3), including a signed near-pi branch."""
    if R.shape[-2:] != (3, 3):
        raise ValueError("R must end in (3,3)")
    trace = R.diagonal(dim1=-2, dim2=-1).sum(-1)
    cos_theta = ((trace - 1.0) * 0.5).clamp(-1.0, 1.0)
    theta = torch.acos(cos_theta)
    skew = 0.5 * (R - R.transpose(-1, -2))
    vee = torch.stack((skew[..., 2, 1], skew[..., 0, 2], skew[..., 1, 0]), dim=-1)
    sin_theta = torch.sin(theta)
    regular = vee * (theta / sin_theta.clamp_min(1e-7)).unsqueeze(-1)

    # For theta -> pi, vee loses the sign of the axis. Recover a signed axis
    # from the largest diagonal term and use an off-diagonal sign.
    diag = torch.diagonal(R, dim1=-2, dim2=-1)
    axis_sq = ((diag + 1.0) * 0.5).clamp_min(0.0)
    axis = axis_sq.sqrt()
    idx = axis_sq.argmax(dim=-1)
    basis = torch.nn.functional.one_hot(idx, num_classes=3).to(R.dtype)
    dominant = axis.gather(-1, idx.unsqueeze(-1)).squeeze(-1).clamp_min(1e-7)
    signs = torch.stack(
        (R[..., 2, 1] - R[..., 1, 2], R[..., 0, 2] - R[..., 2, 0], R[..., 1, 0] - R[..., 0, 1]),
        dim=-1,
    )
    sign = torch.where(signs.gather(-1, idx.unsqueeze(-1)).squeeze(-1) >= 0, 1.0, -1.0)
    signed_axis = axis / dominant.unsqueeze(-1)
    signed_axis = signed_axis * (basis * sign.unsqueeze(-1) + (1.0 - basis))
    near_pi = theta > math.pi - 1e-4
    result = torch.where(near_pi.unsqueeze(-1), theta.unsqueeze(-1) * signed_axis, regular)
    small = theta < 1e-5
    return torch.where(small.unsqueeze(-1), vee, result)


def relative_rotation(R0: torch.Tensor, R1: torch.Tensor) -> torch.Tensor:
    """Return log(R0^T R1), the relative rotation in R0's tangent chart."""
    if R0.shape != R1.shape or R0.shape[-2:] != (3, 3):
        raise ValueError("R0 and R1 must have matching (...,3,3) shapes")
    return so3_log(R0.transpose(-1, -2) @ R1)


__all__ = ["hat", "so3_exp", "so3_log", "relative_rotation"]
