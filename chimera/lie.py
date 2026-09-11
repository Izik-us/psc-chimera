"""Numerically stable SO(3) Lie-group operations used by canonical transport."""

from __future__ import annotations

import math

import torch


def hat(v: torch.Tensor) -> torch.Tensor:
    if v.shape[-1] != 3:
        raise ValueError("v must have final dimension 3")
    x, y, z = v.unbind(-1)
    zero = torch.zeros_like(x)
    return torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), dim=-1).reshape(*v.shape[:-1], 3, 3)


def so3_exp(omega: torch.Tensor) -> torch.Tensor:
    """Rodrigues exponential map with stable small-angle coefficients."""
    if omega.shape[-1] != 3:
        raise ValueError("omega must have final dimension 3")
    theta2 = (omega * omega).sum(-1, keepdim=True)
    theta = theta2.sqrt()
    K = hat(omega)
    I = torch.eye(3, device=omega.device, dtype=omega.dtype)
    while I.ndim < K.ndim:
        I = I.unsqueeze(0)
    eps = torch.finfo(omega.dtype).eps
    theta2_safe = theta2.clamp_min(eps)
    a = torch.where(
        theta2 < 1e-8,
        1.0 - theta2 / 6.0 + theta2 * theta2 / 120.0,
        torch.sin(theta) / theta.clamp_min(eps),
    )
    b = torch.where(
        theta2 < 1e-8,
        0.5 - theta2 / 24.0 + theta2 * theta2 / 720.0,
        (1.0 - torch.cos(theta)) / theta2_safe,
    )
    return I + a.unsqueeze(-1) * K + b.unsqueeze(-1) * (K @ K)


def _near_pi_axis(R: torch.Tensor) -> torch.Tensor:
    """Recover an unoriented rotation axis from the symmetric pi-limit matrix."""
    # At theta=pi, (R + I)/2 = a a^T. The dominant eigenvector is therefore
    # the rotation axis. The sign is immaterial exactly at pi because +a and -a
    # exponentiate to the same rotation.
    symmetric = 0.5 * (R + torch.eye(3, device=R.device, dtype=R.dtype))
    eigenvalues, eigenvectors = torch.linalg.eigh(symmetric)
    axis = eigenvectors[..., :, -1]
    return axis / axis.norm(dim=-1, keepdim=True).clamp_min(1e-7)


def so3_log(R: torch.Tensor) -> torch.Tensor:
    """Principal logarithm on SO(3), stable for small angles and theta≈pi."""
    if R.shape[-2:] != (3, 3):
        raise ValueError("R must end in (3,3)")
    trace = R.diagonal(dim1=-2, dim2=-1).sum(-1)
    cos_theta = ((trace - 1.0) * 0.5).clamp(-1.0, 1.0)
    theta = torch.acos(cos_theta)
    skew = 0.5 * (R - R.transpose(-1, -2))
    vee = torch.stack((skew[..., 2, 1], skew[..., 0, 2], skew[..., 1, 0]), dim=-1)

    sin_theta = torch.sin(theta)
    regular = vee * (theta / sin_theta.clamp_min(1e-7)).unsqueeze(-1)

    near_pi = theta > math.pi - 1e-4
    axis = _near_pi_axis(R)
    near_pi_result = theta.unsqueeze(-1) * axis
    result = torch.where(near_pi.unsqueeze(-1), near_pi_result, regular)

    # The first-order limit is log(R)≈vee.
    small = theta < 1e-5
    return torch.where(small.unsqueeze(-1), vee, result)


def relative_rotation(R0: torch.Tensor, R1: torch.Tensor) -> torch.Tensor:
    """Return log(R0^T R1), the relative rotation in R0's tangent chart."""
    if R0.shape != R1.shape or R0.shape[-2:] != (3, 3):
        raise ValueError("R0 and R1 must have matching (...,3,3) shapes")
    return so3_log(R0.transpose(-1, -2) @ R1)


__all__ = ["hat", "so3_exp", "so3_log", "relative_rotation"]
