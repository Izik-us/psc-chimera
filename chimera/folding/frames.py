"""Rigid-frame <-> N/CA/C/O backbone conversions and an ideal-geometry builder.

Conventions match ``chimera.structure_utils``: the residue frame has
x = unit(C - CA), y = Gram-Schmidt(N - CA), z = x cross y, origin at CA.
Coordinates are ``(B, L, 4, 3)`` in N/CA/C/O order.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

# Engh & Huber ideal backbone geometry (Angstrom / radian).
N_CA, CA_C, C_N, C_O = 1.458, 1.525, 1.329, 1.231
ANG_N_CA_C = math.radians(111.2)
ANG_CA_C_N = math.radians(116.2)
ANG_C_N_CA = math.radians(121.7)
ANG_CA_C_O = math.radians(120.8)

# Local coordinates (CA at origin, x along CA->C, N in the xy plane).
_N_LOCAL = (N_CA * math.cos(ANG_N_CA_C), N_CA * math.sin(ANG_N_CA_C), 0.0)
_C_LOCAL = (CA_C, 0.0, 0.0)
# Carbonyl direction from C in the peptide plane (psi = 180 degrees reference).
_O_DIR = (-math.cos(ANG_CA_C_O), -math.sin(ANG_CA_C_O), 0.0)


def backbone_to_frames(coords: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``(B, L, 4, 3)`` -> rotations ``(B, L, 3, 3)`` and CA translations ``(B, L, 3)``."""
    n, ca, c = coords[..., 0, :], coords[..., 1, :], coords[..., 2, :]
    x = F.normalize(c - ca, dim=-1)
    y0 = n - ca
    y = F.normalize(y0 - (y0 * x).sum(-1, keepdim=True) * x, dim=-1)
    z = torch.cross(x, y, dim=-1)
    return torch.stack((x, y, z), dim=-1), ca


def frames_to_backbone(
    R: torch.Tensor, t: torch.Tensor, psi_sincos: torch.Tensor | None = None
) -> torch.Tensor:
    """Place N/CA/C/O from rigid frames; ``psi_sincos`` rotates O about the CA->C axis."""
    dtype, device = R.dtype, R.device
    n = torch.tensor(_N_LOCAL, dtype=dtype, device=device)
    c = torch.tensor(_C_LOCAL, dtype=dtype, device=device)
    od = torch.tensor(_O_DIR, dtype=dtype, device=device)
    ca = torch.zeros(3, dtype=dtype, device=device)
    shape = R.shape[:-2]
    n_l = n.expand(*shape, 3)
    ca_l = ca.expand(*shape, 3)
    c_l = c.expand(*shape, 3)
    if psi_sincos is None:
        o_dir = od.expand(*shape, 3)
    else:
        sc = F.normalize(psi_sincos, dim=-1)
        sin_p, cos_p = sc[..., 0], sc[..., 1]
        # Rotation about +x (the CA->C axis) of the reference carbonyl direction.
        o_dir = torch.stack(
            (
                od[0].expand(shape),
                od[1] * cos_p - od[2] * sin_p,
                od[1] * sin_p + od[2] * cos_p,
            ),
            dim=-1,
        )
    o_l = c_l + C_O * o_dir
    local = torch.stack((n_l, ca_l, c_l, o_l), dim=-2)  # (..., 4, 3)
    return torch.einsum("...ij,...aj->...ai", R, local) + t.unsqueeze(-2)


def carbonyl_angle_target(coords: torch.Tensor) -> torch.Tensor:
    """``(B, L, 2)`` (sin, cos) of the rotation of O about the CA->C axis (training target for the psi head)."""
    R, t = backbone_to_frames(coords)
    o_local = torch.einsum("blji,blj->bli", R, coords[..., 3, :] - coords[..., 2, :] - 0.0 * t)
    ang = torch.atan2(o_local[..., 2], o_local[..., 1]) - math.atan2(_O_DIR[2], _O_DIR[1])
    return torch.stack((torch.sin(ang), torch.cos(ang)), dim=-1)


def so3_exp_safe(w: torch.Tensor) -> torch.Tensor:
    """Rodrigues exponential with finite gradients at w = 0 (Taylor branch for tiny angles)."""
    th2 = (w * w).sum(-1, keepdim=True)
    small = th2 < 1e-8
    th2s = torch.where(small, torch.ones_like(th2), th2)
    th = th2s.sqrt()
    a = torch.where(small, 1.0 - th2 / 6.0, torch.sin(th) / th)
    b = torch.where(small, 0.5 - th2 / 24.0, (1.0 - torch.cos(th)) / th2s)
    zero = torch.zeros_like(w[..., 0])
    K = torch.stack(
        (
            torch.stack((zero, -w[..., 2], w[..., 1]), -1),
            torch.stack((w[..., 2], zero, -w[..., 0]), -1),
            torch.stack((-w[..., 1], w[..., 0], zero), -1),
        ),
        -2,
    )
    I = torch.eye(3, dtype=w.dtype, device=w.device).expand_as(K)
    return I + a.unsqueeze(-1) * K + b.unsqueeze(-1) * (K @ K)


def _nerf(a, b, c, length, angle, torsion):
    bc = F.normalize(c - b, dim=-1)
    nrm = F.normalize(torch.cross(b - a, bc, dim=-1), dim=-1)
    m = torch.cross(nrm, bc, dim=-1)
    d0 = -length * math.cos(angle)
    sin_a = length * math.sin(angle)
    d1 = sin_a * torch.cos(torsion)
    d2 = sin_a * torch.sin(torsion)
    return c + d0 * bc + d1.unsqueeze(-1) * m + d2.unsqueeze(-1) * nrm


def ideal_backbone_from_torsions(phi: torch.Tensor, psi: torch.Tensor) -> torch.Tensor:
    """Build ideal-geometry trans-peptide backbones from ``(B, L)`` phi/psi (radians)."""
    if phi.shape != psi.shape or phi.ndim != 2:
        raise ValueError("phi and psi must both have shape (B, L)")
    B, L = phi.shape
    dtype, device = phi.dtype, phi.device
    zero = torch.zeros(B, dtype=dtype, device=device)
    n0 = torch.zeros(B, 3, dtype=dtype, device=device)
    ca0 = n0 + torch.tensor([N_CA, 0.0, 0.0], dtype=dtype, device=device)
    c0 = ca0 + torch.tensor(
        [-CA_C * math.cos(ANG_N_CA_C), CA_C * math.sin(ANG_N_CA_C), 0.0],
        dtype=dtype,
        device=device,
    )
    n_i, ca_i, c_i = n0, ca0, c0
    out = []
    omega = zero + math.pi
    for i in range(L):
        o_i = _nerf(n_i, ca_i, c_i, C_O, ANG_CA_C_O, psi[:, i] + math.pi)
        out.append(torch.stack((n_i, ca_i, c_i, o_i), dim=1))
        if i == L - 1:
            break
        n_next = _nerf(n_i, ca_i, c_i, C_N, ANG_CA_C_N, psi[:, i])
        ca_next = _nerf(ca_i, c_i, n_next, N_CA, ANG_C_N_CA, omega)
        c_next = _nerf(c_i, n_next, ca_next, CA_C, ANG_N_CA_C, phi[:, i + 1])
        n_i, ca_i, c_i = n_next, ca_next, c_next
    return torch.stack(out, dim=1)


def backbone_torsions(coords: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """phi/psi in radians, ``(B, L)``; undefined termini are NaN."""
    n, ca, c = coords[..., 0, :], coords[..., 1, :], coords[..., 2, :]

    def dihedral(p0, p1, p2, p3):
        b0, b1, b2 = p0 - p1, p2 - p1, p3 - p2
        b1u = F.normalize(b1, dim=-1)
        v = b0 - (b0 * b1u).sum(-1, keepdim=True) * b1u
        w = b2 - (b2 * b1u).sum(-1, keepdim=True) * b1u
        x = (v * w).sum(-1)
        y = (torch.cross(b1u, v, dim=-1) * w).sum(-1)
        return torch.atan2(y, x)

    nan = torch.full(ca.shape[:-1], float("nan"), dtype=coords.dtype, device=coords.device)
    phi, psi = nan.clone(), nan.clone()
    if coords.shape[-3] > 1:
        phi[:, 1:] = dihedral(c[:, :-1], n[:, 1:], ca[:, 1:], c[:, 1:])
        psi[:, :-1] = dihedral(n[:, :-1], ca[:, :-1], c[:, :-1], n[:, 1:])
    return phi, psi


__all__ = [
    "backbone_to_frames",
    "frames_to_backbone",
    "ideal_backbone_from_torsions",
    "backbone_torsions",
    "carbonyl_angle_target",
    "so3_exp_safe",
]
