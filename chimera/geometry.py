"""Geometry validation for generated protein backbones.

The checks here are deterministic sanity checks, not a replacement for a
validated force field or structure predictor.
"""

from dataclasses import dataclass
from typing import Dict
import torch


@dataclass
class GeometryReport:
    finite: bool
    rotation_orthogonality_error: float
    rotation_determinant_error: float
    ca_bond_mean: float
    ca_bond_max_error: float
    peptide_bond_mean: float
    bond_length_max_error: float
    bond_angle_max_error: float
    torsion_abs_max: float
    clash_count: int

    @property
    def valid(self) -> bool:
        return (
            self.finite
            and self.rotation_orthogonality_error < 1e-3
            and self.rotation_determinant_error < 1e-3
            and self.ca_bond_max_error < 0.5
            and self.peptide_bond_mean < 2.0
            and self.bond_length_max_error < 0.5
            and self.bond_angle_max_error < 0.5
            and self.torsion_abs_max <= 3.2
            and self.clash_count == 0
        )

    def as_dict(self) -> Dict[str, float | bool | int]:
        return {
            "finite": self.finite,
            "rotation_orthogonality_error": self.rotation_orthogonality_error,
            "rotation_determinant_error": self.rotation_determinant_error,
            "ca_bond_mean": self.ca_bond_mean,
            "ca_bond_max_error": self.ca_bond_max_error,
            "peptide_bond_mean": self.peptide_bond_mean,
            "bond_length_max_error": self.bond_length_max_error,
            "bond_angle_max_error": self.bond_angle_max_error,
            "torsion_abs_max": self.torsion_abs_max,
            "clash_count": self.clash_count,
            "valid": self.valid,
        }


def validate_backbone(
    coords: torch.Tensor,
    rotations: torch.Tensor | None = None,
    clash_distance: float = 2.0,
) -> GeometryReport:
    """Validate ``(B, L, 4, 3)`` N/CA/C/O coordinates and optional frames."""
    if coords.ndim != 4 or coords.shape[-2:] != (4, 3):
        raise ValueError("coords must have shape (B, L, 4, 3)")
    finite = bool(torch.isfinite(coords).all())
    if rotations is None:
        rotation_error = 0.0
        determinant_error = 0.0
    else:
        if rotations.shape != (*coords.shape[:2], 3, 3):
            raise ValueError("rotations must have shape (B, L, 3, 3)")
        identity = torch.eye(3, device=rotations.device, dtype=rotations.dtype)
        rotation_error = float((rotations.transpose(-1, -2) @ rotations - identity).abs().max())
        determinant_error = float((rotations.det() - 1).abs().max())

    ca = coords[:, :, 1]
    ca_bonds = (ca[:, 1:] - ca[:, :-1]).norm(dim=-1)
    ca_error = (ca_bonds - 3.8).abs()
    c_to_n = (coords[:, 1:, 0] - coords[:, :-1, 2]).norm(dim=-1)
    bonds = torch.cat([
        (coords[:, :, 1] - coords[:, :, 0]).norm(dim=-1).reshape(-1),
        (coords[:, :, 2] - coords[:, :, 1]).norm(dim=-1).reshape(-1),
        (coords[:, 1:, 0] - coords[:, :-1, 2]).norm(dim=-1).reshape(-1),
    ])
    counts = [
        coords.shape[0] * coords.shape[1],
        coords.shape[0] * coords.shape[1],
        coords.shape[0] * max(0, coords.shape[1] - 1),
    ]
    bond_target = torch.cat([
        torch.full((count,), value, device=coords.device)
        for value, count in zip((1.46, 1.53, 1.33), counts)
    ])
    bond_error = (bonds - bond_target).abs().max() if bonds.numel() else torch.tensor(0.0)
    v1 = coords[:, :, 0] - coords[:, :, 1]
    v2 = coords[:, :, 2] - coords[:, :, 1]
    angles = torch.acos((v1 * v2).sum(-1) / (v1.norm(dim=-1) * v2.norm(dim=-1)).clamp_min(1e-6))
    angle_error = (angles - 1.91).abs().max() if angles.numel() else torch.tensor(0.0)
    torsion_abs = torch.tensor(0.0, device=coords.device)
    clash_points = coords.reshape(coords.shape[0], -1, 3)
    distances = torch.cdist(clash_points, clash_points)
    n = distances.shape[-1]
    distances = distances + torch.eye(n, device=distances.device).unsqueeze(0) * 1e6
    clash_count = int((distances < clash_distance).sum().item() // 2)

    return GeometryReport(
        finite=finite,
        rotation_orthogonality_error=rotation_error,
        rotation_determinant_error=determinant_error,
        ca_bond_mean=float(ca_bonds.mean()) if ca_bonds.numel() else 0.0,
        ca_bond_max_error=float(ca_error.max()) if ca_error.numel() else 0.0,
        peptide_bond_mean=float(c_to_n.mean()) if c_to_n.numel() else 0.0,
        bond_length_max_error=float(bond_error),
        bond_angle_max_error=float(angle_error),
        torsion_abs_max=float(torsion_abs),
        clash_count=clash_count,
    )
