"""Geometry validation for generated protein backbones.

The checks here are deterministic sanity checks, not a replacement for a
validated force field or structure predictor. Steric clashes exclude atoms
that are topologically adjacent or separated by one covalent bond, because
those short distances are expected in a chemically connected backbone.
"""

from dataclasses import dataclass
import math
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
    torsion_planarity_error: float
    clash_count: int
    candidate_valid: tuple[bool, ...]
    candidate_metrics: tuple[dict, ...] = ()

    @property
    def valid(self) -> bool:
        return all(self.candidate_valid)

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
            "torsion_planarity_error": self.torsion_planarity_error,
            "clash_count": self.clash_count,
            "valid": self.valid,
            "candidate_valid": list(self.candidate_valid),
            "candidate_metrics": list(self.candidate_metrics),
        }


def validate_backbone(
    coords: torch.Tensor,
    rotations: torch.Tensor | None = None,
    clash_distance: float = 2.0,
) -> GeometryReport:
    """Validate ``(B, L, 4, 3)`` N/CA/C/O coordinates and optional frames."""
    if coords.ndim != 4 or coords.shape[-2:] != (4, 3):
        raise ValueError("coords must have shape (B, L, 4, 3)")
    if clash_distance <= 0:
        raise ValueError("clash_distance must be positive")
    if coords.shape[0] < 1:
        raise ValueError("coords batch must not be empty")
    if rotations is not None and rotations.shape != (*coords.shape[:2], 3, 3):
        raise ValueError("rotations must have shape (B, L, 3, 3)")
    if coords.shape[0] > 1:
        reports = [
            validate_backbone(
                coords[index:index + 1],
                rotations[index:index + 1] if rotations is not None else None,
                clash_distance,
            )
            for index in range(coords.shape[0])
        ]
        return GeometryReport(
            finite=all(report.finite for report in reports),
            rotation_orthogonality_error=max(report.rotation_orthogonality_error for report in reports),
            rotation_determinant_error=max(report.rotation_determinant_error for report in reports),
            ca_bond_mean=sum(report.ca_bond_mean for report in reports) / len(reports),
            ca_bond_max_error=max(report.ca_bond_max_error for report in reports),
            peptide_bond_mean=sum(report.peptide_bond_mean for report in reports) / len(reports),
            bond_length_max_error=max(report.bond_length_max_error for report in reports),
            bond_angle_max_error=max(report.bond_angle_max_error for report in reports),
            torsion_abs_max=max(report.torsion_abs_max for report in reports),
            torsion_planarity_error=max(report.torsion_planarity_error for report in reports),
            clash_count=sum(report.clash_count for report in reports),
            candidate_valid=tuple(report.valid for report in reports),
            candidate_metrics=tuple(report.as_dict() for report in reports),
        )
    finite = bool(torch.isfinite(coords).all())
    if rotations is None:
        rotation_error = 0.0
        determinant_error = 0.0
    else:
        identity = torch.eye(3, device=rotations.device, dtype=rotations.dtype)
        rotation_error = float(
            (rotations.transpose(-1, -2) @ rotations - identity).abs().max()
        )
        determinant_error = float((rotations.det() - 1).abs().max())

    ca = coords[:, :, 1]
    ca_bonds = (ca[:, 1:] - ca[:, :-1]).norm(dim=-1)
    ca_error = (ca_bonds - 3.8).abs()
    c_to_n = (coords[:, 1:, 0] - coords[:, :-1, 2]).norm(dim=-1)
    bonds = torch.cat(
        [
            (coords[:, :, 1] - coords[:, :, 0]).norm(dim=-1).reshape(-1),
            (coords[:, :, 2] - coords[:, :, 1]).norm(dim=-1).reshape(-1),
            (coords[:, :, 3] - coords[:, :, 2]).norm(dim=-1).reshape(-1),
            (coords[:, 1:, 0] - coords[:, :-1, 2]).norm(dim=-1).reshape(-1),
        ]
    )
    counts = [
        coords.shape[0] * coords.shape[1],
        coords.shape[0] * coords.shape[1],
        coords.shape[0] * coords.shape[1],
        coords.shape[0] * max(0, coords.shape[1] - 1),
    ]
    bond_target = torch.cat(
        [
            torch.full((count,), value, device=coords.device, dtype=coords.dtype)
            for value, count in zip((1.46, 1.53, 1.23, 1.33), counts)
            if count
        ]
    )
    bond_error = (
        (bonds - bond_target).abs().max()
        if bonds.numel()
        else torch.tensor(0.0, device=coords.device)
    )
    v1 = coords[:, :, 0] - coords[:, :, 1]
    v2 = coords[:, :, 2] - coords[:, :, 1]
    cos_angle = (v1 * v2).sum(-1) / (
        v1.norm(dim=-1) * v2.norm(dim=-1)
    ).clamp_min(1e-6)
    angles = torch.acos(cos_angle.clamp(-1.0, 1.0))
    angle_error = (
        (angles - 1.91).abs().max()
        if angles.numel()
        else torch.tensor(0.0, device=coords.device)
    )
    if coords.shape[1] > 1:
        ca_i = coords[:, :-1, 1]
        c_i = coords[:, :-1, 2]
        n_next = coords[:, 1:, 0]
        ca_next = coords[:, 1:, 1]
        axis = n_next - c_i
        axis_norm = axis.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        axis_unit = axis / axis_norm
        first = ca_i - c_i
        second = ca_next - n_next
        first_plane = first - (first * axis_unit).sum(dim=-1, keepdim=True) * axis_unit
        second_plane = second - (second * axis_unit).sum(dim=-1, keepdim=True) * axis_unit
        first_norm = first_plane.norm(dim=-1)
        second_norm = second_plane.norm(dim=-1)
        torsion_degenerate = (first_norm < 1e-6) | (second_norm < 1e-6)
        torsion_y = (
            torch.linalg.cross(axis_unit, first_plane, dim=-1) * second_plane
        ).sum(dim=-1)
        torsion_x = (first_plane * second_plane).sum(dim=-1)
        omega = torch.atan2(torsion_y, torsion_x)
        omega = omega.masked_fill(torsion_degenerate, float("nan"))
        torsion_finite = bool(torch.isfinite(omega).all())
        torsion_abs = omega.abs().nan_to_num(float("inf")).max()
        absolute_omega = omega.abs()
        planarity_error = torch.minimum(
            absolute_omega,
            (math.pi - absolute_omega).abs(),
        ).nan_to_num(float("inf")).max()
    else:
        torsion_finite = True
        torsion_abs = torch.tensor(0.0, device=coords.device)
        planarity_error = torch.tensor(0.0, device=coords.device)
    finite = finite and torsion_finite

    # Build the molecular graph of the four-atom backbone. Steric validation
    # must not flag expected 1-2 or 1-3 distances (for example CA-O in the
    # same residue) as clashes. Only topologically more distant atoms are
    # considered for the simple distance threshold.
    n_res = coords.shape[1]
    n_atoms = 4 * n_res
    atom_type = torch.arange(4, device=coords.device).repeat(n_res)
    residue_id = torch.arange(n_res, device=coords.device).repeat_interleave(4)
    a = atom_type.unsqueeze(0)
    b = atom_type.unsqueeze(1)
    ra = residue_id.unsqueeze(0)
    rb = residue_id.unsqueeze(1)
    same_res = ra == rb
    bonded = same_res & (
        ((a == 0) & (b == 1))
        | ((a == 1) & (b == 0))
        | ((a == 1) & (b == 2))
        | ((a == 2) & (b == 1))
        | ((a == 2) & (b == 3))
        | ((a == 3) & (b == 2))
    )
    bonded = bonded | (
        ((ra + 1 == rb) & (a == 2) & (b == 0))
        | ((rb + 1 == ra) & (b == 2) & (a == 0))
    )
    # Graph distance <= 2 is excluded from steric checking. Matrix
    # multiplication on the boolean adjacency gives the length-2 paths.
    adjacency = bonded.to(torch.int32)
    distance_two = (adjacency @ adjacency) > 0
    excluded = bonded | distance_two
    excluded = excluded | torch.eye(n_atoms, device=coords.device, dtype=torch.bool)

    clash_points = coords.reshape(coords.shape[0], n_atoms, 3)
    distances = torch.cdist(clash_points, clash_points)
    clash_mask = (distances < clash_distance) & ~excluded.unsqueeze(0)
    clash_count = int(clash_mask.sum().item() // 2)

    metric_values = (
        finite
        and rotation_error < 1e-3
        and determinant_error < 1e-3
        and (float(ca_error.max()) if ca_error.numel() else 0.0) < 0.5
        and (float(c_to_n.mean()) if c_to_n.numel() else 0.0) < 2.0
        and float(bond_error) < 0.5
        and float(angle_error) < 0.5
        and float(planarity_error) < 0.5
        and clash_count == 0
    )
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
        torsion_planarity_error=float(planarity_error),
        clash_count=clash_count,
        candidate_valid=(bool(metric_values),),
    )
