"""Deterministic geometric derivatives for canonical protein chains."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch


@dataclass
class GeometryRecord:
    ca_coordinates: torch.Tensor
    backbone_coordinates: torch.Tensor
    residue_frames: torch.Tensor
    frame_mask: torch.Tensor
    residue_mask: torch.Tensor
    pair_mask: torch.Tensor
    distances: torch.Tensor
    relative_positions: torch.Tensor
    relative_rotations: torch.Tensor
    torsions: torch.Tensor
    torsion_mask: torch.Tensor
    contacts: torch.Tensor
    edge_index: torch.Tensor
    edge_features: torch.Tensor
    edge_mask: torch.Tensor


def _normalize(vector: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
    return vector / vector.norm(dim=-1, keepdim=True).clamp_min(epsilon)


def _dihedral(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    b0 = -(b - a)
    b1 = _normalize(c - b)
    b2 = d - c
    v = b0 - (b0 * b1).sum(dim=-1, keepdim=True) * b1
    w = b2 - (b2 * b1).sum(dim=-1, keepdim=True) * b1
    x = (v * w).sum(dim=-1)
    y = (torch.cross(b1, v, dim=-1) * w).sum(dim=-1)
    return torch.atan2(y, x)


def _build_frames(backbone: torch.Tensor, atom_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    n, ca, c = backbone[:, 0], backbone[:, 1], backbone[:, 2]
    x_axis = _normalize(c - ca)
    n_direction = _normalize(n - ca)
    z_axis = _normalize(torch.cross(x_axis, n_direction, dim=-1))
    y_axis = _normalize(torch.cross(z_axis, x_axis, dim=-1))
    frames = torch.stack((x_axis, y_axis, z_axis), dim=-1)
    valid = atom_mask[:, :3].all(dim=-1) & torch.isfinite(frames).all(dim=(-1, -2))
    frames = torch.where(valid[:, None, None], frames, torch.eye(3, dtype=backbone.dtype, device=backbone.device))
    return frames, valid


def derive_geometry(
    backbone_coordinates: torch.Tensor,
    atom_mask: torch.Tensor,
    residue_mask: torch.Tensor | None = None,
    *,
    chain_ids: Sequence[str] | None = None,
    k_neighbors: int = 32,
    contact_cutoff_angstrom: float = 8.0,
    neighbor_cutoff_angstrom: float = 20.0,
    rbf_count: int = 16,
) -> GeometryRecord:
    """Derive masked geometric features from ``(L, 4|5, 3)`` backbone atoms.

    Output vectors and rotations in the pair tensors use residue-i local frames,
    so distances, local directions, relative rotations and torsions are rigid-
    transform invariant. Absolute coordinates remain available separately.
    """
    if backbone_coordinates.ndim != 3 or backbone_coordinates.shape[1] not in {4, 5} or backbone_coordinates.shape[2] != 3:
        raise ValueError("backbone_coordinates must have shape (L, 4|5, 3)")
    length = backbone_coordinates.shape[0]
    if atom_mask.shape != backbone_coordinates.shape[:2] or atom_mask.dtype != torch.bool:
        raise ValueError("atom_mask must be bool with shape (L, atoms)")
    if residue_mask is None:
        residue_mask = atom_mask[:, 1]
    if residue_mask.shape != (length,) or residue_mask.dtype != torch.bool:
        raise ValueError("residue_mask must be bool with shape (L,)")
    if chain_ids is not None and len(chain_ids) != length:
        raise ValueError("chain_ids must contain one identifier per residue")
    if k_neighbors < 1 or rbf_count < 1:
        raise ValueError("k_neighbors and rbf_count must be positive")
    if contact_cutoff_angstrom <= 0.0 or neighbor_cutoff_angstrom <= 0.0:
        raise ValueError("distance cutoffs must be positive")
    if not torch.isfinite(backbone_coordinates[atom_mask]).all():
        raise ValueError("valid atom coordinates must be finite")

    device = backbone_coordinates.device
    dtype = backbone_coordinates.dtype
    backbone = backbone_coordinates[:, :4]
    backbone_valid = atom_mask[:, :4]
    ca_coordinates = backbone[:, 1]
    ca_valid = residue_mask & atom_mask[:, 1]
    frames, frame_mask = _build_frames(backbone, atom_mask)

    delta = ca_coordinates[None, :, :] - ca_coordinates[:, None, :]
    distances = delta.norm(dim=-1)
    pair_mask = ca_valid[:, None] & ca_valid[None, :]
    valid_distances = torch.where(pair_mask, distances, torch.zeros_like(distances))
    local_directions = torch.einsum("lij,lkj->lki", frames.transpose(-1, -2), delta)
    local_directions = _normalize(local_directions)
    relative_rotations = torch.matmul(frames.transpose(-1, -2)[:, None], frames[None, :])
    relative_rotations = torch.where(
        (frame_mask[:, None] & frame_mask[None, :])[:, :, None, None],
        relative_rotations,
        torch.eye(3, dtype=dtype, device=device),
    )

    torsions = torch.zeros((length, 3), dtype=dtype, device=device)
    torsion_mask = torch.zeros((length, 3), dtype=torch.bool, device=device)
    if length > 1:
        # phi_i = C_(i-1), N_i, CA_i, C_i; psi_i = N_i, CA_i, C_i, N_(i+1).
        prev_phi_mask = atom_mask[1:, 0] & atom_mask[1:, 1] & atom_mask[1:, 2] & atom_mask[:-1, 2]
        phi = _dihedral(backbone[:-1, 2], backbone[1:, 0], backbone[1:, 1], backbone[1:, 2])
        torsions[1:, 0] = phi
        torsion_mask[1:, 0] = prev_phi_mask & residue_mask[1:] & residue_mask[:-1]

        next_psi_mask = atom_mask[:-1, 0] & atom_mask[:-1, 1] & atom_mask[:-1, 2] & atom_mask[1:, 0]
        psi = _dihedral(backbone[:-1, 0], backbone[:-1, 1], backbone[:-1, 2], backbone[1:, 0])
        torsions[:-1, 1] = psi
        torsion_mask[:-1, 1] = next_psi_mask & residue_mask[:-1] & residue_mask[1:]

        omega = _dihedral(backbone[:-1, 1], backbone[:-1, 2], backbone[1:, 0], backbone[1:, 1])
        torsions[:-1, 2] = omega
        torsion_mask[:-1, 2] = next_psi_mask & residue_mask[:-1] & residue_mask[1:]
    torsions = torch.where(torsion_mask, torsions, torch.zeros_like(torsions))

    contacts = (distances <= contact_cutoff_angstrom) & pair_mask
    contacts.fill_diagonal_(False)
    no_self = torch.eye(length, dtype=torch.bool, device=device)
    candidate = pair_mask & ~no_self
    if chain_ids is not None:
        chain_equal = torch.tensor(
            [[left == right for right in chain_ids] for left in chain_ids],
            dtype=torch.bool,
            device=device,
        )
    else:
        chain_equal = torch.ones((length, length), dtype=torch.bool, device=device)
    candidate &= distances <= neighbor_cutoff_angstrom

    neighbor_count = min(k_neighbors, max(length - 1, 1))
    ranked_distances = distances.masked_fill(~candidate, float("inf"))
    edge_index = torch.argsort(ranked_distances, dim=-1, stable=True)[:, :neighbor_count]
    edge_mask = candidate.gather(1, edge_index)
    selected_distances = distances.gather(1, edge_index)
    selected_direction = local_directions.gather(
        1, edge_index.unsqueeze(-1).expand(-1, -1, 3)
    )
    selected_rotation = relative_rotations.gather(
        1, edge_index[:, :, None, None].expand(-1, -1, 3, 3)
    ).reshape(length, neighbor_count, 9)
    centers = torch.linspace(0.0, neighbor_cutoff_angstrom, rbf_count, dtype=dtype, device=device)
    rbf = torch.exp(-((selected_distances.unsqueeze(-1) - centers) ** 2) / 2.0)
    residue_positions = torch.arange(length, device=device)
    relative_sequence_position = (residue_positions[:, None] - edge_index).abs().to(dtype)
    relation = chain_equal.gather(1, edge_index).to(dtype)
    edge_features = torch.cat(
        [rbf, selected_direction, selected_rotation, relative_sequence_position.unsqueeze(-1), relation.unsqueeze(-1)],
        dim=-1,
    )
    edge_features = edge_features * edge_mask.unsqueeze(-1).to(dtype)
    edge_index = torch.where(edge_mask, edge_index, torch.zeros_like(edge_index))

    return GeometryRecord(
        ca_coordinates=ca_coordinates,
        backbone_coordinates=backbone,
        residue_frames=frames,
        frame_mask=frame_mask,
        residue_mask=ca_valid,
        pair_mask=pair_mask,
        distances=valid_distances,
        relative_positions=local_directions * pair_mask.unsqueeze(-1).to(dtype),
        relative_rotations=relative_rotations,
        torsions=torsions,
        torsion_mask=torsion_mask,
        contacts=contacts,
        edge_index=edge_index,
        edge_features=edge_features,
        edge_mask=edge_mask,
    )