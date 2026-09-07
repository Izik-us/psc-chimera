"""Icosahedral symmetry and transparent interface-geometry proxies.

The utilities here encode the 20 triangular faces of an icosahedron. They do
not claim to predict physical self-assembly free energy. The interface score is
an explicit geometric orientation/compactness proxy that depends on the chosen
face and module frames.
"""

import math
import torch


def _axis_rotation(axis: torch.Tensor, angle: float) -> torch.Tensor:
    axis = axis / axis.norm()
    x, y, z = axis
    c, s = math.cos(angle), math.sin(angle)
    return torch.tensor(
        [
            [c + x * x * (1 - c), x * y * (1 - c) - z * s, x * z * (1 - c) + y * s],
            [y * x * (1 - c) + z * s, c + y * y * (1 - c), y * z * (1 - c) - x * s],
            [z * x * (1 - c) - y * s, z * y * (1 - c) + x * s, c + z * z * (1 - c)],
        ],
        dtype=torch.float32,
    )


def icosahedral_rotations(dtype=torch.float32) -> torch.Tensor:
    """Return the 60 proper rotational symmetries of an icosahedron."""
    phi = (1.0 + math.sqrt(5.0)) / 2.0
    base = torch.eye(3, dtype=torch.float32)
    generators = [
        _axis_rotation(torch.tensor([0.0, 1.0, phi]), 2 * math.pi / 5),
        _axis_rotation(torch.tensor([1.0, 1.0, 1.0]), 2 * math.pi / 3),
    ]
    generators += [item.transpose(0, 1) for item in generators]
    group = [base]
    queue = [base]
    seen = {tuple(torch.round(base, decimals=5).flatten().tolist())}
    while queue and len(group) < 60:
        current = queue.pop(0)
        for generator in generators:
            candidate = current @ generator
            key = tuple(torch.round(candidate, decimals=5).flatten().tolist())
            if key not in seen:
                seen.add(key)
                group.append(candidate)
                queue.append(candidate)
    if len(group) != 60:
        raise RuntimeError(f"Icosahedral generator produced {len(group)} rotations, expected 60")
    return torch.stack(group).to(dtype)


def icosahedral_face_normals(dtype=torch.float32) -> torch.Tensor:
    """Return 20 normalized outward normals, one for each icosahedron face."""
    phi = (1.0 + math.sqrt(5.0)) / 2.0
    inv_phi = 1.0 / phi
    vertices = []
    for a in (-1.0, 1.0):
        for b in (-1.0, 1.0):
            for c in (-1.0, 1.0):
                vertices.append((a, b, c))
    for a in (-1.0, 1.0):
        for b in (-1.0, 1.0):
            vertices.extend(
                [
                    (0.0, a * inv_phi, b * phi),
                    (a * inv_phi, b * phi, 0.0),
                    (a * phi, 0.0, b * inv_phi),
                ]
            )
    normals = torch.tensor(vertices, dtype=dtype)
    return normals / normals.norm(dim=-1, keepdim=True)


def interface_compatibility(
    module_frames: torch.Tensor,
    face: torch.Tensor,
    interface_points: torch.Tensor,
) -> torch.Tensor:
    """Return a deterministic geometric compatibility proxy in ``[0,1]``.

    ``face`` is an index in ``[0,19]``. The module's local z-axis is compared
    with the selected face normal, while interface-point compactness provides a
    secondary geometric term. This makes the face assignment semantically
    meaningful instead of applying a global rotation that leaves pairwise
    distances unchanged.
    """
    if module_frames.ndim != 4 or module_frames.shape[-2:] != (3, 3):
        raise ValueError("module_frames must have shape (B, N, 3, 3)")
    B, N = module_frames.shape[:2]
    if face.shape != (B,):
        raise ValueError("face must have shape (B,)")
    if torch.any((face < 0) | (face >= 20)):
        raise ValueError("face must contain indices in [0, 19]")
    if interface_points.ndim != 4 or interface_points.shape[0] != B or interface_points.shape[1] != N or interface_points.shape[-1] != 3:
        raise ValueError("interface_points must have shape (B, N, P, 3)")

    normals = icosahedral_face_normals(module_frames.dtype).to(module_frames.device)
    target = normals[face]
    # Average local z-axis over the supplied interface residues.
    z_axis = module_frames[..., :, 2].mean(dim=1)
    z_axis = z_axis / z_axis.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    alignment = ((z_axis * target).sum(dim=-1).clamp(-1.0, 1.0) + 1.0) * 0.5

    points = interface_points.reshape(B, -1, 3)
    if points.shape[1] < 2:
        compactness = torch.ones(B, device=points.device, dtype=points.dtype)
    else:
        distances = torch.cdist(points, points)
        mask = ~torch.eye(points.shape[1], device=points.device, dtype=torch.bool)
        mean_distance = distances.masked_select(mask.unsqueeze(0)).view(B, -1).mean(dim=-1)
        compactness = torch.exp(-mean_distance / 10.0)

    return 0.7 * alignment + 0.3 * compactness
