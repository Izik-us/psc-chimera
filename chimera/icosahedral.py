"""Icosahedral symmetry and interface geometry utilities."""

import math
import torch


def _axis_rotation(axis: torch.Tensor, angle: float) -> torch.Tensor:
    axis = axis / axis.norm()
    x, y, z = axis
    c, s = math.cos(angle), math.sin(angle)
    return torch.tensor([
        [c + x*x*(1-c), x*y*(1-c)-z*s, x*z*(1-c)+y*s],
        [y*x*(1-c)+z*s, c+y*y*(1-c), y*z*(1-c)-x*s],
        [z*x*(1-c)-y*s, z*y*(1-c)+x*s, c+z*z*(1-c)],
    ], dtype=torch.float32)


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
    while queue and len(group) <= 60:
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


def interface_compatibility(
    module_frames: torch.Tensor,
    face: torch.Tensor,
    interface_points: torch.Tensor,
) -> torch.Tensor:
    """Score transformed interface-point separation for one module per batch."""
    if module_frames.ndim != 4 or module_frames.shape[-2:] != (3, 3):
        raise ValueError("module_frames must have shape (B, N, 3, 3)")
    if interface_points.shape[-1] != 3:
        raise ValueError("interface_points must end in 3")
    if face.shape != (module_frames.shape[0],):
        raise ValueError("face must have shape (B,)")
    group = icosahedral_rotations(module_frames.dtype).to(module_frames.device)
    transforms = group[face.remainder(60)]
    points = torch.einsum("bij,bnpj->bnpi", transforms, interface_points)
    distances = torch.cdist(points.reshape(points.shape[0], -1, 3), points.reshape(points.shape[0], -1, 3))
    mask = ~torch.eye(distances.shape[-1], device=distances.device, dtype=torch.bool).unsqueeze(0)
    return torch.exp(-distances.masked_select(mask).view(distances.shape[0], -1).mean(dim=-1) / 10.0)
