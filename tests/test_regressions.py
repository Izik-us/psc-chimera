import math

import pytest
import torch

from chimera.geometry import validate_backbone
from chimera.pcgrad import project_conflicting_gradients
from chimera.structure_utils import load_backbone_coords_pdb


def _ideal_backbone(length: int = 3) -> torch.Tensor:
    """Construct a simple non-clashing backbone with the expected bond lengths."""
    coords = torch.zeros(1, length, 4, 3)
    for i in range(length):
        ca = torch.tensor([3.8 * i, 0.0, 0.0])
        coords[0, i, 1] = ca
        coords[0, i, 0] = ca + torch.tensor([-1.46, 0.0, 0.0])
        coords[0, i, 2] = ca + torch.tensor([1.53, 0.0, 0.0])
        coords[0, i, 3] = ca + torch.tensor([1.53, 1.23, 0.0])
    return coords


def test_geometry_does_not_count_covalent_bonds_as_clashes():
    coords = _ideal_backbone()
    report = validate_backbone(coords, clash_distance=2.0)
    assert report.clash_count == 0


def test_geometry_still_detects_nonbonded_clash():
    coords = _ideal_backbone()
    coords[0, 2, 3] = coords[0, 0, 0]
    report = validate_backbone(coords, clash_distance=2.0)
    assert report.clash_count > 0


def test_pcgrad_releases_graph_after_last_loss():
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    losses = [(parameter - 2).pow(2).sum(), (parameter + 2).pow(2).sum()]
    project_conflicting_gradients(losses, [parameter])
    assert parameter.grad is not None
    assert torch.isfinite(parameter.grad).all()


def test_pcgrad_rejects_empty_parameter_set():
    parameter = torch.nn.Parameter(torch.tensor([1.0]), requires_grad=False)
    with pytest.raises(ValueError, match="trainable parameter"):
        project_conflicting_gradients([parameter.sum()], [parameter])


def test_pdb_loader_rejects_missing_oxygen(tmp_path):
    # Minimal PDB with N/CA/C but no O.
    lines = [
        "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00 20.00           N",
        "ATOM      2  CA  ALA A   1       1.460   0.000   0.000  1.00 20.00           C",
        "ATOM      3  C   ALA A   1       2.990   0.000   0.000  1.00 20.00           C",
    ]
    path = tmp_path / "missing_o.pdb"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="complete N/CA/C/O"):
        load_backbone_coords_pdb(path)
