import torch

from chimera.components import ProteinMPNNBackbone


def _backbone(batch=1, length=5):
    # Non-collinear N-CA-C triplets with distinct residue positions.
    ca = torch.stack(
        [
            torch.arange(length, dtype=torch.float32) * 3.8,
            torch.zeros(length),
            torch.zeros(length),
        ],
        dim=-1,
    )
    n = ca + torch.tensor([-0.5, 1.0, 0.0])
    c = ca + torch.tensor([0.5, 0.0, 1.0])
    o = c + torch.tensor([0.0, 0.5, 0.5])
    coords = torch.stack([n, ca, c, o], dim=1).unsqueeze(0)
    return coords.expand(batch, -1, -1, -1).contiguous()


def test_proteinmpnn_backbone_masks_degenerate_frames_and_keeps_finite_outputs():
    torch.manual_seed(7)
    coords = _backbone()
    # Degenerate N-CA-C frame: all three atoms coincide at residue 2.
    coords[:, 2, :3] = coords[:, 2, 1:2]
    features = torch.randn(1, coords.shape[1], 16)
    model = ProteinMPNNBackbone(
        node_features=16, edge_features=16, max_neighbors=3, n_mp_layers=2
    ).eval()

    output = model(coords, features)

    assert output.shape == features.shape
    assert torch.isfinite(output).all()
    # A degenerate frame is excluded from the effective geometric mask.
    assert torch.equal(output[:, 2], torch.zeros_like(output[:, 2]))


def test_proteinmpnn_backbone_is_rigid_transform_invariant():
    torch.manual_seed(11)
    coords = _backbone()
    features = torch.randn(1, coords.shape[1], 16)
    model = ProteinMPNNBackbone(
        node_features=16, edge_features=16, max_neighbors=3, n_mp_layers=2
    ).eval()

    angle = torch.tensor(0.71)
    rotation = torch.tensor(
        [
            [torch.cos(angle), -torch.sin(angle), 0.0],
            [torch.sin(angle), torch.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    translation = torch.tensor([13.0, -4.0, 2.5])
    transformed = coords @ rotation.T + translation

    with torch.no_grad():
        original_output = model(coords, features)
        transformed_output = model(transformed, features)

    torch.testing.assert_close(
        original_output, transformed_output, rtol=2e-4, atol=2e-4
    )
