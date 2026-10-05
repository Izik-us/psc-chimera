import pytest
import torch

from chimera.components import ProteinMPNNBackbone
from chimera.proteinmpnn import SequenceDecoder, ProteinMPNN, get_protein_graph


def test_fixed_positions_are_hard_constrained():
    decoder = SequenceDecoder(c_node=16, vocab_size=20)
    node_features = torch.randn(1, 4, 16)
    fixed_positions = torch.tensor([[True, False, True, False]])
    fixed_aas = torch.tensor([[3, 0, 7, 0]])
    logits = decoder(node_features, fixed_positions=fixed_positions, fixed_aas=fixed_aas)
    predicted = logits.argmax(dim=-1)
    assert predicted[0, 0].item() == 3
    assert predicted[0, 2].item() == 7
    assert torch.isneginf(logits[0, 0]).sum().item() == 19
    assert torch.isneginf(logits[0, 2]).sum().item() == 19


def test_invalid_fixed_amino_acids_are_rejected():
    decoder = SequenceDecoder(c_node=16, vocab_size=20)
    node_features = torch.randn(1, 2, 16)
    fixed_positions = torch.tensor([[True, False]])
    fixed_aas = torch.tensor([[20, 0]])
    with pytest.raises(ValueError, match="invalid amino-acid IDs"):
        decoder(node_features, fixed_positions=fixed_positions, fixed_aas=fixed_aas)


def test_protein_graph_is_rigid_transform_invariant():
    torch.manual_seed(0)
    coords = torch.randn(1, 8, 3)
    omega = torch.tensor([[0.3, -0.2, 0.1]])
    from chimera.lie import so3_exp
    Q = so3_exp(omega)[0]
    R = Q.unsqueeze(0).expand(1, 8, 3, 3).clone()
    idx1, edge1, mask1 = get_protein_graph(coords, R, k_neighbors=3)
    coords2 = torch.einsum("ij,blj->bli", Q, coords)
    R2 = torch.einsum("ij,bljk->blik", Q, R)
    idx2, edge2, mask2 = get_protein_graph(coords2, R2, k_neighbors=3)
    assert torch.equal(idx1, idx2)
    assert torch.equal(mask1, mask2)
    assert torch.allclose(edge1, edge2, atol=1e-5)


def test_length_one_graph_is_rejected():
    coords = torch.zeros(1, 1, 3)
    frames = torch.eye(3).reshape(1, 1, 3, 3)
    with pytest.raises(ValueError, match="at least two residues"):
        get_protein_graph(coords, frames, k_neighbors=1)


def test_proteinmpnn_forward_shapes():
    model = ProteinMPNN(c_node=16, c_edge=12, n_mp_layers=1, k_neighbors=2)
    coords = torch.randn(2, 5, 3)
    frames = torch.eye(3).reshape(1, 1, 3, 3).expand(2, 5, 3, 3).clone()
    evol = torch.randn(2, 5, 16)
    logits = model(coords, frames, evol)
    assert logits.shape == (2, 5, 20)
    assert torch.isfinite(logits).all()


def test_proteinmpnn_backbone_uses_geometry_to_change_features():
    model = ProteinMPNNBackbone(node_features=16, edge_features=12)
    evol = torch.randn(1, 6, 16)

    backbone_a = torch.zeros(1, 6, 4, 3)
    backbone_a[:, :, 1, :] = torch.tensor([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [3.0, 0.0, 0.0],
        [4.0, 0.0, 0.0],
        [5.0, 0.0, 0.0],
    ])
    for i in range(6):
        backbone_a[:, i, 0, :] = backbone_a[:, i, 1, :] - torch.tensor([0.0, 0.0, 1.0])
        backbone_a[:, i, 2, :] = backbone_a[:, i, 1, :] + torch.tensor([0.0, 1.0, 0.0])
        backbone_a[:, i, 3, :] = backbone_a[:, i, 2, :] + torch.tensor([0.0, 0.0, 1.0])

    backbone_b = backbone_a.clone()
    backbone_b[:, :, 1, 1] = torch.tensor([0.0, 1.0, 0.0, 1.0, 0.0, 1.0], dtype=backbone_b.dtype).view(1, 6)
    backbone_b[:, :, 2, 2] = torch.tensor([0.0, 0.0, 1.0, 1.0, 0.0, 2.0], dtype=backbone_b.dtype).view(1, 6)
    backbone_b[:, :, 3, 0] = torch.tensor([0.5, 1.5, 2.5, 3.5, 4.5, 5.5], dtype=backbone_b.dtype).view(1, 6)

    out_a = model(backbone_a, evol)
    out_b = model(backbone_b, evol)

    assert out_a.shape == (1, 6, 16)
    assert out_b.shape == (1, 6, 16)
    assert not torch.allclose(out_a, out_b, atol=1e-5)
