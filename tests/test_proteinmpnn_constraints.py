import pytest
import torch

from chimera.components import ProteinMPNNBackbone
from chimera.proteinmpnn import NodeMPNN, SequenceDecoder, ProteinMPNN, get_protein_graph


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

def test_proteinmpnn_backbone_is_rigid_transform_invariant():
    torch.manual_seed(11)
    model = ProteinMPNNBackbone(node_features=16, edge_features=12, max_neighbors=4).eval()
    evol = torch.randn(1, 7, 16)
    coords = torch.randn(1, 7, 4, 3)
    from chimera.lie import so3_exp

    Q = so3_exp(torch.tensor([[0.31, -0.22, 0.17]], dtype=coords.dtype))[0]
    translation = torch.tensor([2.5, -1.0, 3.2], dtype=coords.dtype)
    transformed = torch.einsum("ij,blaj->blai", Q, coords) + translation
    with torch.no_grad():
        out_a = model(coords, evol)
        out_b = model(transformed, evol)
    assert torch.allclose(out_a, out_b, atol=2e-5, rtol=2e-5)


def test_proteinmpnn_backbone_mask_blocks_invalid_residues():
    torch.manual_seed(12)
    model = ProteinMPNNBackbone(node_features=16, edge_features=12, max_neighbors=4).eval()
    evol_a = torch.randn(1, 6, 16)
    evol_b = evol_a.clone()
    evol_b[:, 4:] = torch.randn_like(evol_b[:, 4:]) * 100.0
    coords = torch.randn(1, 6, 4, 3)
    residue_mask = torch.tensor([[True, True, True, True, False, False]])
    with torch.no_grad():
        out_a = model(coords, evol_a, residue_mask=residue_mask)
        out_b = model(coords, evol_b, residue_mask=residue_mask)
    assert torch.allclose(out_a[:, :4], out_b[:, :4], atol=1e-5, rtol=1e-5)
    assert torch.equal(out_a[:, 4:], torch.zeros_like(out_a[:, 4:]))


def test_node_mpnn_headwise_attention_handles_empty_neighborhoods_and_gradients():
    torch.manual_seed(31)
    model = NodeMPNN(c_node=16, c_edge=12)
    node = torch.randn(2, 5, 16, requires_grad=True)
    edge = torch.randn(2, 5, 3, 12, requires_grad=True)
    neighbor_index = torch.zeros(2, 5, 3, dtype=torch.long)
    edge_mask = torch.zeros(2, 5, 3, dtype=torch.bool)

    output = model(node, edge, neighbor_index, edge_mask)
    assert output.shape == node.shape
    assert torch.isfinite(output).all()
    output.square().mean().backward()
    assert node.grad is not None and torch.isfinite(node.grad).all()
    assert edge.grad is not None and torch.isfinite(edge.grad).all()
