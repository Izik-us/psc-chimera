import torch

from data.geometry import derive_geometry


def _backbone(length=6):
    coordinates = torch.zeros(length, 4, 3)
    for index in range(length):
        x = index * 3.8
        coordinates[index, 0] = torch.tensor([x, 0.0, 0.0])
        coordinates[index, 1] = torch.tensor([x + 1.4, 0.2, 0.1])
        coordinates[index, 2] = torch.tensor([x + 2.7, 0.0, 0.2])
        coordinates[index, 3] = torch.tensor([x + 3.0, 1.0, 0.3])
    return coordinates


def test_geometry_is_invariant_to_global_rigid_transform():
    coordinates = _backbone()
    atom_mask = torch.ones(6, 4, dtype=torch.bool)
    original = derive_geometry(coordinates, atom_mask, k_neighbors=3)
    rotation = torch.tensor(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )
    transformed = torch.einsum("ij, laj->lai", rotation, coordinates)
    transformed = transformed + torch.tensor([21.0, -4.5, 8.2])
    moved = derive_geometry(transformed, atom_mask, k_neighbors=3)

    assert torch.equal(original.edge_index, moved.edge_index)
    assert torch.equal(original.edge_mask, moved.edge_mask)
    assert torch.equal(original.contacts, moved.contacts)
    assert torch.allclose(original.distances, moved.distances, atol=1e-5)
    assert torch.allclose(original.relative_positions, moved.relative_positions, atol=1e-5)
    assert torch.allclose(original.relative_rotations, moved.relative_rotations, atol=1e-5)
    assert torch.allclose(original.torsions, moved.torsions, atol=1e-5)
    assert torch.allclose(original.edge_features, moved.edge_features, atol=1e-5)


def test_invalid_residues_do_not_contribute_to_pairs_or_neighbor_edges():
    coordinates = _backbone()
    atom_mask = torch.ones(6, 4, dtype=torch.bool)
    residue_mask = torch.ones(6, dtype=torch.bool)
    residue_mask[2] = False
    geometry = derive_geometry(coordinates, atom_mask, residue_mask, k_neighbors=3)

    assert not geometry.pair_mask[2].any()
    assert not geometry.pair_mask[:, 2].any()
    assert not (geometry.edge_mask[2]).any()
    assert not (geometry.edge_index[geometry.edge_mask] == 2).any()
    assert not geometry.contacts[2].any()


def test_knn_graph_preserves_residue_correspondence_after_permutation():
    coordinates = _backbone()
    atom_mask = torch.ones(6, 4, dtype=torch.bool)
    original = derive_geometry(coordinates, atom_mask, k_neighbors=2)
    order = torch.tensor([3, 0, 5, 1, 4, 2])
    permuted = derive_geometry(coordinates[order], atom_mask[order], k_neighbors=2)

    for new_index, old_index in enumerate(order.tolist()):
        original_neighbors = set(original.edge_index[old_index][original.edge_mask[old_index]].tolist())
        permuted_neighbors = set(permuted.edge_index[new_index][permuted.edge_mask[new_index]].tolist())
        mapped_neighbors = {order[new_neighbor].item() for new_neighbor in permuted_neighbors}
        assert mapped_neighbors == original_neighbors


def test_single_residue_is_supported_with_empty_neighbor_mask():
    geometry = derive_geometry(
        _backbone(1), torch.ones(1, 4, dtype=torch.bool), k_neighbors=4
    )

    assert geometry.edge_index.shape == (1, 1)
    assert not geometry.edge_mask.any()
    assert not geometry.torsion_mask.any()