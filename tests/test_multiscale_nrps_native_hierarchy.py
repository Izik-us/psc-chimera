import torch

from chimera.sequence_design import MultiScaleNRPSDesigner


def test_multiscale_segment_pool_returns_zero_for_empty_spans():
    features = torch.randn(2, 5, 8)
    membership = torch.zeros(2, 3, 5, dtype=torch.bool)
    membership[0, 0, :2] = True
    membership[1, 1, 2:5] = True

    pooled, valid = MultiScaleNRPSDesigner._masked_segment_pool(features, membership)

    assert valid.tolist() == [[True, False, False], [False, True, False]]
    assert torch.isfinite(pooled).all()
    assert torch.equal(pooled[~valid], torch.zeros_like(pooled[~valid]))


def test_multiscale_forward_handles_empty_domain_and_padding():
    torch.manual_seed(31)
    model = MultiScaleNRPSDesigner(
        d_residue=16,
        d_domain=16,
        d_module=16,
        d_assembly=16,
        n_domains=3,
        n_modules=2,
        edge_dim=28,
    ).eval()
    batch, length, neighbors = 1, 6, 2
    residue = torch.randn(batch, length, 16)
    evolutionary = torch.randn_like(residue)
    edge_index = torch.tensor([[[1, 2], [0, 2], [1, 3], [2, 4], [3, 5], [4, 3]]])
    edge_features = torch.randn(batch, length, neighbors, 28)
    domains = torch.tensor([[[0, 2], [2, 4], [4, 4]]])
    modules = torch.tensor([[[0, 3], [3, 6]]])
    residue_mask = torch.tensor([[True, True, True, True, True, False]])
    edge_mask = torch.ones(batch, length, neighbors, dtype=torch.bool)

    with torch.no_grad():
        logits = model(
            residue,
            evolutionary,
            edge_features,
            edge_index,
            domains,
            modules,
            torch.tensor([7]),
            edge_mask=edge_mask,
            residue_mask=residue_mask,
        )

    assert logits.shape == (batch, length, 20)
    assert torch.isfinite(logits).all()
    assert torch.equal(logits[:, -1], torch.zeros_like(logits[:, -1]))
