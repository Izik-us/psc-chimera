import torch

from chimera.components import EvoFormerBackbone


def test_msa_padding_values_do_not_affect_valid_representations():
    torch.manual_seed(9)
    encoder = EvoFormerBackbone(d_single=32, d_pair=16).eval()
    tokens_a = torch.tensor(
        [[[1, 2, 3, 4], [1, 5, 6, 22]], [[2, 3, 4, 22], [2, 3, 22, 22]]]
    )
    padding = tokens_a.eq(22)
    tokens_b = tokens_a.clone()
    tokens_b[padding] = 17
    pair_features = torch.zeros(2, 4, 4, 16)

    single_a, pair_a = encoder(tokens_a, pair_features, msa_padding_mask=padding)
    single_b, pair_b = encoder(tokens_b, pair_features, msa_padding_mask=padding)

    residue_valid = (~padding).any(dim=1)
    pair_valid = residue_valid.unsqueeze(1) & residue_valid.unsqueeze(2)
    assert torch.allclose(single_a[residue_valid], single_b[residue_valid], atol=1e-6)
    assert torch.allclose(pair_a[pair_valid], pair_b[pair_valid], atol=1e-6)
    assert torch.equal(single_a[~residue_valid], torch.zeros_like(single_a[~residue_valid]))
    assert torch.equal(pair_a[~pair_valid], torch.zeros_like(pair_a[~pair_valid]))
    assert torch.isfinite(single_a).all() and torch.isfinite(pair_a).all()


def test_msa_mask_semantics_are_true_means_padding():
    encoder = EvoFormerBackbone(d_single=32, d_pair=16).eval()
    tokens = torch.tensor([[[1, 2, 3], [4, 5, 6]]])
    mask = torch.tensor([[[False, False, True], [False, True, True]]])
    pair_features = torch.zeros(1, 3, 3, 16)
    single, pair = encoder(tokens, pair_features, msa_padding_mask=mask)
    assert torch.count_nonzero(single[:, 2]) == 0
    assert torch.count_nonzero(pair[:, 2]) == 0
    assert torch.count_nonzero(pair[:, :, 2]) == 0