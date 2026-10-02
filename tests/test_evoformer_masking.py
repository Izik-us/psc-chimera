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


def test_msa_representation_exchanges_information_across_sequences():
    torch.manual_seed(31)
    encoder = EvoFormerBackbone(d_single=32, d_pair=16).eval()
    tokens_a = torch.tensor([[[1, 2, 3, 4], [5, 6, 7, 8]]])
    tokens_b = tokens_a.clone()
    tokens_b[0, 0, 1] = 9

    encoded_a = encoder.encode_msa(tokens_a)
    encoded_b = encoder.encode_msa(tokens_b)

    assert not torch.allclose(encoded_a[0, 1], encoded_b[0, 1])


def test_masked_msa_reconstruction_trains_encoder_and_head():
    torch.manual_seed(32)
    encoder = EvoFormerBackbone(d_single=32, d_pair=16)
    tokens = torch.tensor([[[1, 2, 3, 4], [5, 6, 7, 8]]])
    loss = encoder.masked_reconstruction_loss(
        tokens,
        mask_probability=1.0,
        generator=torch.Generator().manual_seed(4),
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert encoder.msa_column_encoder.layers[0].self_attn.in_proj_weight.grad is not None
    assert encoder.reconstruction_head.weight.grad is not None