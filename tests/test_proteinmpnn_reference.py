import numpy as np
import pytest
import torch

from chimera.proteinmpnn_reference import (
    AMINO_ACID_ALPHABET,
    ProteinMPNNReference,
)


def _inputs(length=9):
    ca = torch.zeros(1, length, 3)
    ca[0, :, 0] = torch.arange(length) * 3.8
    coords = torch.stack(
        (
            ca + torch.tensor([0.0, 1.2, 0.0]),
            ca,
            ca + torch.tensor([0.0, 0.0, 1.3]),
            ca + torch.tensor([0.0, -0.8, 1.5]),
        ),
        dim=2,
    )
    sequence = torch.arange(length, dtype=torch.long).remainder(20).unsqueeze(0)
    residue_mask = torch.ones(1, length, dtype=torch.bool)
    design_mask = torch.ones_like(residue_mask)
    residue_idx = torch.arange(length, dtype=torch.long).unsqueeze(0)
    chain_encoding = torch.zeros(1, length, dtype=torch.long)
    return coords, sequence, residue_mask, design_mask, residue_idx, chain_encoding


def test_reference_defaults_match_published_standard_architecture():
    model = ProteinMPNNReference()
    native = model.native_model

    assert model.hidden_dim == 128
    assert model.num_encoder_layers == 3
    assert model.num_decoder_layers == 3
    assert model.k_neighbors == 48
    assert len(native.encoder_layers) == 3
    assert len(native.decoder_layers) == 3
    assert native.W_s.num_embeddings == 21
    assert native.features.top_k == 48
    assert "W_e.weight" in native.state_dict()
    assert "encoder_layers.0.W1.weight" in native.state_dict()
    assert "decoder_layers.0.W1.weight" in native.state_dict()


def test_reference_forward_outputs_21_log_probabilities_and_accepts_order():
    model = ProteinMPNNReference().eval()
    coords, sequence, residue_mask, design_mask, residue_idx, chain_encoding = _inputs()
    length = sequence.shape[1]
    order = torch.arange(length).flip(0).unsqueeze(0)
    randn = torch.zeros(1, length)

    log_probs = model(
        coords, sequence, residue_mask, design_mask, residue_idx, chain_encoding,
        randn=randn, decoding_order=order,
    )

    assert log_probs.shape == (1, length, 21)
    assert torch.isfinite(log_probs).all()
    assert torch.allclose(log_probs.exp().sum(dim=-1), torch.ones(1, length), atol=1e-5)


def test_reference_sequence_probabilities_are_rigid_transform_invariant():
    torch.manual_seed(9)
    model = ProteinMPNNReference().eval()
    coords, sequence, residue_mask, design_mask, residue_idx, chain_encoding = _inputs()
    length = sequence.shape[1]
    randn = torch.linspace(-1.0, 1.0, length).unsqueeze(0)
    order = torch.arange(length).unsqueeze(0)

    original = model(
        coords, sequence, residue_mask, design_mask, residue_idx, chain_encoding,
        randn=randn, decoding_order=order,
    )
    angle = torch.tensor(0.73)
    rotation = torch.tensor(
        [[torch.cos(angle), -torch.sin(angle), 0.0],
         [torch.sin(angle), torch.cos(angle), 0.0],
         [0.0, 0.0, 1.0]]
    )
    transformed = coords @ rotation.T + torch.tensor([12.0, -5.0, 2.5])
    moved = model(
        transformed, sequence, residue_mask, design_mask, residue_idx, chain_encoding,
        randn=randn, decoding_order=order,
    )

    assert torch.allclose(original, moved, atol=2e-5, rtol=2e-5)


def test_reference_sampler_preserves_fixed_residues():
    model = ProteinMPNNReference().eval()
    coords, sequence, residue_mask, _, residue_idx, chain_encoding = _inputs(length=7)
    design_mask = torch.tensor([[True, False, True, False, True, False, True]])
    result = model.sample(
        coords, sequence, residue_mask, design_mask, residue_idx, chain_encoding,
        temperature=0.2,
        randn=torch.tensor([[0.2, 0.8, -0.4, 0.5, -0.1, 0.7, -0.9]]),
    )

    assert result["S"].shape == sequence.shape
    assert result["probs"].shape == (1, 7, 21)
    assert result["decoding_order"].shape == sequence.shape
    assert torch.equal(result["S"][~design_mask], sequence[~design_mask])
    assert torch.isfinite(result["probs"]).all()
    assert AMINO_ACID_ALPHABET.endswith("X")


def test_reference_checkpoint_round_trip_loads_strictly(tmp_path):
    torch.manual_seed(13)
    original = ProteinMPNNReference().eval()
    path = tmp_path / "proteinmpnn-reference.pt"
    torch.save(
        {
            "model_state_dict": original.native_model.state_dict(),
            "num_edges": 48,
            "noise_level": 0.2,
        },
        path,
    )

    restored = ProteinMPNNReference.from_pretrained(path)
    for key, value in original.native_model.state_dict().items():
        assert torch.equal(value, restored.native_model.state_dict()[key])


def test_reference_rejects_invalid_decoding_order_and_mask():
    model = ProteinMPNNReference().eval()
    coords, sequence, residue_mask, design_mask, residue_idx, chain_encoding = _inputs()
    with pytest.raises(ValueError, match="permutation"):
        model(
            coords, sequence, residue_mask, design_mask, residue_idx, chain_encoding,
            decoding_order=torch.zeros_like(sequence),
        )
    invalid_design = design_mask.clone()
    invalid_residue_mask = residue_mask.clone()
    invalid_residue_mask[:, -1] = False
    with pytest.raises(ValueError, match="padded or missing"):
        model(
            coords, sequence, invalid_residue_mask, invalid_design, residue_idx, chain_encoding,
        )
