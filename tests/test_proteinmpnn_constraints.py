import pytest
import torch

from chimera.proteinmpnn import SequenceDecoder


def test_fixed_positions_are_hard_constrained():
    decoder = SequenceDecoder(c_node=16, vocab_size=20)
    node_features = torch.randn(1, 4, 16)
    fixed_positions = torch.tensor([[True, False, True, False]])
    fixed_aas = torch.tensor([[3, 0, 7, 0]])

    logits = decoder(
        node_features,
        fixed_positions=fixed_positions,
        fixed_aas=fixed_aas,
    )

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
        decoder(
            node_features,
            fixed_positions=fixed_positions,
            fixed_aas=fixed_aas,
        )
