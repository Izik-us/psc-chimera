import pytest
import torch
import torch.nn.functional as F

from chimera.proteinmpnn_conditioning import ProteinMPNNConditioningAdapter


def _adapter_inputs(batch=2, length=7):
    native = F.log_softmax(torch.randn(batch, length, 21), dim=-1)
    evo = torch.randn(batch, length, 24)
    geometry = torch.randn(batch, length, 32)
    nrps_logits = torch.randn(batch, length, 20)
    return native, evo, geometry, nrps_logits


def test_conditioning_adapter_is_initially_native_distribution_noop():
    adapter = ProteinMPNNConditioningAdapter(
        d_evolutionary=24, d_geometry=32, hidden_dim=16
    )
    native, evo, geometry, nrps = _adapter_inputs()
    actual = adapter(native, evo, geometry, nrps)
    expected = F.log_softmax(native[..., :20], dim=-1)

    assert actual.shape == (2, 7, 20)
    assert torch.allclose(actual, expected, atol=1e-7, rtol=1e-7)
    assert torch.allclose(actual.exp().sum(-1), torch.ones(2, 7), atol=1e-6)


def test_conditioning_adapter_learns_residual_without_mutating_native_inputs():
    adapter = ProteinMPNNConditioningAdapter(
        d_evolutionary=24, d_geometry=32, hidden_dim=16
    )
    native, evo, geometry, nrps = _adapter_inputs()
    original_native = native.clone()
    output = adapter(native, evo, geometry, nrps)
    target = torch.randint(0, 20, (2, 7))
    loss = F.nll_loss(output.reshape(-1, 20), target.reshape(-1))
    loss.backward()

    assert torch.equal(native, original_native)
    assert adapter.residual_head[-1].weight.grad is not None
    assert torch.isfinite(adapter.residual_head[-1].weight.grad).all()


def test_conditioning_adapter_masks_residuals_on_padding():
    adapter = ProteinMPNNConditioningAdapter(
        d_evolutionary=24, d_geometry=32, hidden_dim=16
    )
    native, evo, geometry, nrps = _adapter_inputs()
    mask = torch.tensor([[True, True, False, True, False, True, True]]).expand(2, -1)
    output = adapter(native, evo, geometry, nrps, residue_mask=mask)
    expected = F.log_softmax(native[..., :20], dim=-1)

    assert torch.allclose(output[~mask], expected[~mask], atol=1e-7, rtol=1e-7)


def test_conditioning_adapter_rejects_misaligned_features():
    adapter = ProteinMPNNConditioningAdapter(
        d_evolutionary=24, d_geometry=32, hidden_dim=16
    )
    native, evo, geometry, nrps = _adapter_inputs()
    with pytest.raises(ValueError, match="geometric_features"):
        adapter(native, evo, geometry[..., :12], nrps)
