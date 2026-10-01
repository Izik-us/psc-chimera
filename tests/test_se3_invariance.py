import torch
import pytest

from chimera.flow_matching import FlowMatchingBackbone, InvariantPointAttention, so3_exp
from chimera.lie import relative_rotation, so3_log


def test_canonical_velocity_field_is_translation_invariant():
    torch.manual_seed(7)
    backbone = FlowMatchingBackbone(d_single=64, d_pair=32, n_blocks=1)
    model = backbone.flow_model.velocity_field.eval()
    R = so3_exp(torch.randn(1, 5, 3) * 0.2)
    t = torch.randn(1, 5, 3)
    pair = torch.randn(1, 5, 5, 32)
    single = torch.randn(1, 5, 64)
    time = torch.tensor([0.4])
    v_rot, v_trans = model(R, t, time, pair, single)

    shift = torch.tensor([11.0, -7.0, 4.0])
    v_rot_shift, v_trans_shift = model(R, t + shift, time, pair, single)
    assert torch.allclose(v_rot, v_rot_shift, atol=1e-5, rtol=1e-5)
    assert torch.allclose(v_trans, v_trans_shift, atol=1e-5, rtol=1e-5)


def test_relative_rotation_uses_source_frame():
    R0 = so3_exp(torch.tensor([[0.2, -0.1, 0.3]]))
    R1 = so3_exp(torch.tensor([[-0.4, 0.5, 0.1]]))
    expected = so3_log(R0.transpose(-1, -2) @ R1)
    assert torch.allclose(relative_rotation(R0, R1), expected, atol=1e-6)


def test_ipa_rejects_nondivisible_explicit_head_count():
    with pytest.raises(ValueError, match="must be divisible by n_head"):
        InvariantPointAttention(d_single=30, d_pair=16, n_head=8)


def test_ipa_default_head_count_has_exact_feature_partition():
    module = InvariantPointAttention(d_single=256, d_pair=32)
    assert module.n_head == 8
    assert 256 % module.n_head == 0
    assert module.q_s.out_features == 256


def test_canonical_flow_bridge_is_identity_not_frozen_random_adapter():
    backbone = FlowMatchingBackbone(d_single=32, d_pair=16, n_blocks=1)
    assert isinstance(backbone.frozen_bridge, torch.nn.Identity)
    assert list(backbone.frozen_bridge.parameters()) == []
    features = torch.randn(2, 5, 32)
    assert torch.equal(backbone.frozen_bridge(features), features)
