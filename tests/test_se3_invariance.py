import torch
import pytest

from chimera.flow_matching import FlowMatchingBackbone, InvariantPointAttention, se3_interp, so3_exp, so3_geodesic_interp
from chimera.lie import relative_rotation, so3_log
from chimera.schrodinger_bridge import SE3SchrodingerBridge


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


def test_relative_rotation_identity_inverse_and_composition():
    R0 = so3_exp(torch.tensor([[0.3, -0.2, 0.1]]))
    delta = torch.tensor([[-0.1, 0.25, 0.2]])
    R1 = R0 @ so3_exp(delta)

    assert torch.allclose(relative_rotation(R0, R0), torch.zeros_like(delta), atol=1e-6)
    assert torch.allclose(relative_rotation(R0, R1), delta, atol=1e-5)
    assert torch.allclose(relative_rotation(R1, R0), -delta, atol=1e-5)
    assert torch.allclose(R0 @ so3_exp(relative_rotation(R0, R1)), R1, atol=1e-5)


def test_geodesic_and_se3_interpolation_endpoints():
    R0 = so3_exp(torch.tensor([[0.2, -0.1, 0.0]]))
    R1 = so3_exp(torch.tensor([[-0.3, 0.15, 0.4]]))
    t0 = torch.tensor([[1.0, 2.0, 3.0]])
    t1 = torch.tensor([[-1.0, 0.5, 4.0]])

    assert torch.allclose(so3_geodesic_interp(R0, R1, 0.0), R0, atol=1e-6)
    assert torch.allclose(so3_geodesic_interp(R0, R1, 1.0), R1, atol=1e-6)
    R_start, t_start = se3_interp(R0, t0, R1, t1, 0.0)
    R_end, t_end = se3_interp(R0, t0, R1, t1, 1.0)
    assert torch.allclose(R_start, R0, atol=1e-6) and torch.equal(t_start, t0)
    assert torch.allclose(R_end, R1, atol=1e-6) and torch.equal(t_end, t1)


def test_sb_bridge_target_uses_canonical_source_relative_rotation():
    R0 = so3_exp(torch.tensor([[[0.2, 0.1, -0.1]]]))
    R1 = so3_exp(torch.tensor([[[-0.1, 0.25, 0.3]]]))
    expected = relative_rotation(R0, R1)
    assert torch.allclose(SE3SchrodingerBridge._relative_rotation(R0, R1), expected)


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
