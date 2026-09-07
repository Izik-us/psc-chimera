import torch

from chimera.flow_matching import VelocityField, so3_exp


def test_velocity_field_is_translation_invariant_for_rotational_state_features():
    torch.manual_seed(7)
    model = VelocityField(d_single=64, d_pair=32, n_blocks=1, n_head=8).eval()
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
