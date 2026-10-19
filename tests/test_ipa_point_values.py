import math

import torch

from chimera.se3_flow import InvariantPointAttention


def _rotation(axis, angle):
    axis = torch.as_tensor(axis, dtype=torch.float32)
    axis = axis / axis.norm()
    x, y, z = axis
    c = math.cos(angle)
    s = math.sin(angle)
    C = 1.0 - c
    return torch.tensor(
        [
            [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
        ]
    )


def _inputs(batch=1, length=6, d_single=32, d_pair=16):
    torch.manual_seed(19)
    s = torch.randn(batch, length, d_single)
    z = torch.randn(batch, length, length, d_pair)
    R = torch.eye(3).expand(batch, length, 3, 3).clone()
    t = torch.randn(batch, length, 3)
    return s, z, R, t


def test_ipa_outputs_are_invariant_to_global_se3_transform():
    model = InvariantPointAttention(
        d_single=32, d_pair=16, n_head=4, n_qk_pts=2, n_v_pts=3
    ).eval()
    s, z, R, t = _inputs()

    Q = _rotation((1.0, -2.0, 0.5), 0.73)
    shift = torch.tensor([7.0, -3.5, 2.25])
    transformed_R = torch.einsum("ij,bljk->blik", Q, R)
    transformed_t = torch.einsum("ij,blj->bli", Q, t) + shift

    with torch.no_grad():
        reference = model(s, z, R, t)
        transformed = model(s, z, transformed_R, transformed_t)

    assert torch.allclose(reference, transformed, atol=2e-5, rtol=2e-5)


def test_ipa_output_changes_when_value_point_projection_is_removed():
    model = InvariantPointAttention(
        d_single=32, d_pair=16, n_head=4, n_qk_pts=2, n_v_pts=3
    ).eval()
    s, z, R, t = _inputs()

    with torch.no_grad():
        reference = model(s, z, R, t)
        saved = model.v_p.weight.detach().clone()
        model.v_p.weight.zero_()
        without_value_points = model(s, z, R, t)
        model.v_p.weight.copy_(saved)

    assert not torch.allclose(reference, without_value_points, atol=1e-7, rtol=1e-7)


def test_ipa_value_points_receive_gradients():
    model = InvariantPointAttention(
        d_single=32, d_pair=16, n_head=4, n_qk_pts=2, n_v_pts=3
    )
    s, z, R, t = _inputs()
    output = model(s, z, R, t)
    loss = output.square().mean()
    loss.backward()

    grad = model.v_p.weight.grad
    assert grad is not None
    assert torch.isfinite(grad).all()
    assert grad.abs().sum() > 0


def test_ipa_supports_independent_qk_and_value_point_counts():
    model = InvariantPointAttention(
        d_single=48, d_pair=20, n_head=6, n_qk_pts=3, n_v_pts=5
    )
    s, z, R, t = _inputs(d_single=48, d_pair=20)
    output = model(s, z, R, t)
    assert output.shape == s.shape
    assert torch.isfinite(output).all()
