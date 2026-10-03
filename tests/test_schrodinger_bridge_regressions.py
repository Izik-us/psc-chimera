import torch
from torch import nn

from chimera.schrodinger_bridge import SE3SchrodingerBridge
from chimera.lie import so3_exp


class ZeroDrift(nn.Module):
    def forward(self, R, t, time, pair_cond, evol_single, R0, t0=None, substrate_coords=None, evol_conditioning_fn=None):
        return torch.zeros(R.shape[0], R.shape[1], 3, device=R.device, dtype=R.dtype), torch.zeros_like(t)


def test_se3_endpoint_coupling_uses_pairwise_relative_rotation():
    bridge = SE3SchrodingerBridge(ZeroDrift(), diffusion=0.05, sinkhorn_iters=30)
    R0 = torch.eye(3).repeat(2, 1, 1, 1)
    R1 = torch.stack([
        torch.eye(3),
        so3_exp(torch.tensor([0.0, 0.0, 1.0])),
    ], dim=0).unsqueeze(1)
    R0 = R0[:, :1]
    t0 = torch.zeros(2, 1, 3)
    t1 = torch.zeros(2, 1, 3)
    target_R, target_t = bridge._coupled_targets(R0, t0, R1, t1)
    assert target_R.shape == R0.shape
    assert target_t.shape == t0.shape


def test_se3_bridge_sample_rejects_invalid_step_count():
    bridge = SE3SchrodingerBridge(ZeroDrift())
    R0 = torch.eye(3).reshape(1, 1, 3, 3)
    t0 = torch.zeros(1, 1, 3)
    pair = torch.zeros(1, 1, 1, 4)
    single = torch.zeros(1, 1, 4)
    try:
        bridge.sample(R0, t0, pair, single, n_steps=1)
    except ValueError as exc:
        assert "at least 2" in str(exc)
    else:
        raise AssertionError("n_steps=1 must be rejected")


def test_se3_sampler_is_inference_only_and_bridge_loss_remains_trainable():
    drift = torch.nn.Linear(4, 6)

    class Drift(torch.nn.Module):
        def __init__(self, layer):
            super().__init__()
            self.layer = layer

        def forward(self, R, t, time, pair_cond, evol_single, **kwargs):
            values = self.layer(evol_single)
            return values[..., :3], values[..., 3:]

    bridge = SE3SchrodingerBridge(Drift(drift), diffusion=0.01, sinkhorn_iters=2)
    R0 = torch.eye(3).reshape(1, 1, 3, 3).expand(1, 2, -1, -1).clone().requires_grad_()
    t0 = torch.zeros(1, 2, 3, requires_grad=True)
    pair = torch.zeros(1, 2, 2, 4)
    single = torch.randn(1, 2, 4, requires_grad=True)
    R1 = so3_exp(torch.randn(1, 2, 3) * 0.1)
    t1 = torch.randn(1, 2, 3)

    sampled_R, sampled_t = bridge.sample(R0, t0, pair, single, n_steps=2)
    assert not sampled_R.requires_grad and not sampled_t.requires_grad

    loss = bridge.bridge_loss(R0, t0, R1, t1, pair, single)
    loss.backward()
    assert drift.weight.grad is not None
    assert torch.isfinite(drift.weight.grad).all()
