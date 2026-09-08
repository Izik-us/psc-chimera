import torch

from chimera.schrodinger_bridge import SE3SchrodingerBridge


class ZeroDrift(torch.nn.Module):
    def forward(self, R, t, time, pair_cond, evol_single, R0, t0, substrate_coords=None, evol_conditioning_fn=None):
        return torch.zeros_like(t), torch.zeros_like(t)


def test_bridge_interpolation_keeps_fixed_frames_exact():
    model = SE3SchrodingerBridge(ZeroDrift(), diffusion=0.01, sinkhorn_iters=10)
    R0 = torch.eye(3).view(1, 1, 3, 3).expand(1, 3, 3, 3).clone()
    R1 = R0.clone()
    t0 = torch.zeros(1, 3, 3)
    t1 = torch.ones(1, 3, 3)
    mask = torch.tensor([[True, False, True]])
    time = torch.tensor([0.5])

    Rt, xt, vr, vt = model.interpolate(R0, t0, R1, t1, time, fixed_mask=mask)

    assert torch.equal(Rt[:, mask[0]], R0[:, mask[0]])
    assert torch.equal(xt[:, mask[0]], t0[:, mask[0]])
    assert torch.equal(vr[:, mask[0]], torch.zeros_like(vr[:, mask[0]]))
    assert torch.equal(vt[:, mask[0]], torch.zeros_like(vt[:, mask[0]]))
