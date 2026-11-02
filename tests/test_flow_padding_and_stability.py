"""Regression tests for padding masks, the bridge time cap, and gradient clipping."""

import pytest
import torch

from chimera import CHIMERAv2
from chimera.lie import so3_exp
from chimera.schrodinger_bridge import SE3SchrodingerBridge
from chimera.se3_flow import InvariantPointAttention, VelocityField
from chimera.training import CanonicalTrainer, CanonicalTrainingBatch


def _frames(batch, length, scale=8.0):
    return so3_exp(torch.randn(batch, length, 3) * 0.5), torch.randn(batch, length, 3) * scale


def _pad_frames(R, t, n_pad, far=0.0):
    batch = R.shape[0]
    R_pad = torch.cat([R, torch.eye(3).expand(batch, n_pad, 3, 3)], dim=1)
    t_pad = torch.cat([t, torch.full((batch, n_pad, 3), far)], dim=1)
    return R_pad, t_pad


def _mask(batch, real, total):
    mask = torch.zeros(batch, total, dtype=torch.bool)
    mask[:, :real] = True
    return mask


def test_ipa_with_mask_ignores_padding_and_without_mask_does_not():
    torch.manual_seed(0)
    real, total = 12, 16
    ipa = InvariantPointAttention(d_single=64, d_pair=32, n_head=4).eval()
    s, z = torch.randn(1, real, 64), torch.randn(1, real, real, 32)
    R, t = _frames(1, real)
    with torch.no_grad():
        reference = ipa(s, z, R, t)
        s_pad = torch.cat([s, torch.randn(1, total - real, 64)], dim=1)
        z_pad = torch.randn(1, total, total, 32)
        z_pad[:, :real, :real] = z
        R_pad, t_pad = _pad_frames(R, t, total - real)
        mask = _mask(1, real, total)
        masked = ipa(s_pad, z_pad, R_pad, t_pad, residue_mask=mask)
        unmasked = ipa(s_pad, z_pad, R_pad, t_pad)
    assert torch.allclose(masked[:, :real], reference, atol=1e-5)
    assert masked[:, real:].abs().max() == 0
    # Documents the original defect: without a mask padding leaks into real residues.
    assert (unmasked[:, :real] - reference).abs().max() > 1e-3


def test_velocity_field_is_padding_invariant_with_mask():
    torch.manual_seed(1)
    real, total = 12, 17
    field = VelocityField(d_single=32, d_pair=16, n_blocks=2, n_head=4).eval()
    R, t = _frames(1, real)
    R0, t0 = _frames(1, real)
    pair, evo = torch.randn(1, real, real, 16), torch.randn(1, real, 32)
    time = torch.tensor([0.4])
    with torch.no_grad():
        ref_r, ref_t = field(R, t, time, pair, evo, R0, t0)
        R_p, t_p = _pad_frames(R, t, total - real)
        R0_p, t0_p = _pad_frames(R0, t0, total - real)
        pair_p = torch.randn(1, total, total, 16)
        pair_p[:, :real, :real] = pair
        evo_p = torch.cat([evo, torch.randn(1, total - real, 32)], dim=1)
        out_r, out_t = field(R_p, t_p, time, pair_p, evo_p, R0_p, t0_p, residue_mask=_mask(1, real, total))
    assert torch.allclose(out_r[:, :real], ref_r, atol=1e-5)
    assert torch.allclose(out_t[:, :real], ref_t, atol=1e-4)
    assert out_r[:, real:].abs().max() == 0 and out_t[:, real:].abs().max() == 0


def test_bridge_loss_is_padding_invariant_with_mask():
    real, total = 10, 14
    torch.manual_seed(2)
    field = VelocityField(d_single=32, d_pair=16, n_blocks=2, n_head=4).eval()
    bridge = SE3SchrodingerBridge(field, diffusion=1e-12)
    R0, t0 = _frames(1, real)
    R1, t1 = _frames(1, real)
    pair, evo = torch.randn(1, real, real, 16), torch.randn(1, real, 32)
    torch.manual_seed(123)
    ref = bridge.bridge_loss(R0, t0, R1, t1, pair, evo)
    R0p, t0p = _pad_frames(R0, t0, total - real)
    R1p, t1p = _pad_frames(R1, t1, total - real)
    pair_p = torch.randn(1, total, total, 16)
    pair_p[:, :real, :real] = pair
    evo_p = torch.cat([evo, torch.randn(1, total - real, 32)], dim=1)
    torch.manual_seed(123)
    padded = bridge.bridge_loss(R0p, t0p, R1p, t1p, pair_p, evo_p, residue_mask=_mask(1, real, total))
    assert torch.isfinite(padded)
    assert torch.allclose(padded, ref, rtol=1e-3, atol=1e-3)


def test_coupling_is_unaffected_by_padding_contents():
    torch.manual_seed(3)
    batch, real, total = 3, 8, 12
    R0, t0 = _frames(batch, real)
    R1, t1 = _frames(batch, real)
    mask = _mask(batch, real, total)
    bridge = SE3SchrodingerBridge(torch.nn.Identity(), diffusion=0.05)
    picks = []
    for far in (0.0, 500.0):
        R0p, t0p = _pad_frames(R0, t0, total - real, far=far)
        R1p, t1p = _pad_frames(R1, t1, total - real, far=-far)
        torch.manual_seed(7)
        _, tt = bridge._coupled_targets(R0p, t0p, R1p, t1p, residue_mask=mask)
        picks.append(tt[:, :real])
    assert torch.equal(picks[0], picks[1])


class _RecordingDrift(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.times = []

    def forward(self, R, t, time, pair_cond, evol_single, R0=None, t0=None, substrate_coords=None, evol_conditioning_fn=None):
        self.times.append(float(time.max()))
        return torch.zeros_like(t), torch.zeros_like(t)


def test_training_and_sampling_times_respect_the_cap(monkeypatch):
    drift = _RecordingDrift()
    bridge = SE3SchrodingerBridge(drift, diffusion=0.05, train_time_max=0.99)
    R0, t0 = _frames(1, 6)
    R1, t1 = _frames(1, 6)
    pair, evo = torch.zeros(1, 6, 6, 4), torch.zeros(1, 6, 4)
    monkeypatch.setattr(torch, "rand", lambda *a, **k: torch.ones(*a, **{x: y for x, y in k.items() if x in ("device", "dtype")}))
    bridge.bridge_loss(R0, t0, R1, t1, pair, evo)
    assert drift.times[-1] == pytest.approx(0.99)
    drift.times.clear()
    bridge.sample(R0, t0, pair, evo, n_steps=400)
    assert max(drift.times) <= 0.99 + 1e-6
    with pytest.raises(ValueError):
        SE3SchrodingerBridge(drift, train_time_max=1.0)


def test_masked_drift_interface_is_enforced():
    class NoMask(torch.nn.Module):
        def forward(self, R, t, time, pair_cond, evol_single, R0=None, t0=None, substrate_coords=None, evol_conditioning_fn=None):
            return torch.zeros_like(t), torch.zeros_like(t)

    bridge = SE3SchrodingerBridge(NoMask())
    R0, t0 = _frames(1, 6)
    with pytest.raises(TypeError):
        bridge.bridge_loss(R0, t0, R0, t0, torch.zeros(1, 6, 6, 4), torch.zeros(1, 6, 4), residue_mask=torch.ones(1, 6, dtype=torch.bool))


def _tiny_model():
    torch.manual_seed(0)
    return CHIMERAv2(d_evo_single=32, d_evo_pair=16, d_se3=32, d_pair_out=32, d_mpnn=32,
                     n_flow_blocks=2, evoformer_n_blocks=2, n_domains=2, n_modules=2, n_mc_dropout=2)


def _flow_batch(length=16, real=16):
    R1, t1 = _frames(1, length)
    R0 = R1 @ so3_exp(torch.randn(1, length, 3) * 0.2)
    t0 = t1 + torch.randn(1, length, 3)
    msa = torch.randint(0, 20, (1, 1, length))
    msa[..., real:] = 22
    return CanonicalTrainingBatch(
        msa_tokens=msa, pair_features=torch.zeros(1, length, length, 16),
        source_R=R0, source_t=t0, target_R=R1, target_t=t1,
        target_sequence=torch.randint(0, 20, (1, length)),
    )


def test_trainer_clips_and_reports_gradient_norm():
    model = _tiny_model()
    trainer = CanonicalTrainer(model, "flow", learning_rate=1e-3, random_seed=0, max_grad_norm=0.5)
    out = trainer.train_step(_flow_batch())
    assert out["grad_norm"] is not None and out["grad_norm"] > 0.5
    assert out["grad_clipped"] is True
    total = torch.sqrt(sum(p.grad.square().sum() for p in model.parameters() if p.grad is not None))
    assert float(total) <= 0.5 * 1.001


def test_trainer_clipping_can_be_disabled_and_validated():
    trainer = CanonicalTrainer(_tiny_model(), "flow", learning_rate=1e-3, random_seed=0, max_grad_norm=None)
    out = trainer.train_step(_flow_batch())
    assert out["grad_norm"] is None and out["grad_clipped"] is False
    with pytest.raises(ValueError):
        CanonicalTrainer(_tiny_model(), "flow", max_grad_norm=0.0)


def test_flow_training_step_with_padded_batch_passes_gradient_contract():
    trainer = CanonicalTrainer(_tiny_model(), "flow", learning_rate=1e-3, random_seed=0)
    out = trainer.train_step(_flow_batch(length=16, real=11))
    assert out["loss"] == out["loss"] and out["grad_norm"] is not None


def test_ipa_gradient_checkpointing_matches_plain_backward():
    torch.manual_seed(5)
    field = VelocityField(d_single=32, d_pair=16, n_blocks=2, n_head=4).train()
    length = 10
    R, t = _frames(1, length)
    R0, t0 = _frames(1, length)
    pair, evo, time = torch.randn(1, length, length, 16), torch.randn(1, length, 32), torch.tensor([0.3])

    def grads(enabled):
        field.gradient_checkpointing = enabled
        field.zero_grad(set_to_none=True)
        v_r, v_t = field(R, t, time, pair, evo, R0, t0)
        (v_r.square().sum() + v_t.square().sum()).backward()
        return [p.grad.clone() for p in field.parameters() if p.grad is not None]

    plain, checkpointed = grads(False), grads(True)
    assert len(plain) == len(checkpointed) > 0
    for a, b in zip(plain, checkpointed):
        assert torch.allclose(a, b, atol=1e-5, rtol=1e-4)
