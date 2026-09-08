"""Regression tests for the merge-readiness scientific contracts."""

import copy

import torch
import torch.nn as nn

from chimera.bayesian import BayesianUncertaintyEstimator
from chimera.dpo import DPOBatch, DPOTrainer
from chimera.pcgrad import PCGradOptimizer
from chimera.schrodinger_bridge import SE3SchrodingerBridge


def test_gaussian_ei_uses_standard_deviation():
    mean = torch.tensor([0.8, 0.5])
    std = torch.tensor([0.2, 0.01])
    ei = BayesianUncertaintyEstimator.expected_improvement(mean, std, best_observed=0.6)
    assert ei.shape == mean.shape
    assert torch.isfinite(ei).all()
    assert (ei >= 0).all()


def test_dpo_reference_is_not_updated():
    class Policy(nn.Module):
        def __init__(self):
            super().__init__()
            self.logits = nn.Parameter(torch.zeros(1, 4, 3))

        def token_logprobs(self, context, tokens):
            logits = self.logits.expand(tokens.shape[0], -1, -1)
            return torch.log_softmax(logits, dim=-1).gather(-1, tokens.unsqueeze(-1)).squeeze(-1)

    policy = Policy()
    reference = copy.deepcopy(policy)
    for p in reference.parameters():
        p.requires_grad = False
    batch = DPOBatch(
        context=torch.zeros(2, 4, 2),
        chosen=torch.tensor([[0, 1, 2, 1], [0, 0, 1, 2]]),
        rejected=torch.tensor([[1, 1, 2, 1], [2, 0, 1, 2]]),
        chosen_mask=torch.ones(2, 4, dtype=torch.bool),
        rejected_mask=torch.ones(2, 4, dtype=torch.bool),
    )
    before = [p.detach().clone() for p in reference.parameters()]
    trainer = DPOTrainer(beta=0.1)
    optimizer = torch.optim.SGD(policy.parameters(), lr=0.01)
    trainer.step(optimizer, policy, reference, batch)
    assert all(torch.equal(a, b) for a, b in zip(before, reference.parameters()))


def test_pcgrad_optimizer_preserves_parameter_shape():
    model = nn.Linear(3, 1, bias=False)
    optimizer = PCGradOptimizer(model.parameters(), lr=1e-2)
    x = torch.tensor([[1.0, 0.0, 0.0]])
    y1 = model(x).sum()
    y2 = -model(x).sum()
    optimizer.pc_backward([y1, y2])
    optimizer.step()
    assert model.weight.shape == (1, 3)
    assert torch.isfinite(model.weight).all()


def test_sb_fixed_residue_is_preserved():
    class ZeroVelocity(nn.Module):
        def forward(self, R, t, time, pair, single, **kwargs):
            return torch.zeros_like(t), torch.zeros_like(t)

    bridge = SE3SchrodingerBridge(ZeroVelocity(), diffusion=0.01, sinkhorn_iters=2)
    R0 = torch.eye(3).view(1, 1, 3, 3).expand(1, 2, -1, -1).clone()
    t0 = torch.zeros(1, 2, 3)
    pair = torch.zeros(1, 2, 2, 1)
    single = torch.zeros(1, 2, 1)
    fixed = torch.tensor([[True, False]])
    R, t = bridge.sample(R0, t0, pair, single, n_steps=2, fixed_mask=fixed)
    assert torch.allclose(R[:, 0], R0[:, 0])
    assert torch.allclose(t[:, 0], t0[:, 0])
