"""Regression tests for the merge-readiness scientific contracts."""

import copy

import pytest
import torch
import torch.nn as nn

from chimera.bayesian import BayesianUncertaintyEstimator
from chimera.dpo import DPOBatch, DPOTrainer
from chimera.pcgrad import PCGradOptimizer
from chimera.schrodinger_bridge import SE3SchrodingerBridge
from chimera.pareto_pcgrad import MergeReadyParetoMultiObjectiveHead
from chimera.multi_objective import ParetoObjectives, ParetoMultiObjectiveHead


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
    base_optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
    optimizer = PCGradOptimizer(base_optimizer)
    x = torch.tensor([[1.0, 0.0, 0.0]])
    y1 = model(x).sum()
    y2 = -model(x).sum()
    optimizer.zero_grad()
    optimizer.step([y1, y2])
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


def test_sb_euler_maruyama_noise_variance_scales_with_diffusion_time():
    class ZeroVelocity(nn.Module):
        def forward(self, R, t, time, pair, single, **kwargs):
            return torch.zeros_like(t), torch.zeros_like(t)

    diffusion = 0.01
    sample_count = 4096
    steps = 8
    bridge = SE3SchrodingerBridge(
        ZeroVelocity(),
        diffusion=diffusion,
        sinkhorn_iters=2,
    )
    rotations = torch.eye(3).view(1, 1, 3, 3).expand(sample_count, 1, -1, -1).clone()
    translations = torch.zeros(sample_count, 1, 3)
    pair = torch.zeros(sample_count, 1, 1, 1)
    single = torch.zeros(sample_count, 1, 1)

    sampled_rotations, sampled_translations = bridge.sample(
        rotations,
        translations,
        pair,
        single,
        n_steps=steps,
        generator=torch.Generator().manual_seed(19),
    )

    translation_variance = sampled_translations[:, 0].var(dim=0, unbiased=False).mean()
    rotation_vectors = torch.linalg.vector_norm(
        torch.stack(
            (
                sampled_rotations[:, 0, 2, 1] - sampled_rotations[:, 0, 1, 2],
                sampled_rotations[:, 0, 0, 2] - sampled_rotations[:, 0, 2, 0],
                sampled_rotations[:, 0, 1, 0] - sampled_rotations[:, 0, 0, 1],
            ),
            dim=-1,
        ) * 0.5,
        dim=-1,
    )
    rotation_variance = rotation_vectors.square().mean()

    expected_component_variance = 2.0 * diffusion
    assert abs(float(translation_variance) - expected_component_variance) < 0.0015
    assert abs(float(rotation_variance) - 3.0 * expected_component_variance) < 0.004


def test_canonical_pareto_head_performs_gradient_surgery():
    torch.manual_seed(12)
    head = MergeReadyParetoMultiObjectiveHead(d_model=8)
    features = {
        "sequence": torch.randn(2, 8, requires_grad=True),
        "evolutionary": torch.randn(2, 8, requires_grad=True),
        "structural": torch.randn(2, 8, requires_grad=True),
        "substrate": torch.randn(2, 8, requires_grad=True),
    }
    objectives = head(features)
    labels = {
        "evol": torch.ones(2),
        "stab": torch.zeros(2),
        "expr": torch.ones(2),
        "sel": torch.zeros(2),
        "asm": torch.ones(2),
    }
    total, metrics = head.pcgrad_loss(objectives, labels)
    total.backward()
    assert torch.isfinite(total)
    assert metrics["pcgrad_task_count"] == 5
    for feature in features.values():
        assert feature.grad is not None and torch.isfinite(feature.grad).all()


def test_legacy_pcgrad_name_is_explicitly_deprecated():
    head = ParetoMultiObjectiveHead(d_model=8)
    objectives = head(torch.randn(2, 3, 8))
    with pytest.warns(DeprecationWarning, match="heuristic, not PCGrad"):
        loss, _ = head.pcgrad_loss(objectives, {"evol": torch.zeros(2)})
    assert torch.isfinite(loss)
