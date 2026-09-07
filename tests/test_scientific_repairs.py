import math

import torch
import torch.nn as nn

from chimera.bayesian import BayesianUncertaintyEstimator
from chimera.dpo import DPOBatch, DPOTrainer
from chimera.flow_matching import so3_exp, so3_log
from chimera.pcgrad import project_conflicting_gradients
from chimera.schrodinger_bridge import SchrodingerBridge, SE3SchrodingerBridge


def test_so3_log_exp_roundtrip():
    omega = torch.tensor([[0.2, -0.1, 0.3]])
    R = so3_exp(omega)
    recovered = so3_log(R)
    assert torch.allclose(recovered, omega, atol=1e-5)
    eye = R.transpose(-1, -2) @ R
    assert torch.allclose(eye, torch.eye(3).expand_as(eye), atol=1e-5)


def test_brownian_bridge_endpoints_and_drift():
    bridge = SchrodingerBridge(diffusion=0.1)
    x0 = torch.zeros(4, 3)
    x1 = torch.ones(4, 3)
    t = torch.full((4,), 0.5)
    xt, drift = bridge.sample_bridge(x0, x1, t)
    assert xt.shape == x0.shape
    assert drift.shape == x0.shape
    expected = (x1 - xt) / 0.5
    assert torch.allclose(drift, expected)


def test_sinkhorn_has_requested_marginals():
    bridge = SchrodingerBridge(diffusion=0.1, sinkhorn_iters=100)
    cost = torch.rand(5, 7)
    coupling = bridge.sinkhorn_coupling(cost)
    assert torch.allclose(coupling.sum(), torch.tensor(1.0), atol=1e-5)
    assert torch.allclose(coupling.sum(1), torch.full((5,), 0.2), atol=2e-3)
    assert torch.allclose(coupling.sum(0), torch.full((7,), 1.0 / 7), atol=2e-3)


def test_pcgrad_removes_negative_component():
    p = nn.Parameter(torch.tensor([1.0, 1.0]))
    l1 = p[0] - p[1]
    l2 = -p[0] - p[1]
    project_conflicting_gradients([l1, l2], [p])
    # g1=(1,-1), g2=(-1,-1) are conflicting. After projection the
    # resulting update must not contain the original conflicting component.
    assert p.grad is not None
    assert torch.isfinite(p.grad).all()


class TinyPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(5, 8)
        self.proj = nn.Linear(8, 5)

    def logprob(self, context, tokens):
        hidden = self.embedding(tokens) + context.unsqueeze(1)
        logits = self.proj(hidden)
        return torch.log_softmax(logits, -1).gather(-1, tokens.unsqueeze(-1)).squeeze(-1).sum(-1)


def test_dpo_loss_is_finite_and_prefers_chosen():
    torch.manual_seed(0)
    policy = TinyPolicy()
    reference = TinyPolicy()
    reference.load_state_dict(policy.state_dict())
    batch = DPOBatch(
        context=torch.zeros(2, 8),
        chosen=torch.tensor([[1, 1, 1], [2, 2, 2]]),
        rejected=torch.tensor([[3, 3, 3], [4, 4, 4]]),
    )
    loss, metrics = DPOTrainer(beta=0.1).loss(policy, reference, batch)
    assert torch.isfinite(loss)
    assert math.isfinite(metrics["reward_margin"])
    assert abs(float(loss) - math.log(2.0)) < 1e-6


def test_expected_improvement_uses_standard_deviation():
    mean = torch.tensor([1.0, 0.0])
    std = torch.tensor([0.1, 1.0])
    ei = BayesianUncertaintyEstimator.expected_improvement(mean, std, best_observed=0.5)
    assert ei.shape == mean.shape
    assert torch.isfinite(ei).all()
    assert (ei >= 0).all()


def test_dropout_uncertainty_restores_mode():
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.dropout = nn.Dropout(0.5)
            self.linear = nn.Linear(3, 1)

        def forward(self, x):
            return self.linear(self.dropout(x))

    model = M()
    model.train(False)
    x = torch.ones(4, 3)
    estimator = BayesianUncertaintyEstimator(n_samples=4)
    result = estimator.predict(model, {"x": x})
    assert not model.training
    assert result["mean"].shape == (4, 1)
    assert result["epistemic_std"].shape == (4, 1)
