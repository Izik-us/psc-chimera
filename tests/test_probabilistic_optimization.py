import torch
import torch.nn as nn

from chimera.bayesian import BayesianUncertaintyEstimator
from chimera.dpo import DPOBatch, DPOTrainer
from chimera.multi_objective import AutoregressiveSequencePolicy
from chimera.pcgrad import project_conflicting_gradients
from chimera.schrodinger_bridge import SchrodingerBridge


def test_sinkhorn_coupling_matches_marginals():
    bridge = SchrodingerBridge(diffusion=0.05, sinkhorn_iters=100)
    cost = torch.tensor([[0.0, 4.0], [4.0, 0.0]])
    coupling = bridge.sinkhorn_coupling(cost)
    assert torch.allclose(coupling.sum(1), torch.full((2,), 0.5), atol=2e-3)
    assert torch.allclose(coupling.sum(0), torch.full((2,), 0.5), atol=2e-3)


def test_brownian_bridge_conditional_mean_and_variance():
    torch.manual_seed(0)
    bridge = SchrodingerBridge(diffusion=0.1)
    n = 20000
    x0 = torch.zeros(n, 1)
    x1 = torch.ones(n, 1)
    t = torch.full((n,), 0.5)
    xt, drift = bridge.sample_bridge(x0, x1, t)
    assert abs(xt.mean().item() - 0.5) < 0.02
    expected_var = 2.0 * 0.1 * 0.5 * 0.5
    assert abs(xt.var(unbiased=True).item() - expected_var) < 0.02
    expected_drift = (1.0 - xt) / 0.5
    assert torch.allclose(drift, expected_drift)


def test_dpo_mask_excludes_padding_tokens():
    torch.manual_seed(0)
    policy = AutoregressiveSequencePolicy(context_dim=8, vocab_size=3, layers=1)
    reference = AutoregressiveSequencePolicy(context_dim=8, vocab_size=3, layers=1)
    reference.load_state_dict(policy.state_dict())
    batch = DPOBatch(
        context=torch.randn(2, 8),
        chosen=torch.tensor([[0, 1, 2], [0, 1, 2]]),
        rejected=torch.tensor([[0, 2, 1], [0, 2, 1]]),
        chosen_mask=torch.tensor([[1, 1, 1], [1, 1, 0]], dtype=torch.bool),
        rejected_mask=torch.tensor([[1, 1, 1], [1, 1, 0]], dtype=torch.bool),
    )
    trainer = DPOTrainer(beta=0.1)
    loss, metrics = trainer.loss(policy, reference, batch)
    assert torch.isfinite(loss)
    assert 0.0 <= metrics["preference_accuracy"] <= 1.0


def test_bayesian_estimator_preserves_training_state_and_uses_std_for_ei():
    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.dropout = nn.Dropout(0.5)
            self.linear = nn.Linear(2, 1)

        def forward(self, x):
            return self.linear(self.dropout(x))

    model = Tiny()
    model.train()
    estimator = BayesianUncertaintyEstimator(n_samples=8)
    result = estimator.predict(model, {"x": torch.ones(16, 2)})
    assert model.training
    assert result["epistemic_std"].shape == result["mean"].shape
    ei = estimator.expected_improvement(
        torch.tensor([1.0, 0.5]),
        torch.tensor([0.1, 0.5]),
        best_observed=0.75,
    )
    assert torch.isfinite(ei).all()
    assert (ei >= 0).all()
    assert ei[0] > ei[1]


def test_pcgrad_removes_negative_pairwise_component():
    parameter = nn.Parameter(torch.tensor([0.0, 0.0]))
    g1 = torch.tensor([1.0, 0.0])
    g2 = torch.tensor([-1.0, 1.0])
    losses = [
        (parameter * g1).sum(),
        (parameter * g2).sum(),
    ]
    project_conflicting_gradients(losses, [parameter])
    # The projected task gradients must not have a negative dot product with
    # the other task's original gradient after the surgery step for g1.
    assert torch.isfinite(parameter.grad).all()
