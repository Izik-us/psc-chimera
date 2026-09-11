import torch
import torch.nn as nn

from chimera.checkpoint import CheckpointManifest, config_hash, state_schema_hash
from chimera.lie import so3_exp, so3_log
from chimera.pcgrad import project_conflicting_gradients


def test_so3_log_exp_roundtrip_near_pi():
    axis = torch.tensor([[0.3, -0.7, 0.64]], dtype=torch.float64)
    axis = axis / axis.norm(dim=-1, keepdim=True)
    omega = axis * (torch.pi - 1e-6)
    recovered = so3_log(so3_exp(omega))
    assert torch.allclose(so3_exp(recovered), so3_exp(omega), atol=1e-7, rtol=1e-7)


def test_pcgrad_generator_is_reproducible():
    p = nn.Parameter(torch.tensor([1.0, 2.0]))
    losses = [p[0] * p[1], -p[0] * p[1]]
    g1 = torch.Generator().manual_seed(123)
    _, conflicts1 = project_conflicting_gradients(losses, [p], retain_graph=True, generator=g1, return_conflicts=True)
    grad1 = p.grad.detach().clone()
    p.grad = None
    g2 = torch.Generator().manual_seed(123)
    _, conflicts2 = project_conflicting_gradients(losses, [p], retain_graph=True, generator=g2, return_conflicts=True)
    grad2 = p.grad.detach().clone()
    assert conflicts1 == conflicts2
    assert torch.equal(grad1, grad2)


def test_checkpoint_fingerprint_is_order_independent():
    a = {"b": torch.zeros(2, 3), "a": torch.ones(4, dtype=torch.float32)}
    b = {"a": torch.ones(4, dtype=torch.float32), "b": torch.zeros(2, 3)}
    assert state_schema_hash(a) == state_schema_hash(b)
    assert config_hash({"layers": 2, "hidden": 128}) == config_hash({"hidden": 128, "layers": 2})


def test_checkpoint_manifest_accepts_v2_defaults():
    manifest = CheckpointManifest()
    manifest.validate()
    assert manifest.format_version == 2
    assert manifest.geometry_edge_dim == 28
    assert manifest.objective_count == 5
