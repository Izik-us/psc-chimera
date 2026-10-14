import math

import pytest
import torch
from torch import nn

from chimera.lie import so3_exp, so3_log
from chimera.schrodinger_bridge import SchrodingerBridge, SE3SchrodingerBridge


@pytest.mark.parametrize(
    "angle",
    [0.0, 1e-8, 1e-5, 1e-3, math.pi - 1e-5, math.pi],
)
def test_so3_exp_log_round_trip_across_numerical_regimes(angle):
    axis = torch.tensor([0.3, -0.8, 0.5], dtype=torch.float64)
    axis = axis / torch.linalg.vector_norm(axis)
    rotation = so3_exp((axis * angle).reshape(1, 3))[0]
    recovered = so3_exp(so3_log(rotation))
    assert torch.isfinite(recovered).all()
    assert torch.allclose(recovered, rotation, atol=2e-6, rtol=2e-6)
    identity = torch.eye(3, dtype=rotation.dtype)
    assert torch.allclose(rotation.T @ rotation, identity, atol=2e-6, rtol=2e-6)
    assert torch.allclose(
        torch.linalg.det(rotation),
        torch.tensor(1.0, dtype=rotation.dtype),
        atol=2e-6,
    )


def test_sinkhorn_coupling_stays_finite_for_large_endpoint_costs():
    bridge = SchrodingerBridge(diffusion=0.05, sinkhorn_iters=100)
    cost = torch.tensor(
        [[1.0e4, 1.0e4 + 3.0], [1.0e4 + 2.0, 1.0e4]],
        dtype=torch.float64,
    )
    coupling = bridge.sinkhorn_coupling(cost)
    assert torch.isfinite(coupling).all()
    assert torch.all(coupling >= 0)
    assert torch.allclose(
        coupling.sum(), torch.tensor(1.0, dtype=coupling.dtype), atol=1e-8
    )
    assert torch.allclose(
        coupling.sum(0), torch.full((2,), 0.5, dtype=coupling.dtype), atol=2e-3
    )
    assert torch.allclose(
        coupling.sum(1), torch.full((2,), 0.5, dtype=coupling.dtype), atol=2e-3
    )


class _ZeroDrift(nn.Module):
    def forward(
        self,
        rotations,
        translations,
        time,
        pair_cond,
        evol_single,
        source_rotations,
        source_translations,
        substrate_coords,
        evol_conditioning_fn,
    ):
        del rotations, time, pair_cond, evol_single, source_rotations
        del source_translations, substrate_coords, evol_conditioning_fn
        return torch.zeros_like(translations), torch.zeros_like(translations)


def test_sampler_preserves_fixed_frames_and_valid_so3_over_long_trajectory():
    bridge = SE3SchrodingerBridge(_ZeroDrift(), diffusion=0.01)
    source_rotations = so3_exp(torch.randn(1, 6, 3) * 0.2)
    source_translations = torch.randn(1, 6, 3)
    fixed_mask = torch.tensor([[True, False, False, False, False, True]])
    args = (
        source_rotations,
        source_translations,
        torch.zeros(1, 6, 8),
        torch.zeros(1, 6, 8),
    )

    generator = torch.Generator().manual_seed(101)
    rotations, translations = bridge.sample(
        *args,
        n_steps=200,
        fixed_mask=fixed_mask,
        generator=generator,
    )

    assert torch.equal(rotations[fixed_mask], source_rotations[fixed_mask])
    assert torch.equal(translations[fixed_mask], source_translations[fixed_mask])
    gram = rotations.transpose(-1, -2) @ rotations
    identity = torch.eye(3).expand_as(gram)
    assert torch.allclose(gram, identity, atol=2e-5, rtol=2e-5)
    assert torch.allclose(
        torch.linalg.det(rotations), torch.ones(1, 6), atol=2e-5, rtol=2e-5
    )
    assert torch.isfinite(translations).all()


def test_sampler_is_reproducible_with_explicit_local_generator():
    bridge = SE3SchrodingerBridge(_ZeroDrift(), diffusion=0.01)
    rotations = so3_exp(torch.zeros(1, 4, 3))
    translations = torch.zeros(1, 4, 3)
    args = (rotations, translations, torch.zeros(1, 4, 8), torch.zeros(1, 4, 8))
    first = bridge.sample(
        *args, n_steps=12, generator=torch.Generator().manual_seed(9)
    )
    second = bridge.sample(
        *args, n_steps=12, generator=torch.Generator().manual_seed(9)
    )
    assert torch.equal(first[0], second[0])
    assert torch.equal(first[1], second[1])
