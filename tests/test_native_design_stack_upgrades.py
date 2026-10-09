import torch

from chimera.autoregressive_policy import AutoregressiveSequencePolicy
from chimera.sequence_design import MultiScaleNRPSDesigner


def test_multiscale_designer_handles_empty_spans_padding_and_gradients():
    torch.manual_seed(21)
    model = MultiScaleNRPSDesigner(
        d_residue=16,
        d_domain=16,
        d_module=16,
        d_assembly=16,
        n_domains=3,
        n_modules=2,
    )
    batch, length, neighbors = 2, 8, 3
    residue = torch.randn(batch, length, 16, requires_grad=True)
    evo = torch.randn(batch, length, 16, requires_grad=True)
    edge = torch.randn(batch, length, neighbors, 28)
    edge_index = torch.arange(length).view(1, length, 1).expand(batch, -1, neighbors)
    edge_index = (edge_index + torch.arange(neighbors).view(1, 1, neighbors)) % length
    edge_mask = torch.zeros(batch, length, neighbors, dtype=torch.bool)
    domain_boundaries = torch.tensor(
        [[[0, 2], [2, 2], [4, 8]], [[0, 3], [3, 5], [5, 8]]]
    )
    module_boundaries = torch.tensor(
        [[[0, 4], [4, 8], [0, 0]], [[0, 0], [0, 8], [0, 0]]]
    )
    residue_mask = torch.tensor(
        [[True, True, True, True, True, True, False, False],
         [True, True, True, True, True, True, True, True]]
    )
    faces = torch.tensor([4, 19])

    logits = model(
        residue, evo, edge, edge_index, domain_boundaries, module_boundaries, faces,
        edge_mask=edge_mask, residue_mask=residue_mask,
    )
    assert logits.shape == (batch, length, 20)
    assert torch.isfinite(logits).all()
    assert torch.equal(logits[0, 6:], torch.zeros_like(logits[0, 6:]))
    logits.square().mean().backward()
    assert residue.grad is not None and torch.isfinite(residue.grad).all()
    assert evo.grad is not None and torch.isfinite(evo.grad).all()


def test_policy_teacher_forcing_remains_causal_with_structure_bias():
    torch.manual_seed(22)
    policy = AutoregressiveSequencePolicy(
        context_dim=32, vocab_size=20, layers=2, heads=4
    ).eval()
    context = torch.randn(2, 7, 32)
    tokens = torch.randint(0, 20, (2, 7))
    changed_future = tokens.clone()
    changed_future[:, 5:] = (changed_future[:, 5:] + 3) % 20
    with torch.no_grad():
        logits_a = policy(context, tokens)
        logits_b = policy(context, changed_future)
    assert torch.allclose(logits_a[:, :6], logits_b[:, :6], atol=1e-5, rtol=1e-5)


def test_policy_incremental_kv_logits_match_full_causal_logits():
    torch.manual_seed(23)
    policy = AutoregressiveSequencePolicy(
        context_dim=32, vocab_size=20, layers=2, heads=4
    ).eval()
    context = torch.randn(2, 6, 32)
    tokens = torch.randint(0, 20, (2, 6))
    with torch.no_grad():
        expected = policy._causal_logits(context, tokens)
        projected = policy.context(context)
        cache = [None for _ in policy.decoder.layers]
        bos = torch.full((tokens.shape[0],), policy.vocab_size, dtype=torch.long)
        actual = []
        for position in range(tokens.shape[1]):
            previous = bos if position == 0 else tokens[:, position - 1]
            step_logits, cache = policy._incremental_step(
                projected, previous, position, cache
            )
            actual.append(step_logits)
        actual = torch.stack(actual, dim=1)
    assert torch.allclose(actual, expected, atol=2e-5, rtol=2e-5)


def test_policy_generation_preserves_fixed_tokens_and_rejects_invalid_constraints():
    torch.manual_seed(24)
    policy = AutoregressiveSequencePolicy(
        context_dim=32, vocab_size=20, layers=1, heads=4
    ).eval()
    context = torch.randn(2, 6, 32)
    fixed = torch.full((2, 6), -1, dtype=torch.long)
    fixed[:, 0] = 3
    fixed[1, 4] = 17
    generated = policy.generate(context, length=6, fixed_tokens=fixed)
    assert torch.equal(generated[:, 0], torch.tensor([3, 3]))
    assert generated[1, 4].item() == 17
    invalid = fixed.clone()
    invalid[0, 3] = 20
    try:
        policy.generate(context, length=6, fixed_tokens=invalid)
    except ValueError as exc:
        assert "valid amino-acid IDs" in str(exc)
    else:
        raise AssertionError("invalid fixed token was not rejected")
