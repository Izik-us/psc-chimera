import torch

from chimera.autoregressive_policy import AutoregressiveSequencePolicy


def test_policy_teacher_forcing_is_causal():
    torch.manual_seed(19)
    policy = AutoregressiveSequencePolicy(context_dim=32, layers=2, heads=4).eval()
    context = torch.randn(2, 7, 32)
    tokens = torch.randint(0, 20, (2, 7))
    changed_future = tokens.clone()
    changed_future[:, 4:] = (changed_future[:, 4:] + 1) % 20

    with torch.no_grad():
        logits = policy(context, tokens)
        changed_logits = policy(context, changed_future)

    torch.testing.assert_close(logits[:, :4], changed_logits[:, :4])


def test_policy_incremental_cache_matches_full_causal_logits():
    torch.manual_seed(23)
    policy = AutoregressiveSequencePolicy(context_dim=32, layers=2, heads=4).eval()
    context = torch.randn(2, 6, 32)
    tokens = torch.randint(0, 20, (2, 6))

    with torch.no_grad():
        full_logits = policy(context, tokens)
        projected = policy.context(context)
        caches = [None for _ in policy.decoder.layers]
        bos = torch.full((tokens.shape[0],), policy.vocab_size, dtype=torch.long)
        incremental = []
        for position in range(tokens.shape[1]):
            previous = bos if position == 0 else tokens[:, position - 1]
            step_logits, caches = policy._incremental_step(
                projected, previous, position, caches
            )
            incremental.append(step_logits)

    incremental_logits = torch.stack(incremental, dim=1)
    torch.testing.assert_close(
        incremental_logits, full_logits, rtol=2e-5, atol=2e-5
    )


def test_policy_fixed_tokens_are_preserved_during_generation():
    torch.manual_seed(29)
    policy = AutoregressiveSequencePolicy(context_dim=32, layers=2, heads=4).eval()
    context = torch.randn(2, 5, 32)
    fixed = torch.full((2, 5), -1, dtype=torch.long)
    fixed[0, 1] = 3
    fixed[1, 4] = 17

    generated = policy.generate(context, length=5, fixed_tokens=fixed)

    assert generated[0, 1].item() == 3
    assert generated[1, 4].item() == 17
