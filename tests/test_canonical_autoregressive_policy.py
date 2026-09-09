import torch

from chimera.autoregressive_policy import AutoregressiveSequencePolicy


def test_causal_policy_exposes_teacher_forced_token_logprobs():
    torch.manual_seed(0)
    policy = AutoregressiveSequencePolicy(context_dim=32, vocab_size=20, layers=1, heads=4)
    context = torch.randn(2, 7, 32)
    tokens = torch.randint(0, 20, (2, 7))
    token_lp = policy.token_logprobs(context, tokens)
    assert token_lp.shape == tokens.shape
    assert torch.isfinite(token_lp).all()


def test_generation_is_left_to_right_and_respects_fixed_tokens():
    torch.manual_seed(1)
    policy = AutoregressiveSequencePolicy(context_dim=32, vocab_size=20, layers=1, heads=4)
    context = torch.randn(2, 6, 32)
    fixed = torch.full((2, 6), -1, dtype=torch.long)
    fixed[:, 0] = 3
    fixed[1, 4] = 17
    generated = policy.generate(context, length=6, temperature=1.0, fixed_tokens=fixed)
    assert generated.shape == (2, 6)
    assert torch.equal(generated[:, 0], torch.tensor([3, 3]))
    assert generated[1, 4].item() == 17
    assert generated.min().item() >= 0 and generated.max().item() < 20
