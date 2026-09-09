from pathlib import Path

import torch

from chimera import CHIMERAv2, AutoregressiveSequencePolicy


def test_package_boundary_has_no_runtime_compat_shim():
    root = Path(__file__).resolve().parents[1]
    assert not (root / "chimera" / "runtime_compat.py").exists()
    init_text = (root / "chimera" / "__init__.py").read_text(encoding="utf-8").lower()
    assert "runtime_compat" not in init_text


def test_public_chimera_is_canonical_and_has_causal_policy():
    model = CHIMERAv2(
        d_evo_single=32,
        d_evo_pair=16,
        d_se3=32,
        d_pair_out=32,
        d_mpnn=32,
        n_flow_blocks=1,
        n_domains=1,
        n_modules=1,
        n_mpnn_seqs=1,
        n_mc_dropout=2,
    )
    assert getattr(model, "_canonical_components", False)
    assert isinstance(model.sequence_policy, AutoregressiveSequencePolicy)


def test_autoregressive_policy_has_finite_logprob():
    policy = AutoregressiveSequencePolicy(context_dim=32, vocab_size=20, layers=1, heads=4)
    context = torch.randn(2, 5, 32)
    tokens = torch.randint(0, 20, (2, 5))
    logp = policy.logprob(context, tokens)
    assert logp.shape == (2,)
    assert torch.isfinite(logp).all()
