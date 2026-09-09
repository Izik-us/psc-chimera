import torch

from chimera import CHIMERAv2
from chimera.chimera_v2 import CHIMERAv2 as LegacyCHIMERAv2


def test_public_chimera_is_explicit_canonical_subclass():
    assert CHIMERAv2 is not LegacyCHIMERAv2
    model = CHIMERAv2(
        d_evo_single=32,
        d_evo_pair=16,
        d_se3=32,
        d_pair_out=32,
        d_mpnn=16,
        n_flow_blocks=1,
        n_domains=2,
        n_modules=2,
        n_mc_dropout=2,
    )
    assert getattr(model, "_canonical_components", False) is True


def test_canonical_replacement_preserves_freeze_contract():
    model = CHIMERAv2(
        d_evo_single=32,
        d_evo_pair=16,
        d_se3=32,
        d_pair_out=32,
        d_mpnn=16,
        n_flow_blocks=1,
        n_domains=2,
        n_modules=2,
        n_mc_dropout=2,
    )
    assert all(not p.requires_grad for p in model.flow_model.parameters())
    assert all(not p.requires_grad for p in model.multi_scale_designer.parameters())
    assert all(not p.requires_grad for p in model.pareto_head.parameters())
    assert any(p.requires_grad for p in model._seq_to_repr.parameters())
