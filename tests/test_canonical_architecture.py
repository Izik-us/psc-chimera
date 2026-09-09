from pathlib import Path

import torch

from chimera import CHIMERAv2
from chimera.chimera_v2 import CHIMERAv2 as LegacyCHIMERAv2
from chimera.bayesian import BayesianUncertaintyEstimator


def test_public_chimera_is_explicit_canonical_subclass():
    import chimera
    import chimera.architecture as architecture

    assert CHIMERAv2 is architecture.CanonicalCHIMERAv2
    assert CHIMERAv2 is not LegacyCHIMERAv2
    assert not hasattr(chimera, "runtime_compat")
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


def test_bayesian_chimera_normalization_uses_fixed_scales():
    output = {
        "evol_plausibility": torch.tensor([[0.0, 2.0]]),
        "structural_stability": torch.tensor([[50.0, 100.0]]),
        "expression_efficiency": torch.tensor([[0.2, 0.8]]),
        "substrate_selectivity": torch.tensor([[0.1, 0.9]]),
        "assembly_compat": torch.tensor([[0.3, 0.7]]),
    }
    normalized = BayesianUncertaintyEstimator._normalize_chimera_objectives(output)
    assert normalized.shape == (1, 2, 5)
    assert torch.all((normalized >= 0) & (normalized <= 1))
    assert torch.allclose(
        normalized[0, :, 0],
        torch.sigmoid(output["evol_plausibility"]).squeeze(0),
    )


def test_weight_downloaders_are_atomic():
    root = Path(__file__).resolve().parents[1]
    sh = (root / "scripts" / "download_weights.sh").read_text(encoding="utf-8")
    ps1 = (root / "scripts" / "download_weights.ps1").read_text(encoding="utf-8")
    assert "${out}.part" in sh
    assert "$Output.part" in ps1
    assert 'mv -f "$part" "$out"' in sh
    assert "Move-Item -Force $part $Output" in ps1
