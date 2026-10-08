"""Plumbing tests for the opt-in fold validation inside ``CHIMERAv2.design``.

The folder here is a test double returning a fixed ideal backbone; these tests check selection,
column replacement and fail-closed behaviour, NOT folding accuracy.
"""
import math

import pytest
import torch

from chimera import CHIMERAv2
from chimera.folding.frames import ideal_backbone_from_torsions
from chimera.folding.types import FoldingBackend, FoldPrediction
from chimera.validation import FoldValidationConfig, FoldValidator

L = 8


class Const(FoldingBackend):
    name, independence_class = "const", "test_double"

    def __init__(self, phi, psi):
        self.phi, self.psi = phi, psi

    def predict(self, tokens, mask=None, *, seed=0, msa_tokens=None):
        B, Lx = tokens.shape
        c = ideal_backbone_from_torsions(torch.full((B, Lx), self.phi), torch.full((B, Lx), self.psi))
        c = c + 0.02 * torch.randn(c.shape, generator=torch.Generator().manual_seed(seed))
        return FoldPrediction(tokens, torch.ones(B, Lx, dtype=torch.bool), c, torch.full((B, Lx), 80.0),
                              ptm=torch.full((B,), 0.7), backend="const", independence_class="test_double", seed=seed)


def run(validator=None, **kw):
    torch.manual_seed(0)
    model = CHIMERAv2(d_evo_single=32, d_evo_pair=16, d_se3=32, d_pair_out=32, d_mpnn=16, n_flow_blocks=1,
                      n_flow_steps=2, n_mpnn_seqs=1, n_domains=2, n_modules=2, n_mc_dropout=2)
    R = torch.eye(3).reshape(1, 1, 3, 3).expand(1, L, -1, -1).clone()
    t = torch.arange(L).reshape(1, L, 1).float() * torch.tensor([3.8, 0.0, 0.0])
    return model.design(nrps_msa=torch.randint(0, 20, (1, 2, L)), source_backbone=(R, t),
                        initial_pair_features=torch.zeros(1, L, L, 16), n_designs=6, n_pareto_samples=3,
                        flow_steps=2, experimental=True, fold_validator=validator, **kw)


def permissive():
    return FoldValidationConfig(tm_reject=0.0, plddt_reject=0.0, tm_accept=2.0)  # nothing ACCEPTs, nothing hard-rejects


def test_default_path_is_unchanged():
    r = run()
    assert r["fold_validation"] is None and r["ranking_source"] == "deterministic_proxy_evaluator"
    assert r["objective_names"][1] == "backbone_sanity_validity"


def test_fold_validation_replaces_structural_column_and_labels_uncalibrated_as_proxy():
    r = run(FoldValidator([Const(-math.radians(57), -math.radians(47))], config=permissive()), fold_budget=4)
    fv = r["fold_validation"]
    assert fv["n_folded"] == 4 and fv["label_kind"] == "proxy" and fv["biological_measurement"] is False
    assert "UNCALIBRATED" in fv["status"]
    assert r["ranking_source"].startswith("fold_validated[proxy]+")
    assert r["objective_names"][1] == "fold_validated_structural_score"
    assert 1 <= r["pareto_sequences"].shape[0] <= 3
    assert r["pareto_scores"].shape == (r["pareto_sequences"].shape[0], 5)
    assert all(rep["provenance"]["independence_class"] == "test_double" for rep in fv["reports"])


def test_fails_closed_when_every_fold_is_rejected_or_accept_is_required_but_uncalibrated():
    r = run(FoldValidator([Const(math.radians(-120), math.radians(130))]), fold_budget=4)  # wrong fold type
    assert r["pareto_sequences"].shape[0] == 0 and "no candidate survived" in r["fold_validation"]["status"]
    r2 = run(FoldValidator([Const(-math.radians(57), -math.radians(47))], config=permissive()),
             fold_budget=4, fold_require_accept=True)
    assert r2["pareto_sequences"].shape[0] == 0 and "ACCEPT" in r2["fold_validation"]["status"]
