import inspect
import math

import numpy as np
import pytest
import torch

from chimera.errors import ConfigurationError, InputValidationError
from chimera.folding.adapters import ExternalFoldingBackend, parse_backbone_pdb
from chimera.folding.frames import (
    backbone_to_frames, carbonyl_angle_target, frames_to_backbone, ideal_backbone_from_torsions, so3_exp_safe,
)
from chimera.folding.losses import fape, folding_loss
from chimera.folding.model import CHIMERAFold, CHIMERAFoldBackend, FoldConfig
from chimera.folding.types import FoldingBackend, FoldPrediction
from chimera.geometry import validate_backbone
from chimera.lie import so3_exp
from chimera.validation import ConfidenceCalibrator, Decision, FoldValidator, pareto_funnel, structural_objective_labels
from chimera.validation.metrics import (
    contact_f1, domain_arrangement, frechet_mean_so3, kabsch, lddt_ca, rmsd, tm_score, torsion_agreement,
)

TINY = FoldConfig(d_single=32, d_pair=16, n_pairformer=1, n_structure_layers=2, n_heads=4, c_tri_hidden=8, n_recycle=0)


def helix(B=2, L=40):
    return ideal_backbone_from_torsions(torch.full((B, L), math.radians(-57)), torch.full((B, L), math.radians(-47)))


def strand(B=2, L=40):
    return ideal_backbone_from_torsions(torch.full((B, L), math.radians(-120)), torch.full((B, L), math.radians(130)))


def rigid(x, seed=0):
    g = torch.Generator().manual_seed(seed)
    R = so3_exp(torch.randn(x.shape[0], 3, generator=g))
    return torch.einsum("bij,blaj->blai", R, x) + torch.randn(x.shape[0], 1, 1, 3, generator=g) * 10


class Fake(FoldingBackend):
    name, independence_class = "fake", "test_double"

    def __init__(self, target, noise=0.05):
        self.t, self.n, self.seen = target, noise, []

    def predict(self, tokens, mask=None, *, seed=0, msa_tokens=None):
        self.seen.append((tokens, mask, msa_tokens))
        g = torch.Generator().manual_seed(seed)
        c = self.t + self.n * torch.randn(self.t.shape, generator=g)
        return FoldPrediction(tokens, torch.ones(tokens.shape, dtype=torch.bool), c, torch.full(tokens.shape, 85.0),
                              ptm=torch.full((tokens.shape[0],), 0.8), backend="fake",
                              independence_class="test_double", seed=seed)


# ---- geometry / frames -------------------------------------------------
def test_ideal_backbone_is_valid_and_frames_roundtrip():
    c = helix()
    assert validate_backbone(c).as_dict()["valid"]
    R, t = backbone_to_frames(c)
    back = frames_to_backbone(R, t, carbonyl_angle_target(c))
    assert (back - c).abs().max() < 1e-3


def test_so3_exp_safe_matches_repo_and_has_finite_gradient_at_zero():
    w = torch.randn(5, 3)
    assert (so3_exp_safe(w) - so3_exp(w)).abs().max() < 1e-5
    z = torch.zeros(3, 3, requires_grad=True)
    so3_exp_safe(z).sum().backward()
    assert torch.isfinite(z.grad).all()


# ---- metrics ------------------------------------------------------------
def test_metrics_invariant_to_rigid_motion_and_detect_mirror():
    c = helix()
    a, b = c[:, :, 1], rigid(c)[:, :, 1]
    assert rmsd(b, a).max() < 1e-3
    assert (tm_score(b, a).tm > 0.999).all()
    assert (lddt_ca(b, a)[1] > 0.999).all()
    assert (contact_f1(b, a) > 0.999).all()
    assert (tm_score(a * torch.tensor([1.0, 1.0, -1.0]), a).tm < 0.7).all()


def test_kabsch_returns_proper_rotation():
    P = torch.randn(3, 30, 3)
    R, _ = kabsch(P * torch.tensor([1.0, 1.0, -1.0]), P)  # reflection must not yield det -1
    assert torch.allclose(torch.det(R), torch.ones(3), atol=1e-4)


def test_helix_vs_strand_is_different_fold():
    assert (tm_score(strand()[:, :, 1], helix()[:, :, 1]).tm < 0.3).all()
    assert (torsion_agreement(helix(), strand())[0] < 0.1).all()


def test_domain_arrangement_detects_hinge():
    d = helix(1, 80)[0, :, 1]
    f = d.clone()
    Rr = so3_exp(torch.tensor([0.0, 0.0, 0.8]))
    f[40:] = (f[40:] - f[40:].mean(0)) @ Rr.T + f[40:].mean(0)
    same = domain_arrangement(d, d, [(0, 40), (40, 80)])
    hinged = domain_arrangement(d, f, [(0, 40), (40, 80)])
    assert same["max_arrangement_rotation_deg"] < 1e-2 and min(same["domain_tm"]) > 0.99
    assert hinged["max_arrangement_rotation_deg"] > 20 and min(hinged["domain_tm"]) > 0.9


def test_karcher_mean_recovers_center():
    mu0 = so3_exp(torch.tensor([0.3, -0.2, 0.5]))
    rots = torch.stack([mu0 @ so3_exp(0.1 * torch.randn(3)) for _ in range(200)])
    mu, disp = frechet_mean_so3(rots)
    assert torch.linalg.matrix_norm(mu - mu0) < 0.05 and disp < 0.05


# ---- losses / model ------------------------------------------------------
def test_fape_is_invariant_to_global_rigid_transform_of_target():
    c = helix(1, 20)
    R, t = backbone_to_frames(c)
    R2, t2 = backbone_to_frames(rigid(c))
    m = torch.ones(1, 20, dtype=torch.bool)
    assert fape(R, t, R2, t2, m).abs().max() < 2e-3  # eps floor of the sqrt is 1e-3


def test_model_trains_and_backend_fails_closed_without_checkpoint():
    torch.manual_seed(0)
    m = CHIMERAFold(TINY)
    tok, mask, true = torch.randint(0, 20, (2, 14)), torch.ones(2, 14, dtype=torch.bool), helix(2, 14)
    opt = torch.optim.Adam(m.parameters(), 1e-3)
    first = None
    for _ in range(25):
        opt.zero_grad()
        loss, _ = folding_loss(m(tok), true, mask, m.heads)
        first = first or float(loss.detach())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
    assert float(loss) < first and all(torch.isfinite(p).all() for p in m.parameters())
    with pytest.raises(ConfigurationError):
        CHIMERAFoldBackend(m)
    p = CHIMERAFoldBackend(m, allow_untrained=True).predict(tok)
    assert p.independence_class == "chimera_native_untrained" and p.coords.shape == (2, 14, 4, 3)


# ---- calibration --------------------------------------------------------
def _synth(n, rng):
    raw = rng.uniform(0.2, 0.95, n)
    return raw, np.clip(raw ** 1.5 + rng.normal(0, 0.07, n), 0, 1), rng.integers(50, 700, n)


def test_conformal_lower_bound_attains_coverage_and_fails_closed_when_uncalibrated():
    rng = np.random.default_rng(3)
    cal = ConfidenceCalibrator().fit(*_synth(4000, rng), alpha=0.1)
    r, t, L = _synth(5000, rng)
    cov = np.mean([t[i] >= cal.predict(r[i], int(L[i]))[1] for i in range(len(r))])
    assert cov >= 0.88
    assert cal.state.ece_after < cal.state.ece_before
    assert ConfidenceCalibrator.from_json(cal.to_json()).predict(0.8, 300) == cal.predict(0.8, 300)
    assert ConfidenceCalibrator().predict(0.9, 300)[1] == float("-inf")


# ---- validator ----------------------------------------------------------
def test_firewall_signature_and_validator_only_passes_sequence():
    params = set(inspect.signature(FoldingBackend.predict).parameters)
    assert params == {"self", "tokens", "mask", "seed", "msa_tokens"}
    d, tok = helix(), torch.randint(0, 20, (2, 40))
    fake = Fake(d)
    FoldValidator([fake]).validate(d, tok)
    for tokens, mask, msa in fake.seen:
        assert tokens.dtype == torch.long and msa is None
        assert not any(isinstance(x, torch.Tensor) and x.shape[-2:] == (4, 3) for x in (tokens, mask))


def test_matching_fold_is_withheld_until_calibrated_then_accepted():
    d, tok = helix(), torch.randint(0, 20, (2, 40))
    r = FoldValidator([Fake(d)]).validate(d, tok)
    assert all(x.decision is Decision.PENALIZE and x.tm_design_fold > 0.95 for x in r)
    assert structural_objective_labels(r)[1]["label_kind"] == "proxy"
    rng = np.random.default_rng(0)
    raw = rng.uniform(0.3, 0.99, 600)
    cal = ConfidenceCalibrator().fit(raw, np.clip(raw + rng.normal(0, 0.03, 600), 0, 1), np.full(600, 40), alpha=0.1)
    r2 = FoldValidator([Fake(d)], calibrator=cal).validate(d, tok)
    assert all(x.decision is Decision.ACCEPT for x in r2)
    assert structural_objective_labels(r2)[1]["label_kind"] == "deterministic_evaluator"


def test_wrong_fold_rejected_and_bad_inputs_refused():
    d, tok = helix(), torch.randint(0, 20, (2, 40))
    r = FoldValidator([Fake(strand())]).validate(d, tok)
    assert all(x.decision is Decision.REJECT and x.structural_score == 0.0 for x in r)
    with pytest.raises(InputValidationError):
        FoldValidator([Fake(d)]).validate(d, torch.full((2, 40), 25))
    with pytest.raises(InputValidationError):
        FoldValidator([Fake(d)]).validate(d[..., :3, :], tok)


def test_pareto_funnel_returns_front_first():
    s = torch.tensor([[1.0, 1.0], [0.9, 0.9], [0.2, 0.2], [1.0, 0.1], [0.1, 1.0]])
    idx = set(pareto_funnel(s, 3).tolist())
    assert 0 in idx and 2 not in idx


# ---- adapters -----------------------------------------------------------
def test_pdb_parse_roundtrip_and_external_backend_fails_closed(tmp_path):
    c = helix(1, 12)[0]
    lines = []
    for i in range(12):
        for nm, xyz in zip(("N", "CA", "C", "O"), c[i]):
            lines.append(f"ATOM  {len(lines)+1:5d} {nm:<4s} ALA A{i+1:4d}    {xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}  1.00 {0.9:5.2f}           {nm[0]}")
    p = tmp_path / "x.pdb"
    p.write_text("\n".join(lines))
    coords, plddt = parse_backbone_pdb(p)
    assert (coords - c).abs().max() < 2e-3 and abs(float(plddt.mean()) - 90.0) < 1e-3
    be = ExternalFoldingBackend(["/nonexistent/folder", "{fasta}", "{out_pdb}"], name="esmfold")
    with pytest.raises(ConfigurationError):
        be.predict(torch.randint(0, 20, (1, 12)))
    with pytest.raises(ConfigurationError):
        ExternalFoldingBackend(["x"], name="esmfold", checkpoint_path=tmp_path / "missing.pt")
