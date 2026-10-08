"""Independent fold validation: designed sequence -> folder -> compare to intended backbone.

The validator hands the folder *only* the sequence (``FoldingBackend.predict``
has no structural parameter). It then asks six separable questions and keeps
them separate in the report:

1. Does the fold match the design?         (TM, GDT, lDDT, contacts, frames, torsions)
2. Does the folder believe its own answer? (pLDDT, pTM -> calibrated, conformal lower bound)
3. Is the fold reproducible?               (multi-seed / multi-backend consensus)
4. Is the fold physically sane?            (``chimera.geometry.validate_backbone``)
5. Is the match better than chance?        (sequence-shuffle null, optional)
6. Is the multi-domain arrangement kept?   (per-domain TM + inter-domain pose error)

Default thresholds are *unvalidated engineering defaults*; they are recorded in
every report. With ``require_calibration=True`` (default) no candidate can be
``ACCEPT``ed until a ``ConfidenceCalibrator`` fitted on cluster-held-out data is
supplied -- consistent with the repo's fail-closed production posture.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Sequence

import torch

from chimera.errors import InputValidationError
from ..folding.ensemble import FoldEnsemble
from ..folding.frames import backbone_to_frames
from ..folding.types import PAD_IDX, FoldingBackend
from chimera.geometry import validate_backbone
from .calibration import ConfidenceCalibrator
from .metrics import (
    contact_f1,
    domain_arrangement,
    frame_geodesic_error,
    lddt_ca,
    rmsd,
    tm_score,
    torsion_agreement,
)

VALIDATOR_VERSION = "fold-validator-v1"


class Decision(str, Enum):
    ACCEPT = "accept"
    PENALIZE = "penalize"
    REJECT = "reject"


@dataclass(frozen=True)
class FoldValidationConfig:
    tm_accept: float = 0.80
    tm_reject: float = 0.50
    plddt_accept: float = 70.0
    plddt_reject: float = 50.0
    ensemble_min: float = 0.85
    conformal_floor: float = 0.50  # lower bound on realised TM of the fold itself
    domain_tm_min: float = 0.70
    arrangement_rot_max_deg: float = 20.0
    arrangement_trans_max_A: float = 6.0
    null_z_min: float = 3.0
    require_calibration: bool = True
    require_ensemble: bool = False
    score_weights: dict = field(
        default_factory=lambda: {"tm": 3.0, "plddt": 1.0, "ensemble": 1.0, "lower": 1.0, "torsion": 0.5}
    )
    thresholds_provenance: str = "engineering_default_unvalidated"


@dataclass
class FoldValidationReport:
    decision: Decision
    structural_score: float  # 0-100, non-compensatory (weighted geometric mean)
    reasons: list[str]
    tm_design_fold: float
    gdt_ts: float
    lddt: float
    ca_rmsd: float
    contact_f1: float
    torsion_frac: float
    frame_median_deg: float
    mean_plddt: float
    ptm: float | None
    expected_tm: float | None
    conformal_lower_tm: float
    calibrated: bool
    geometry_valid: bool
    clash_count: int
    ensemble_agreement: float | None
    effective_members: int
    null_z: float | None = None
    null_p: float | None = None
    domain: dict | None = None
    reference: dict | None = None  # three-way triangle when an experimental structure is supplied
    provenance: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        d = asdict(self)
        d["decision"] = self.decision.value
        return d


def _geo_check(coords: torch.Tensor):
    g = validate_backbone(coords.unsqueeze(0)).as_dict()
    return bool(g["valid"]), int(g.get("clash_count", 0))


class FoldValidator:
    def __init__(
        self,
        backends: Sequence[FoldingBackend],
        *,
        calibrator: ConfidenceCalibrator | None = None,
        config: FoldValidationConfig | None = None,
        seeds: Sequence[int] = (0, 1, 2),
    ):
        self.ensemble = FoldEnsemble(backends, seeds)
        self.calibrator = calibrator or ConfidenceCalibrator()
        self.cfg = config or FoldValidationConfig()

    def validate(
        self,
        design_coords: torch.Tensor,
        tokens: torch.Tensor,
        mask: torch.Tensor | None = None,
        *,
        domain_spans: Sequence[Sequence[tuple[int, int]]] | None = None,
        reference_coords: torch.Tensor | None = None,
        n_null: int = 0,
        null_seed: int = 0,
    ) -> list[FoldValidationReport]:
        if design_coords.ndim != 4 or design_coords.shape[-2:] != (4, 3):
            raise InputValidationError("design_coords must have shape (B, L, 4, 3)")
        if tokens.shape != design_coords.shape[:2]:
            raise InputValidationError("tokens must have shape (B, L) matching design_coords")
        if mask is None:
            mask = tokens != PAD_IDX
        if bool(((tokens < 0) | ((tokens > 19) & mask)).any()):
            raise InputValidationError("sequence tokens must be amino-acid ids in [0, 19]")
        cfg = self.cfg
        ens = self.ensemble.fold(tokens, mask)  # sequence-only: firewall enforced by the interface
        fold = ens.medoid
        d_ca, f_ca = design_coords[:, :, 1], fold.ca
        tm = tm_score(f_ca, d_ca, mask)
        _, lddt = lddt_ca(f_ca, d_ca, mask)
        rms = rmsd(f_ca, d_ca, mask)
        cf1 = contact_f1(f_ca, d_ca, mask)
        tors, _ = torsion_agreement(design_coords, fold.coords, mask)
        fr = frame_geodesic_error(design_coords, fold.coords, tm.R)
        null_tm = self._null(tokens, mask, d_ca, n_null, null_seed) if n_null > 0 else None
        ref = self._reference(design_coords, fold, reference_coords, mask) if reference_coords is not None else None

        reports = []
        for b in range(tokens.shape[0]):
            Lb = int(mask[b].sum())
            valid = mask[b]
            plddt = float(fold.mean_plddt[b])
            ptm = None if fold.ptm is None else float(fold.ptm[b])
            raw = ptm if ptm is not None else plddt / 100.0
            exp_tm, lower = self.calibrator.predict(raw, Lb)
            geo_ok, clashes = _geo_check(fold.coords[b][valid])
            agree = float(ens.agreement[b])
            agree = None if math.isnan(agree) else agree
            dom = None
            if domain_spans is not None and domain_spans[b]:
                dom = domain_arrangement(d_ca[b][valid], f_ca[b][valid], domain_spans[b])
            nz = npv = None
            if null_tm is not None:
                mu, sd = null_tm[b].mean(), null_tm[b].std(unbiased=True).clamp_min(1e-3)
                nz = float((tm.tm[b] - mu) / sd)
                npv = float((1 + (null_tm[b] >= tm.tm[b]).sum()) / (1 + null_tm.shape[1]))
            r = FoldValidationReport(
                decision=Decision.REJECT, structural_score=0.0, reasons=[],
                tm_design_fold=float(tm.tm[b]), gdt_ts=float(tm.gdt_ts[b]), lddt=float(lddt[b]),
                ca_rmsd=float(rms[b]), contact_f1=float(cf1[b]), torsion_frac=float(tors[b]),
                frame_median_deg=float(torch.rad2deg(fr[b][valid].median())),
                mean_plddt=plddt, ptm=ptm, expected_tm=None if math.isnan(exp_tm) else exp_tm,
                conformal_lower_tm=lower, calibrated=self.calibrator.calibrated,
                geometry_valid=geo_ok, clash_count=clashes, ensemble_agreement=agree,
                effective_members=int(ens.effective_members[b]), null_z=nz, null_p=npv, domain=dom,
                reference=None if ref is None else ref[b],
                provenance={
                    "validator": VALIDATOR_VERSION, "backend": fold.backend,
                    "independence_class": fold.independence_class, "members": fold.provenance.get("members"),
                    "calibration_sha256": self.calibrator.state.sha256() if self.calibrator.calibrated else None,
                    "thresholds": cfg.thresholds_provenance,
                },
            )
            self._decide(r)
            reports.append(r)
        return reports

    # ------------------------------------------------------------------
    def _null(self, tokens, mask, d_ca, n_null, seed):
        """TM(design, fold(shuffled sequence)): composition-preserving chance level."""
        g = torch.Generator().manual_seed(seed)
        B, L = tokens.shape
        backend = self.ensemble.backends[0]
        out = torch.zeros(B, n_null)
        for k in range(n_null):
            sh = tokens.clone()
            for b in range(B):
                idx = mask[b].nonzero(as_tuple=True)[0]
                sh[b, idx] = tokens[b, idx[torch.randperm(len(idx), generator=g)]]
            p = backend.predict(sh, mask, seed=0)
            out[:, k] = tm_score(p.ca, d_ca, mask).tm
        return out

    def _reference(self, design, fold, ref, mask):
        """Triangle: design<->fold, fold<->experimental, design<->experimental (TM normalised to reference)."""
        r_ca = ref[:, :, 1]
        t_fr = tm_score(fold.ca, r_ca, mask).tm
        t_dr = tm_score(design[:, :, 1], r_ca, mask).tm
        _, l_fr = lddt_ca(fold.ca, r_ca, mask)
        return [
            {"tm_fold_ref": float(t_fr[b]), "tm_design_ref": float(t_dr[b]), "lddt_fold_ref": float(l_fr[b])}
            for b in range(design.shape[0])
        ]

    def _decide(self, r: FoldValidationReport) -> None:
        c, why = self.cfg, r.reasons
        hard = False
        if not r.geometry_valid:
            why.append("predicted fold fails backbone geometry validation"), (hard := True)
        if r.tm_design_fold < c.tm_reject:
            why.append(f"design-fold TM {r.tm_design_fold:.2f} < {c.tm_reject}: different fold"), (hard := True)
        if r.mean_plddt < c.plddt_reject:
            why.append(f"mean pLDDT {r.mean_plddt:.1f} < {c.plddt_reject}: folder does not support a fold"), (hard := True)
        soft = []
        if r.tm_design_fold < c.tm_accept:
            soft.append(f"design-fold TM {r.tm_design_fold:.2f} < {c.tm_accept}")
        if r.mean_plddt < c.plddt_accept:
            soft.append(f"mean pLDDT {r.mean_plddt:.1f} < {c.plddt_accept}")
        if r.ensemble_agreement is None:
            if c.require_ensemble:
                soft.append("no independent ensemble members (deterministic or single backend)")
        elif r.ensemble_agreement < c.ensemble_min:
            soft.append(f"ensemble agreement {r.ensemble_agreement:.2f} < {c.ensemble_min}")
        if r.calibrated:
            if r.conformal_lower_tm < c.conformal_floor:
                soft.append(f"conformal lower-bound TM {r.conformal_lower_tm:.2f} < {c.conformal_floor}")
        if r.domain:
            if r.domain["min_domain_tm"] < c.domain_tm_min:
                soft.append(f"min domain TM {r.domain['min_domain_tm']:.2f} < {c.domain_tm_min}")
            if r.domain["max_arrangement_rotation_deg"] > c.arrangement_rot_max_deg:
                soft.append("inter-domain rotation exceeds limit")
            if r.domain["max_arrangement_translation_A"] > c.arrangement_trans_max_A:
                soft.append("inter-domain translation exceeds limit")
        if r.null_z is not None and r.null_z < c.null_z_min:
            soft.append(f"match not distinguishable from shuffled-sequence null (z={r.null_z:.1f})")
        r.structural_score = 0.0 if hard else self._score(r)
        if hard:
            r.decision = Decision.REJECT
        elif soft:
            r.decision, r.reasons = Decision.PENALIZE, why + soft
        elif c.require_calibration and not r.calibrated:
            r.decision = Decision.PENALIZE
            r.reasons = why + ["all checks pass but confidence is uncalibrated: accept withheld (fail-closed)"]
        else:
            r.decision = Decision.ACCEPT

    def _score(self, r: FoldValidationReport) -> float:
        w = self.cfg.score_weights
        terms = {
            "tm": r.tm_design_fold,
            "plddt": r.mean_plddt / 100.0,
            "torsion": r.torsion_frac,
        }
        if r.ensemble_agreement is not None:
            terms["ensemble"] = r.ensemble_agreement
        if r.calibrated and math.isfinite(r.conformal_lower_tm):
            terms["lower"] = max(r.conformal_lower_tm, 0.0)
        num = sum(w[k] * math.log(max(min(v, 1.0), 1e-3)) for k, v in terms.items())
        return float(100.0 * math.exp(num / sum(w[k] for k in terms)))


def structural_objective_labels(reports: Sequence[FoldValidationReport]) -> tuple[torch.Tensor, dict]:
    """Per-sequence scores on the schema's 0-100 scale for ``structural_stability``.

    The label kind is only ``deterministic_evaluator`` when every report used a
    calibrated confidence; otherwise it stays ``proxy`` so it cannot be mistaken
    for validated evidence downstream.
    """
    scores = torch.tensor([r.structural_score for r in reports], dtype=torch.float32)
    kind = "deterministic_evaluator" if reports and all(r.calibrated for r in reports) else "proxy"
    return scores, {
        "source": VALIDATOR_VERSION, "label_kind": kind, "biological_measurement": False,
        "decisions": [r.decision.value for r in reports],
    }


def pareto_funnel(cheap_scores: torch.Tensor, budget: int) -> torch.Tensor:
    """Indices of ``budget`` candidates chosen by non-dominated rank (higher is better) for folding.

    ``cheap_scores``: ``(N, K)``. Ties within a front are broken by crowding distance.
    """
    N = cheap_scores.shape[0]
    remaining, chosen = list(range(N)), []
    while remaining and len(chosen) < budget:
        S = cheap_scores[remaining]
        dom = ((S[:, None] >= S[None]).all(-1) & (S[:, None] > S[None]).any(-1)).any(0)  # dominated by someone
        front = [remaining[i] for i in range(len(remaining)) if not dom[i]]
        if len(chosen) + len(front) > budget:
            F_ = cheap_scores[front]
            crowd = torch.zeros(len(front))
            for k in range(F_.shape[1]):
                o = F_[:, k].argsort()
                span = (F_[o[-1], k] - F_[o[0], k]).clamp_min(1e-9)
                crowd[o[0]] = crowd[o[-1]] = float("inf")
                if len(front) > 2:
                    crowd[o[1:-1]] += (F_[o[2:], k] - F_[o[:-2], k]) / span
            front = [front[i] for i in crowd.argsort(descending=True)[: budget - len(chosen)]]
        chosen += front
        remaining = [i for i in remaining if i not in set(front)]
    return torch.tensor(chosen, dtype=torch.long)
