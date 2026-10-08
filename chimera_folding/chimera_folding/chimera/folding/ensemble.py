"""Multi-seed / multi-backend fold ensembles with SO(3)-aware consensus."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import torch

from ..validation.metrics import apply_rigid, frechet_mean_so3, kabsch, tm_score
from .frames import backbone_to_frames
from .types import FoldingBackend, FoldPrediction


@dataclass
class EnsembleFold:
    members: list[FoldPrediction]
    medoid: FoldPrediction  # per-batch-item medoid, assembled into one prediction
    tm_matrix: torch.Tensor  # (B, K, K) symmetrised TM between members
    effective_members: torch.Tensor  # (B,) number of numerically distinct members
    agreement: torch.Tensor  # (B,) mean off-diagonal TM; NaN if < 2 distinct members
    ca_rmsf: torch.Tensor  # (B, L) per-residue Calpha fluctuation about the medoid
    frame_dispersion: torch.Tensor  # (B, L) Karcher-mean geodesic variance (rad^2)
    meta: dict = field(default_factory=dict)


class FoldEnsemble:
    def __init__(self, backends: Sequence[FoldingBackend], seeds: Sequence[int] = (0, 1, 2), distinct_tol: float = 1e-3):
        if not backends:
            raise ValueError("at least one backend is required")
        self.backends, self.seeds, self.tol = list(backends), tuple(seeds), distinct_tol

    def fold(self, tokens, mask=None) -> EnsembleFold:
        members = [b.predict(tokens, mask, seed=s) for b in self.backends for s in self.seeds]
        B, L = tokens.shape
        K = len(members)
        m = members[0].mask
        tm = torch.eye(K).expand(B, K, K).clone()
        distinct = torch.ones(B, K, dtype=torch.bool)
        for i in range(K):
            for j in range(i + 1, K):
                a, b = members[i].ca, members[j].ca
                v = 0.5 * (tm_score(a, b, m).tm + tm_score(b, a, m).tm)
                tm[:, i, j] = tm[:, j, i] = v
                same = ((a - b).abs().amax((-1, -2)) < self.tol)
                same = same | (v > 1.0 - 1e-6) & (a - b).abs().amax((-1, -2)).lt(self.tol)
                distinct[:, j] &= ~same
        eff = distinct.sum(-1)
        off = (tm.sum((-1, -2)) - K) / max(K * (K - 1), 1)
        # Agreement over distinct members only (identical deterministic reruns carry no information).
        agree = torch.full((B,), float("nan"))
        for b in range(B):
            idx = distinct[b].nonzero(as_tuple=True)[0]
            if len(idx) >= 2:
                sub = tm[b][idx][:, idx]
                agree[b] = (sub.sum() - len(idx)) / (len(idx) * (len(idx) - 1))
        med_idx = tm.mean(-1).argmax(-1)
        pick = lambda f: torch.stack([f(members[int(med_idx[b])])[b] for b in range(B)])
        medoid = FoldPrediction(
            tokens=tokens, mask=m, coords=pick(lambda p: p.coords), plddt=pick(lambda p: p.plddt),
            pae=None if members[0].pae is None else pick(lambda p: p.pae),
            ptm=None if members[0].ptm is None else pick(lambda p: p.ptm),
            backend="+".join(sorted({p.backend for p in members})),
            independence_class="/".join(sorted({p.independence_class for p in members})),
            checkpoint_sha256=None, seed=None,
            provenance={"members": [(p.backend, p.seed, p.checkpoint_sha256) for p in members]},
        )
        rmsf = torch.zeros(B, L)
        disp = torch.zeros(B, L)
        for b in range(B):
            ref = medoid.ca[b : b + 1]
            devs, rots = [], []
            for p in members:
                R, t = kabsch(p.ca[b : b + 1], ref, m[b : b + 1])
                aligned = apply_rigid(p.ca[b : b + 1], R, t)
                devs.append(((aligned - ref) ** 2).sum(-1)[0])
                Rf, _ = backbone_to_frames(p.coords[b : b + 1])
                rots.append(R[0] @ Rf[0])
            rmsf[b] = torch.stack(devs).mean(0).sqrt()
            if K >= 2:
                _, d = frechet_mean_so3(torch.stack(rots))
                disp[b] = d
        return EnsembleFold(members, medoid, tm, eff, agree, rmsf, disp, {"n_members": K})
