"""Superposition-aware and superposition-free structural comparison metrics.

All functions take Calpha ``(B, L, 3)`` (or N/CA/C/O ``(B, L, 4, 3)`` where
stated) tensors in matching residue order, with an optional ``(B, L)`` mask.
Gradients are not propagated; these are evaluators, not training losses.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from ..folding.frames import backbone_to_frames, backbone_torsions
from ..lie import so3_log


def _mask(mask, ref):
    if mask is None:
        return torch.ones(ref.shape[:2], dtype=torch.bool, device=ref.device)
    return mask.bool()


def kabsch(P: torch.Tensor, Q: torch.Tensor, w: torch.Tensor | None = None):
    """Optimal proper rotation/translation mapping ``P`` onto ``Q`` (weighted).

    Returns ``R (B,3,3), t (B,3)`` with ``Q ~ P @ R^T + t``. The determinant
    correction guarantees ``R in SO(3)`` (no reflections).
    """
    if w is None:
        w = torch.ones(P.shape[:2], dtype=P.dtype, device=P.device)
    w = w.to(P.dtype)
    ws = w.sum(-1, keepdim=True).clamp_min(1e-8)
    cp = (P * w.unsqueeze(-1)).sum(1) / ws
    cq = (Q * w.unsqueeze(-1)).sum(1) / ws
    Pc, Qc = P - cp.unsqueeze(1), Q - cq.unsqueeze(1)
    H = torch.einsum("bn,bni,bnj->bij", w, Pc, Qc)
    U, _, Vh = torch.linalg.svd(H.double())
    V = Vh.transpose(-1, -2)
    d = torch.sign(torch.det(V @ U.transpose(-1, -2)))
    d = torch.where(d == 0, torch.ones_like(d), d)
    D = torch.diag_embed(torch.stack((torch.ones_like(d), torch.ones_like(d), d), dim=-1))
    R = (V @ D @ U.transpose(-1, -2)).to(P.dtype)
    t = cq - torch.einsum("bij,bj->bi", R, cp)
    return R, t


def apply_rigid(P: torch.Tensor, R: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    return torch.einsum("bij,bnj->bni", R, P) + t.unsqueeze(1)


def rmsd(P, Q, mask=None, superpose: bool = True) -> torch.Tensor:
    m = _mask(mask, P)
    if superpose:
        R, t = kabsch(P, Q, m)
        P = apply_rigid(P, R, t)
    sq = ((P - Q) ** 2).sum(-1) * m
    return (sq.sum(-1) / m.sum(-1).clamp_min(1)).sqrt()


def tm_d0(length: torch.Tensor | int) -> torch.Tensor:
    L = torch.as_tensor(length, dtype=torch.float64)
    d0 = 1.24 * (L.clamp_min(19.0) - 15.0).pow(1.0 / 3.0) - 1.8
    return d0.clamp_min(0.5)


@dataclass
class TMResult:
    tm: torch.Tensor  # (B,)
    R: torch.Tensor  # (B,3,3) mobile -> reference
    t: torch.Tensor  # (B,3)
    gdt_ts: torch.Tensor  # (B,)


def tm_score(P, Q, mask=None, max_starts: int = 48, max_iter: int = 20) -> TMResult:
    """TM-score of ``P`` (mobile) against ``Q`` (reference), normalised by reference length.

    Uses the TM-score iterative superposition search (fragment seeds of length
    L, L/2, L/4, ... >= 4, re-weighted by a distance cutoff until the selected
    set is stable) and keeps the best-scoring superposition. The score is
    therefore a lower bound on the true maximum, as in the reference program.
    """
    B = P.shape[0]
    m = _mask(mask, P)
    tms, Rs, ts, gdts = [], [], [], []
    for b in range(B):
        idx_valid = m[b].nonzero(as_tuple=True)[0]
        p, q = P[b, idx_valid], Q[b, idx_valid]
        tm, R, t, gdt = _tm_single(p, q, max_starts, max_iter)
        tms.append(tm), Rs.append(R), ts.append(t), gdts.append(gdt)
    return TMResult(torch.stack(tms), torch.stack(Rs), torch.stack(ts), torch.stack(gdts))


def _tm_single(p, q, max_starts, max_iter):
    N = p.shape[0]
    dev, dt = p.device, p.dtype
    if N < 3:
        z = torch.zeros((), dtype=dt, device=dev)
        return z, torch.eye(3, dtype=dt, device=dev), torch.zeros(3, dtype=dt, device=dev), z
    d0 = float(tm_d0(N))
    d_search = min(max(d0 + 1.5, 4.5), 8.0)
    best = (-1.0, None, None, None)

    def fit(sel):
        R, t = kabsch(p[sel].unsqueeze(0), q[sel].unsqueeze(0))
        return R[0], t[0]

    def evaluate(R, t):
        d = (torch.einsum("ij,nj->ni", R, p) + t - q).norm(dim=-1)
        tm = float((1.0 / (1.0 + (d / d0) ** 2)).sum() / N)
        return tm, d

    lengths, fl = [], N
    while fl >= 4:
        lengths.append(fl)
        fl //= 2
    starts_budget = max(1, max_starts // max(1, len(lengths)))
    for fl in lengths:
        n_pos = N - fl + 1
        stride = max(1, math.ceil(n_pos / starts_budget))
        for s in range(0, n_pos, stride):
            sel = torch.arange(s, s + fl, device=dev)
            R, t = fit(sel)
            prev = None
            for _ in range(max_iter):
                tm, d = evaluate(R, t)
                if tm > best[0]:
                    gdt = float(sum((d < c).double().mean() for c in (1.0, 2.0, 4.0, 8.0)) / 4)
                    best = (tm, R, t, gdt)
                cut = d_search
                chosen = (d < cut).nonzero(as_tuple=True)[0]
                while chosen.numel() < 3 and cut < 1e3:
                    cut += 0.5
                    chosen = (d < cut).nonzero(as_tuple=True)[0]
                key = tuple(chosen.tolist())
                if key == prev or chosen.numel() < 3:
                    break
                prev = key
                R, t = fit(chosen)
    tm, R, t, gdt = best
    return (
        torch.tensor(tm, dtype=dt, device=dev),
        R,
        t,
        torch.tensor(gdt, dtype=dt, device=dev),
    )


def lddt_ca(P, Q, mask=None, cutoff: float = 15.0, thresholds=(0.5, 1.0, 2.0, 4.0)):
    """Superposition-free Calpha lDDT of ``P`` against reference ``Q``.

    Returns ``(per_residue (B,L) in [0,1], mean (B,))``.
    """
    m = _mask(mask, P)
    dp, dq = torch.cdist(P, P), torch.cdist(Q, Q)
    pair = m.unsqueeze(1) & m.unsqueeze(2)
    eye = torch.eye(P.shape[1], dtype=torch.bool, device=P.device).unsqueeze(0)
    incl = (dq < cutoff) & pair & ~eye
    diff = (dp - dq).abs()
    score = sum((diff < th).to(P.dtype) for th in thresholds) / len(thresholds)
    num = (score * incl).sum(-1)
    den = incl.sum(-1).clamp_min(1)
    per_res = num / den
    per_res = torch.where(incl.any(-1), per_res, torch.zeros_like(per_res))
    mean = (per_res * m).sum(-1) / m.sum(-1).clamp_min(1)
    return per_res, mean


def contact_f1(P, Q, mask=None, threshold: float = 8.0, min_sep: int = 6) -> torch.Tensor:
    m = _mask(mask, P)
    L = P.shape[1]
    sep = (torch.arange(L, device=P.device)[:, None] - torch.arange(L, device=P.device)[None]).abs() >= min_sep
    valid = sep.unsqueeze(0) & m.unsqueeze(1) & m.unsqueeze(2)
    cp, cq = (torch.cdist(P, P) < threshold) & valid, (torch.cdist(Q, Q) < threshold) & valid
    tp = (cp & cq).sum((-1, -2)).to(P.dtype)
    prec = tp / cp.sum((-1, -2)).clamp_min(1)
    rec = tp / cq.sum((-1, -2)).clamp_min(1)
    f1 = 2 * prec * rec / (prec + rec).clamp_min(1e-8)
    return torch.where(cq.sum((-1, -2)) > 0, f1, torch.ones_like(f1))


def distance_map_mae(P, Q, mask=None, cutoff: float = 15.0) -> torch.Tensor:
    m = _mask(mask, P)
    dp, dq = torch.cdist(P, P), torch.cdist(Q, Q)
    valid = (dq < cutoff) & m.unsqueeze(1) & m.unsqueeze(2)
    return ((dp - dq).abs() * valid).sum((-1, -2)) / valid.sum((-1, -2)).clamp_min(1)


def frame_geodesic_error(design_coords, fold_coords, R_align, mask=None) -> torch.Tensor:
    """Per-residue SO(3) geodesic angle (radians) between design and aligned fold frames."""
    Rd, _ = backbone_to_frames(design_coords)
    Rf, _ = backbone_to_frames(fold_coords)
    Rf = torch.einsum("bij,bljk->blik", R_align, Rf)
    rel = Rd.transpose(-1, -2) @ Rf
    return so3_log(rel).norm(dim=-1)


def torsion_agreement(design_coords, fold_coords, mask=None, tol_deg: float = 30.0):
    """Fraction of defined phi/psi within ``tol_deg`` (circular distance) and mean circular error (deg)."""
    pd, sd = backbone_torsions(design_coords)
    pf, sf = backbone_torsions(fold_coords)
    out_frac, out_err = [], []
    for a, b in ((pd, pf), (sd, sf)):
        d = torch.atan2(torch.sin(a - b), torch.cos(a - b)).abs()
        valid = torch.isfinite(d)
        if mask is not None:
            valid = valid & mask.bool()
        d = d.nan_to_num(0.0)
        out_frac.append(((d < math.radians(tol_deg)) & valid).sum(-1) / valid.sum(-1).clamp_min(1))
        out_err.append(torch.rad2deg((d * valid).sum(-1) / valid.sum(-1).clamp_min(1)))
    return torch.stack(out_frac).mean(0), torch.stack(out_err).mean(0)


def domain_arrangement(design_ca, fold_ca, spans, min_len: int = 8):
    """Per-domain TM and inter-domain arrangement error (single candidate, ``(L,3)`` inputs).

    Each fold domain is rigidly fitted onto its design domain (``T_k``). If the
    domain arrangement is conserved all ``T_k`` coincide; the pairwise
    discrepancy is reported as a rotation angle (deg) and centroid displacement (A).
    """
    fits = []
    domain_tm = []
    for lo, hi in spans:
        lo, hi = int(lo), int(hi)
        if hi - lo < min_len:
            continue
        p, q = fold_ca[lo:hi].unsqueeze(0), design_ca[lo:hi].unsqueeze(0)
        res = tm_score(p, q)
        domain_tm.append(float(res.tm[0]))
        fits.append((res.R[0], res.t[0], p[0].mean(0)))
    max_rot, max_trans = 0.0, 0.0
    for i in range(len(fits)):
        for j in range(i + 1, len(fits)):
            Ri, ti, ci = fits[i]
            Rj, tj, _ = fits[j]
            rot = float(so3_log(Ri.transpose(-1, -2) @ Rj).norm())
            trans = float(((Ri @ ci + ti) - (Rj @ ci + tj)).norm())
            max_rot, max_trans = max(max_rot, math.degrees(rot)), max(max_trans, trans)
    return {
        "domain_tm": domain_tm,
        "min_domain_tm": min(domain_tm) if domain_tm else float("nan"),
        "max_arrangement_rotation_deg": max_rot,
        "max_arrangement_translation_A": max_trans,
    }


def frechet_mean_so3(R: torch.Tensor, iters: int = 30, tol: float = 1e-7):
    """Karcher mean of rotations ``R (K, ..., 3, 3)`` and geodesic dispersion (rad^2).

    Iterates ``mu <- mu exp(mean_k log(mu^T R_k))``; converges for ensembles
    contained in a geodesic ball of radius < pi/2.
    """
    from ..lie import so3_exp

    mu = R[0]
    for _ in range(iters):
        step = so3_log(mu.transpose(-1, -2).unsqueeze(0) @ R).mean(0)
        mu = mu @ so3_exp(step)
        if float(step.norm(dim=-1).max()) < tol:
            break
    disp = (so3_log(mu.transpose(-1, -2).unsqueeze(0) @ R) ** 2).sum(-1).mean(0)
    return mu, disp


__all__ = [
    "kabsch",
    "apply_rigid",
    "rmsd",
    "tm_d0",
    "tm_score",
    "TMResult",
    "lddt_ca",
    "contact_f1",
    "distance_map_mae",
    "frame_geodesic_error",
    "torsion_agreement",
    "domain_arrangement",
    "frechet_mean_so3",
]
