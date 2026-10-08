"""Training objectives for the CHIMERA folding generator.

Frame-aligned point error (FAPE), distogram, SO(3) geodesic, distributional
confidence (pLDDT / PAE trained against *realised* error so that confidence is
learned, not asserted), and stereochemical violation terms.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .frames import carbonyl_angle_target, backbone_to_frames


def _pair_mask(mask):
    return (mask.unsqueeze(1) & mask.unsqueeze(2)).to(torch.float32)


def local_points(R, t):
    """x_ij = R_i^T (t_j - t_i): residue j expressed in residue i's frame."""
    rel = t.unsqueeze(1) - t.unsqueeze(2)
    return torch.einsum("bikl,bijk->bijl", R, rel)


def fape(R_pred, t_pred, R_true, t_true, mask, clamp: float = 10.0, scale: float = 10.0, eps: float = 1e-4):
    xp, xt = local_points(R_pred, t_pred), local_points(R_true, t_true)
    err = ((xp - xt) ** 2).sum(-1).add(eps).sqrt()
    err = err.clamp(max=clamp) / scale
    pm = _pair_mask(mask)
    return (err * pm).sum((-1, -2)) / pm.sum((-1, -2)).clamp_min(1)


def aligned_error(R_pred, t_pred, R_true, t_true, eps: float = 1e-4):
    xp, xt = local_points(R_pred, t_pred), local_points(R_true, t_true)
    return ((xp - xt) ** 2).sum(-1).add(eps).sqrt()


def so3_angle(Rrel, eps: float = 1e-8):
    """Geodesic angle of a relative rotation; gradient-safe at 0 and pi (atan2 form)."""
    tr = Rrel.diagonal(dim1=-2, dim2=-1).sum(-1)
    v = torch.stack(
        (Rrel[..., 2, 1] - Rrel[..., 1, 2], Rrel[..., 0, 2] - Rrel[..., 2, 0], Rrel[..., 1, 0] - Rrel[..., 0, 1]),
        dim=-1,
    )
    return torch.atan2(0.5 * (v.pow(2).sum(-1) + eps).sqrt(), 0.5 * (tr - 1.0))


def so3_geodesic_loss(R_pred, R_true, mask):
    ang = so3_angle(R_pred.transpose(-1, -2) @ R_true)
    m = mask.to(ang.dtype)
    return (ang * m).sum(-1) / m.sum(-1).clamp_min(1)


def distogram_loss(logits, ca_true, mask, edges):
    d = torch.cdist(ca_true, ca_true)
    target = torch.bucketize(d, edges)
    ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target.reshape(-1), reduction="none").view_as(d)
    pm = _pair_mask(mask)
    return (ce * pm).sum((-1, -2)) / pm.sum((-1, -2)).clamp_min(1)


def realised_lddt(ca_pred, ca_true, mask, cutoff: float = 15.0):
    from ..validation.metrics import lddt_ca

    per_res, _ = lddt_ca(ca_pred, ca_true, mask, cutoff=cutoff)
    return per_res


def plddt_loss(logits, ca_pred, ca_true, mask, n_bins):
    with torch.no_grad():
        target = (realised_lddt(ca_pred.detach(), ca_true, mask) * 100.0).clamp(0, 100 - 1e-4)
        idx = (target / (100.0 / n_bins)).long()
    ce = F.cross_entropy(logits.reshape(-1, n_bins), idx.reshape(-1), reduction="none").view_as(idx)
    m = mask.to(ce.dtype)
    return (ce * m).sum(-1) / m.sum(-1).clamp_min(1)


def pae_loss(logits, R_pred, t_pred, R_true, t_true, mask, n_bins, pae_max):
    with torch.no_grad():
        err = aligned_error(R_pred.detach(), t_pred.detach(), R_true, t_true).clamp(0, pae_max - 1e-4)
        idx = (err / (pae_max / n_bins)).long()
    ce = F.cross_entropy(logits.reshape(-1, n_bins), idx.reshape(-1), reduction="none").view_as(idx)
    pm = _pair_mask(mask)
    return (ce * pm).sum((-1, -2)) / pm.sum((-1, -2)).clamp_min(1)


def violation_loss(coords, mask):
    """Flat-bottom penalties: peptide bond, intra-residue bonds, and Calpha non-bonded clashes."""
    n, ca, c, o = (coords[:, :, i] for i in range(4))
    m = mask.to(coords.dtype)
    pm = m[:, 1:] * m[:, :-1]
    pep = ((n[:, 1:] - c[:, :-1]).norm(dim=-1) - 1.329).abs()
    pep = F.relu(pep - 0.05)
    intra = sum(
        F.relu(((a - b).norm(dim=-1) - ref).abs() - 0.05) * m
        for a, b, ref in ((n, ca, 1.458), (ca, c, 1.525), (c, o, 1.231))
    )
    L = ca.shape[1]
    d = torch.cdist(ca, ca)
    sep = (torch.arange(L, device=ca.device)[:, None] - torch.arange(L, device=ca.device)[None]).abs() >= 2
    valid = sep.unsqueeze(0) & mask.unsqueeze(1) & mask.unsqueeze(2)
    clash = F.relu(3.6 - d) * valid
    loss = (pep * pm).sum(-1) / pm.sum(-1).clamp_min(1) + intra.sum(-1) / m.sum(-1).clamp_min(1)
    return loss + clash.sum((-1, -2)) / valid.sum((-1, -2)).clamp_min(1)


def psi_loss(psi_sincos, coords_true, mask):
    target = carbonyl_angle_target(coords_true)
    pred = F.normalize(psi_sincos, dim=-1)
    err = ((pred - target) ** 2).sum(-1)
    m = mask.to(err.dtype)
    return (err * m).sum(-1) / m.sum(-1).clamp_min(1)


DEFAULT_WEIGHTS = {
    "fape": 1.0,
    "fape_aux": 0.5,
    "geodesic": 0.1,
    "distogram": 0.3,
    "plddt": 0.1,
    "pae": 0.1,
    "psi": 0.1,
    "violation": 0.05,
}


def folding_loss(out: dict, coords_true: torch.Tensor, mask: torch.Tensor, heads, weights=None):
    """Return ``(total, components)``; ``coords_true`` is ``(B, L, 4, 3)`` reference N/CA/C/O."""
    w = {**DEFAULT_WEIGHTS, **(weights or {})}
    R_true, t_true = backbone_to_frames(coords_true)
    ca_true = coords_true[:, :, 1]
    R, t = out["R"], out["t"]
    comp = {}
    comp["fape"] = fape(R, t, R_true, t_true, mask).mean()
    aux = [fape(Ri, ti, R_true, t_true, mask).mean() for Ri, ti in out["trajectory"]]
    comp["fape_aux"] = torch.stack(aux).mean()
    comp["geodesic"] = so3_geodesic_loss(R, R_true, mask).mean()
    comp["distogram"] = distogram_loss(
        out["distogram_logits"], ca_true, mask, heads.dist_edges(R.device, R.dtype)
    ).mean()
    comp["plddt"] = plddt_loss(out["plddt_logits"], t, ca_true, mask, heads.n_plddt_bins).mean()
    comp["pae"] = pae_loss(out["pae_logits"], R, t, R_true, t_true, mask, heads.n_pae_bins, heads.pae_max).mean()
    comp["psi"] = psi_loss(out["psi_sincos"], coords_true, mask).mean()
    comp["violation"] = violation_loss(out["coords"], mask).mean()
    total = sum(w[k] * v for k, v in comp.items())
    return total, {k: float(v.detach()) for k, v in comp.items()}
