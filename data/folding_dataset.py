"""Clean-data gates for folding generator/validator training and calibration.

Folder accuracy and confidence calibration are only as honest as their data,
so every rule here is a *reject-with-reason* gate, never silent repair. The
leakage-safe split itself stays with ``data.leakage_splits`` (identity
clustering + provenance); this module adds structure-quality gates, sampling
weights, and the carving of a disjoint **calibration** partition.
"""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")


@dataclass(frozen=True)
class FoldingQC:
    max_resolution_xray: float = 2.5
    max_resolution_cryoem: float = 3.0
    max_rfree: float = 0.28
    min_length: int = 30
    max_length: int = 1500
    min_modelled_fraction: float = 0.90
    max_nonstandard_fraction: float = 0.02
    max_ca_gap_A: float = 4.3  # consecutive CA > this means a chain break
    max_chain_breaks: int = 2
    max_bfactor_z: float = 3.0  # drop records whose mean B is an outlier within the method
    allowed_methods: tuple = ("X-RAY DIFFRACTION", "ELECTRON MICROSCOPY")


@dataclass
class QCResult:
    record_id: str
    passed: bool
    reasons: list[str] = field(default_factory=list)


def _ca_breaks(ca: Sequence[Sequence[float]], max_gap: float) -> int:
    n = 0
    for a, b in zip(ca[:-1], ca[1:]):
        if math.dist(a, b) > max_gap:
            n += 1
    return n


def evaluate_record(rec: Mapping, qc: FoldingQC = FoldingQC()) -> QCResult:
    """``rec`` keys: id, sequence, method, resolution, rfree?, ca (list of xyz or None per residue), mean_bfactor?"""
    rid, why = str(rec.get("id", "?")), []
    seq = str(rec.get("sequence", "")).upper()
    L = len(seq)
    method = str(rec.get("method", "")).upper()
    if method not in qc.allowed_methods:
        why.append(f"method {method or 'unknown'} not allowed")
    res = rec.get("resolution")
    if res is None:
        why.append("resolution missing")
    else:
        cap = qc.max_resolution_cryoem if "ELECTRON" in method else qc.max_resolution_xray
        if res > cap:
            why.append(f"resolution {res} > {cap}")
    rfree = rec.get("rfree")
    if "X-RAY" in method:
        if rfree is None:
            why.append("R-free missing")
        elif rfree > qc.max_rfree:
            why.append(f"R-free {rfree} > {qc.max_rfree}")
    if not qc.min_length <= L <= qc.max_length:
        why.append(f"length {L} outside [{qc.min_length}, {qc.max_length}]")
    if L:
        if sum(c not in STANDARD_AA for c in seq) / L > qc.max_nonstandard_fraction:
            why.append("too many non-standard residues")
        ca = rec.get("ca") or []
        if len(ca) != L:
            why.append("coordinate count != sequence length (unaligned record)")
        else:
            present = [c for c in ca if c is not None]
            if len(present) / L < qc.min_modelled_fraction:
                why.append(f"modelled fraction {len(present)/L:.2f} < {qc.min_modelled_fraction}")
            if any(not all(math.isfinite(x) for x in c) for c in present):
                why.append("non-finite coordinates")
            elif _ca_breaks(present, qc.max_ca_gap_A) > qc.max_chain_breaks:
                why.append("too many chain breaks")
    return QCResult(rid, not why, why)


def apply_qc(records: Iterable[Mapping], qc: FoldingQC = FoldingQC()):
    """Return ``(kept, report)``; B-factor outliers are removed per experimental method."""
    results = [(r, evaluate_record(r, qc)) for r in records]
    by_method: dict[str, list[float]] = {}
    for r, q in results:
        if q.passed and r.get("mean_bfactor") is not None:
            by_method.setdefault(str(r.get("method", "")).upper(), []).append(float(r["mean_bfactor"]))
    stats = {}
    for m, v in by_method.items():
        mu = sum(v) / len(v)
        sd = math.sqrt(sum((x - mu) ** 2 for x in v) / max(len(v) - 1, 1)) or 1.0
        stats[m] = (mu, sd)
    kept, rejected = [], []
    for r, q in results:
        if q.passed and r.get("mean_bfactor") is not None:
            mu, sd = stats[str(r.get("method", "")).upper()]
            if abs((float(r["mean_bfactor"]) - mu) / sd) > qc.max_bfactor_z:
                q.passed, q.reasons = False, [f"mean B-factor outlier (|z|>{qc.max_bfactor_z})"]
        (kept if q.passed else rejected).append((r, q))
    reasons = Counter(x for _, q in rejected for x in q.reasons)
    report = {"n_in": len(results), "n_kept": len(kept), "n_rejected": len(rejected), "reject_reasons": dict(reasons)}
    return [r for r, _ in kept], report


def cluster_balanced_weights(cluster_ids: Sequence[str], family: Sequence[str] | None = None) -> list[float]:
    """Inverse-cluster-size sampling weights (so large redundant families do not dominate);
    optionally further balanced across protein families. Normalised to mean 1."""
    sizes = Counter(cluster_ids)
    w = [1.0 / sizes[c] for c in cluster_ids]
    if family is not None:
        fam_mass = Counter()
        for wi, f in zip(w, family):
            fam_mass[f] += wi
        w = [wi / math.sqrt(fam_mass[f]) for wi, f in zip(w, family)]
    mean = sum(w) / len(w)
    return [x / mean for x in w]


def carve_calibration_partition(
    record_ids: Sequence[str], split_of: Mapping[str, str], group_of: Mapping[str, str], fraction: float = 0.5,
    source: str = "validation",
) -> dict[str, str]:
    """Split a held-out partition into ``validation`` and ``calibration`` by *cluster* (never by record),
    using a stable hash so the carve-out is reproducible and leak-free."""
    out = dict(split_of)
    for rid in record_ids:
        if split_of[rid] != source:
            continue
        h = int(hashlib.sha256(f"calibration::{group_of[rid]}".encode()).hexdigest(), 16) % 10_000
        out[rid] = "calibration" if h < fraction * 10_000 else source
    return out


def assert_no_cluster_overlap(split_of: Mapping[str, str], group_of: Mapping[str, str]) -> None:
    seen: dict[str, str] = {}
    for rid, sp in split_of.items():
        g = group_of[rid]
        if g in seen and seen[g] != sp:
            raise ValueError(f"cluster {g} appears in both {seen[g]} and {sp}: leakage")
        seen[g] = sp


def exclude_design_overlap(train_seqs: Mapping[str, str], design_seqs: Sequence[str], min_identity: float = 0.5):
    """Drop training records that resemble a CHIMERA design (identity over the shorter length, ungapped
    position-wise proxy) so the folder is not scored on sequences it effectively trained on.
    For real use replace the proxy with MMseqs2/Biopython alignment identity."""
    drop = set()
    for rid, s in train_seqs.items():
        for d in design_seqs:
            n = min(len(s), len(d))
            if n and sum(a == b for a, b in zip(s, d)) / n >= min_identity:
                drop.add(rid)
                break
    return drop
