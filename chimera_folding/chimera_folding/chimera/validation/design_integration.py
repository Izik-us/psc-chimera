"""Glue between ``CHIMERAv2.design`` and the fold validator (opt-in; default behaviour unchanged).

Flow: cheap objectives -> ``pareto_funnel`` picks the folding budget -> independent folder ->
per-sequence structural score replaces the constant backbone-sanity column -> candidates whose fold
is REJECTed (or, optionally, not ACCEPTed) are removed -> Pareto front over the survivors.
Candidates that were never folded cannot be compared on the new objective and are excluded.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import torch

from .fold_validator import Decision, FoldValidationReport, FoldValidator, pareto_funnel, structural_objective_labels


@dataclass
class FoldRerank:
    folded_indices: torch.Tensor  # global candidate ids that were folded
    reports: list[FoldValidationReport]
    selected: torch.Tensor  # global candidate ids, best first
    selected_scores: torch.Tensor  # (S, K) objectives with the structural column replaced
    front_size: int
    label_kind: str
    status: str

    def summary(self) -> dict:
        counts = {d.value: sum(r.decision is d for r in self.reports) for d in Decision}
        return {
            "status": self.status,
            "label_kind": self.label_kind,
            "biological_measurement": False,
            "n_folded": int(self.folded_indices.numel()),
            "decision_counts": counts,
            "folded_indices": self.folded_indices.tolist(),
            "reports": [r.as_dict() for r in self.reports],
        }


def fold_rerank(
    validator: FoldValidator,
    sequences: torch.Tensor,  # (N, L) amino-acid ids 0-19
    backbones: torch.Tensor,  # (N, L, 4, 3) the backbone each sequence was designed for
    objectives: torch.Tensor,  # (N, K) cheap objectives, higher is better
    *,
    pareto_fn: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
    budget: int,
    structural_column: int = 1,
    residue_mask: torch.Tensor | None = None,  # (L,) bool
    domain_spans: Sequence[tuple[int, int]] | None = None,
    n_null: int = 0,
    require_accept: bool = False,
    chunk: int = 8,
) -> FoldRerank:
    if sequences.shape[0] != backbones.shape[0] or sequences.shape[0] != objectives.shape[0]:
        raise ValueError("sequences, backbones and objectives must share the candidate dimension")
    idx = pareto_funnel(objectives, min(budget, objectives.shape[0]))
    reports: list[FoldValidationReport] = []
    for lo in range(0, len(idx), chunk):
        sel = idx[lo : lo + chunk]
        mask = None if residue_mask is None else residue_mask.unsqueeze(0).expand(len(sel), -1)
        spans = None if domain_spans is None else [list(domain_spans)] * len(sel)
        reports += validator.validate(backbones[sel], sequences[sel], mask, domain_spans=spans, n_null=n_null)
    scores, meta = structural_objective_labels(reports)
    adjusted = objectives[idx].clone()
    adjusted[:, structural_column] = scores / 100.0
    ok = [
        i for i, r in enumerate(reports)
        if r.decision is Decision.ACCEPT or (r.decision is Decision.PENALIZE and not require_accept)
    ]
    if not ok:
        return FoldRerank(
            idx, reports, torch.empty(0, dtype=torch.long), adjusted[:0], 0, meta["label_kind"],
            "no candidate survived fold validation" + (" with ACCEPT" if require_accept else ""),
        )
    _, local_front = pareto_fn(adjusted[ok])
    front = torch.tensor([ok[int(j)] for j in local_front], dtype=torch.long)
    order = front[torch.argsort(scores[front], descending=True)]
    return FoldRerank(
        idx, reports, idx[order], adjusted[order], int(front.numel()), meta["label_kind"],
        "ranked by fold-validated structural score" + ("" if meta["label_kind"] == "deterministic_evaluator"
                                                         else " (UNCALIBRATED: proxy label, not validated evidence)"),
    )
