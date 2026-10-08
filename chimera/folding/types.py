"""Structured fold-prediction contract and the sequence-only backend interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import torch

AA = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {a: i for i, a in enumerate(AA)}
UNK_IDX = 20
PAD_IDX = 21


def tokenize(sequences: Sequence[str]) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode equal-or-ragged sequences to ``(B, L)`` tokens plus a ``(B, L)`` bool mask."""
    L = max(len(s) for s in sequences)
    tok = torch.full((len(sequences), L), PAD_IDX, dtype=torch.long)
    for i, s in enumerate(sequences):
        for j, ch in enumerate(s.upper()):
            tok[i, j] = AA_TO_IDX.get(ch, UNK_IDX)
    return tok, tok != PAD_IDX


def detokenize(tokens: torch.Tensor) -> list[str]:
    table = AA + "XX"
    return ["".join(table[int(i)] for i in row if int(i) != PAD_IDX) for row in tokens]


@dataclass
class FoldPrediction:
    """Everything an independent folder returns about one batch of sequences."""

    tokens: torch.Tensor  # (B, L)
    mask: torch.Tensor  # (B, L) bool
    coords: torch.Tensor  # (B, L, 4, 3) N/CA/C/O
    plddt: torch.Tensor  # (B, L) in [0, 100]
    pae: torch.Tensor | None = None  # (B, L, L) Angstrom, expected aligned error
    ptm: torch.Tensor | None = None  # (B,) predicted TM-score
    distogram_logits: torch.Tensor | None = None
    frames: tuple[torch.Tensor, torch.Tensor] | None = None  # (R, t)
    backend: str = ""
    independence_class: str = ""
    checkpoint_sha256: str | None = None
    seed: int | None = None
    provenance: Mapping[str, Any] = field(default_factory=dict)

    @property
    def mean_plddt(self) -> torch.Tensor:
        m = self.mask.to(self.plddt.dtype)
        return (self.plddt * m).sum(-1) / m.sum(-1).clamp_min(1)

    @property
    def ca(self) -> torch.Tensor:
        return self.coords[:, :, 1]


class FoldingBackend(ABC):
    """Sequence-in, structure-out folder.

    The signature is the *independence firewall*: there is deliberately no
    parameter through which a design backbone, source frame, or any
    design-derived structural tensor could reach the folder. Validators call
    only ``predict``.
    """

    name: str = "folding-backend"
    #: "external_pretrained" | "chimera_native" | "chimera_native_untrained"
    independence_class: str = "unspecified"
    checkpoint_sha256: str | None = None

    @abstractmethod
    def predict(
        self,
        tokens: torch.Tensor,
        mask: torch.Tensor | None = None,
        *,
        seed: int = 0,
        msa_tokens: torch.Tensor | None = None,
    ) -> FoldPrediction:
        raise NotImplementedError


__all__ = [
    "AA",
    "tokenize",
    "detokenize",
    "FoldPrediction",
    "FoldingBackend",
    "PAD_IDX",
    "UNK_IDX",
]
