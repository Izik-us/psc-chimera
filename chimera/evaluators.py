"""Independent, deterministic objective evaluators.

These are transparent proxies. They must not be presented as PoET, pLDDT,
experimental activity, or assembly free energy without the corresponding
external model and calibration data.
"""

from typing import Dict

import torch
import torch.nn.functional as F

from .codon_optimizer import sliding_window_optimize
from .geometry import validate_backbone
from .icosahedral import interface_compatibility


class BiologicalObjectiveEvaluator:
    """Compute auditable objective proxies outside the neural objective head."""

    def __init__(
        self,
        target_gc: tuple[float, float] = (0.58, 0.65),
        target_profile: dict[int, int] | None = None,
    ):
        lo, hi = target_gc
        if not 0 <= lo <= hi <= 1:
            raise ValueError("target_gc must be an ordered interval inside [0,1]")
        self.target_gc = target_gc
        self.target_profile = target_profile or {}

    def evaluate(
        self,
        sequences: torch.Tensor,
        backbone_coords: torch.Tensor,
        rotations: torch.Tensor | None = None,
        faces: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor | dict]:
        """Return one score per candidate sequence plus transparent geometry metadata."""
        if sequences.ndim != 2 or sequences.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.long):
            raise ValueError("sequences must be an integer tensor with shape (N, L)")
        if torch.any((sequences < 0) | (sequences >= 20)):
            raise ValueError("sequences must contain amino-acid IDs in [0,19]")
        geometry = validate_backbone(backbone_coords, rotations)
        n = sequences.shape[0]
        if backbone_coords.shape[0] not in (1, n):
            raise ValueError("backbone batch must be 1 or match candidate count")
        if faces is not None and faces.shape != (n,):
            raise ValueError("faces must have shape (N,) when supplied")

        expression = []
        aa = "ACDEFGHIKLMNPQRSTVWY"
        for sequence in sequences.detach().cpu():
            protein = "".join(aa[int(token)] for token in sequence)
            _, metrics = sliding_window_optimize(protein)
            gc = float(metrics["gc_content"])
            lo, hi = self.target_gc
            expression.append(max(0.0, 1.0 - max(lo - gc, gc - hi) / 0.5))

        structural = torch.full(
            (n,), 1.0 if geometry.valid else 0.0, dtype=torch.float32, device=sequences.device
        )
        selectivity = self._selectivity(sequences).to(sequences.device)
        assembly = self._assembly(backbone_coords, rotations, faces, n)
        return {
            # This is sequence entropy/complexity, not a PoET likelihood.
            "evolutionary_plausibility_proxy": self._sequence_entropy(sequences),
            "structural_validity": structural,
            "expression_proxy": torch.tensor(expression, device=sequences.device),
            "selectivity_proxy": selectivity,
            "assembly_proxy": assembly,
            "geometry": geometry.as_dict(),
        }

    @staticmethod
    def _sequence_entropy(sequences: torch.Tensor) -> torch.Tensor:
        counts = F.one_hot(sequences, 20).float().mean(dim=1)
        entropy = -(counts.clamp_min(1e-8) * counts.clamp_min(1e-8).log()).sum(dim=-1)
        return entropy / torch.log(torch.tensor(20.0, device=sequences.device, dtype=entropy.dtype))

    def _selectivity(self, sequences: torch.Tensor) -> torch.Tensor:
        if not self.target_profile:
            return torch.zeros(sequences.shape[0], device=sequences.device)
        positions = torch.tensor(list(self.target_profile), dtype=torch.long, device=sequences.device)
        expected = torch.tensor(list(self.target_profile.values()), dtype=torch.long, device=sequences.device)
        valid = (positions >= 0) & (positions < sequences.shape[1])
        if not valid.any():
            return torch.zeros(sequences.shape[0], device=sequences.device)
        return (sequences[:, positions[valid]] == expected[valid]).float().mean(dim=-1)

    @staticmethod
    def _assembly(
        coords: torch.Tensor,
        rotations: torch.Tensor | None,
        faces: torch.Tensor | None,
        n_candidates: int,
    ) -> torch.Tensor:
        if rotations is None:
            return torch.zeros(n_candidates, device=coords.device)
        if faces is None:
            faces = torch.zeros(coords.shape[0], dtype=torch.long, device=coords.device)
        if coords.shape[0] == 1 and n_candidates > 1:
            coords = coords.expand(n_candidates, -1, -1, -1)
            rotations = rotations.expand(n_candidates, -1, -1, -1)
            faces = faces.expand(n_candidates)
        elif coords.shape[0] != n_candidates:
            raise ValueError("assembly geometry batch must be 1 or match candidates")
        points = coords[:, :, 1].unsqueeze(2)
        return interface_compatibility(rotations, faces, points).nan_to_num(0.0)
