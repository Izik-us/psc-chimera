"""Independent, deterministic objective evaluators.

These are transparent proxies. They must not be presented as PoET, pLDDT,
experimental activity, or assembly free energy without the corresponding
external model and calibration data.
"""

from typing import Dict, Sequence
import torch

from .geometry import validate_backbone
from .codon_optimizer import sliding_window_optimize
from .icosahedral import interface_compatibility


class BiologicalObjectiveEvaluator:
    """Compute auditable objective proxies outside the neural objective head."""

    def __init__(self, target_gc: tuple[float, float] = (0.58, 0.65), target_profile: dict[int, int] | None = None):
        self.target_gc = target_gc
        self.target_profile = target_profile or {}

    def evaluate(
        self,
        sequences: torch.Tensor,
        backbone_coords: torch.Tensor,
        rotations: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor | dict]:
        """Return one score per ``(candidate, residue sequence)`` row."""
        if sequences.ndim != 2:
            raise ValueError("sequences must have shape (N, L)")
        geometry = validate_backbone(backbone_coords, rotations)
        n = sequences.shape[0]
        if backbone_coords.shape[0] not in (1, n):
            raise ValueError("backbone batch must be 1 or match candidate count")

        expression = []
        for sequence in sequences.detach().cpu():
            aa = "ACDEFGHIKLMNPQRSTVWY"
            protein = "".join(aa[int(token)] for token in sequence)
            _, metrics = sliding_window_optimize(protein)
            gc = float(metrics["gc_content"])
            lo, hi = self.target_gc
            expression.append(max(0.0, 1.0 - max(lo - gc, gc - hi) / 0.5))

        structural = torch.full((n,), 1.0 if geometry.valid else 0.0, dtype=torch.float32)
        selectivity = self._selectivity(sequences)
        assembly = self._assembly(backbone_coords, rotations)
        return {
            "evolutionary_plausibility_proxy": self._sequence_entropy(sequences),
            "structural_validity": structural,
            "expression_proxy": torch.tensor(expression),
            "selectivity_proxy": selectivity,
            "assembly_proxy": assembly,
            "geometry": geometry.as_dict(),
        }

    @staticmethod
    def _sequence_entropy(sequences: torch.Tensor) -> torch.Tensor:
        counts = torch.nn.functional.one_hot(sequences.clamp(0, 19), 20).float().mean(dim=1)
        return -(counts.clamp_min(1e-8) * counts.clamp_min(1e-8).log()).sum(dim=-1) / 3.0

    def _selectivity(self, sequences: torch.Tensor) -> torch.Tensor:
        if not self.target_profile:
            return torch.zeros(sequences.shape[0])
        positions = torch.tensor(list(self.target_profile), dtype=torch.long)
        expected = torch.tensor(list(self.target_profile.values()), dtype=torch.long)
        valid = positions < sequences.shape[1]
        if not valid.any():
            return torch.zeros(sequences.shape[0])
        return (sequences[:, positions[valid]] == expected[valid]).float().mean(dim=-1)

    @staticmethod
    def _assembly(coords: torch.Tensor, rotations: torch.Tensor | None) -> torch.Tensor:
        if rotations is None:
            return torch.zeros(coords.shape[0])
        points = coords[:, :, 1]
        frame = rotations[:, :, :, :]
        return interface_compatibility(frame, torch.zeros(coords.shape[0], dtype=torch.long), points.unsqueeze(2)).nan_to_num(0.0)
