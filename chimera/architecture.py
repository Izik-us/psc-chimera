"""Canonical public architecture boundary for PSC-CHIMERA v2.

The supported CHIMERAv2 entry point is assembled explicitly here. Corrected
components are ordinary imports, not runtime monkey-patches.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from .chimera_v2 import CHIMERAv2 as _LegacyCHIMERAv2
from .bayesian import BayesianUncertaintyEstimator
from .dpo import DPOBatch, DPOTrainer
from .multi_objective import ParetoObjectives
from .pareto_pcgrad import MergeReadyParetoMultiObjectiveHead
from .pcgrad import PCGradOptimizer, pcgrad_step, project_conflicting_gradients
from .reproducibility import make_generator, seed_everything, seed_worker
from .schrodinger_bridge import SchrodingerBridge, SE3SchrodingerBridge
from .lie import so3_log
from .autoregressive_policy import AutoregressiveSequencePolicy
from .canonical_components import (
    FlowMatchingBackbone,
    MultiScaleNRPSDesigner,
    SubstratePocketConditioner,
    expected_improvement,
    update_from_proteus,
    set_best_observed,
)


class CanonicalCHIMERAv2(_LegacyCHIMERAv2):
    """CHIMERAv2 with explicit canonical component composition."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        d_single = self.evoformer.d_single
        d_pair = self.pair_connector.input_proj.out_features
        d_mpnn = self.base_mpnn.node_features
        n_blocks = getattr(self.flow_model.flow_model.velocity_field, "n_blocks", 8)
        n_domains = self.multi_scale_designer.n_domains
        n_modules = self.multi_scale_designer.n_modules

        self.flow_model = FlowMatchingBackbone(d_single=d_single, d_pair=d_pair, n_blocks=n_blocks)
        self.multi_scale_designer = MultiScaleNRPSDesigner(
            d_residue=d_mpnn, d_domain=256, d_module=512, d_assembly=256,
            n_domains=n_domains, n_modules=n_modules,
        )
        self.substrate_conditioner = SubstratePocketConditioner(d_pair=d_pair)
        self.pareto_head = MergeReadyParetoMultiObjectiveHead(d_model=d_mpnn)
        self.uncertainty_estimator = BayesianUncertaintyEstimator(n_samples=30)
        self.sequence_policy = AutoregressiveSequencePolicy(d_mpnn)
        self._canonical_components = True
        self.freeze_pretrained()

    def _autoregressive_context(self, logits: torch.Tensor) -> torch.Tensor:
        if logits.ndim != 3 or logits.shape[-1] != self.sequence_policy.vocab_size:
            raise ValueError("sequence logits must have shape (B,L,vocab_size)")
        return self._seq_to_repr(logits)

    def forward(self, *args, **kwargs):
        """Generate structures and then sample sequences from the causal policy."""
        generator = kwargs.pop("generator", None)
        temperature = kwargs.pop("temperature", 1.0)
        outputs = super().forward(*args, **kwargs)
        logits = outputs["sequences"]
        B, n_seqs, L, V = logits.shape
        context = self._autoregressive_context(logits.reshape(B * n_seqs, L, V))

        fixed_sequence = None
        constraints = kwargs.get("constraints")
        if constraints is not None and constraints.fixed_sequence is not None:
            fixed_sequence = constraints.fixed_sequence.to(logits.device)
            fixed_sequence = fixed_sequence[:, None, :].expand(B, n_seqs, L).reshape(B * n_seqs, L)

        sampled = self.sequence_policy.generate(
            context, length=L, temperature=temperature,
            generator=generator, fixed_tokens=fixed_sequence,
        ).reshape(B, n_seqs, L)
        sequence_features = self._seq_to_repr(F.one_hot(sampled, num_classes=V).to(logits.dtype))
        objective_flat = self.pareto_head(sequence_features.reshape(B * n_seqs, L, -1))
        outputs["sequence_tokens"] = sampled
        outputs["sequences"] = F.one_hot(sampled, num_classes=V).to(logits.dtype)
        outputs["evol_plausibility"] = objective_flat.evolutionary_plausibility.reshape(B, n_seqs)
        outputs["structural_stability"] = objective_flat.structural_stability.reshape(B, n_seqs)
        outputs["expression_efficiency"] = objective_flat.expression_efficiency.reshape(B, n_seqs)
        outputs["substrate_selectivity"] = objective_flat.substrate_selectivity.reshape(B, n_seqs)
        outputs["assembly_compat"] = objective_flat.assembly_compatibility.reshape(B, n_seqs)
        outputs["pareto_objectives"] = ParetoObjectives(
            evolutionary_plausibility=outputs["evol_plausibility"],
            structural_stability=outputs["structural_stability"],
            expression_efficiency=outputs["expression_efficiency"],
            substrate_selectivity=outputs["substrate_selectivity"],
            assembly_compatibility=outputs["assembly_compat"],
        )
        return outputs

    def update_from_proteus(self, *args, **kwargs):
        return update_from_proteus(self, *args, **kwargs)

    def set_best_observed(self, value: Optional[float]) -> None:
        set_best_observed(self, value)

    def compute_expected_improvement(self, mean, std, best_observed):
        return expected_improvement(self, mean, std, best_observed)


CHIMERAv2 = CanonicalCHIMERAv2

__all__ = [
    "CHIMERAv2", "CanonicalCHIMERAv2", "BayesianUncertaintyEstimator",
    "DPOBatch", "DPOTrainer", "MergeReadyParetoMultiObjectiveHead",
    "PCGradOptimizer", "pcgrad_step", "project_conflicting_gradients",
    "seed_everything", "seed_worker", "make_generator", "SchrodingerBridge",
    "SE3SchrodingerBridge", "FlowMatchingBackbone", "MultiScaleNRPSDesigner",
    "SubstratePocketConditioner", "so3_log", "AutoregressiveSequencePolicy",
]
