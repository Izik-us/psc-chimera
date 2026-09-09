"""Canonical public architecture boundary for PSC-CHIMERA v2.

The supported CHIMERAv2 entry point is assembled explicitly here.  The package
initializer must not mutate classes in other modules at import time.  Legacy
implementations remain importable for backwards compatibility, but the
canonical model is constructed through :class:`CanonicalCHIMERAv2`.
"""

from __future__ import annotations

from typing import Optional

from .chimera_v2 import CHIMERAv2 as _LegacyCHIMERAv2
from .bayesian import BayesianUncertaintyEstimator
from .dpo import DPOBatch, DPOTrainer
from .pareto_pcgrad import MergeReadyParetoMultiObjectiveHead
from .pcgrad import PCGradOptimizer, pcgrad_step, project_conflicting_gradients
from .reproducibility import make_generator, seed_everything, seed_worker
from .schrodinger_bridge import SchrodingerBridge, SE3SchrodingerBridge
from .runtime_compat import (
    MergeReadyFlowMatchingBackbone,
    MergeReadyInvariantPointAttention,
    MergeReadyMultiScaleNRPSDesigner,
    MergeReadySubstratePocketConditioner,
    merge_ready_expected_improvement,
    merge_ready_update_from_proteus,
    set_best_observed,
    merge_ready_so3_log,
)


class CanonicalCHIMERAv2(_LegacyCHIMERAv2):
    """CHIMERAv2 with explicit canonical component composition.

    This class replaces the previous import-time monkey-patching strategy.
    Corrected components are selected during construction, so importing the
    package cannot mutate the semantics of unrelated legacy modules.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Preserve the dimensions selected by the parent constructor while
        # replacing only components whose repaired implementations have a
        # different public contract.
        self.flow_model = MergeReadyFlowMatchingBackbone(
            d_single=self.evoformer.d_single,
            d_pair=self.pair_connector.input_proj.out_features,
            n_blocks=self.flow_model.flow_model.n_blocks,
        )
        self.multi_scale_designer = MergeReadyMultiScaleNRPSDesigner(
            d_residue=self.base_mpnn.node_features,
            d_domain=256,
            d_module=512,
            d_assembly=256,
            n_domains=self.multi_scale_designer.n_domains,
            n_modules=self.multi_scale_designer.n_modules,
        )
        self.substrate_conditioner = MergeReadySubstratePocketConditioner(
            d_pair=self.pair_connector.input_proj.out_features,
        )
        self.pareto_head = MergeReadyParetoMultiObjectiveHead(
            d_model=self.base_mpnn.node_features,
        )
        self.uncertainty_estimator = BayesianUncertaintyEstimator(
            n_samples=getattr(self, "n_mc_dropout", 30),
        )
        self._canonical_components = True

    def update_from_proteus(self, *args, **kwargs):
        return merge_ready_update_from_proteus(self, *args, **kwargs)

    def set_best_observed(self, value: Optional[float]) -> None:
        return set_best_observed(self, value)

    def compute_expected_improvement(self, mean, std, best_observed):
        return merge_ready_expected_improvement(self, mean, std, best_observed)


# Explicit aliases used by new code.  No class attributes are rewritten in
# legacy modules at import time.
CHIMERAv2 = CanonicalCHIMERAv2

__all__ = [
    "CHIMERAv2",
    "CanonicalCHIMERAv2",
    "BayesianUncertaintyEstimator",
    "DPOBatch",
    "DPOTrainer",
    "MergeReadyParetoMultiObjectiveHead",
    "PCGradOptimizer",
    "pcgrad_step",
    "project_conflicting_gradients",
    "seed_everything",
    "seed_worker",
    "make_generator",
    "SchrodingerBridge",
    "SE3SchrodingerBridge",
    "MergeReadyFlowMatchingBackbone",
    "MergeReadyInvariantPointAttention",
    "MergeReadyMultiScaleNRPSDesigner",
    "MergeReadySubstratePocketConditioner",
    "merge_ready_so3_log",
]
