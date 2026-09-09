"""Canonical public architecture boundary for PSC-CHIMERA v2.

The supported CHIMERAv2 entry point is assembled explicitly here.  The package
initializer is side-effect free: corrected components are ordinary imports,
not runtime monkey-patches.
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
from .lie import so3_log
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
        # VelocityField does not expose n_blocks as mutable metadata; its
        # ModuleList is the authoritative architectural count.
        n_blocks = len(self.flow_model.flow_model.velocity_field.ipa_blocks)
        n_domains = self.multi_scale_designer.n_domains
        n_modules = self.multi_scale_designer.n_modules

        self.flow_model = FlowMatchingBackbone(
            d_single=d_single,
            d_pair=d_pair,
            n_blocks=n_blocks,
        )
        self.multi_scale_designer = MultiScaleNRPSDesigner(
            d_residue=d_mpnn,
            d_domain=256,
            d_module=512,
            d_assembly=256,
            n_domains=n_domains,
            n_modules=n_modules,
        )
        self.substrate_conditioner = SubstratePocketConditioner(d_pair=d_pair)
        self.pareto_head = MergeReadyParetoMultiObjectiveHead(d_model=d_mpnn)
        self.uncertainty_estimator = BayesianUncertaintyEstimator(n_samples=30)
        self._canonical_components = True

        # Re-apply the intended frozen/trainable policy after replacing child modules.
        self.freeze_pretrained()

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
    "SubstratePocketConditioner", "so3_log",
]
