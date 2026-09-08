"""Canonical public architecture boundary for PSC-CHIMERA v2.

This module centralizes the implementations used by the supported v2 API.
Legacy implementations remain importable for backwards compatibility, but new
code should import model components from here instead of relying on runtime
monkey-patching in ``chimera.__init__``.
"""

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
    merge_ready_so3_log,
    merge_ready_expected_improvement,
    merge_ready_update_from_proteus,
    set_best_observed,
)

__all__ = [
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
    "merge_ready_expected_improvement",
    "merge_ready_update_from_proteus",
    "set_best_observed",
]
