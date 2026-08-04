"""
PSC-CHIMERA
===========
Pharmacosynthetic Constructor — CHIMERA Engineering Pipeline

CHIMERA: Compositional Hierarchical Inference Model for
         Evolutionary Representation and Architecture

Stage 1 computational design component of the PSC Engineering Pipeline.
Designs NRPS A-domain sequences for mammalian intracellular expression
and icosahedral self-assembly within the PSC.
"""

__version__ = "2.0.0"
__author__  = "PSC Engineering Pipeline"

from chimera.chimera_v2 import CHIMERAv2, NRPSConstraints
from chimera.codon_optimizer import CodonOptimizer, optimize_nrps_for_mammalian_expression
from chimera.flow_matching import SE3FlowMatching
from chimera.multi_objective import (
    StructuralRetriever,
    DPOTrainer,
    ParetoMultiObjectiveHead,
    BayesianUncertaintyEstimator,
    MultiScaleNRPSDesigner,
)

__all__ = [
    "CHIMERAv2",
    "NRPSConstraints",
    "CodonOptimizer",
    "optimize_nrps_for_mammalian_expression",
    "SE3FlowMatching",
    "StructuralRetriever",
    "DPOTrainer",
    "ParetoMultiObjectiveHead",
    "BayesianUncertaintyEstimator",
    "MultiScaleNRPSDesigner",
]
