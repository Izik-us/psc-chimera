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
__author__ = "PSC Engineering Pipeline"

from chimera.chimera_v2 import CHIMERAv2, NRPSConstraints
from chimera.codon_optimizer import (
    CodonOptimizer,
    optimize_nrps_for_mammalian_expression,
)
from chimera.flow_matching import SE3FlowMatching
from chimera.geometry import GeometryReport, validate_backbone
from chimera.evaluators import BiologicalObjectiveEvaluator
from chimera.pcgrad import PCGradOptimizer, project_conflicting_gradients
from chimera.domain_schema import DomainType, DomainSpan, AssemblySchema
from chimera.multi_objective import (
    StructuralRetriever,
    DPOTrainer,
    ParetoMultiObjectiveHead,
    BayesianUncertaintyEstimator,
    MultiScaleNRPSDesigner,
    AutoregressiveSequencePolicy,
)
from chimera.adapters import (
    BackboneEncoder,
    StructureGenerator,
    SequenceDesigner,
    OpenFoldAdapter,
    OpenFoldCLIAdapter,
    RFdiffusionAdapter,
    RFdiffusionCLIAdapter,
    ProteinMPNNAdapter,
)

__all__ = [
    "CHIMERAv2",
    "NRPSConstraints",
    "CodonOptimizer",
    "optimize_nrps_for_mammalian_expression",
    "SE3FlowMatching",
    "GeometryReport",
    "validate_backbone",
    "BiologicalObjectiveEvaluator",
    "PCGradOptimizer",
    "project_conflicting_gradients",
    "DomainType",
    "DomainSpan",
    "AssemblySchema",
    "StructuralRetriever",
    "DPOTrainer",
    "ParetoMultiObjectiveHead",
    "BayesianUncertaintyEstimator",
    "MultiScaleNRPSDesigner",
    "AutoregressiveSequencePolicy",
    "BackboneEncoder",
    "StructureGenerator",
    "SequenceDesigner",
    "OpenFoldAdapter",
    "OpenFoldCLIAdapter",
    "RFdiffusionAdapter",
    "RFdiffusionCLIAdapter",
    "ProteinMPNNAdapter",
]
