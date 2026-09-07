"""
PSC-CHIMERA
===========
Pharmacosynthetic Constructor — CHIMERA Engineering Pipeline

CHIMERA: Compositional Hierarchical Inference Model for
         Evolutionary Representation and Architecture
"""

__version__ = "2.0.0"
__author__ = "PSC Engineering Pipeline"

# Import the implementation module first so its global component references can
# be repaired before a CHIMERAv2 instance is constructed.
from chimera import chimera_v2 as _chimera_v2
from chimera.runtime_compat import (
    MergeReadyMultiScaleNRPSDesigner,
    MergeReadySubstratePocketConditioner,
)

_chimera_v2.MultiScaleNRPSDesigner = MergeReadyMultiScaleNRPSDesigner
_chimera_v2.SubstratePocketConditioner = MergeReadySubstratePocketConditioner
CHIMERAv2 = _chimera_v2.CHIMERAv2
NRPSConstraints = _chimera_v2.NRPSConstraints

from chimera.codon_optimizer import CodonOptimizer, optimize_nrps_for_mammalian_expression
from chimera.protein_fitness import ESMProteinFitnessScorer
from chimera.flow_matching import SE3FlowMatching
from chimera.schrodinger_bridge import SchrodingerBridge, SE3SchrodingerBridge
from chimera.geometry import GeometryReport, validate_backbone
from chimera.evaluators import BiologicalObjectiveEvaluator
from chimera.pcgrad import PCGradOptimizer, pcgrad_step, project_conflicting_gradients
from chimera.dpo import DPOBatch, DPOTrainer
from chimera.bayesian import BayesianUncertaintyEstimator
from chimera.domain_schema import DomainType, DomainSpan, AssemblySchema
from chimera.multi_objective import (
    StructuralRetriever,
    DPOTrainer as LegacyDPOTrainer,
    ParetoMultiObjectiveHead,
    BayesianUncertaintyEstimator as LegacyBayesianUncertaintyEstimator,
    MultiScaleNRPSDesigner as LegacyMultiScaleNRPSDesigner,
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
    "CHIMERAv2", "NRPSConstraints", "CodonOptimizer", "optimize_nrps_for_mammalian_expression",
    "ESMProteinFitnessScorer", "SE3FlowMatching", "SchrodingerBridge", "SE3SchrodingerBridge",
    "GeometryReport", "validate_backbone", "BiologicalObjectiveEvaluator",
    "PCGradOptimizer", "pcgrad_step", "project_conflicting_gradients",
    "DPOBatch", "DPOTrainer", "BayesianUncertaintyEstimator",
    "LegacyDPOTrainer", "LegacyBayesianUncertaintyEstimator", "LegacyMultiScaleNRPSDesigner",
    "DomainType", "DomainSpan", "AssemblySchema", "StructuralRetriever",
    "ParetoMultiObjectiveHead", "AutoregressiveSequencePolicy",
    "BackboneEncoder", "StructureGenerator", "SequenceDesigner", "OpenFoldAdapter",
    "OpenFoldCLIAdapter", "RFdiffusionAdapter", "RFdiffusionCLIAdapter", "ProteinMPNNAdapter",
]
