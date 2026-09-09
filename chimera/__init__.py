"""
PSC-CHIMERA
===========
Pharmacosynthetic Constructor — CHIMERA Engineering Pipeline

CHIMERA: Compositional Hierarchical Inference Model for
         Evolutionary Representation and Architecture
"""

__version__ = "2.0.0"
__author__ = "PSC Engineering Pipeline"

# The package boundary is intentionally side-effect free. Importing chimera
# only exposes public classes and performs no import-time module mutation.
from .architecture import CHIMERAv2, CanonicalCHIMERAv2
from .chimera_v2 import NRPSConstraints
from .codon_optimizer import CodonOptimizer, optimize_nrps_for_mammalian_expression
from .protein_fitness import ESMProteinFitnessScorer
from .flow_matching import SE3FlowMatching
from .schrodinger_bridge import SchrodingerBridge, SE3SchrodingerBridge
from .geometry import GeometryReport, validate_backbone
from .evaluators import BiologicalObjectiveEvaluator
from .pcgrad import PCGradOptimizer, pcgrad_step, project_conflicting_gradients
from .dpo import DPOBatch, DPOTrainer
from .bayesian import BayesianUncertaintyEstimator
from .reproducibility import seed_everything, seed_worker, make_generator
from .checkpoint import CheckpointManifest, save_manifest, load_manifest
from .domain_schema import DomainType, DomainSpan, AssemblySchema
from .multi_objective import StructuralRetriever
from .autoregressive_policy import AutoregressiveSequencePolicy
from .pareto_pcgrad import MergeReadyParetoMultiObjectiveHead
from .adapters import (
    BackboneEncoder,
    StructureGenerator,
    SequenceDesigner,
    OpenFoldAdapter,
    OpenFoldCLIAdapter,
    RFdiffusionAdapter,
    RFdiffusionCLIAdapter,
    ProteinMPNNAdapter,
)

# Public name retained for callers using the historical package API.
ParetoMultiObjectiveHead = MergeReadyParetoMultiObjectiveHead

__all__ = [
    "CHIMERAv2", "CanonicalCHIMERAv2", "NRPSConstraints",
    "CodonOptimizer", "optimize_nrps_for_mammalian_expression",
    "ESMProteinFitnessScorer", "SE3FlowMatching", "SchrodingerBridge",
    "SE3SchrodingerBridge", "GeometryReport", "validate_backbone",
    "BiologicalObjectiveEvaluator", "PCGradOptimizer", "pcgrad_step",
    "project_conflicting_gradients", "DPOBatch", "DPOTrainer",
    "BayesianUncertaintyEstimator", "CheckpointManifest", "save_manifest",
    "load_manifest", "seed_everything", "seed_worker", "make_generator",
    "DomainType", "DomainSpan", "AssemblySchema", "StructuralRetriever",
    "ParetoMultiObjectiveHead", "AutoregressiveSequencePolicy",
    "BackboneEncoder", "StructureGenerator", "SequenceDesigner",
    "OpenFoldAdapter", "OpenFoldCLIAdapter", "RFdiffusionAdapter",
    "RFdiffusionCLIAdapter", "ProteinMPNNAdapter",
]
