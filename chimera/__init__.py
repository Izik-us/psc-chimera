"""
PSC-CHIMERA
===========
Pharmacosynthetic Constructor — CHIMERA Engineering Pipeline

CHIMERA: Compositional Hierarchical Inference Model for
         Evolutionary Representation and Architecture
"""

from ._version import __version__
from .configuration import InferenceConfig, CONFIGURATION_SCHEMA_VERSION
from .errors import (
    ArtifactError,
    CheckpointCompatibilityError,
    ChimeraError,
    ConfigurationError,
    InferenceError,
    InputValidationError,
    IntegrityError,
    ProvenanceError,
    ResourceError,
    UnsupportedRuntimeError,
)
from .production_gate import run_production_gate

__author__ = "PSC Engineering Pipeline"

# The package boundary is intentionally side-effect free. Importing chimera
# only exposes public classes and performs no import-time module mutation.
from .architecture import CHIMERAv2, CanonicalCHIMERAv2
from .domain_schema import NRPSConstraints
from .codon_optimizer import CodonOptimizer, optimize_nrps_for_mammalian_expression
from .protein_fitness import ESMProteinFitnessScorer
from .flow_matching import SE3FlowMatching, FlowMatchingBackbone, InvariantPointAttention
from .schrodinger_bridge import SchrodingerBridge, SE3SchrodingerBridge
from .geometry import GeometryReport, validate_backbone
from .evaluators import BiologicalObjectiveEvaluator, ObjectiveOutput
from .pcgrad import PCGradOptimizer, pcgrad_step, project_conflicting_gradients
from .dpo import DPOBatch, DPOTrainer
from .bayesian import BayesianUncertaintyEstimator
from .reproducibility import seed_everything, seed_worker, make_generator
from .checkpoint import CheckpointManifest, save_manifest, load_manifest, config_hash, state_schema_hash, validate_checkpoint_compatibility
from .domain_schema import DomainType, DomainSpan, AssemblySchema
from .retrieval import StructuralRetriever
from .sequence_design import MultiScaleNRPSDesigner
from .conditioning import SubstratePocketConditioner
from .components import MSARepresentationBackbone
from .autoregressive_policy import AutoregressiveSequencePolicy
from .pareto_pcgrad import (
    MergeReadyParetoMultiObjectiveHead,
    ObjectiveFeatureEncoder,
    ObjectiveFeatureSource,
    OBJECTIVE_FEATURE_PLAN,
)
from .training import (
    CanonicalTrainingBatch,
    CanonicalTrainer,
    ObjectiveLabelKind,
    TrainingRegime,
    gradient_flow_report,
)
from .objective_schema import (
    OBJECTIVE_SCHEMA,
    OBJECTIVE_SCHEMA_VERSION,
    objective_schema_hash,
    validate_objective_target,
)
from .readiness import PreProductionGateEvidence, preproduction_readiness_report
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
    "FlowMatchingBackbone", "InvariantPointAttention", "MultiScaleNRPSDesigner",
    "SubstratePocketConditioner",
    "MSARepresentationBackbone",
    "SE3SchrodingerBridge", "GeometryReport", "validate_backbone",
    "BiologicalObjectiveEvaluator", "ObjectiveOutput", "PCGradOptimizer", "pcgrad_step",
    "project_conflicting_gradients", "DPOBatch", "DPOTrainer",
    "BayesianUncertaintyEstimator", "CheckpointManifest", "save_manifest",
    "load_manifest", "config_hash", "state_schema_hash", "validate_checkpoint_compatibility",
    "seed_everything", "seed_worker", "make_generator",
    "DomainType", "DomainSpan", "AssemblySchema", "StructuralRetriever",
    "ParetoMultiObjectiveHead", "AutoregressiveSequencePolicy",
    "ObjectiveFeatureEncoder", "ObjectiveFeatureSource", "OBJECTIVE_FEATURE_PLAN",
    "BackboneEncoder", "StructureGenerator", "SequenceDesigner",
    "OpenFoldAdapter", "OpenFoldCLIAdapter", "RFdiffusionAdapter",
    "RFdiffusionCLIAdapter", "ProteinMPNNAdapter",
    "CanonicalTrainingBatch", "CanonicalTrainer", "TrainingRegime", "ObjectiveLabelKind",
    "gradient_flow_report",
    "OBJECTIVE_SCHEMA", "OBJECTIVE_SCHEMA_VERSION", "objective_schema_hash",
    "validate_objective_target", "PreProductionGateEvidence", "preproduction_readiness_report",
    "__version__", "InferenceConfig", "CONFIGURATION_SCHEMA_VERSION",
    "ChimeraError", "ConfigurationError", "ArtifactError", "CheckpointCompatibilityError",
    "InputValidationError", "UnsupportedRuntimeError", "ResourceError", "InferenceError",
    "ProvenanceError", "IntegrityError", "run_production_gate",
]
