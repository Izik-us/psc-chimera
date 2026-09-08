"""
PSC-CHIMERA
===========
Pharmacosynthetic Constructor — CHIMERA Engineering Pipeline

CHIMERA: Compositional Hierarchical Inference Model for
         Evolutionary Representation and Architecture
"""

__version__ = "2.0.0"
__author__ = "PSC Engineering Pipeline"

from chimera import chimera_v2 as _chimera_v2
from chimera.runtime_compat import (
    MergeReadyFlowMatchingBackbone,
    MergeReadyInvariantPointAttention,
    MergeReadyMultiScaleNRPSDesigner,
    MergeReadySubstratePocketConditioner,
    merge_ready_expected_improvement,
    merge_ready_so3_log,
    merge_ready_update_from_proteus,
    set_best_observed,
)
from chimera.pareto_pcgrad import MergeReadyParetoMultiObjectiveHead
from chimera.bayesian_module import BayesianUncertaintyModule
from chimera import flow_matching as _flow_matching

# The corrected implementations remain isolated at the public v2 boundary
# while the historical module definitions are being folded into their
# canonical files. No model behavior is silently changed for legacy imports.
_flow_matching.InvariantPointAttention = MergeReadyInvariantPointAttention
_flow_matching.so3_log = merge_ready_so3_log
_chimera_v2.FlowMatchingBackbone = MergeReadyFlowMatchingBackbone
_chimera_v2.MultiScaleNRPSDesigner = MergeReadyMultiScaleNRPSDesigner
_chimera_v2.SubstratePocketConditioner = MergeReadySubstratePocketConditioner
_chimera_v2.ParetoMultiObjectiveHead = MergeReadyParetoMultiObjectiveHead
_chimera_v2.BayesianUncertaintyEstimator = BayesianUncertaintyModule
_chimera_v2.CHIMERAv2.update_from_proteus = merge_ready_update_from_proteus
_chimera_v2.CHIMERAv2.set_best_observed = set_best_observed
_chimera_v2.CHIMERAv2.compute_expected_improvement = merge_ready_expected_improvement

# Historical constructor names are accepted only as compatibility aliases.
_original_chimera_init = _chimera_v2.CHIMERAv2.__init__


def _compat_chimera_init(self, *args, **kwargs):
    legacy_aliases = {
        "evoformer_layers": None,
        "flow_blocks": "n_flow_blocks",
        "mpnn_layers": None,
    }
    for legacy_name, canonical_name in legacy_aliases.items():
        if legacy_name in kwargs:
            value = kwargs.pop(legacy_name)
            if canonical_name is not None and canonical_name not in kwargs:
                kwargs[canonical_name] = value
    _original_chimera_init(self, *args, **kwargs)


_chimera_v2.CHIMERAv2.__init__ = _compat_chimera_init
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
from chimera.reproducibility import seed_everything, seed_worker, make_generator
from chimera.checkpoint import CheckpointManifest, save_manifest, load_manifest
from chimera.domain_schema import DomainType, DomainSpan, AssemblySchema
from chimera.multi_objective import (
    StructuralRetriever,
    DPOTrainer as LegacyDPOTrainer,
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

ParetoMultiObjectiveHead = MergeReadyParetoMultiObjectiveHead
BayesianUncertaintyEstimator = BayesianUncertaintyModule

__all__ = [
    "CHIMERAv2", "NRPSConstraints", "CodonOptimizer", "optimize_nrps_for_mammalian_expression",
    "ESMProteinFitnessScorer", "SE3FlowMatching", "SchrodingerBridge", "SE3SchrodingerBridge",
    "GeometryReport", "validate_backbone", "BiologicalObjectiveEvaluator",
    "PCGradOptimizer", "pcgrad_step", "project_conflicting_gradients",
    "DPOBatch", "DPOTrainer", "BayesianUncertaintyEstimator",
    "CheckpointManifest", "save_manifest", "load_manifest",
    "seed_everything", "seed_worker", "make_generator",
    "LegacyDPOTrainer", "LegacyBayesianUncertaintyEstimator", "LegacyMultiScaleNRPSDesigner",
    "DomainType", "DomainSpan", "AssemblySchema", "StructuralRetriever",
    "ParetoMultiObjectiveHead", "AutoregressiveSequencePolicy",
    "BackboneEncoder", "StructureGenerator", "SequenceDesigner", "OpenFoldAdapter",
    "OpenFoldCLIAdapter", "RFdiffusionAdapter", "RFdiffusionCLIAdapter", "ProteinMPNNAdapter",
]
