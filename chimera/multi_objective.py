"""Compatibility imports for the former mixed multi-objective module.

New code should import canonical sequence and objective components from their
own modules. Historical implementations remain isolated in
``legacy_optimization``; this module only preserves established import paths.
"""

from .legacy_optimization import (
    AutoregressiveSequencePolicy,
    BayesianUncertaintyEstimator,
    DPOTrainer,
    LegacyAutoregressiveSequencePolicy,
    LegacyBayesianUncertaintyEstimator,
    LegacyDPOTrainer,
    LegacyParetoMultiObjectiveHead,
    ParetoMultiObjectiveHead,
    ProteusPreferencePair,
)
from .objective_schema import ParetoObjectives
from .retrieval import StructuralRetriever
from .sequence_design import MultiScaleNRPSDesigner

__all__ = [
    "AutoregressiveSequencePolicy",
    "BayesianUncertaintyEstimator",
    "DPOTrainer",
    "LegacyAutoregressiveSequencePolicy",
    "LegacyBayesianUncertaintyEstimator",
    "LegacyDPOTrainer",
    "LegacyParetoMultiObjectiveHead",
    "MultiScaleNRPSDesigner",
    "ParetoMultiObjectiveHead",
    "ParetoObjectives",
    "ProteusPreferencePair",
    "StructuralRetriever",
]
