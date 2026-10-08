"""Fold validation. Exports are lazy so ``chimera.validation.metrics`` stays import-light
(``chimera.folding`` depends on the metrics, the validator depends on ``chimera.folding``)."""

from importlib import import_module

_EXPORTS = {
    "ConfidenceCalibrator": "calibration",
    "IsotonicCalibrator": "calibration",
    "expected_calibration_error": "calibration",
    "Decision": "fold_validator",
    "FoldValidationConfig": "fold_validator",
    "FoldValidationReport": "fold_validator",
    "FoldValidator": "fold_validator",
    "pareto_funnel": "fold_validator",
    "structural_objective_labels": "fold_validator",
}


def __getattr__(name):
    if name in _EXPORTS:
        return getattr(import_module(f".{_EXPORTS[name]}", __name__), name)
    raise AttributeError(name)


__all__ = list(_EXPORTS)
