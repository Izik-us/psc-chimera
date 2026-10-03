"""Versioned semantic contracts for CHIMERA objective predictions and targets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch

from .checkpoint import config_hash


OBJECTIVE_SCHEMA_VERSION = "objective-schema-v2"


@dataclass
class ParetoObjectives:
    """Named objective predictions plus the channels available on a candidate."""

    evolutionary_plausibility: torch.Tensor
    structural_stability: torch.Tensor
    expression_efficiency: torch.Tensor
    substrate_selectivity: torch.Tensor
    assembly_compatibility: torch.Tensor
    available_objectives: tuple[str, ...] | None = None


OBJECTIVE_SCHEMA: dict[str, dict[str, Any]] = {
    "evolutionary_plausibility": {
        "version": OBJECTIVE_SCHEMA_VERSION,
        "feature_sources": ("sequence", "evolutionary"),
        "prediction_shape": ("B",),
        "target_shape": ("B",),
        "dtype": "torch.float32",
        "finite": True,
        "target_type": "normalized_sequence_entropy_proxy",
        "target_normalization": "divide by log(20)",
        "target_scale": "unit interval",
        "target_range": (0.0, 1.0),
        "activation": "identity",
        "loss": "mean_squared_error",
        "loss_config": {"reduction": "mean"},
        "label_kinds": ("proxy", "surrogate", "validated_surrogate", "deterministic_evaluator", "experimental_measurement"),
        "biological_measurement": False,
        "calibration": "heldout_required_for_uncertainty",
    },
    "structural_stability": {
        "version": OBJECTIVE_SCHEMA_VERSION,
        "feature_sources": ("sequence", "structural"),
        "prediction_shape": ("B",),
        "target_shape": ("B",),
        "dtype": "torch.float32",
        "finite": True,
        "target_type": "structure_quality_proxy_or_explicit_stability_label",
        "target_normalization": "none; target is on the 0-100 model output scale",
        "target_scale": "0-100",
        "target_range": (0.0, 100.0),
        "activation": "100 * sigmoid",
        "loss": "mean_squared_error",
        "loss_config": {"reduction": "mean"},
        "label_kinds": ("proxy", "surrogate", "validated_surrogate", "deterministic_evaluator", "experimental_measurement"),
        "biological_measurement": False,
        "calibration": "heldout_required_for_uncertainty",
    },
    "expression_efficiency": {
        "version": OBJECTIVE_SCHEMA_VERSION,
        "feature_sources": ("sequence",),
        "prediction_shape": ("B",),
        "target_shape": ("B",),
        "dtype": "torch.float32",
        "finite": True,
        "target_type": "normalized_expression_label_or_proxy",
        "target_normalization": "caller supplies values on the unit interval",
        "target_scale": "unit interval",
        "target_range": (0.0, 1.0),
        "activation": "sigmoid",
        "loss": "binary_cross_entropy",
        "loss_config": {"reduction": "mean"},
        "label_kinds": ("proxy", "surrogate", "validated_surrogate", "deterministic_evaluator", "experimental_measurement"),
        "biological_measurement": False,
        "calibration": "heldout_required_for_uncertainty",
    },
    "substrate_selectivity": {
        "version": OBJECTIVE_SCHEMA_VERSION,
        "feature_sources": ("sequence", "substrate"),
        "prediction_shape": ("B",),
        "target_shape": ("B",),
        "dtype": "torch.float32",
        "finite": True,
        "target_type": "target_profile_match_proxy_or_explicit_selectivity_label",
        "target_normalization": "caller supplies values on the unit interval",
        "target_scale": "unit interval",
        "target_range": (0.0, 1.0),
        "activation": "sigmoid",
        "loss": "mean_squared_error",
        "loss_config": {"reduction": "mean"},
        "label_kinds": ("proxy", "surrogate", "validated_surrogate", "deterministic_evaluator", "experimental_measurement"),
        "biological_measurement": False,
        "calibration": "heldout_required_for_uncertainty",
    },
    "assembly_compatibility": {
        "version": OBJECTIVE_SCHEMA_VERSION,
        "feature_sources": ("structural",),
        "prediction_shape": ("B",),
        "target_shape": ("B",),
        "dtype": "torch.float32",
        "finite": True,
        "target_type": "icosahedral_interface_geometry_proxy",
        "target_normalization": "deterministic evaluator output in unit interval",
        "target_scale": "unit interval",
        "target_range": (0.0, 1.0),
        "activation": "none; direct deterministic evaluator",
        "loss": "none; deterministic inference-time objective",
        "loss_config": {},
        "label_kinds": ("deterministic_evaluator",),
        "biological_measurement": False,
        "calibration": "not_applicable_to_deterministic_evaluator",
    },
}


def objective_schema_hash(schema: Mapping[str, Mapping[str, Any]] | None = None) -> str:
    """Fingerprint objective meaning, not just tensor shape or feature widths."""
    return config_hash(schema or OBJECTIVE_SCHEMA)


def validate_objective_target(
    name: str,
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> None:
    """Enforce the objective's exact batch-vector, dtype, finite, and range contract."""
    try:
        schema = OBJECTIVE_SCHEMA[name]
    except KeyError as exc:
        raise ValueError(f"unknown objective target {name!r}") from exc
    if schema["loss"].startswith("none;"):
        raise ValueError(f"{name} is deterministic-only and cannot enter a neural objective loss")
    if prediction.ndim != 1:
        raise ValueError(f"{name} prediction must have shape (B,)")
    if target.ndim != 1 or target.shape != prediction.shape:
        raise ValueError(f"{name} target must have exact shape {tuple(prediction.shape)}")
    if prediction.dtype != torch.float32 or target.dtype != torch.float32:
        raise TypeError(f"{name} prediction and target must both have dtype torch.float32")
    if not torch.isfinite(prediction).all() or not torch.isfinite(target).all():
        raise ValueError(f"{name} prediction and target must be finite")
    low, high = schema["target_range"]
    if torch.any((target < low) | (target > high)):
        raise ValueError(f"{name} target must be within [{low}, {high}]")