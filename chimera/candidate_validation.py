"""Fail-closed computational validation for individual CHIMERA candidates."""

from __future__ import annotations

from typing import Any

import torch


def validate_candidate(
    *,
    sequence_tokens: torch.Tensor,
    backbone_coordinates: torch.Tensor,
    geometry_requested: bool,
    geometry_valid: bool | None,
    geometry_metrics: dict[str, Any] | None = None,
    constraints: Any | None = None,
) -> dict[str, Any]:
    """Validate sequence/coordinate contracts and produce ranking disposition.

    This is a computational sanity gate. It does not establish folding,
    stability, expression, substrate activity, or biological validity.
    """
    if not isinstance(geometry_requested, bool):
        raise TypeError("geometry_requested must be a boolean")

    tokens_ok = (
        torch.is_tensor(sequence_tokens)
        and sequence_tokens.ndim == 1
        and sequence_tokens.dtype
        in {torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64}
        and sequence_tokens.numel() > 0
        and bool(torch.all((sequence_tokens >= 0) & (sequence_tokens < 20)))
    )
    length = (
        int(sequence_tokens.shape[0])
        if torch.is_tensor(sequence_tokens) and sequence_tokens.ndim
        else 0
    )
    shape_ok = (
        torch.is_tensor(backbone_coordinates)
        and backbone_coordinates.ndim == 3
        and tuple(backbone_coordinates.shape) == (length, 4, 3)
    )
    finite_ok = (
        shape_ok
        and torch.is_floating_point(backbone_coordinates)
        and bool(torch.isfinite(backbone_coordinates).all())
    )

    checks: dict[str, str] = {
        "sequence_token_range": "PASS" if tokens_ok else "FAIL",
        "backbone_shape": "PASS" if shape_ok else "FAIL",
        "backbone_finiteness": "PASS" if finite_ok else "FAIL",
    }

    fixed_sequence = getattr(constraints, "fixed_sequence", None)
    if fixed_sequence is None:
        checks["fixed_sequence_constraints"] = "NOT_APPLICABLE"
    elif not tokens_ok or not torch.is_tensor(fixed_sequence):
        checks["fixed_sequence_constraints"] = "FAIL"
    else:
        expected = fixed_sequence.detach().to(
            device=sequence_tokens.device, dtype=torch.long
        )
        if expected.ndim == 2 and expected.shape[0] == 1:
            expected = expected[0]
        if expected.shape != sequence_tokens.shape:
            checks["fixed_sequence_constraints"] = "FAIL"
        else:
            fixed_positions = expected >= 0
            specified_are_valid = bool(
                torch.all(
                    (expected[fixed_positions] >= 0)
                    & (expected[fixed_positions] < 20)
                )
            )
            fixed_positions_match = bool(
                torch.equal(
                    sequence_tokens[fixed_positions].long(),
                    expected[fixed_positions],
                )
            )
            checks["fixed_sequence_constraints"] = (
                "PASS" if specified_are_valid and fixed_positions_match else "FAIL"
            )

    if geometry_requested:
        checks["geometry_sanity"] = "PASS" if geometry_valid is True else "FAIL"
    else:
        checks["geometry_sanity"] = "NOT_REQUESTED"

    accepted_states = {"PASS", "NOT_APPLICABLE", "NOT_REQUESTED"}
    eligible = all(state in accepted_states for state in checks.values())
    reasons = [name for name, state in checks.items() if state == "FAIL"]
    return {
        "status": "VALIDATED" if geometry_requested else "NOT_REQUESTED",
        "valid": geometry_valid if geometry_requested else None,
        "type": (
            "canonical_backbone_geometry_sanity" if geometry_requested else None
        ),
        "scientific_validation": "NOT ESTABLISHED",
        "metrics": geometry_metrics if geometry_requested else None,
        "checks": checks,
        "eligible_for_ranking": eligible,
        "disposition": (
            "ACCEPTED_FOR_COMPUTATIONAL_RANKING" if eligible else "REJECTED"
        ),
        "rejection_reasons": reasons,
    }
