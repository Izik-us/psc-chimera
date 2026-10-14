"""One fail-closed readiness policy shared by direct and wrapped inference."""

from __future__ import annotations

from typing import Any, Mapping


READINESS_POLICIES = frozenset({"exploratory", "require_trained"})


def apply_inference_readiness_policy(
    readiness: Mapping[str, Any],
    *,
    allow_experimental: bool,
    context: str = "Canonical CHIMERA",
) -> dict[str, Any]:
    """Validate a readiness report and enforce the shared strict/exploratory policy."""
    if not isinstance(readiness, Mapping):
        raise TypeError("readiness must be a mapping")
    if not isinstance(allow_experimental, bool):
        raise TypeError("allow_experimental must be a boolean")
    state = readiness.get("state")
    if not isinstance(state, str) or not state:
        state = "UNKNOWN"
    if state != "TRAINED" and not allow_experimental:
        raise RuntimeError(
            f"{context} is {state}; production inference requires trained components "
            "and calibrated objectives. Use explicit experimental mode to proceed."
        )
    return dict(readiness)


def exploratory_readiness_warning(readiness: Mapping[str, Any]) -> str | None:
    """Return the common warning used when research inference is allowed."""
    state = readiness.get("state", "UNKNOWN")
    if state == "TRAINED":
        return None
    if not isinstance(state, str) or not state:
        state = "UNKNOWN"
    return (
        f"model readiness is {state}; this result is exploratory, "
        "not production-valid"
    )


__all__ = [
    "READINESS_POLICIES",
    "apply_inference_readiness_policy",
    "exploratory_readiness_warning",
]
