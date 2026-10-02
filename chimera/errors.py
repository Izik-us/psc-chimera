"""Small, explicit error taxonomy for production-facing CHIMERA boundaries."""

from __future__ import annotations


class ChimeraError(RuntimeError):
    """Base class for failures with a stable production category."""

    category = "chimera_error"
    retryable = False

    def __init__(self, message: str, *, corrective_action: str | None = None) -> None:
        super().__init__(message)
        self.corrective_action = corrective_action


class ConfigurationError(ChimeraError):
    category = "configuration_error"


class ArtifactError(ChimeraError):
    category = "artifact_error"


class IntegrityError(ArtifactError):
    category = "integrity_error"


class CheckpointCompatibilityError(ArtifactError):
    category = "checkpoint_compatibility_error"


class InputValidationError(ChimeraError, ValueError):
    category = "input_validation_error"


class UnsupportedRuntimeError(ChimeraError):
    category = "unsupported_runtime_error"


class ResourceError(ChimeraError):
    category = "resource_error"
    retryable = True


class InferenceError(ChimeraError):
    category = "inference_error"


class ProvenanceError(ArtifactError):
    category = "provenance_error"
