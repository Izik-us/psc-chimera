"""Canonical preparation boundary for CHIMERA inference inputs.

This module deliberately does not invent an MSA or a starting structure. Those
artifacts come from an upstream, versioned preprocessing/structure provider.
The purpose of this boundary is to make their provenance and tensor contracts
explicit before the neural model is allowed to run.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any, Mapping

import torch

from .checkpoint import config_hash
from .errors import InputValidationError


def validate_rotation_frames(rotations: torch.Tensor, *, tolerance: float = 5e-3) -> None:
    """Require finite, proper rotation matrices rather than shape-compatible tensors."""
    if not torch.is_tensor(rotations) or rotations.ndim < 2 or rotations.shape[-2:] != (3, 3):
        raise InputValidationError("source_rotations must end in (3,3)")
    if not torch.is_floating_point(rotations) or not torch.isfinite(rotations).all():
        raise InputValidationError("source_rotations must contain finite floating-point values")
    check_dtype = torch.float64 if rotations.dtype == torch.float64 else torch.float32
    value = rotations.detach().to(dtype=check_dtype)
    identity = torch.eye(3, dtype=value.dtype, device=value.device).expand_as(value)
    gram_error = torch.linalg.matrix_norm(
        value.transpose(-1, -2) @ value - identity, dim=(-2, -1)
    )
    determinant = torch.linalg.det(value)
    if torch.any(gram_error > tolerance) or torch.any(
        torch.abs(determinant - 1.0) > tolerance
    ):
        raise InputValidationError(
            "source_rotations must be numerically valid SO(3) frames "
            "(orthogonality and determinant checks failed)"
        )


@dataclass(frozen=True)
class PreparedInputManifest:
    """Identity of the upstream artifacts used to construct an inference request."""

    preprocessing_id: str
    sequence_id: str
    msa_source: str
    pair_feature_source: str
    structure_source: str
    structure_id: str
    residue_mapping_id: str
    schema_version: int = 1

    def __post_init__(self) -> None:
        for name in (
            "preprocessing_id",
            "sequence_id",
            "msa_source",
            "pair_feature_source",
            "structure_source",
            "structure_id",
            "residue_mapping_id",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise InputValidationError(f"{name} must be a non-empty identity string")
        if self.schema_version != 1:
            raise InputValidationError("unsupported PreparedInputManifest schema_version")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "preprocessing_id": self.preprocessing_id,
            "sequence_id": self.sequence_id,
            "msa_source": self.msa_source,
            "pair_feature_source": self.pair_feature_source,
            "structure_source": self.structure_source,
            "structure_id": self.structure_id,
            "residue_mapping_id": self.residue_mapping_id,
        }

    @property
    def identity(self) -> str:
        return config_hash(self.to_dict())


@dataclass(frozen=True)
class PreparedInferenceInputs:
    """Validated tensors plus the identity of the upstream preparation pipeline."""

    msa_tokens: torch.Tensor
    pair_features: torch.Tensor
    source_rotations: torch.Tensor
    source_translations: torch.Tensor
    manifest: PreparedInputManifest
    msa_padding_mask: torch.Tensor | None = None
    residue_mask: torch.Tensor | None = None
    residue_index: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if self.msa_tokens.ndim != 3:
            raise InputValidationError("msa_tokens must have shape (B,N,L)")
        batch, _, length = self.msa_tokens.shape
        if batch < 1 or length < 2:
            raise InputValidationError("MSA must contain at least one batch item and two residues")
        if self.msa_tokens.dtype not in {
            torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64
        }:
            raise InputValidationError("msa_tokens must use an integer dtype")
        if self.msa_tokens.numel() and (
            int(self.msa_tokens.min()) < 0 or int(self.msa_tokens.max()) > 23
        ):
            raise InputValidationError("msa_tokens contain token IDs outside [0,23]")
        if self.pair_features.ndim != 4 or self.pair_features.shape[:3] != (batch, length, length):
            raise InputValidationError("pair_features must have shape (B,L,L,d_pair)")
        if self.source_rotations.shape != (batch, length, 3, 3):
            raise InputValidationError("source_rotations must have shape (B,L,3,3)")
        if self.source_translations.shape != (batch, length, 3):
            raise InputValidationError("source_translations must have shape (B,L,3)")
        for name, tensor in (
            ("pair_features", self.pair_features),
            ("source_rotations", self.source_rotations),
            ("source_translations", self.source_translations),
        ):
            if not torch.is_floating_point(tensor) or not torch.isfinite(tensor).all():
                raise InputValidationError(f"{name} must contain finite floating-point values")
        if self.msa_padding_mask is not None and (
            self.msa_padding_mask.shape != self.msa_tokens.shape
            or self.msa_padding_mask.dtype != torch.bool
        ):
            raise InputValidationError("msa_padding_mask must be bool with shape (B,N,L)")
        if self.residue_mask is not None and (
            self.residue_mask.shape != (batch, length)
            or self.residue_mask.dtype != torch.bool
        ):
            raise InputValidationError("residue_mask must be bool with shape (B,L)")
        if self.residue_index is not None and self.residue_index.shape not in {(length,), (batch, length)}:
            raise InputValidationError("residue_index must have shape (L,) or (B,L)")
        if self.residue_index is not None and self.residue_index.dtype not in {
            torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64
        }:
            raise InputValidationError("residue_index must use an integer dtype")
        if self.residue_index is not None and batch > 1 and self.residue_index.shape == (length,):
            object.__setattr__(self, "residue_index", self.residue_index.unsqueeze(0).expand(batch, -1))
        self._validate_frames()

    def _validate_frames(self) -> None:
        """Check that source rotations are proper SO(3) matrices."""
        validate_rotation_frames(self.source_rotations)

    @property
    def batch_size(self) -> int:
        return int(self.msa_tokens.shape[0])

    @property
    def length(self) -> int:
        return int(self.msa_tokens.shape[-1])

    @staticmethod
    def _tensor_identity(value: torch.Tensor | None) -> str | None:
        if value is None:
            return None
        tensor = value.detach().cpu().contiguous()
        digest = hashlib.sha256()
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("utf-8"))
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        return digest.hexdigest()

    @property
    def identity(self) -> str:
        """Hash both preparation provenance and every tensor's actual contents."""
        payload = {
            "manifest": self.manifest.to_dict(),
            "tensors": {
                name: self._tensor_identity(getattr(self, name))
                for name in (
                    "msa_tokens",
                    "pair_features",
                    "source_rotations",
                    "source_translations",
                    "msa_padding_mask",
                    "residue_mask",
                    "residue_index",
                )
            },
        }
        return config_hash(payload)


def prepare_inference_inputs(
    *,
    msa_tokens: torch.Tensor,
    pair_features: torch.Tensor,
    source_rotations: torch.Tensor,
    source_translations: torch.Tensor,
    manifest: PreparedInputManifest,
    msa_padding_mask: torch.Tensor | None = None,
    residue_mask: torch.Tensor | None = None,
    residue_index: torch.Tensor | None = None,
) -> PreparedInferenceInputs:
    """Validate and package upstream preprocessing outputs.

    The function intentionally requires a source backbone. It will not fabricate
    identity frames or zero pair features and label them as a biological input.
    A future structure-prediction adapter can feed this contract explicitly.
    """
    return PreparedInferenceInputs(
        msa_tokens=msa_tokens,
        pair_features=pair_features,
        source_rotations=source_rotations,
        source_translations=source_translations,
        manifest=manifest,
        msa_padding_mask=msa_padding_mask,
        residue_mask=residue_mask,
        residue_index=residue_index,
    )




def build_inference_request(prepared: PreparedInferenceInputs, config, **kwargs):
    """Convert validated preparation output into the public InferenceRequest.

    ``preprocessing_id`` is taken from the preparation manifest and cannot be
    silently replaced by a caller-provided value.
    """
    from .inference import InferenceRequest

    if config.preprocessing_id != prepared.manifest.preprocessing_id:
        raise InputValidationError(
            "InferenceConfig.preprocessing_id does not match the prepared-input manifest"
        )
    reserved = {"prepared_input_manifest", "prepared_input_identity"} & set(kwargs)
    if reserved:
        raise InputValidationError(
            "prepared-input provenance fields are assigned by build_inference_request"
        )
    return InferenceRequest(
        msa_tokens=prepared.msa_tokens,
        initial_pair_features=prepared.pair_features,
        source_rotations=prepared.source_rotations,
        source_translations=prepared.source_translations,
        config=config,
        msa_padding_mask=prepared.msa_padding_mask,
        residue_mask=prepared.residue_mask,
        residue_index=prepared.residue_index,
        prepared_input_manifest=prepared.manifest.to_dict(),
        prepared_input_identity=prepared.identity,
        **kwargs,
    )
