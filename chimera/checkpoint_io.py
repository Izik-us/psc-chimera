"""Safe reconstruction of canonical CHIMERA models from training checkpoints.

Training checkpoints remain resumable trainer artifacts. This module adds the
model-reconstruction half of that contract without changing component-transfer
checkpoint semantics.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from .checkpoint import CheckpointManifest, config_hash, state_schema_hash
from .errors import CheckpointCompatibilityError


def canonical_model_from_configuration(architecture: dict[str, Any]):
    """Instantiate a canonical model from its complete persisted configuration."""
    from .architecture import CanonicalCHIMERAv2

    constructor_config = dict(architecture)
    evoformer_config = constructor_config.pop("evoformer", None)
    if not isinstance(evoformer_config, dict):
        raise CheckpointCompatibilityError("architecture_config.evoformer must be a mapping")
    expected_evoformer_architecture = "chimera_af2_evoformer_core_v2"
    if evoformer_config.get("architecture") != expected_evoformer_architecture:
        raise CheckpointCompatibilityError("unsupported canonical EvoFormer architecture identity")
    for saved_name, model_name in (
        ("d_msa", "d_evo_single"),
        ("d_pair", "d_evo_pair"),
        ("d_single", "d_evo_single"),
    ):
        if evoformer_config.get(saved_name) != constructor_config.get(model_name):
            raise CheckpointCompatibilityError(
                f"architecture_config.evoformer.{saved_name} conflicts with {model_name}"
            )
    evoformer_keys = {
        "n_blocks": "evoformer_n_blocks",
        "gradient_checkpointing": "evoformer_gradient_checkpointing",
        "attention_chunk_size": "evoformer_attention_chunk_size",
        "opm_chunk_size": "evoformer_opm_chunk_size",
        "dropout_msa_row": "evoformer_dropout_msa_row",
        "dropout_msa_column": "evoformer_dropout_msa_column",
        "dropout_triangle": "evoformer_dropout_triangle",
        "transition_factor": "evoformer_transition_factor",
        "opm_epsilon": "evoformer_opm_epsilon",
        "n_heads_msa": "evoformer_n_heads_msa",
        "n_heads_pair": "evoformer_n_heads_pair",
        "c_hidden_opm": "evoformer_c_hidden_opm",
        "c_hidden_triangle": "evoformer_c_hidden_triangle",
        "max_relative_position": "evoformer_max_relative_position",
    }
    metadata_keys = {"architecture", "d_msa", "d_pair", "d_single"}
    unknown_evoformer_keys = set(evoformer_config) - set(evoformer_keys) - metadata_keys
    if unknown_evoformer_keys:
        raise CheckpointCompatibilityError(
            f"unsupported EvoFormer architecture fields: {sorted(unknown_evoformer_keys)}"
        )
    for saved_name, constructor_name in evoformer_keys.items():
        if saved_name not in evoformer_config:
            raise CheckpointCompatibilityError(
                f"architecture_config.evoformer is missing {saved_name}"
            )
        constructor_config[constructor_name] = evoformer_config[saved_name]
    try:
        return CanonicalCHIMERAv2(**constructor_config)
    except (TypeError, ValueError, AttributeError) as exc:
        raise CheckpointCompatibilityError(
            f"saved canonical architecture cannot be reconstructed: {exc}"
        ) from exc

def load_canonical_model_from_checkpoint(
    path: str | Path,
    *,
    map_location: str = "cpu",
):
    """Reconstruct and strictly load CanonicalCHIMERAv2 from a v2 training artifact.

    Older canonical checkpoints did not persist their full architecture
    configuration. They remain loadable through the existing caller-instantiated
    trainer/inference APIs, but cannot be reconstructed automatically.
    """
    from .inference import _safe_torch_load

    checkpoint_path = Path(path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise CheckpointCompatibilityError(
            f"canonical checkpoint does not exist: {checkpoint_path}"
        )
    try:
        payload: Any = _safe_torch_load(checkpoint_path)
    except Exception as exc:
        raise CheckpointCompatibilityError(
            f"canonical checkpoint could not be safely deserialized: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise CheckpointCompatibilityError("canonical checkpoint payload must be a mapping")
    if payload.get("artifact_type") != "canonical_training_checkpoint":
        raise CheckpointCompatibilityError(
            "automatic reconstruction requires a canonical_training_checkpoint"
        )
    if payload.get("checkpoint_schema_version") != 2:
        raise CheckpointCompatibilityError(
            "checkpoint does not contain the architecture-reconstruction schema; "
            "use the legacy caller-instantiated loading path"
        )
    architecture = payload.get("architecture_config")
    if not isinstance(architecture, dict) or not architecture:
        raise CheckpointCompatibilityError(
            "canonical checkpoint is missing architecture_config"
        )
    try:
        manifest = CheckpointManifest(**payload["manifest"])
        manifest.validate()
    except (KeyError, TypeError, ValueError) as exc:
        raise CheckpointCompatibilityError(
            f"canonical checkpoint manifest is invalid: {exc}"
        ) from exc
    actual_config_hash = config_hash(architecture)
    if manifest.config_hash != actual_config_hash:
        raise CheckpointCompatibilityError(
            "architecture_config SHA-256 does not match checkpoint manifest"
        )
    model = canonical_model_from_configuration(architecture)
    model_state = payload.get("model_state")
    if not isinstance(model_state, dict):
        raise CheckpointCompatibilityError("canonical checkpoint is missing model_state")
    actual_state_schema = state_schema_hash(model_state)
    if actual_state_schema != manifest.state_schema_hash:
        raise CheckpointCompatibilityError(
            "checkpoint model_state does not match its declared state schema"
        )
    expected_schema = state_schema_hash(model.state_dict())
    if actual_state_schema != expected_schema:
        raise CheckpointCompatibilityError(
            "reconstructed architecture state schema does not match checkpoint weights"
        )
    if config_hash(model.model_configuration()) != actual_config_hash:
        raise CheckpointCompatibilityError(
            "reconstructed model configuration does not match saved architecture identity"
        )
    try:
        model.load_state_dict(model_state, strict=True)
    except RuntimeError as exc:
        raise CheckpointCompatibilityError(
            f"strict checkpoint weight loading failed: {exc}"
        ) from exc
    training = payload.get("training")
    if not isinstance(training, dict):
        raise CheckpointCompatibilityError("canonical checkpoint is missing training metadata")
    for field in ("component_status", "objective_calibration", "objective_status"):
        if field not in training:
            raise CheckpointCompatibilityError(
                f"canonical checkpoint is missing training.{field}"
            )
    model.component_status = training["component_status"]
    model.objective_calibration = training["objective_calibration"]
    model.objective_status = training["objective_status"]
    model.eval()
    return model


def checkpoint_file_sha256(path: str | Path) -> str:
    """Return the SHA-256 digest of the exact serialized checkpoint bytes."""
    digest = hashlib.sha256()
    with Path(path).expanduser().resolve().open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = ["canonical_model_from_configuration", "checkpoint_file_sha256", "load_canonical_model_from_checkpoint"]
