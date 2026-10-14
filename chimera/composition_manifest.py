"""Reproducible research-composition manifests for canonical CHIMERA."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import torch

from ._version import __version__
from .checkpoint import config_hash, runtime_provenance
from .objective_schema import OBJECTIVE_SCHEMA_VERSION, objective_schema_hash


def _state_dict_digest(state_dict: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state_dict):
        value = state_dict[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("utf-8"))
        digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("utf-8"))
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _json_digest(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_composition_manifest(
    model: torch.nn.Module,
    config: Any,
    *,
    prepared_inputs: Any | None = None,
    checkpoint_sha256: str | None = None,
    validation_policy: Mapping[str, Any] | None = None,
    validation_evidence: Mapping[str, Any] | None = None,
    optional_backends: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Describe the exact runnable research composition without granting release approval."""
    if not hasattr(config, "to_dict") or not hasattr(config, "config_id"):
        raise TypeError("config must be a canonical InferenceConfig")
    if not hasattr(model, "model_configuration") or not hasattr(model, "state_dict"):
        raise TypeError("model must expose canonical model configuration and state")

    prepared_identity = None
    if prepared_inputs is not None:
        input_manifest = getattr(prepared_inputs, "manifest", None)
        prepared_preprocessing_id = getattr(input_manifest, "preprocessing_id", None)
        if prepared_preprocessing_id != config.preprocessing_id:
            raise ValueError(
                "InferenceConfig.preprocessing_id does not match prepared-input provenance"
            )
        prepared_identity = getattr(prepared_inputs, "identity", None)
        if not isinstance(prepared_identity, str) or not prepared_identity:
            raise ValueError("prepared_inputs must expose a stable identity")

    root = Path(__file__).resolve().parent
    dependency_path = root / "architecture_dependencies.json"
    dependency_bytes = dependency_path.read_bytes() if dependency_path.is_file() else b""
    dependency_manifest: dict[str, Any] = {}
    if dependency_bytes:
        dependency_manifest = json.loads(dependency_bytes.decode("utf-8"))

    retrieval = getattr(model, "structural_retriever", None)
    retrieval_embeddings = getattr(retrieval, "index_embs", None)
    retrieval_index_present = (
        retrieval is not None
        and retrieval_embeddings is not None
        and len(getattr(retrieval, "index_meta", [])) > 0
    )
    readiness = (
        model.inference_readiness()
        if callable(getattr(model, "inference_readiness", None))
        else {"state": "UNKNOWN"}
    )
    component_status = getattr(model, "component_status", {})
    objective_provenance = (
        model.objective_provenance()
        if callable(getattr(model, "objective_provenance", None))
        else {}
    )

    manifest: dict[str, Any] = {
        "artifact_type": "chimera_composition_manifest",
        "schema_version": 1,
        "software": {"version": __version__},
        "objective_schema": {
            "version": OBJECTIVE_SCHEMA_VERSION,
            "sha256": objective_schema_hash(),
        },
        "composition": {
            "id": "canonical-chimerav2-research-composition-v1",
            "architecture_class": (
                f"{model.__class__.__module__}.{model.__class__.__qualname__}"
            ),
            "architecture_configuration": model.model_configuration(),
            "architecture_configuration_sha256": config_hash(
                model.model_configuration()
            ),
            "model_state_sha256": _state_dict_digest(model.state_dict()),
            "inference_configuration": config.to_dict(),
            "inference_configuration_sha256": config.config_id,
            "preprocessing_id": config.preprocessing_id,
            "prepared_input_identity": prepared_identity,
            "checkpoint_id": config.checkpoint_id,
            "checkpoint_sha256": checkpoint_sha256,
        },
        "dependencies": {
            "manifest_path": "chimera/architecture_dependencies.json",
            "manifest_sha256": (
                hashlib.sha256(dependency_bytes).hexdigest()
                if dependency_bytes
                else None
            ),
            "inventory_status": dependency_manifest.get(
                "inventory_status", "MISSING"
            ),
            "declared_production_composition": dependency_manifest.get(
                "production_composition"
            ),
            "candidate_composition": dependency_manifest.get("candidate_composition"),
        },
        "components": {
            "status": component_status,
            "objective_provenance": objective_provenance,
        },
        "retrieval": {
            "requested": config.retrieval_id is not None,
            "activated": False,
            "index_present": retrieval_index_present,
            "index_identity": getattr(retrieval, "index_identity", None),
            "encoder_identity": getattr(retrieval, "index_encoder_identity", None),
            "status": (
                "PRESENT_BUT_NOT_ACTIVATED"
                if retrieval_index_present
                else "OPTIONAL_UNAVAILABLE"
            ),
        },
        "validation": {
            "policy": dict(validation_policy or {}),
            "evidence": dict(validation_evidence or {}),
            "scientific_validation": "NOT ESTABLISHED",
        },
        "runtime": runtime_provenance(),
        "readiness": readiness,
        "optional_backends": dict(optional_backends or {}),
        "release_decision": {
            "status": "RESEARCH_ONLY",
            "production_eligible": False,
            "reason": (
                "The canonical composition remains a research prototype; "
                "the repository dependency closure and scientific approval are not established."
            ),
        },
    }
    manifest["composition_sha256"] = _json_digest(manifest)
    return manifest


def write_composition_manifest(
    manifest: Mapping[str, Any],
    path: str | Path,
) -> str:
    """Write an immutable JSON manifest and return the file digest."""
    payload = dict(manifest)
    expected = payload.get("composition_sha256")
    unhashed = {
        key: value for key, value in payload.items() if key != "composition_sha256"
    }
    actual = _json_digest(unhashed)
    if not isinstance(expected, str) or expected != actual:
        raise ValueError("composition manifest identity is missing or does not match its content")
    encoded = (
        json.dumps(
            payload,
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("xb") as stream:
        stream.write(encoded)
        stream.flush()
    return hashlib.sha256(encoded).hexdigest()
