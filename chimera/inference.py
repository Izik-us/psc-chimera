"""Configured canonical inference with isolated randomness and auditable provenance."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, fields, is_dataclass
import hashlib
import json
import os
from pathlib import Path
import secrets
import time
import threading
from typing import Any, Mapping

import torch

from ._version import __version__
from .checkpoint import (
    CheckpointManifest,
    config_hash,
    git_provenance,
    runtime_provenance,
    state_schema_hash,
    validate_checkpoint_compatibility,
)
from .configuration import InferenceConfig
from .errors import (
    CheckpointCompatibilityError,
    ConfigurationError,
    InputValidationError,
    ResourceError,
)
from .domain_schema import NRPSConstraints
from .objective_schema import OBJECTIVE_SCHEMA_VERSION, objective_schema_hash


OBJECTIVE_NAMES = (
    "evolutionary_plausibility",
    "structural_stability",
    "expression_efficiency",
    "substrate_selectivity",
    "assembly_compatibility",
)
_PROXY_KEYS = (
    "normalized_sequence_entropy_proxy",
    "backbone_sanity_validity",
    "rule_based_codon_optimization_proxy",
    "target_profile_match_proxy",
    "icosahedral_interface_geometry_proxy",
)
_SUBSTRATE_TOKENS = {
    name: index
    for index, name in enumerate(
        (
            "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
            "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
        )
    )
}
_RUNTIME_POLICY_LOCK = threading.RLock()
_AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


@dataclass(frozen=True)
class ValidationOptions:
    """Candidate checks supported by the current canonical inference path."""

    geometry: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.geometry, bool):
            raise ConfigurationError("validation_options.geometry must be a boolean")


@dataclass(frozen=True)
class InferenceRequest:
    """Caller-prepared canonical inputs plus one validated inference configuration."""

    msa_tokens: torch.Tensor
    initial_pair_features: torch.Tensor
    source_rotations: torch.Tensor
    source_translations: torch.Tensor
    config: InferenceConfig
    num_candidates: int = 1
    constraints: Any | None = None
    target_substrate: str = "PHE"
    substrate_coordinates: torch.Tensor | None = None
    substrate_types: torch.Tensor | None = None
    validation_options: ValidationOptions = ValidationOptions()
    checkpoint_path: str | Path | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.config, InferenceConfig):
            raise ConfigurationError("config must be an InferenceConfig")
        if (
            isinstance(self.num_candidates, bool)
            or not isinstance(self.num_candidates, int)
            or self.num_candidates < 1
        ):
            raise InputValidationError("num_candidates must be a positive integer")
        if not isinstance(self.validation_options, ValidationOptions):
            raise ConfigurationError("validation_options must be a ValidationOptions instance")
        if not all(
            torch.is_tensor(value)
            for value in (
                self.msa_tokens,
                self.initial_pair_features,
                self.source_rotations,
                self.source_translations,
            )
        ):
            raise InputValidationError("MSA, pair features, and source frames must be tensors")
        if self.msa_tokens.ndim != 3 or self.msa_tokens.shape[0] != 1:
            raise InputValidationError("msa_tokens must have shape (1,N,L)")
        if self.msa_tokens.shape[1] < 1 or self.msa_tokens.shape[2] < 2:
            raise InputValidationError("msa_tokens must contain at least one row and two residues")
        if self.msa_tokens.dtype not in {
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        }:
            raise InputValidationError("msa_tokens must use an integer token dtype")
        if self.msa_tokens.numel() and (
            int(self.msa_tokens.min()) < 0 or int(self.msa_tokens.max()) > 23
        ):
            raise InputValidationError("msa_tokens contain token IDs outside [0,23]")
        if self.msa_tokens.shape[-1] > self.config.max_sequence_length:
            raise InputValidationError(
                "input sequence length exceeds config.max_sequence_length"
            )
        length = self.msa_tokens.shape[-1]
        if self.initial_pair_features.ndim != 4 or self.initial_pair_features.shape[:3] != (
            1,
            length,
            length,
        ):
            raise InputValidationError(
                "initial_pair_features must have shape (1,L,L,d_pair)"
            )
        if self.source_rotations.shape != (1, length, 3, 3):
            raise InputValidationError("source_rotations must have shape (1,L,3,3)")
        if self.source_translations.shape != (1, length, 3):
            raise InputValidationError("source_translations must have shape (1,L,3)")
        for name, value in (
            ("initial_pair_features", self.initial_pair_features),
            ("source_rotations", self.source_rotations),
            ("source_translations", self.source_translations),
        ):
            if not torch.is_floating_point(value) or not torch.isfinite(value).all():
                raise InputValidationError(f"{name} must contain finite floating-point values")
        if self.constraints is not None and not isinstance(self.constraints, NRPSConstraints):
            raise InputValidationError("constraints must be NRPSConstraints or None")
        if (self.substrate_coordinates is None) != (self.substrate_types is None):
            raise InputValidationError(
                "substrate_coordinates and substrate_types must be supplied together"
            )
        if self.substrate_coordinates is not None:
            if (
                self.substrate_coordinates.ndim != 3
                or self.substrate_coordinates.shape[0] != 1
                or self.substrate_coordinates.shape[-1] != 3
            ):
                raise InputValidationError("substrate_coordinates must have shape (1,N,3)")
            if self.substrate_types.shape != (*self.substrate_coordinates.shape[:2], 8):
                raise InputValidationError("substrate_types must have shape (1,N,8)")
            if (
                not torch.is_floating_point(self.substrate_coordinates)
                or not torch.isfinite(self.substrate_coordinates).all()
                or not torch.is_floating_point(self.substrate_types)
                or not torch.isfinite(self.substrate_types).all()
            ):
                raise InputValidationError("substrate inputs must be finite floating-point tensors")
        if self.target_substrate not in _SUBSTRATE_TOKENS:
            raise InputValidationError(f"unsupported target_substrate {self.target_substrate!r}")


@dataclass(frozen=True)
class InferenceCandidate:
    sequence: str
    sequence_tokens: torch.Tensor
    backbone_coordinates: torch.Tensor
    digest: str


@dataclass(frozen=True)
class InferenceResult:
    candidates: tuple[InferenceCandidate, ...]
    validation_results: tuple[Mapping[str, Any], ...]
    evaluation_outputs: Mapping[str, tuple[float, ...]]
    config_hash: str
    checkpoint_identity: str
    checkpoint_sha256: str | None
    checkpoint_compatibility: str
    random_seed: int
    provenance: Mapping[str, Any]
    warnings: tuple[str, ...]
    optional_backends: Mapping[str, str]
    approximate_components: tuple[str, ...]
    production_validated: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "artifact_type": "canonical_inference_result",
            "artifact_schema": "canonical-inference-result-v1",
            "candidates": [
                {
                    "sequence": candidate.sequence,
                    "sequence_tokens": candidate.sequence_tokens.tolist(),
                    "backbone_coordinates": candidate.backbone_coordinates.tolist(),
                    "sha256": candidate.digest,
                }
                for candidate in self.candidates
            ],
            "validation_results": list(self.validation_results),
            "evaluation_outputs": {
                name: list(values) for name, values in self.evaluation_outputs.items()
            },
            "config_hash": self.config_hash,
            "checkpoint_identity": self.checkpoint_identity,
            "checkpoint_sha256": self.checkpoint_sha256,
            "checkpoint_compatibility": self.checkpoint_compatibility,
            "random_seed": self.random_seed,
            "provenance": dict(self.provenance),
            "warnings": list(self.warnings),
            "optional_backends": dict(self.optional_backends),
            "approximate_components": list(self.approximate_components),
            "production_validated": self.production_validated,
        }

    def write_json(self, path: str | Path) -> str:
        """Write an immutable inference artifact and return its SHA-256 digest."""
        encoded = (
            json.dumps(
                self.as_dict(),
                sort_keys=True,
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
        with Path(path).open("xb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        return hashlib.sha256(encoded).hexdigest()

    def as_evidence(self) -> dict[str, Any]:
        """Return structured inference evidence; it is not itself release approval."""
        return {
            "evidence_type": "canonical_inference_result",
            "config_hash": self.config_hash,
            "checkpoint_identity": self.checkpoint_identity,
            "checkpoint_sha256": self.checkpoint_sha256,
            "checkpoint_compatibility": self.checkpoint_compatibility,
            "result_sha256": self.provenance["result_sha256"],
            "candidate_count": len(self.candidates),
            "all_geometry_valid": (
                all(row.get("valid") is True for row in self.validation_results)
                if self.validation_results
                and all(row.get("status") == "VALIDATED" for row in self.validation_results)
                else None
            ),
            "production_validated": self.production_validated,
            "scientific_validation": "NOT ESTABLISHED",
            "optional_backends": dict(self.optional_backends),
        }


def _update_tensor_hash(digest: Any, tensor: torch.Tensor) -> None:
    value = tensor.detach().to(device="cpu").contiguous()
    digest.update(str(value.dtype).encode())
    digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode())
    digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())


def _hash_input_value(digest: Any, value: Any) -> None:
    if torch.is_tensor(value):
        _update_tensor_hash(digest, value)
    elif is_dataclass(value):
        for field in fields(value):
            digest.update(field.name.encode())
            _hash_input_value(digest, getattr(value, field.name))
    elif isinstance(value, Mapping):
        for key in sorted(value):
            digest.update(str(key).encode())
            _hash_input_value(digest, value[key])
    elif isinstance(value, (tuple, list)):
        digest.update(str(len(value)).encode())
        for item in value:
            _hash_input_value(digest, item)
    elif value is None or isinstance(value, (str, bool, int, float)):
        digest.update(json.dumps(value, sort_keys=True, allow_nan=False).encode())
    else:
        raise InputValidationError(
            f"cannot create stable input identity for {type(value).__name__}"
        )


def _input_hash(request: InferenceRequest) -> str:
    digest = hashlib.sha256()
    for name in (
        "msa_tokens",
        "initial_pair_features",
        "source_rotations",
        "source_translations",
        "constraints",
        "target_substrate",
        "substrate_coordinates",
        "substrate_types",
    ):
        digest.update(name.encode())
        _hash_input_value(digest, getattr(request, name))
    return digest.hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _state_dict_sha256(state_dict: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state_dict):
        digest.update(name.encode())
        _update_tensor_hash(digest, state_dict[name])
    return digest.hexdigest()


def _safe_torch_load(path: Path) -> Any:
    import numpy as np

    numpy_core = getattr(np, "_core", None) or np.core
    reconstruct = numpy_core.multiarray._reconstruct
    safe_types = (
        reconstruct,
        np.ndarray,
        np.dtype,
        type(np.dtype("uint32")),
    )
    with torch.serialization.safe_globals(list(safe_types)):
        return torch.load(path, map_location="cpu", weights_only=True)


def _load_checkpoint(
    model: torch.nn.Module, path: Path, fingerprint: str
) -> dict[str, Any]:
    from dataclasses import replace

    payload = _safe_torch_load(path)
    if not isinstance(payload, dict):
        raise CheckpointCompatibilityError("checkpoint payload must be a mapping")
    artifact_type = payload.get("artifact_type")
    if artifact_type == "canonical_training_checkpoint":
        try:
            actual = CheckpointManifest(**payload["manifest"])
            actual.validate()
            expected = replace(
                actual,
                config_hash=config_hash(model.model_configuration()),
                state_schema_hash=state_schema_hash(model.state_dict()),
                objective_schema_hash=objective_schema_hash(),
                chimera_version=__version__,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise CheckpointCompatibilityError(
                f"canonical checkpoint manifest is invalid: {exc}"
            ) from exc
        verdict = validate_checkpoint_compatibility(actual, expected)
        if verdict in {"incompatible", "unverified-provenance"}:
            raise CheckpointCompatibilityError(
                f"canonical checkpoint is not compatible with this model: {verdict}"
            )
        if not actual.has_complete_provenance():
            raise CheckpointCompatibilityError(
                "canonical checkpoint does not have complete training provenance"
            )
        if not isinstance(payload.get("model_state"), dict):
            raise CheckpointCompatibilityError("canonical checkpoint is missing model_state")
        if state_schema_hash(payload["model_state"]) != actual.state_schema_hash:
            raise CheckpointCompatibilityError(
                "canonical checkpoint model_state does not match its declared state schema"
            )
        training = payload.get("training")
        if not isinstance(training, dict):
            raise CheckpointCompatibilityError("canonical checkpoint is missing training metadata")
        for name in ("component_status", "objective_calibration", "objective_status"):
            if name not in training:
                raise CheckpointCompatibilityError(
                    f"canonical checkpoint is missing training.{name}"
                )
        model.load_state_dict(payload["model_state"], strict=True)
        model.component_status = training["component_status"]
        model.objective_calibration = training["objective_calibration"]
        model.objective_status = training["objective_status"]
        return {
            "sha256": fingerprint,
            "compatibility": verdict,
            "state_schema_hash": actual.state_schema_hash,
            "manifest": actual,
            "kind": artifact_type,
        }
    if artifact_type not in (None, "component_transfer_checkpoint"):
        raise CheckpointCompatibilityError(
            f"unsupported inference checkpoint artifact type: {artifact_type!r}"
        )
    model.load_connectors(str(path))
    return {
        "sha256": fingerprint,
        "compatibility": "strict-component-load-unverified-training-provenance",
        "state_schema_hash": state_schema_hash(model.state_dict()),
        "manifest": None,
        "kind": artifact_type or "legacy-component-map",
    }


@contextmanager
def _runtime_policy(config: InferenceConfig):
    with _RUNTIME_POLICY_LOCK:
        previous_threads = torch.get_num_threads()
        previous_deterministic = torch.are_deterministic_algorithms_enabled()
        previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
        previous_cudnn_deterministic = torch.backends.cudnn.deterministic
        previous_cudnn_benchmark = torch.backends.cudnn.benchmark
        previous_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
        previous_cudnn_tf32 = torch.backends.cudnn.allow_tf32
        try:
            torch.set_num_threads(config.cpu_threads)
            torch.use_deterministic_algorithms(config.deterministic, warn_only=False)
            torch.backends.cudnn.deterministic = config.deterministic
            torch.backends.cudnn.benchmark = not config.deterministic
            torch.backends.cuda.matmul.allow_tf32 = not config.deterministic
            torch.backends.cudnn.allow_tf32 = not config.deterministic
            yield
        finally:
            torch.set_num_threads(previous_threads)
            torch.use_deterministic_algorithms(
                previous_deterministic, warn_only=previous_warn_only
            )
            torch.backends.cudnn.deterministic = previous_cudnn_deterministic
            torch.backends.cudnn.benchmark = previous_cudnn_benchmark
            torch.backends.cuda.matmul.allow_tf32 = previous_matmul_tf32
            torch.backends.cudnn.allow_tf32 = previous_cudnn_tf32


def _validate_runtime_request(config: InferenceConfig) -> tuple[torch.device, torch.dtype]:
    if config.padding_policy != "longest_in_batch":
        raise ConfigurationError(
            "only padding_policy='longest_in_batch' is supported; preprocessing is caller-owned"
        )
    if config.worker_count != 0:
        raise ConfigurationError("worker_count is unsupported by the synchronous inference API")
    if config.memory_limit_bytes is not None:
        raise ConfigurationError(
            "memory_limit_bytes cannot be enforced by the current inference runtime"
        )
    device = torch.device(config.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise ConfigurationError(f"configured device {config.device!r} is unavailable")
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise ConfigurationError(f"configured device {config.device!r} is unavailable")
    return device, torch.float32


def _run_inference_locked(model: torch.nn.Module, request: InferenceRequest) -> InferenceResult:
    """Run canonical candidate generation under the request's explicit policy.

    The currently verified compute dtype is float32. Timeout is cooperative
    between candidate microbatches and cannot interrupt a kernel.
    """
    from .architecture import CanonicalCHIMERAv2

    if not isinstance(model, CanonicalCHIMERAv2):
        raise InputValidationError("run_inference requires a CanonicalCHIMERAv2 instance")
    config = request.config
    device, _ = _validate_runtime_request(config)
    if config.retrieval_id is not None:
        retrieval_warning = (
            f"retrieval_id {config.retrieval_id!r} was requested but retrieval is unavailable "
            "for canonical inference because no aligned index/query evidence is configured"
        )
    else:
        retrieval_warning = None
    checkpoint_info: dict[str, Any] = {
        "sha256": None,
        "compatibility": "not-supplied",
        "state_schema_hash": state_schema_hash(model.state_dict()),
        "manifest": None,
        "kind": None,
    }
    checkpoint_identity = config.checkpoint_id
    if request.checkpoint_path is not None:
        checkpoint_path = Path(request.checkpoint_path).expanduser().resolve()
        if not checkpoint_path.is_file():
            raise CheckpointCompatibilityError(
                f"configured checkpoint does not exist: {checkpoint_path}"
            )
        checkpoint_fingerprint = _file_sha256(checkpoint_path)
        checkpoint_identity = config.checkpoint_id
        checkpoint_digest_id = config.checkpoint_id.rsplit(":", 1)[-1]
        if (
            len(checkpoint_digest_id) == 64
            and all(character in "0123456789abcdefABCDEF" for character in checkpoint_digest_id)
            and checkpoint_digest_id.lower() != checkpoint_fingerprint
        ):
            raise CheckpointCompatibilityError(
                "checkpoint file SHA-256 does not match config.checkpoint_id"
            )
        checkpoint_info = _load_checkpoint(
            model, checkpoint_path, checkpoint_fingerprint
        )
    original_device = next(model.parameters()).device
    training_modes = [(module, module.training) for module in model.modules()]
    seed = (
        config.random_seed
        if config.random_seed is not None
        else secrets.randbits(63)
    )
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    started = time.monotonic()
    microbatch_size = config.microbatch_size or config.batch_size
    candidate_rows: list[tuple[InferenceCandidate, dict[str, Any], dict[str, float]]] = []
    old_num_threads = torch.get_num_threads()
    warnings_list = [
        "Geometry checks and deterministic evaluator scores are engineering proxies, not biological validation.",
        "Local MSA and ProteinMPNN-inspired components are approximations; no native external model backend was used.",
        f"timeout_seconds={config.timeout_seconds} is checked between microbatches and cannot interrupt an active model kernel.",
    ]
    readiness_before = model.inference_readiness()
    if readiness_before["state"] != "TRAINED":
        warnings_list.append(
            f"model readiness is {readiness_before['state']}; this result is exploratory, not production-valid"
        )
    if checkpoint_info["compatibility"] == "not-supplied":
        warnings_list.append(
            "no unified trained CHIMERA checkpoint was supplied; checkpoint identity is an unresolved reference"
        )
    if retrieval_warning is not None:
        warnings_list.append(retrieval_warning)
    input_digest = _input_hash(request)
    request_digest = config_hash(
        {
            "config": config.to_dict(),
            "input_sha256": input_digest,
            "num_candidates": request.num_candidates,
            "validation_options": {"geometry": request.validation_options.geometry},
        }
    )
    try:
        model.to(device=device)
        model.eval()
        with _runtime_policy(config):
            for start in range(0, request.num_candidates, microbatch_size):
                if time.monotonic() - started > config.timeout_seconds:
                    raise ResourceError("inference exceeded its cooperative timeout")
                count = min(microbatch_size, request.num_candidates - start)
                with torch.inference_mode():
                    output = model(
                        request.msa_tokens.to(device=device, dtype=torch.long).expand(count, -1, -1),
                        request.initial_pair_features.to(device=device, dtype=torch.float32).expand(
                            count, -1, -1, -1
                        ),
                        request.source_rotations.to(device=device, dtype=torch.float32).expand(
                            count, -1, -1, -1
                        ),
                        request.source_translations.to(device=device, dtype=torch.float32).expand(
                            count, -1, -1
                        ),
                        constraints=request.constraints,
                        substrate_id=torch.full(
                            (count,),
                            _SUBSTRATE_TOKENS[request.target_substrate],
                            device=device,
                            dtype=torch.long,
                        ),
                        substrate_coords=(
                            request.substrate_coordinates.to(
                                device=device, dtype=torch.float32
                            ).expand(count, -1, -1)
                            if request.substrate_coordinates is not None else None
                        ),
                        substrate_types=(
                            request.substrate_types.to(
                                device=device, dtype=torch.float32
                            ).expand(count, -1, -1)
                            if request.substrate_types is not None else None
                        ),
                        n_flow_steps=config.inference_steps,
                        n_mpnn_seqs=1,
                        use_rag=False,
                        temperature=config.sequence_temperature,
                        generator=generator,
                        validate_geometry=request.validation_options.geometry,
                    )
                geometry_report = output["geometry_report"]
                if geometry_report is not None:
                    geometry_metrics = geometry_report["candidate_metrics"]
                    if not geometry_metrics:
                        geometry_metrics = [
                            {
                                name: value
                                for name, value in geometry_report.items()
                                if name not in {"candidate_metrics", "candidate_valid"}
                            }
                        ]
                    geometry_valid = geometry_report["candidate_valid"]
                else:
                    geometry_metrics = []
                    geometry_valid = []
                proxy_scores = output["objective_proxy_scores"][:, 0].detach().to("cpu")
                tokens = output["sequence_tokens"][:, 0].detach().to("cpu")
                coordinates = output["backbone_coords"].detach().to("cpu")
                for index in range(count):
                    sequence_tokens = tokens[index].contiguous()
                    backbone = coordinates[index].contiguous()
                    candidate_digest = hashlib.sha256()
                    _update_tensor_hash(candidate_digest, sequence_tokens)
                    _update_tensor_hash(candidate_digest, backbone)
                    sequence = "".join(_AMINO_ACIDS[int(token)] for token in sequence_tokens)
                    candidate = InferenceCandidate(
                        sequence=sequence,
                        sequence_tokens=sequence_tokens,
                        backbone_coordinates=backbone,
                        digest=candidate_digest.hexdigest(),
                    )
                    if request.validation_options.geometry:
                        validation = {
                            "status": "VALIDATED",
                            "valid": bool(geometry_valid[index]),
                            "type": "canonical_backbone_geometry_sanity",
                            "scientific_validation": "NOT ESTABLISHED",
                            "metrics": geometry_metrics[index],
                        }
                    else:
                        validation = {
                            "status": "NOT_REQUESTED",
                            "valid": None,
                            "type": None,
                            "scientific_validation": "NOT ESTABLISHED",
                            "metrics": None,
                        }
                    values = {
                        name: float(proxy_scores[index, objective_index])
                        for objective_index, name in enumerate(OBJECTIVE_NAMES)
                        if name in config.objective_set
                    }
                    candidate_rows.append((candidate, validation, values))
                if time.monotonic() - started > config.timeout_seconds:
                    raise ResourceError("inference exceeded its cooperative timeout")
            runtime = runtime_provenance()
            runtime.update(
                {
                    "device": str(device),
                    "compute_dtype": config.dtype,
                    "model_parameter_dtype": str(next(model.parameters()).dtype),
                    "cpu_threads": config.cpu_threads,
                }
            )
            runtime["environment_hash"] = config_hash(
                {key: value for key, value in runtime.items() if key != "environment_hash"}
            )
    finally:
        model.to(device=original_device)
        for module, training in training_modes:
            module.training = training
        if torch.get_num_threads() != old_num_threads:
            torch.set_num_threads(old_num_threads)

    candidates = tuple(row[0] for row in candidate_rows)
    validations = tuple(row[1] for row in candidate_rows)
    evaluation_outputs = {
        name: tuple(row[2][name] for row in candidate_rows)
        for name in config.objective_set
        if name in OBJECTIVE_NAMES
    }
    result_hash = hashlib.sha256(
        "".join(candidate.digest for candidate in candidates).encode()
    ).hexdigest()
    git = git_provenance(Path(__file__).resolve().parents[1])
    readiness = model.inference_readiness()
    production_validated = False
    provenance = {
        "schema": "chimera-inference-provenance-v1",
        "software": {
            "version": __version__,
            **git,
        },
        "configuration": {
            "schema_version": config.schema_version,
            "value": config.to_dict(),
            "sha256": config.config_id,
            "request_sha256": request_digest,
            "num_candidates": request.num_candidates,
        },
        "model": {
            "model_id": config.model_id,
            "artifact_id": config.artifact_id,
            "architecture_config_sha256": config_hash(model.model_configuration()),
            "model_weights_sha256": _state_dict_sha256(model.state_dict()),
            "checkpoint_id": checkpoint_identity,
            "checkpoint_sha256": checkpoint_info["sha256"],
            "checkpoint_kind": checkpoint_info["kind"],
            "checkpoint_compatibility": checkpoint_info["compatibility"],
            "checkpoint_state_schema_hash": checkpoint_info["state_schema_hash"],
            "objective_schema_version": OBJECTIVE_SCHEMA_VERSION,
            "objective_schema_hash": objective_schema_hash(),
            "readiness": readiness,
            "production_validated": production_validated,
        },
        "inputs": {
            "sha256": input_digest,
            "preprocessing_id": config.preprocessing_id,
            "preprocessing_identity_sha256": config_hash(
                {"preprocessing_id": config.preprocessing_id}
            ),
            "raw_inputs_stored": False,
        },
        "runtime": runtime,
        "randomness": {
            "requested_seed": config.random_seed,
            "effective_seed": seed,
            "deterministic": config.deterministic,
            "generator_device": str(device),
            "global_python_numpy_torch_rng_seeded": False,
            "policy": (
                "deterministic torch algorithms and backend flags during inference"
                if config.deterministic
                else "seeded local generator; nondeterministic kernels permitted"
            ),
        },
        "components": {
            "canonical_local_components_used": [
                "MSARepresentationBackbone",
                "SE3SchrodingerBridge",
                "ProteinMPNN-inspired local sequence-recovery component",
                "AutoregressiveSequencePolicy",
            ],
            "native_external_backends_used": [],
            "approximate_components_used": [
                "local MSA representation",
                "local ProteinMPNN-inspired sequence recovery",
                "deterministic evaluator proxy scores",
            ],
            "retrieval": (
                "requested_but_unavailable"
                if config.retrieval_id is not None
                else "not_requested"
            ),
            "unavailable_optional_systems": ["RFdiffusion", "OpenFold", "ESMFold", "native ProteinMPNN"],
        },
        "validation": {
            "candidate_geometry": "requested" if request.validation_options.geometry else "not_requested",
            "experimental_validation": "NOT ESTABLISHED",
        },
        "evaluation": {
            "requested_objectives": list(config.objective_set),
            "sources": {
                name: _PROXY_KEYS[index]
                for index, name in enumerate(OBJECTIVE_NAMES)
                if name in config.objective_set
            },
        },
        "result_sha256": result_hash,
        "production_evidence": {
            "inference_executed": True,
            "checkpoint_provenance_complete": checkpoint_info["compatibility"]
            in {"exact-compatible", "expected-compatible"},
            "held_out_validation_evidence": readiness["state"] == "TRAINED",
            "production_validated": False,
            "release_gate_decision": "not evaluated; production gate remains authoritative",
        },
    }
    optional_backends = {
        "native_proteinmpnn": "not_used",
        "rfdiffusion": "not_used",
        "openfold": "not_used",
        "esmfold": "not_used",
        "retrieval": "requested_but_unavailable" if retrieval_warning else "not_requested",
    }
    return InferenceResult(
        candidates=candidates,
        validation_results=validations,
        evaluation_outputs=evaluation_outputs,
        config_hash=config.config_id,
        checkpoint_identity=checkpoint_identity,
        checkpoint_sha256=checkpoint_info["sha256"],
        checkpoint_compatibility=checkpoint_info["compatibility"],
        random_seed=seed,
        provenance=provenance,
        warnings=tuple(warnings_list),
        optional_backends=optional_backends,
        approximate_components=tuple(provenance["components"]["approximate_components_used"]),
        production_validated=production_validated,
    )


def run_inference(model: torch.nn.Module, request: InferenceRequest) -> InferenceResult:
    """Serialize the process-global runtime policy and model placement changes."""
    with _RUNTIME_POLICY_LOCK:
        return _run_inference_locked(model, request)


__all__ = [
    "InferenceCandidate",
    "InferenceRequest",
    "InferenceResult",
    "ValidationOptions",
    "run_inference",
]
