"""Synchronous, auditable execution for bounded canonical design tasks."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import torch

from ..configuration import canonical_config_json
from ..domain_schema import NRPSConstraints
from ..errors import ArtifactError, ConfigurationError, InputValidationError
from ..inference import ValidationOptions, run_inference_from_checkpoint
from ..input_contract import (
    PreparedInputManifest,
    build_inference_request,
    prepare_inference_inputs,
)
from .manager import inspect_design_run
from .planner import load_design_task


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Any) -> str:
    encoded = (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    return hashlib.sha256(encoded).hexdigest()


def _event(directory: Path, name: str, **details: Any) -> None:
    record = {
        "event_schema_version": 1,
        "event": name,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        **details,
    }
    with (directory / "events.jsonl").open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(canonical_config_json(record) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _tensor(bundle: dict[str, Any], name: str, *, required: bool, dtype: torch.dtype):
    value = bundle.get(name)
    if value is None:
        if required:
            raise InputValidationError(f"prepared input bundle is missing tensor {name!r}")
        return None
    try:
        return torch.tensor(value, dtype=dtype)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise InputValidationError(f"prepared input tensor {name!r} is malformed: {exc}") from exc


def _load_request(task, input_path: Path):
    try:
        document = json.loads(input_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"cannot read prepared input bundle {input_path}: {exc}") from exc
    if not isinstance(document, dict) or document.get("schema_version") != 1:
        raise InputValidationError("prepared input bundle must be a schema_version=1 JSON object")
    manifest_data = document.get("manifest")
    tensors = document.get("tensors")
    if not isinstance(manifest_data, dict) or not isinstance(tensors, dict):
        raise InputValidationError("prepared input bundle requires manifest and tensors objects")
    manifest = PreparedInputManifest(**manifest_data)
    if manifest.preprocessing_id != task.inference_config.preprocessing_id:
        raise InputValidationError(
            "prepared-input preprocessing_id does not match inference_config.preprocessing_id"
        )
    prepared = prepare_inference_inputs(
        msa_tokens=_tensor(tensors, "msa_tokens", required=True, dtype=torch.long),
        pair_features=_tensor(tensors, "pair_features", required=True, dtype=torch.float32),
        source_rotations=_tensor(tensors, "source_rotations", required=True, dtype=torch.float32),
        source_translations=_tensor(tensors, "source_translations", required=True, dtype=torch.float32),
        manifest=manifest,
        msa_padding_mask=_tensor(tensors, "msa_padding_mask", required=False, dtype=torch.bool),
        residue_mask=_tensor(tensors, "residue_mask", required=False, dtype=torch.bool),
        residue_index=_tensor(tensors, "residue_index", required=False, dtype=torch.long),
    )
    if prepared.batch_size != 1:
        raise InputValidationError("canonical InferenceRequest currently requires exactly one prepared input")
    expected = task.input_shape
    actual = {
        "sequence_length": prepared.length,
        "msa_depth": int(prepared.msa_tokens.shape[1]),
        "pair_channels": int(prepared.pair_features.shape[-1]),
    }
    if actual != dict(expected):
        raise InputValidationError(
            f"prepared input shape {actual} does not match task.input_shape {dict(expected)}"
        )

    constraint_data = document.get("constraints")
    constraints = None
    if constraint_data is not None:
        if not isinstance(constraint_data, dict):
            raise InputValidationError("constraints must be a JSON object")
        tensor_fields = {
            "fixed_mask": torch.bool,
            "stachelhaus_positions": torch.long,
            "domain_boundaries": torch.long,
            "module_boundaries": torch.long,
            "icosahedral_face": torch.long,
            "hotspot_coords": torch.float32,
            "hotspot_indices": torch.long,
            "fixed_sequence": torch.long,
            "domain_types": torch.long,
        }
        values = dict(constraint_data)
        for name, dtype in tensor_fields.items():
            if name in values and values[name] is not None:
                values[name] = torch.tensor(values[name], dtype=dtype)
        required = {
            "fixed_mask", "stachelhaus_positions", "domain_boundaries",
            "module_boundaries", "icosahedral_face", "ppt_serine_position",
            "hotspot_coords", "hotspot_indices", "target_substrate",
        }
        missing = sorted(required - set(values))
        if missing:
            raise InputValidationError(f"constraints are missing fields: {missing}")
        try:
            constraints = NRPSConstraints(**values)
        except TypeError as exc:
            raise InputValidationError(f"constraints fields are invalid: {exc}") from exc

    substrate = str(task.target_context.get("substrate", "PHE")).upper()
    return build_inference_request(
        prepared,
        task.inference_config,
        num_candidates=task.candidate_count,
        constraints=constraints,
        target_substrate=substrate,
        validation_options=ValidationOptions(geometry=True),
    )


def _validate_declared_constraints(task, request) -> None:
    supported = {"sequence_length"}
    for item in task.hard_constraints:
        kind = item.get("id")
        if kind not in supported:
            raise ConfigurationError(
                f"hard constraint {kind!r} has no implemented canonical evaluator; refusing to ignore it"
            )
        operator = item.get("kind")
        value = item.get("value")
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigurationError("sequence_length hard constraint value must be an integer")
        length = int(request.msa_tokens.shape[-1])
        if operator == "maximum" and length > value:
            raise InputValidationError("prepared sequence length violates sequence_length maximum")
        if operator == "minimum" and length < value:
            raise InputValidationError("prepared sequence length violates sequence_length minimum")
        if operator not in {"maximum", "minimum"}:
            raise ConfigurationError("sequence_length hard constraint kind must be maximum or minimum")
    unknown_objectives = set(task.soft_objectives) - set(task.inference_config.objective_set)
    if unknown_objectives:
        raise ConfigurationError(
            f"soft objectives are not enabled in inference_config.objective_set: {sorted(unknown_objectives)}"
        )


def execute_design_run(run_dir: str | Path) -> dict[str, Any]:
    """Execute one prepared run synchronously; preserve failures as durable states."""
    directory = Path(run_dir).expanduser().resolve()
    lock = directory / ".execution.lock"
    try:
        with lock.open("x", encoding="utf-8") as lock_stream:
            lock_stream.write(f"pid={os.getpid()}\n")
            lock_stream.flush()
            os.fsync(lock_stream.fileno())
    except FileExistsError as exc:
        raise ArtifactError(f"design run is already executing or has a stale lock: {directory}") from exc

    try:
        snapshot = inspect_design_run(directory)
        if snapshot["status"] != "PLANNED":
            raise ArtifactError(
                f"only PLANNED runs can execute; current status is {snapshot['status']}"
            )
        task = load_design_task(directory / "task.json")
        plan = snapshot["plan"]
        input_path = Path(plan["input_bundle"]["path"])
        checkpoint_path = Path(plan["checkpoint"]["path"])
        if not input_path.is_file() or _sha256(input_path) != plan["input_bundle"]["sha256"]:
            raise ArtifactError("prepared input bundle changed after planning; create a new run")
        if not checkpoint_path.is_file() or _sha256(checkpoint_path) != plan["checkpoint"]["sha256"]:
            raise ArtifactError("checkpoint changed after planning; create a new run")

        manifest = dict(snapshot["manifest"])
        manifest.update({
            "status": "RUNNING",
            "phase": "input_validation",
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "generation_started": False,
            "checkpoint_loaded": False,
        })
        _atomic_json(directory / "manifest.json", manifest)
        _event(directory, "run_started", run_id=snapshot["run_id"], status="RUNNING")

        request = _load_request(task, input_path)
        _validate_declared_constraints(task, request)
        manifest["phase"] = "inference"
        manifest["generation_started"] = True
        _atomic_json(directory / "manifest.json", manifest)
        _event(directory, "generation_started", run_id=snapshot["run_id"],
               requested_candidates=task.candidate_count)

        result = run_inference_from_checkpoint(checkpoint_path, request)
        result_path = directory / "inference-result.json"
        result_sha256 = result.write_json(result_path)
        result_data = result.as_dict()
        report = {
            "schema_version": 1,
            "run_id": snapshot["run_id"],
            "status": "COMPLETED",
            "requested_candidates": task.candidate_count,
            "generated_candidates": len(result.candidates),
            "eligible_candidate_indices": list(result.eligible_candidate_indices),
            "rejected_candidate_indices": list(result.rejected_candidate_indices),
            "objective_names": sorted(result.evaluation_outputs),
            "checkpoint_compatibility": result.checkpoint_compatibility,
            "checkpoint_sha256": result.checkpoint_sha256,
            "input_bundle_sha256": plan["input_bundle"]["sha256"],
            "inference_result_sha256": result_sha256,
            "warnings": list(result.warnings),
            "optional_backends": dict(result.optional_backends),
            "scientific_validation": "NOT ESTABLISHED",
            "experimental_validation": "NOT ESTABLISHED",
            "production_validated": False,
        }
        report_sha256 = _atomic_json(directory / "report.json", report)
        manifest.update({
            "status": "COMPLETED",
            "phase": "complete",
            "checkpoint_loaded": True,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "requested_candidate_count": task.candidate_count,
            "generated_candidate_count": len(result.candidates),
            "eligible_candidate_count": len(result.eligible_candidate_indices),
            "rejected_candidate_count": len(result.rejected_candidate_indices),
            "inference_result": {
                "path": result_path.name,
                "sha256": result_sha256,
            },
            "report": {"path": "report.json", "sha256": report_sha256},
            "checkpoint_compatibility": result.checkpoint_compatibility,
            "scientific_validation": "NOT ESTABLISHED",
        })
        _atomic_json(directory / "manifest.json", manifest)
        _event(directory, "run_completed", run_id=snapshot["run_id"],
               generated_candidates=len(result.candidates),
               eligible_candidates=len(result.eligible_candidate_indices),
               inference_result_sha256=result_sha256)
        return inspect_design_run(directory)
    except Exception as exc:
        try:
            manifest_path = directory / "manifest.json"
            if manifest_path.is_file():
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                if manifest.get("status") == "RUNNING":
                    manifest.update({
                        "status": "FAILED",
                        "phase": "failed",
                        "failed_at_utc": datetime.now(timezone.utc).isoformat(),
                        "error": {"type": type(exc).__name__, "message": str(exc)},
                        "scientific_validation": "NOT ESTABLISHED",
                    })
                    _atomic_json(manifest_path, manifest)
                    _event(directory, "run_failed", run_id=manifest.get("run_id"),
                           error_type=type(exc).__name__, message=str(exc))
        except Exception:
            pass
        raise
    finally:
        try:
            lock.unlink()
        except FileNotFoundError:
            pass


def run_design_task(task_file: str | Path, *, runs_root: str | Path = "runs/design"):
    """Create the immutable plan and execute it as one bounded task."""
    from .manager import create_design_run

    task_path = Path(task_file).expanduser().resolve()
    task = load_design_task(task_path)
    created = create_design_run(task, task_file=task_path, runs_root=runs_root)
    if created["manifest"]["status"] == "BLOCKED":
        return inspect_design_run(created["run_dir"])
    return execute_design_run(created["run_dir"])


__all__ = ["execute_design_run", "run_design_task"]
