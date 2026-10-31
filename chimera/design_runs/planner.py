"""Side-effect-light preflight planner for CHIMERA design tasks.

The planner never imports a model or starts generation. Resource figures are
explicit tensor-storage lower bounds, not predictions of total runtime memory.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from ..configuration import canonical_config_json
from ..errors import ConfigurationError
from .schema import DesignTask

PLANNER_VERSION = "1"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_reference(reference: str, task_file: Path) -> Path:
    path = Path(reference).expanduser()
    if not path.is_absolute():
        path = task_file.parent / path
    return path.resolve()


def build_design_plan(task: DesignTask, *, task_file: str | Path) -> dict[str, Any]:
    """Return a deterministic plan and blockers without loading checkpoints."""
    task_path = Path(task_file).expanduser().resolve()
    input_path = _resolve_reference(task.input_bundle, task_path)
    checkpoint_path = _resolve_reference(task.checkpoint_path, task_path)
    blockers: list[str] = []
    warnings: list[str] = []

    def artifact_record(path: Path, label: str) -> dict[str, Any]:
        if not path.is_file():
            blockers.append(f"{label} file does not exist: {path}")
            return {"path": str(path), "exists": False, "size_bytes": None, "sha256": None}
        try:
            size = path.stat().st_size
            digest = _sha256_file(path)
        except OSError as exc:
            blockers.append(f"cannot inspect {label} file {path}: {exc}")
            return {"path": str(path), "exists": True, "size_bytes": None, "sha256": None}
        if size == 0:
            blockers.append(f"{label} file is empty: {path}")
        return {"path": str(path), "exists": True, "size_bytes": size, "sha256": digest}

    input_record = artifact_record(input_path, "input bundle")
    checkpoint_record = artifact_record(checkpoint_path, "checkpoint")
    config = task.inference_config
    shape = task.input_shape
    length = shape["sequence_length"]
    depth = shape["msa_depth"]
    channels = shape["pair_channels"]

    # Float32 pair features + float32 rotations/translations + int64 MSA token
    # IDs. This deliberately excludes model activations, parameters, workspaces,
    # optimizer state and allocator overhead.
    lower_bound = (
        length * length * channels * 4
        + length * 9 * 4
        + length * 3 * 4
        + depth * length * 8
    )
    candidate_coordinate_bytes = task.candidate_count * length * 3 * 4
    if config.worker_count:
        blockers.append(
            "inference_config.worker_count is unsupported by canonical inference; set it to 0"
        )
    if config.memory_limit_bytes is not None:
        blockers.append(
            "inference_config.memory_limit_bytes is not enforced by canonical inference"
        )
    if config.retrieval_id:
        warnings.append(
            "retrieval_id is recorded in configuration, but canonical run_inference currently disables RAG"
        )
    if config.batch_size < task.candidate_count:
        warnings.append(
            "candidate_count exceeds inference_config.batch_size; planner does not assume automatic multi-batch execution"
        )
    if not task_path.is_file():
        blockers.append(f"task file does not exist: {task_path}")

    plan_payload = {
        "planner_version": PLANNER_VERSION,
        "task_schema_version": task.schema_version,
        "task_id": task.task_id,
        "objective": task.objective,
        "target_context": dict(task.target_context),
        "task_file": str(task_path),
        "task_sha256": _sha256_file(task_path) if task_path.is_file() else None,
        "input_bundle": input_record,
        "checkpoint": checkpoint_record,
        "model_id": config.model_id,
        "artifact_id": config.artifact_id,
        "checkpoint_id": config.checkpoint_id,
        "preprocessing_id": config.preprocessing_id,
        "inference_config": config.to_dict(),
        "inference_config_sha256": config.config_id,
        "candidate_count_requested": task.candidate_count,
        "input_shape": dict(shape),
        "hard_constraints": [dict(x) for x in task.hard_constraints],
        "soft_objectives": list(task.soft_objectives),
        "resource_estimate": {
            "status": "LOWER_BOUND_ONLY",
            "input_tensor_storage_lower_bound_bytes": lower_bound,
            "candidate_coordinate_storage_bytes": candidate_coordinate_bytes,
            "assumptions": [
                "float32 pair features, source rotations, source translations",
                "int64 MSA token IDs",
                "candidate coordinates stored as float32",
            ],
            "excludes": [
                "model parameters",
                "intermediate activations",
                "attention workspaces",
                "allocator fragmentation",
                "runtime overhead",
            ],
        },
        "backend": "canonical_run_inference",
        "generation_started": False,
        "checkpoint_loaded": False,
        "readiness": "BLOCKED" if blockers else "PLAN_ONLY",
        "blockers": blockers,
        "warnings": warnings,
        "scientific_validation": "NOT ESTABLISHED",
    }
    plan_payload["plan_sha256"] = hashlib.sha256(
        canonical_config_json(plan_payload).encode("utf-8")
    ).hexdigest()
    return plan_payload


def load_design_task(path: str | Path) -> DesignTask:
    task_path = Path(path).expanduser().resolve()
    try:
        document = json.loads(task_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ConfigurationError(f"cannot read design task {task_path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ConfigurationError(
            f"design task is not valid JSON at line {exc.lineno}, column {exc.colno}: {exc.msg}"
        ) from exc
    return DesignTask.from_dict(document)
