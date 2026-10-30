"""Durable creation and inspection of planned design runs.

This module prepares auditable run directories only. It does not load models,
generate candidates, or claim that a scientific run has executed.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any
from uuid import uuid4

from ..configuration import canonical_config_json
from ..errors import ArtifactError, ConfigurationError
from .planner import build_design_plan
from .schema import DesignTask

RUN_SCHEMA_VERSION = 1
_RUN_ID_PATTERN = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]{0,127}$")


def _write_json_new(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _plan_digest(plan: dict[str, Any]) -> str:
    """Recompute the planner identity, excluding its self-referential digest."""
    payload = dict(plan)
    payload.pop("plan_sha256", None)
    return hashlib.sha256(canonical_config_json(payload).encode("utf-8")).hexdigest()


def create_design_run(
    task: DesignTask,
    *,
    task_file: str | Path,
    runs_root: str | Path = "runs/design",
) -> dict[str, Any]:
    """Create a durable planned/blocked run without starting generation."""
    task_path = Path(task_file).expanduser().resolve()
    root = Path(runs_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    plan = build_design_plan(task, task_file=task_path)
    created_at = datetime.now(timezone.utc).isoformat()
    run_id = f"{task.task_id}-{uuid4().hex[:12]}"
    if not _RUN_ID_PATTERN.fullmatch(run_id):
        raise ConfigurationError("generated run ID contains unsupported characters")

    task_artifact = (json.dumps(
        task.to_dict(), indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False
    ) + "\n").encode("utf-8")
    manifest = {
        "run_schema_version": RUN_SCHEMA_VERSION,
        "run_id": run_id,
        "task_id": task.task_id,
        "created_at_utc": created_at,
        "status": "BLOCKED" if plan["readiness"] == "BLOCKED" else "PLANNED",
        "phase": "preflight",
        "generation_started": False,
        "checkpoint_loaded": False,
        "task_file_sha256": plan["task_sha256"],
        "task_artifact_sha256": hashlib.sha256(task_artifact).hexdigest(),
        "plan_sha256": plan["plan_sha256"],
        "inference_config_sha256": plan["inference_config_sha256"],
        "backend": plan["backend"],
        "scientific_validation": "NOT ESTABLISHED",
    }
    event = {
        "event_schema_version": 1,
        "event": "run_prepared",
        "timestamp_utc": created_at,
        "run_id": run_id,
        "status": manifest["status"],
        "blockers": plan["blockers"],
        "warnings": plan["warnings"],
    }

    staging = Path(tempfile.mkdtemp(prefix=".design-run-staging-", dir=root))
    destination = root / run_id
    try:
        with (staging / "task.json").open("xb") as stream:
            stream.write(task_artifact)
            stream.flush()
            os.fsync(stream.fileno())
        _write_json_new(staging / "plan.json", plan)
        _write_json_new(staging / "manifest.json", manifest)
        with (staging / "events.jsonl").open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(canonical_config_json(event) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        if destination.exists():
            raise ArtifactError(f"refusing to overwrite existing design run: {destination}")
        os.rename(staging, destination)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        raise

    return {"run_id": run_id, "run_dir": str(destination), "manifest": manifest, "plan": plan}


def inspect_design_run(run_dir: str | Path) -> dict[str, Any]:
    """Read and integrity-check the durable records for a prepared run."""
    directory = Path(run_dir).expanduser().resolve()
    required = ("task.json", "plan.json", "manifest.json", "events.jsonl")
    missing = [name for name in required if not (directory / name).is_file()]
    if missing:
        raise ArtifactError(f"design run is missing required artifacts: {', '.join(missing)}")
    try:
        task_bytes = (directory / "task.json").read_bytes()
        plan = json.loads((directory / "plan.json").read_text(encoding="utf-8"))
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        events = [
            json.loads(line)
            for line in (directory / "events.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"cannot read design run artifacts in {directory}: {exc}") from exc
    if hashlib.sha256(task_bytes).hexdigest() != manifest.get("task_artifact_sha256"):
        raise ArtifactError("task.json SHA-256 does not match run manifest")
    recorded_plan_digest = plan.get("plan_sha256")
    if not isinstance(recorded_plan_digest, str) or recorded_plan_digest != manifest.get("plan_sha256"):
        raise ArtifactError("plan.json identity does not match run manifest")
    try:
        actual_plan_digest = _plan_digest(plan)
    except ConfigurationError as exc:
        raise ArtifactError(f"plan.json is not canonically hashable: {exc}") from exc
    if actual_plan_digest != recorded_plan_digest:
        raise ArtifactError("plan.json content SHA-256 does not match its recorded identity")
    if not events or events[0].get("run_id") != manifest.get("run_id"):
        raise ArtifactError("events.jsonl does not contain the expected initial run event")
    return {
        "run_dir": str(directory),
        "run_id": manifest["run_id"],
        "status": manifest["status"],
        "generation_started": manifest["generation_started"],
        "scientific_validation": manifest["scientific_validation"],
        "manifest": manifest,
        "plan": plan,
        "event_count": len(events),
        "last_event": events[-1],
    }
