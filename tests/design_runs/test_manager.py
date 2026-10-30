import json

import pytest

from chimera.configuration import InferenceConfig
from chimera.design_runs import DesignTask, create_design_run, inspect_design_run
from chimera.errors import ArtifactError


def _task(**overrides):
    values = {
        "task_id": "nrps-linker",
        "objective": "NRPS module/domain-linker compatibility",
        "target_context": {"substrate": "PHE"},
        "input_bundle": "inputs.json",
        "checkpoint_path": "checkpoint.pt",
        "inference_config": InferenceConfig(
            model_id="chimera",
            artifact_id="artifact-v1",
            checkpoint_id="checkpoint-v1",
            preprocessing_id="inputs-v1",
            max_sequence_length=64,
        ),
        "candidate_count": 2,
        "input_shape": {"sequence_length": 16, "msa_depth": 4, "pair_channels": 8},
    }
    values.update(overrides)
    return DesignTask(**values)


def test_create_and_inspect_publishes_complete_planned_run(tmp_path):
    task_file = tmp_path / "task.json"
    (tmp_path / "inputs.json").write_text('{"inputs": "fixture"}', encoding="utf-8")
    (tmp_path / "checkpoint.pt").write_bytes(b"not loaded by planner")
    task = _task()
    task_file.write_text(json.dumps(task.to_dict()), encoding="utf-8")

    created = create_design_run(task, task_file=task_file, runs_root=tmp_path / "runs")
    inspected = inspect_design_run(created["run_dir"])

    assert inspected["run_id"] == created["run_id"]
    assert inspected["status"] == "PLANNED"
    assert inspected["generation_started"] is False
    assert inspected["scientific_validation"] == "NOT ESTABLISHED"
    assert inspected["event_count"] == 1
    assert (tmp_path / "runs" / created["run_id"] / "task.json").is_file()


def test_missing_artifacts_create_blocked_run_not_false_success(tmp_path):
    task_file = tmp_path / "task.json"
    task = _task()
    task_file.write_text(json.dumps(task.to_dict()), encoding="utf-8")

    created = create_design_run(task, task_file=task_file, runs_root=tmp_path / "runs")
    inspected = inspect_design_run(created["run_dir"])

    assert inspected["status"] == "BLOCKED"
    assert inspected["plan"]["blockers"]


def test_inspection_detects_task_tampering(tmp_path):
    task_file = tmp_path / "task.json"
    (tmp_path / "inputs.json").write_text("{}", encoding="utf-8")
    (tmp_path / "checkpoint.pt").write_bytes(b"fixture")
    task = _task()
    task_file.write_text(json.dumps(task.to_dict()), encoding="utf-8")
    created = create_design_run(task, task_file=task_file, runs_root=tmp_path / "runs")
    task_artifact = tmp_path / "runs" / created["run_id"] / "task.json"
    task_artifact.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ArtifactError, match="SHA-256"):
        inspect_design_run(created["run_dir"])


def test_inspection_detects_plan_content_tampering(tmp_path):
    task_file = tmp_path / "task.json"
    (tmp_path / "inputs.json").write_text("{}", encoding="utf-8")
    (tmp_path / "checkpoint.pt").write_bytes(b"fixture")
    task = _task()
    task_file.write_text(json.dumps(task.to_dict()), encoding="utf-8")
    created = create_design_run(task, task_file=task_file, runs_root=tmp_path / "runs")
    plan_path = tmp_path / "runs" / created["run_id"] / "plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    plan["objective"] = "tampered objective"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")

    with pytest.raises(ArtifactError, match="content SHA-256"):
        inspect_design_run(created["run_dir"])


def test_inspection_detects_plan_digest_tampering(tmp_path):
    task_file = tmp_path / "task.json"
    (tmp_path / "inputs.json").write_text("{}", encoding="utf-8")
    (tmp_path / "checkpoint.pt").write_bytes(b"fixture")
    task = _task()
    task_file.write_text(json.dumps(task.to_dict()), encoding="utf-8")
    created = create_design_run(task, task_file=task_file, runs_root=tmp_path / "runs")
    plan_path = tmp_path / "runs" / created["run_id"] / "plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    plan["plan_sha256"] = "0" * 64
    plan_path.write_text(json.dumps(plan), encoding="utf-8")

    with pytest.raises(ArtifactError, match="identity"):
        inspect_design_run(created["run_dir"])
