import json

import pytest

from chimera.configuration import InferenceConfig
from chimera.design_runs import DesignTask, build_design_plan, load_design_task
from chimera.errors import ConfigurationError


def _config(**overrides):
    values = {
        "model_id": "canonical-chimera-v2",
        "artifact_id": "test-artifact",
        "checkpoint_id": "test-checkpoint",
        "preprocessing_id": "prepared-input-v1",
        "batch_size": 4,
        "max_sequence_length": 128,
        "random_seed": 13,
    }
    values.update(overrides)
    return InferenceConfig(**values)


def _task(**overrides):
    values = {
        "task_id": "nrps-linker-smoke",
        "objective": "NRPS module/domain-linker compatibility",
        "target_context": {"substrate": "PHE", "module": "test-module"},
        "input_bundle": "inputs.json",
        "checkpoint_path": "checkpoint.pt",
        "inference_config": _config(),
        "candidate_count": 4,
        "input_shape": {"sequence_length": 32, "msa_depth": 8, "pair_channels": 16},
        "hard_constraints": [{"id": "sequence_length", "kind": "maximum", "value": 128}],
        "soft_objectives": ["structural_stability"],
    }
    values.update(overrides)
    return DesignTask(**values)


def test_task_round_trip_preserves_versioned_contract():
    task = _task()
    loaded = DesignTask.from_dict(task.to_dict())
    assert loaded.to_dict() == task.to_dict()


def test_task_rejects_unbounded_or_invalid_candidate_count():
    with pytest.raises(ConfigurationError, match="candidate_count"):
        _task(candidate_count=0)
    with pytest.raises(ConfigurationError, match="candidate_count"):
        _task(candidate_count=100001)


def test_planner_hashes_existing_inputs_and_checkpoint_without_loading_model(tmp_path):
    task_file = tmp_path / "task.json"
    input_path = tmp_path / "inputs.json"
    checkpoint_path = tmp_path / "checkpoint.pt"
    input_path.write_text('{"fixture": true}', encoding="utf-8")
    checkpoint_path.write_bytes(b"checkpoint fixture")
    task = _task()
    task_file.write_text(json.dumps(task.to_dict()), encoding="utf-8")

    plan = build_design_plan(task, task_file=task_file)

    assert plan["readiness"] == "PLAN_ONLY"
    assert plan["generation_started"] is False
    assert plan["checkpoint_loaded"] is False
    assert plan["input_bundle"]["sha256"]
    assert plan["checkpoint"]["sha256"]
    assert plan["plan_sha256"]
    assert plan["resource_estimate"]["status"] == "LOWER_BOUND_ONLY"
    assert plan["scientific_validation"] == "NOT ESTABLISHED"


def test_planner_blocks_missing_references_and_discloses_unsupported_controls(tmp_path):
    task_file = tmp_path / "task.json"
    task = _task(
        inference_config=_config(worker_count=1, memory_limit_bytes=1024, retrieval_id="index-1")
    )
    task_file.write_text(json.dumps(task.to_dict()), encoding="utf-8")

    plan = build_design_plan(task, task_file=task_file)

    assert plan["readiness"] == "BLOCKED"
    assert any("input bundle file does not exist" in item for item in plan["blockers"])
    assert any("checkpoint file does not exist" in item for item in plan["blockers"])
    assert any("worker_count is unsupported" in item for item in plan["blockers"])
    assert any("memory_limit_bytes is not enforced" in item for item in plan["blockers"])
    assert any("currently disables RAG" in item for item in plan["warnings"])


def test_load_task_rejects_invalid_json(tmp_path):
    task_file = tmp_path / "task.json"
    task_file.write_text("{", encoding="utf-8")
    with pytest.raises(ConfigurationError, match="valid JSON"):
        load_design_task(task_file)
