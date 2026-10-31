import json
from pathlib import Path

import pytest

from chimera.configuration import InferenceConfig
from chimera.design_runs import DesignTask
from chimera.errors import ConfigurationError
from chimera.design_runs import runner


def _task(**overrides):
    values = {
        "task_id": "runner-smoke",
        "objective": "NRPS module/domain-linker compatibility",
        "target_context": {"substrate": "PHE"},
        "input_bundle": "inputs.json",
        "checkpoint_path": "checkpoint.pt",
        "inference_config": InferenceConfig(
            model_id="canonical-chimera-v2",
            artifact_id="fixture-artifact",
            checkpoint_id="fixture-checkpoint",
            preprocessing_id="prep-v1",
            batch_size=2,
            max_sequence_length=8,
            objective_set=("structural_stability",),
        ),
        "candidate_count": 2,
        "input_shape": {"sequence_length": 2, "msa_depth": 1, "pair_channels": 3},
        "soft_objectives": ["structural_stability"],
    }
    values.update(overrides)
    return DesignTask(**values)


def _write_inputs(path: Path):
    document = {
        "schema_version": 1,
        "manifest": {
            "preprocessing_id": "prep-v1",
            "sequence_id": "fixture-sequence",
            "msa_source": "fixture-msa",
            "pair_feature_source": "fixture-pair",
            "structure_source": "fixture-structure",
            "structure_id": "fixture-structure-v1",
            "residue_mapping_id": "fixture-map-v1",
        },
        "tensors": {
            "msa_tokens": [[[0, 1]]],
            "pair_features": [[[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                               [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]],
            "source_rotations": [[
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            ]],
            "source_translations": [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]],
        },
    }
    path.write_text(json.dumps(document), encoding="utf-8")


class _FakeResult:
    candidates = (object(), object())
    validation_results = ({}, {})
    evaluation_outputs = {"structural_stability": (0.2, 0.3)}
    eligible_candidate_indices = (0, 1)
    rejected_candidate_indices = ()
    checkpoint_compatibility = "exact-compatible"
    checkpoint_sha256 = "a" * 64
    warnings = ()
    optional_backends = {"retrieval": "not_requested"}

    def as_dict(self):
        return {"artifact_type": "fake-result"}

    def write_json(self, path):
        path = Path(path)
        path.write_text(json.dumps(self.as_dict()), encoding="utf-8")
        import hashlib
        return hashlib.sha256(path.read_bytes()).hexdigest()


def test_run_design_task_executes_and_publishes_auditable_result(tmp_path, monkeypatch):
    task_path = tmp_path / "task.json"
    input_path = tmp_path / "inputs.json"
    checkpoint_path = tmp_path / "checkpoint.pt"
    _write_inputs(input_path)
    checkpoint_path.write_bytes(b"checkpoint-fixture")
    task = _task()
    task_path.write_text(json.dumps(task.to_dict()), encoding="utf-8")
    calls = []

    def fake_inference(checkpoint, request):
        calls.append((Path(checkpoint), request))
        return _FakeResult()

    monkeypatch.setattr(runner, "run_inference_from_checkpoint", fake_inference)
    report = runner.run_design_task(task_path, runs_root=tmp_path / "runs")

    assert report["status"] == "COMPLETED"
    assert report["generation_started"] is True
    assert report["manifest"]["checkpoint_loaded"] is True
    assert report["manifest"]["generated_candidate_count"] == 2
    assert (Path(report["run_dir"]) / "inference-result.json").is_file()
    assert (Path(report["run_dir"]) / "report.json").is_file()
    assert len(calls) == 1
    assert calls[0][1].num_candidates == 2
    assert report["event_count"] == 4


def test_run_fails_closed_for_unsupported_hard_constraint(tmp_path, monkeypatch):
    task_path = tmp_path / "task.json"
    _write_inputs(tmp_path / "inputs.json")
    (tmp_path / "checkpoint.pt").write_bytes(b"checkpoint-fixture")
    task = _task(hard_constraints=({"id": "substrate-specificity", "kind": "must_match"},))
    task_path.write_text(json.dumps(task.to_dict()), encoding="utf-8")
    monkeypatch.setattr(
        runner, "run_inference_from_checkpoint",
        lambda *_: pytest.fail("inference must not start for unsupported hard constraints"),
    )

    with pytest.raises(ConfigurationError, match="no implemented canonical evaluator"):
        runner.run_design_task(task_path, runs_root=tmp_path / "runs")

    run_dirs = list((tmp_path / "runs").iterdir())
    manifest = json.loads((run_dirs[0] / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "FAILED"
    assert manifest["generation_started"] is False
