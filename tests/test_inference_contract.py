import hashlib
import json
import random
from dataclasses import asdict

import numpy as np
import pytest
import torch

from chimera import CanonicalCHIMERAv2
from chimera.checkpoint import CheckpointManifest, config_hash, state_schema_hash
from chimera.configuration import InferenceConfig
from chimera.errors import (
    CheckpointCompatibilityError,
    ConfigurationError,
    InputValidationError,
)
from chimera.inference import InferenceRequest, ValidationOptions, run_inference
from chimera.objective_schema import objective_schema_hash


def _model():
    return CanonicalCHIMERAv2(
        d_evo_single=32,
        d_evo_pair=16,
        d_se3=32,
        d_pair_out=16,
        d_mpnn=16,
        n_flow_blocks=1,
        n_flow_steps=2,
        n_mpnn_seqs=1,
        n_domains=2,
        n_modules=2,
        n_mc_dropout=2,
    )


def _inputs():
    length = 8
    rotations = torch.eye(3).reshape(1, 1, 3, 3).expand(1, length, -1, -1).clone()
    translations = (
        torch.arange(length, dtype=torch.float32).reshape(1, length, 1)
        * torch.tensor([3.8, 0.0, 0.0])
    )
    return {
        "msa_tokens": torch.randint(0, 20, (1, 2, length)),
        "initial_pair_features": torch.zeros(1, length, length, 16),
        "source_rotations": rotations,
        "source_translations": translations,
    }


def _config(**overrides):
    values = {
        "model_id": "canonical-chimera-v2",
        "artifact_id": "test-artifact",
        "checkpoint_id": "test-checkpoint",
        "preprocessing_id": "caller-prepared-features-v1",
        "inference_steps": 2,
        "batch_size": 2,
        "random_seed": 19,
    }
    values.update(overrides)
    return InferenceConfig(**values)


def _request(config=None, **overrides):
    values = _inputs()
    values.update(
        config=config or _config(),
        num_candidates=2,
        target_substrate="PHE",
    )
    values.update(overrides)
    return InferenceRequest(**values)


def _complete_manifest(model):
    return CheckpointManifest(
        config_hash=config_hash(model.model_configuration()),
        state_schema_hash=state_schema_hash(model.state_dict()),
        objective_schema_hash=objective_schema_hash(),
        git_commit="test-commit",
        git_worktree_clean=True,
        source_tree_hash="test-source-tree",
        dataset_manifest="test-dataset-manifest",
        dataset_version="test-dataset-v1",
        dataset_hash="test-dataset-hash",
        preprocessing_hash="test-preprocessing-hash",
        environment_hash="test-environment-hash",
        python_version="3.12",
        torch_version="test-torch",
        platform="test-platform",
        numpy_version="test-numpy",
        cuda_available=False,
        deterministic_algorithms_enabled=False,
        cudnn_deterministic=False,
        cudnn_benchmark=False,
        matmul_allow_tf32=False,
        cudnn_allow_tf32=False,
        training_regime="sequence",
        min_validation_examples=30,
        optimizer_class="test-optimizer",
        optimizer_hash="test-optimizer-hash",
        scheduler_class="none",
        scheduler_hash="test-scheduler-hash",
    )


def test_same_seed_reproduces_candidate_bytes_and_provenance(tmp_path):
    torch.manual_seed(41)
    model = _model()
    request = _request()

    first = run_inference(model, request)
    second = run_inference(model, request)

    assert [candidate.digest for candidate in first.candidates] == [
        candidate.digest for candidate in second.candidates
    ]
    assert first.provenance == second.provenance
    assert first.provenance["result_sha256"] == second.provenance["result_sha256"]
    assert first.production_validated is False
    assert first.as_evidence()["scientific_validation"] == "NOT ESTABLISHED"
    artifact_path = tmp_path / "inference-result.json"
    artifact_hash = first.write_json(artifact_path)
    assert hashlib.sha256(artifact_path.read_bytes()).hexdigest() == artifact_hash
    assert json.loads(artifact_path.read_text(encoding="utf-8"))["provenance"] == first.provenance
    with pytest.raises(FileExistsError):
        first.write_json(artifact_path)


def test_different_seed_changes_stochastic_candidate_generation():
    torch.manual_seed(41)
    model = _model()
    first = run_inference(model, _request(_config(random_seed=3)))
    second = run_inference(model, _request(_config(random_seed=4)))

    assert [candidate.digest for candidate in first.candidates] != [
        candidate.digest for candidate in second.candidates
    ]


def test_config_controls_microbatch_flow_steps_and_sequence_temperature(monkeypatch):
    torch.manual_seed(41)
    model = _model()
    original_forward = model.forward
    calls = []
    propagated_masks = []

    def observe_forward(*args, **kwargs):
        propagated_masks.append(
            (
                kwargs["msa_padding_mask"],
                kwargs["residue_mask"],
                kwargs["residue_index"],
            )
        )
        calls.append(
            (
                args[0].shape[0],
                kwargs["n_flow_steps"],
                kwargs["temperature"],
                kwargs["generator"],
                torch.get_num_threads(),
            )
        )
        return original_forward(*args, **kwargs)

    monkeypatch.setattr(model, "forward", observe_forward)
    result = run_inference(
        model,
        _request(
            _config(
                inference_steps=3,
                sequence_temperature=0.6,
                batch_size=2,
                cpu_threads=1,
            ),
            num_candidates=3,
            msa_padding_mask=torch.zeros(1, 2, 8, dtype=torch.bool),
            residue_mask=torch.ones(1, 8, dtype=torch.bool),
            residue_index=torch.arange(8),
        ),
    )

    assert [(count, steps, temperature) for count, steps, temperature, _, _ in calls] == [
        (2, 3, 0.6),
        (1, 3, 0.6),
    ]
    assert all(generator is not None for _, _, _, generator, _ in calls)
    assert all(threads == 1 for _, _, _, _, threads in calls)
    assert len(result.candidates) == 3
    assert propagated_masks[0][0].shape == (2, 2, 8)
    assert propagated_masks[0][1].shape == (2, 8)
    assert propagated_masks[0][2].shape == (2, 8)
    calls.clear()
    run_inference(
        model,
        _request(
            _config(
                inference_steps=3,
                sequence_temperature=0.6,
                batch_size=2,
                microbatch_size=1,
                cpu_threads=1,
            ),
            num_candidates=3,
        ),
    )
    assert [count for count, _, _, _, _ in calls] == [1, 1, 1]


def test_inference_preserves_caller_rng_and_reports_proxy_validation():
    torch.manual_seed(41)
    model = _model()
    request = _request(
        _config(retrieval_id="aligned-index-not-installed"),
        validation_options=ValidationOptions(geometry=True),
    )
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state().clone()

    result = run_inference(model, request)

    assert random.getstate() == python_state
    after_numpy_state = np.random.get_state()
    assert after_numpy_state[0] == numpy_state[0]
    assert np.array_equal(after_numpy_state[1], numpy_state[1])
    assert after_numpy_state[2:] == numpy_state[2:]
    assert torch.equal(torch.get_rng_state(), torch_state)
    assert result.validation_results[0]["type"] == "canonical_backbone_geometry_sanity"
    assert result.optional_backends["retrieval"] == "requested_but_unavailable"
    assert result.provenance["inputs"]["sha256"]
    assert result.provenance["runtime"]["torch_version"]
    assert result.provenance["randomness"]["effective_seed"] == 19
    assert result.provenance["components"]["native_external_backends_used"] == []


def test_inference_rejects_incompatible_checkpoint_manifest(tmp_path):
    model = _model()
    checkpoint_path = tmp_path / "incompatible.pt"
    torch.save(
        {
            "artifact_type": "canonical_training_checkpoint",
            "manifest": asdict(CheckpointManifest()),
            "model_state": {},
            "training": {},
        },
        checkpoint_path,
    )
    request = _request(checkpoint_path=checkpoint_path)

    with pytest.raises(CheckpointCompatibilityError, match="not compatible"):
        run_inference(model, request)


def test_component_checkpoint_identity_is_hashed_and_strictly_loaded(tmp_path):
    torch.manual_seed(41)
    model = _model()
    checkpoint_path = tmp_path / "components.pt"
    model.save(str(checkpoint_path))
    fingerprint = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    request = _request(
        _config(checkpoint_id=f"sha256:{fingerprint}"),
        checkpoint_path=checkpoint_path,
    )

    result = run_inference(_model(), request)

    assert result.checkpoint_sha256 == fingerprint
    assert result.checkpoint_compatibility == "strict-component-load-unverified-training-provenance"
    assert result.provenance["model"]["checkpoint_kind"] == "component_transfer_checkpoint"


def test_complete_training_checkpoint_uses_existing_manifest_contract(tmp_path):
    source_model = _model()
    checkpoint_path = tmp_path / "canonical-training.pt"
    torch.save(
        {
            "artifact_type": "canonical_training_checkpoint",
            "manifest": asdict(_complete_manifest(source_model)),
            "model_state": source_model.state_dict(),
            "training": {
                "component_status": source_model.component_status,
                "objective_calibration": source_model.objective_calibration,
                "objective_status": source_model.objective_status,
            },
        },
        checkpoint_path,
    )
    fingerprint = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    request = _request(
        _config(checkpoint_id=f"sha256:{fingerprint}"),
        checkpoint_path=checkpoint_path,
    )

    result = run_inference(_model(), request)

    assert result.checkpoint_compatibility == "exact-compatible"
    assert result.checkpoint_sha256 == fingerprint
    assert result.provenance["production_evidence"]["checkpoint_provenance_complete"] is True
    assert result.production_validated is False


def test_checkpoint_payload_must_match_its_declared_state_schema(tmp_path):
    source_model = _model()
    checkpoint_path = tmp_path / "wrong-state-schema.pt"
    torch.save(
        {
            "artifact_type": "canonical_training_checkpoint",
            "manifest": asdict(_complete_manifest(source_model)),
            "model_state": {},
            "training": {
                "component_status": source_model.component_status,
                "objective_calibration": source_model.objective_calibration,
                "objective_status": source_model.objective_status,
            },
        },
        checkpoint_path,
    )

    with pytest.raises(CheckpointCompatibilityError, match="does not match its declared state schema"):
        run_inference(
            _model(),
            _request(checkpoint_path=checkpoint_path),
        )


def test_unsupported_resource_knobs_fail_instead_of_being_ignored():
    with pytest.raises(ConfigurationError, match="memory_limit_bytes"):
        run_inference(
            _model(),
            _request(_config(memory_limit_bytes=1024)),
        )


def test_invalid_request_shapes_fail_before_model_execution():
    inputs = _inputs()
    inputs["initial_pair_features"] = torch.zeros(1, 8, 7, 16)
    with pytest.raises(InputValidationError, match="initial_pair_features"):
        InferenceRequest(config=_config(), **inputs)


def test_request_accepts_and_validates_msa_masks_and_residue_indices():
    request = _request(
        msa_padding_mask=torch.zeros(1, 2, 8, dtype=torch.bool),
        residue_mask=torch.ones(1, 8, dtype=torch.bool),
        residue_index=torch.tensor([[0, 1, 2, 5, 6, 7, 8, 9]]),
    )
    assert request.msa_padding_mask.shape == request.msa_tokens.shape
    assert request.residue_mask.shape == (1, 8)
    assert request.residue_index.shape == (1, 8)

    with pytest.raises(InputValidationError, match="residue_mask"):
        _request(residue_mask=torch.ones(1, 7, dtype=torch.bool))


def test_validation_can_be_explicitly_not_requested():
    result = run_inference(
        _model(),
        _request(validation_options=ValidationOptions(geometry=False)),
    )

    assert all(row["status"] == "NOT_REQUESTED" for row in result.validation_results)
    assert result.as_evidence()["all_geometry_valid"] is None
