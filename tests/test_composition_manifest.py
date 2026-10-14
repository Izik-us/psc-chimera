import json

import pytest
import torch

from chimera import (
    InferenceConfig,
    PreparedInputManifest,
    build_composition_manifest,
    prepare_inference_inputs,
    write_composition_manifest,
)
from chimera.architecture import CHIMERAv2


def _model():
    return CHIMERAv2(
        d_evo_single=32,
        d_evo_pair=16,
        d_se3=32,
        d_pair_out=32,
        d_mpnn=16,
        n_flow_blocks=1,
        n_domains=2,
        n_modules=2,
        n_mc_dropout=2,
        evoformer_n_blocks=1,
    )


def _config():
    return InferenceConfig(
        model_id="composition-test-model",
        artifact_id="research-candidate-v1",
        checkpoint_id="untrained-test-checkpoint",
        preprocessing_id="prep-test-v1",
        max_sequence_length=16,
    )


def test_composition_manifest_is_reproducible_and_never_self_approves():
    manifest = build_composition_manifest(_model(), _config())
    assert manifest["artifact_type"] == "chimera_composition_manifest"
    assert manifest["composition_sha256"]
    assert manifest["release_decision"]["status"] == "RESEARCH_ONLY"
    assert manifest["release_decision"]["production_eligible"] is False
    assert manifest["dependencies"]["manifest_sha256"]
    assert manifest["composition"]["model_state_sha256"]


def test_manifest_rejects_mismatched_preprocessing_identity():
    inputs = prepare_inference_inputs(
        msa_tokens=torch.zeros(1, 1, 4, dtype=torch.long),
        pair_features=torch.zeros(1, 4, 4, 16),
        source_rotations=torch.eye(3).reshape(1, 1, 3, 3).expand(1, 4, 3, 3).clone(),
        source_translations=torch.zeros(1, 4, 3),
        manifest=PreparedInputManifest(
            preprocessing_id="prep-other",
            sequence_id="sequence-test",
            msa_source="synthetic-test",
            pair_feature_source="synthetic-test",
            structure_source="synthetic-test",
            structure_id="synthetic-test",
            residue_mapping_id="mapping-test",
        ),
    )
    with pytest.raises(ValueError, match="preprocessing_id"):
        build_composition_manifest(_model(), _config(), prepared_inputs=inputs)


def test_manifest_artifact_is_immutable_and_integrity_checked(tmp_path):
    manifest = build_composition_manifest(_model(), _config())
    path = tmp_path / "manifest.json"
    digest = write_composition_manifest(manifest, path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["composition_sha256"] == manifest["composition_sha256"]
    assert len(digest) == 64
    with pytest.raises(FileExistsError):
        write_composition_manifest(manifest, path)
    tampered = dict(manifest)
    tampered["readiness"] = {"state": "TRAINED"}
    with pytest.raises(ValueError, match="identity"):
        write_composition_manifest(tampered, tmp_path / "tampered.json")
