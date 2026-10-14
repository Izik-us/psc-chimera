import pytest
import torch

from chimera import (
    InferenceConfig,
    InputValidationError,
    PreparedInputManifest,
    build_inference_request,
    prepare_inference_inputs,
)


def _manifest():
    return PreparedInputManifest(
        preprocessing_id="prep-v1",
        sequence_id="seq-test",
        msa_source="synthetic-test-msa",
        pair_feature_source="synthetic-test-pair",
        structure_source="synthetic-test-frames",
        structure_id="test-structure",
        residue_mapping_id="mapping-v1",
    )


def _inputs(length=8):
    msa = torch.zeros(1, 1, length, dtype=torch.long)
    pair = torch.zeros(1, length, length, 128)
    R = torch.eye(3).reshape(1, 1, 3, 3).expand(1, length, 3, 3).clone()
    t = torch.zeros(1, length, 3)
    return msa, pair, R, t


def test_prepared_inputs_validate_so3_frames():
    msa, pair, R, t = _inputs()
    prepared = prepare_inference_inputs(
        msa_tokens=msa,
        pair_features=pair,
        source_rotations=R,
        source_translations=t,
        manifest=_manifest(),
    )
    assert prepared.length == 8
    assert prepared.identity


def test_prepared_inputs_reject_non_rotation_frames():
    msa, pair, R, t = _inputs()
    R[:, 0, 0, 0] = 2.0
    with pytest.raises(InputValidationError, match=r"SO\(3\)"):
        prepare_inference_inputs(
            msa_tokens=msa,
            pair_features=pair,
            source_rotations=R,
            source_translations=t,
            manifest=_manifest(),
        )


def test_request_requires_matching_preprocessing_identity():
    msa, pair, R, t = _inputs()
    prepared = prepare_inference_inputs(
        msa_tokens=msa,
        pair_features=pair,
        source_rotations=R,
        source_translations=t,
        manifest=_manifest(),
    )
    config = InferenceConfig(
        model_id="canonical-test",
        artifact_id="artifact-test",
        checkpoint_id="checkpoint-test",
        preprocessing_id="different-preprocessor",
        max_sequence_length=32,
    )
    with pytest.raises(InputValidationError, match="preprocessing_id"):
        build_inference_request(prepared, config)
