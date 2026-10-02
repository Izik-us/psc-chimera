import json

import pytest

from chimera.configuration import InferenceConfig, canonical_config_json
from chimera.errors import ConfigurationError


def _config(**overrides):
    values = {
        "model_id": "chimera-v2",
        "artifact_id": "artifact-sha256:abc",
        "checkpoint_id": "checkpoint-sha256:def",
        "preprocessing_id": "nrps-preprocessing-v1",
    }
    values.update(overrides)
    return InferenceConfig(**values)


def test_equivalent_canonical_configuration_has_same_identity():
    left = _config(objective_set=("expression_efficiency", "substrate_selectivity"))
    right = InferenceConfig.from_dict(
        {
            **left.to_dict(),
            "objective_set": ["expression_efficiency", "substrate_selectivity"],
        }
    )
    assert left.to_json() == right.to_json()
    assert left.config_id == right.config_id


def test_semantically_different_inference_options_change_identity():
    baseline = _config()
    different_batch = _config(batch_size=2)
    different_seed = _config(random_seed=17)
    assert baseline.config_id != different_batch.config_id
    assert baseline.config_id != different_seed.config_id


def test_configuration_records_resource_and_randomness_settings():
    config = _config(
        device="cuda:0",
        dtype="bfloat16",
        batch_size=8,
        microbatch_size=2,
        random_seed=9,
        memory_limit_bytes=4_000_000_000,
    )
    decoded = json.loads(config.to_json())
    assert decoded["device"] == "cuda:0"
    assert decoded["dtype"] == "bfloat16"
    assert decoded["microbatch_size"] == 2
    assert decoded["random_seed"] == 9
    assert decoded["memory_limit_bytes"] == 4_000_000_000


@pytest.mark.parametrize(
    "overrides",
    [
        {"schema_version": 2},
        {"device": "cudaish"},
        {"dtype": "float16"},
        {"deterministic": True, "random_seed": None},
        {"batch_size": 0},
        {"batch_size": 2, "microbatch_size": 3},
        {"objective_set": ("same", "same")},
    ],
)
def test_invalid_inference_configuration_fails_with_field_error(overrides):
    with pytest.raises(ConfigurationError):
        _config(**overrides)


def test_canonical_json_rejects_python_objects_and_non_finite_numbers():
    with pytest.raises(ConfigurationError, match="non-JSON value"):
        canonical_config_json({"device": object()})
    with pytest.raises(ConfigurationError, match="NaN or infinity"):
        canonical_config_json({"threshold": float("nan")})
