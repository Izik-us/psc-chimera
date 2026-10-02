from copy import deepcopy

import pytest
import torch

from chimera.lie import so3_exp
from chimera.objective_schema import (
    OBJECTIVE_SCHEMA,
    objective_schema_hash,
    validate_objective_target,
)
from chimera.pareto_pcgrad import (
    OBJECTIVE_FEATURE_PLAN,
    MergeReadyParetoMultiObjectiveHead,
    ObjectiveFeatureEncoder,
)


def test_structural_objective_features_are_invariant_to_global_rigid_transforms():
    torch.manual_seed(31)
    rotations = so3_exp(torch.randn(2, 7, 3) * 0.2)
    translations = torch.randn(2, 7, 3)
    reference = ObjectiveFeatureEncoder.structural_invariants(rotations, translations)
    global_rotation = so3_exp(torch.randn(2, 3) * 0.8)
    global_translation = torch.randn(2, 3)

    cases = (
        (rotations, translations + global_translation[:, None]),
        (
            global_rotation[:, None] @ rotations,
            torch.einsum("bij,blj->bli", global_rotation, translations),
        ),
        (
            global_rotation[:, None] @ rotations,
            torch.einsum("bij,blj->bli", global_rotation, translations)
            + global_translation[:, None],
        ),
    )
    for transformed_rotations, transformed_translations in cases:
        transformed = ObjectiveFeatureEncoder.structural_invariants(
            transformed_rotations, transformed_translations
        )
        assert torch.allclose(reference, transformed, atol=1e-5, rtol=1e-5)

    assert reference.shape == (2, 7, 32)


def test_structural_objective_features_respond_to_internal_geometry_change():
    rotations = torch.eye(3).reshape(1, 1, 3, 3).expand(1, 6, -1, -1).clone()
    translations = torch.arange(6, dtype=torch.float32).reshape(1, 6, 1) * torch.tensor([3.8, 0.0, 0.0])
    reference = ObjectiveFeatureEncoder.structural_invariants(rotations, translations)
    changed = translations.clone()
    changed[:, 3, 1] += 0.5
    altered = ObjectiveFeatureEncoder.structural_invariants(rotations, changed)

    assert not torch.allclose(reference, altered, atol=1e-5, rtol=1e-5)


def test_objective_attention_pooling_preserves_positional_arrangement():
    torch.manual_seed(4)
    encoder = ObjectiveFeatureEncoder(d_model=16, evolutionary_dim=12).eval()
    sequence = torch.nn.functional.one_hot(torch.tensor([[0, 1, 2, 3, 4]]), 20).float()
    reordered = sequence.flip(1)

    original_embedding = encoder({"sequence": sequence})["sequence"]
    reordered_embedding = encoder({"sequence": reordered})["sequence"]

    assert torch.equal(sequence.sum(dim=1), reordered.sum(dim=1))
    assert not torch.allclose(original_embedding, reordered_embedding, atol=1e-6, rtol=1e-6)


def test_learned_objective_head_declares_only_computed_channels():
    encoder = ObjectiveFeatureEncoder(d_model=16, evolutionary_dim=12)
    head = MergeReadyParetoMultiObjectiveHead(d_model=16)
    encoded = encoder({
        "sequence": torch.randn(2, 5, 20),
        "evolutionary": torch.randn(2, 5, 12),
        "structural": torch.randn(2, 5, 52),
        "substrate": torch.randn(2, 20),
    })
    output = head(encoded)

    assert output.available_objectives == tuple(sorted(OBJECTIVE_FEATURE_PLAN))
    assert "assembly_compatibility" not in output.available_objectives


def test_objective_schema_hash_changes_when_semantics_change_without_shape_change():
    changed_schema = deepcopy(OBJECTIVE_SCHEMA)
    before = changed_schema["expression_efficiency"]["target_shape"]
    changed_schema["expression_efficiency"]["target_normalization"] = "z-score"

    assert changed_schema["expression_efficiency"]["target_shape"] == before
    assert objective_schema_hash(changed_schema) != objective_schema_hash(OBJECTIVE_SCHEMA)


@pytest.mark.parametrize(
    "target,error_type",
    [
        (torch.tensor([[0.2], [0.8]], dtype=torch.float32), ValueError),
        (torch.tensor([0, 1], dtype=torch.int64), TypeError),
        (torch.tensor([float("nan"), 0.5], dtype=torch.float32), ValueError),
        (torch.tensor([0.5, float("inf")], dtype=torch.float32), ValueError),
        (torch.tensor([-0.1, 0.5], dtype=torch.float32), ValueError),
        (torch.tensor([0.5, 1.1], dtype=torch.float32), ValueError),
    ],
)
def test_objective_targets_reject_invalid_shape_dtype_finite_and_range(target, error_type):
    prediction = torch.tensor([0.25, 0.75], dtype=torch.float32)
    with pytest.raises(error_type):
        validate_objective_target("expression_efficiency", prediction, target)


def test_objective_targets_reject_nonfinite_predictions_and_assembly_training():
    target = torch.tensor([0.25, 0.75], dtype=torch.float32)
    with pytest.raises(ValueError, match="finite"):
        validate_objective_target(
            "expression_efficiency", torch.tensor([float("nan"), 0.5]), target
        )
    with pytest.raises(ValueError, match="deterministic-only"):
        validate_objective_target("assembly_compatibility", target, target)