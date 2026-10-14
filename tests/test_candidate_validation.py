from types import SimpleNamespace

import pytest
import torch

from chimera.candidate_validation import validate_candidate
from chimera.inference import InferenceCandidate, InferenceResult


def _valid_candidate():
    return torch.tensor([1, 2, 3]), torch.zeros(3, 4, 3)


def test_valid_candidate_is_eligible_for_computational_ranking():
    tokens, coordinates = _valid_candidate()
    report = validate_candidate(
        sequence_tokens=tokens,
        backbone_coordinates=coordinates,
        geometry_requested=True,
        geometry_valid=True,
        geometry_metrics={"bond_length_sanity": 1.0},
        constraints=SimpleNamespace(fixed_sequence=torch.tensor([[1, -1, 3]])),
    )
    assert report["eligible_for_ranking"] is True
    assert report["disposition"] == "ACCEPTED_FOR_COMPUTATIONAL_RANKING"
    assert report["checks"]["fixed_sequence_constraints"] == "PASS"


def test_invalid_geometry_is_excluded_from_ranking():
    tokens, coordinates = _valid_candidate()
    report = validate_candidate(
        sequence_tokens=tokens,
        backbone_coordinates=coordinates,
        geometry_requested=True,
        geometry_valid=False,
    )
    assert report["eligible_for_ranking"] is False
    assert report["disposition"] == "REJECTED"
    assert "geometry_sanity" in report["rejection_reasons"]


def test_fixed_sequence_mismatch_is_rejected():
    tokens, coordinates = _valid_candidate()
    report = validate_candidate(
        sequence_tokens=tokens,
        backbone_coordinates=coordinates,
        geometry_requested=False,
        geometry_valid=None,
        constraints=SimpleNamespace(fixed_sequence=torch.tensor([[0, -1, 3]])),
    )
    assert report["checks"]["fixed_sequence_constraints"] == "FAIL"
    assert report["eligible_for_ranking"] is False


def test_nonfinite_backbone_and_invalid_sequence_tokens_are_rejected():
    tokens, coordinates = _valid_candidate()
    tokens[0] = 20
    coordinates[1, 0, 0] = float("nan")
    report = validate_candidate(
        sequence_tokens=tokens,
        backbone_coordinates=coordinates,
        geometry_requested=False,
        geometry_valid=None,
    )
    assert report["checks"]["sequence_token_range"] == "FAIL"
    assert report["checks"]["backbone_finiteness"] == "FAIL"
    assert report["eligible_for_ranking"] is False


def test_geometry_can_be_not_requested_without_misrepresenting_validation():
    tokens, coordinates = _valid_candidate()
    report = validate_candidate(
        sequence_tokens=tokens,
        backbone_coordinates=coordinates,
        geometry_requested=False,
        geometry_valid=None,
    )
    assert report["status"] == "NOT_REQUESTED"
    assert report["valid"] is None
    assert report["eligible_for_ranking"] is True



def test_inference_ranking_excludes_rejected_and_nonfinite_candidates():
    candidates = tuple(
        InferenceCandidate(
            sequence="AAA",
            sequence_tokens=torch.tensor([0, 0, 0]),
            backbone_coordinates=torch.zeros(3, 4, 3),
            digest=f"{index:064x}",
        )
        for index in range(4)
    )
    result = InferenceResult(
        candidates=candidates,
        validation_results=(
            {"eligible_for_ranking": True},
            {"eligible_for_ranking": False},
            {"eligible_for_ranking": True},
            {"eligible_for_ranking": True},
        ),
        evaluation_outputs={
            "expression_efficiency": (0.4, 0.99, 0.8, float("nan"))
        },
        config_hash="a" * 64,
        checkpoint_identity="test-checkpoint",
        checkpoint_sha256=None,
        checkpoint_compatibility="not-supplied",
        random_seed=7,
        provenance={},
        warnings=(),
        optional_backends={},
        approximate_components=(),
        production_validated=False,
    )
    assert result.rank_candidates("expression_efficiency") == (2, 0)
    assert result.rank_candidates("expression_efficiency", descending=False) == (0, 2)
    assert result.eligible_candidate_indices == (0, 2, 3)
    assert result.rankable_candidate_indices_by_objective["expression_efficiency"] == (0, 2)
    assert result.rankable_evaluation_outputs["expression_efficiency"] == (0.4, 0.8)
    with pytest.raises(KeyError, match="was not evaluated"):
        result.rank_candidates("substrate_selectivity")



def test_invalid_fixed_sequence_sentinel_is_rejected():
    tokens, coordinates = _valid_candidate()
    report = validate_candidate(
        sequence_tokens=tokens,
        backbone_coordinates=coordinates,
        geometry_requested=False,
        geometry_valid=None,
        constraints=SimpleNamespace(fixed_sequence=torch.tensor([[-2, -1, 3]])),
    )
    assert report["checks"]["fixed_sequence_constraints"] == "FAIL"
    assert report["eligible_for_ranking"] is False
