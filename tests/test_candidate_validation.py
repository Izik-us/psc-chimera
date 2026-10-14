from types import SimpleNamespace

import torch

from chimera.candidate_validation import validate_candidate


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
