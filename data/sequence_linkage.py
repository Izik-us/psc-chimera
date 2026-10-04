"""Alignment-based structure-to-sequence database linkage."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable

from Bio.Align import PairwiseAligner


@dataclass(frozen=True)
class SequenceCandidate:
    accession: str
    sequence: str
    source: str
    source_version: str


@dataclass(frozen=True)
class SequenceLinkage:
    structure_id: str
    chain_id: str
    entity_id: str
    sequence_sha256: str
    external_accession: str | None
    match_method: str
    sequence_identity: float
    coverage: float
    mismatches: int
    gaps: int
    confidence: float
    source: str | None
    source_version: str | None
    reported_accession: str | None = None
    target_to_candidate_positions: tuple[int | None, ...] = ()


def _alignment_metrics(target: str, candidate: str) -> tuple[float, float, int, int, tuple[int | None, ...]]:
    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.match_score = 2.0
    aligner.mismatch_score = -1.0
    aligner.open_gap_score = -5.0
    aligner.extend_gap_score = -0.5
    alignment = aligner.align(target, candidate)[0]
    matches = mismatches = gaps = aligned_target = 0
    target_to_candidate: list[int | None] = [None] * len(target)
    coordinates = alignment.coordinates
    for start in range(coordinates.shape[1] - 1):
        target_start, target_end = coordinates[0, start:start + 2]
        candidate_start, candidate_end = coordinates[1, start:start + 2]
        target_delta = int(target_end - target_start)
        candidate_delta = int(candidate_end - candidate_start)
        if target_delta and candidate_delta:
            width = min(target_delta, candidate_delta)
            for offset in range(width):
                target_to_candidate[int(target_start) + offset] = int(candidate_start) + offset
                if target[int(target_start) + offset] == candidate[int(candidate_start) + offset]:
                    matches += 1
                else:
                    mismatches += 1
            aligned_target += width
            gaps += abs(target_delta - candidate_delta)
        elif target_delta or candidate_delta:
            gaps += max(target_delta, candidate_delta)
    identity = matches / max(matches + mismatches, 1)
    coverage = aligned_target / max(len(target), 1)
    return identity, min(coverage, 1.0), mismatches, gaps, tuple(target_to_candidate)


def link_structure_sequence(
    *,
    structure_id: str,
    chain_id: str,
    entity_id: str,
    sequence: str,
    candidates: Iterable[SequenceCandidate],
    minimum_coverage: float = 0.5,
    reported_accession: str | None = None,
) -> SequenceLinkage:
    """Select the best sequence-aligned accession and retain explicit confidence metrics."""
    target = "".join(letter for letter in sequence.upper() if letter.isalpha())
    if not target:
        raise ValueError("target sequence must contain amino-acid letters")
    evaluated = []
    for candidate in candidates:
        candidate_sequence = "".join(letter for letter in candidate.sequence.upper() if letter.isalpha())
        if not candidate_sequence:
            continue
        identity, coverage, mismatches, gaps, position_map = _alignment_metrics(target, candidate_sequence)
        evaluated.append((coverage, identity, -gaps, candidate, mismatches, gaps, position_map))

    sequence_hash = hashlib.sha256(target.encode("ascii")).hexdigest()
    if not evaluated:
        return SequenceLinkage(
            structure_id, chain_id, entity_id, sequence_hash, None, "no_candidate",
            0.0, 0.0, 0, 0, 0.0, None, None, reported_accession, tuple(None for _ in target),
        )
    coverage, identity, negative_gaps, candidate, mismatches, gaps, position_map = max(
        evaluated, key=lambda item: (item[0], item[1], item[2], item[3].accession)
    )
    accepted = coverage >= minimum_coverage
    return SequenceLinkage(
        structure_id=structure_id,
        chain_id=chain_id,
        entity_id=entity_id,
        sequence_sha256=sequence_hash,
        external_accession=candidate.accession if accepted else None,
        match_method="global_pairwise_alignment",
        sequence_identity=identity,
        coverage=coverage,
        mismatches=mismatches,
        gaps=gaps,
        confidence=identity * coverage,
        source=candidate.source if accepted else None,
        source_version=candidate.source_version if accepted else None,
        reported_accession=reported_accession,
        target_to_candidate_positions=position_map,
    )