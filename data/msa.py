"""Provenance-preserving MSA normalization and deterministic model inputs."""

from __future__ import annotations

import hashlib
import random
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence

import torch

AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_TOKEN = {letter: index for index, letter in enumerate(AA_ALPHABET)}
AA_TO_TOKEN.update({"X": 20, "-": 21, ".": 21})
PAD_TOKEN = 22


@dataclass(frozen=True)
class MSAConfig:
    minimum_coverage: float = 0.5
    maximum_gap_fraction: float = 0.5
    identity_cluster_threshold: float = 0.9
    maximum_depth: int = 512
    pad_to_maximum_depth: bool = True
    seed: int = 0

    def __post_init__(self) -> None:
        if not 0.0 <= self.minimum_coverage <= 1.0:
            raise ValueError("minimum_coverage must be in [0, 1]")
        if not 0.0 <= self.maximum_gap_fraction <= 1.0:
            raise ValueError("maximum_gap_fraction must be in [0, 1]")
        if not 0.0 <= self.identity_cluster_threshold <= 1.0:
            raise ValueError("identity_cluster_threshold must be in [0, 1]")
        if self.maximum_depth < 1 or self.seed < 0:
            raise ValueError("maximum_depth must be positive and seed non-negative")


@dataclass
class MSARecord:
    msa_family_id: str
    target_sequence_id: str
    source: str
    source_version: str
    raw_alignment: str
    raw_rows: list[str]
    normalized_rows: list[str]
    model_ready_tokens: torch.Tensor
    model_ready_mask: torch.Tensor
    query_residue_to_msa_column: list[int]
    num_rows: int
    alignment_length: int
    effective_sequence_count: int
    neff: float
    identity_distribution: dict[str, float]
    coverage_statistics: dict[str, float]
    gap_statistics: dict[str, float]
    taxonomy_distribution: dict[str, int]
    generation_method: str
    generation_version: str
    raw_sha256: str
    model_ready_row_count: int
    dropped_row_count: int
    metadata: dict[str, str] = field(default_factory=dict)
    raw_sequence_records: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self, *, include_arrays: bool = False) -> dict:
        payload = asdict(self)
        if include_arrays:
            payload["model_ready_tokens"] = self.model_ready_tokens.tolist()
            payload["model_ready_mask"] = self.model_ready_mask.tolist()
        else:
            payload.pop("model_ready_tokens")
            payload.pop("model_ready_mask")
        return payload


def _parse_fasta_rows(raw_alignment: str) -> list[str]:
    rows: list[str] = []
    current: list[str] = []
    for line in raw_alignment.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if current:
                rows.append("".join(current))
                current = []
        else:
            current.append(line)
    if current:
        rows.append("".join(current))
    if not rows:
        raise ValueError("alignment contains no FASTA rows")
    return rows


def _normalize_a3m_row(row: str) -> str:
    normalized = "".join(character for character in row if not character.islower())
    normalized = normalized.upper().replace(".", "-")
    invalid = set(normalized) - set(AA_ALPHABET + "X-")
    if invalid:
        raise ValueError(f"alignment contains unsupported symbols: {sorted(invalid)}")
    return normalized


def _identity(left: str, right: str) -> float:
    paired = [(a, b) for a, b in zip(left, right) if a != "-" and b != "-"]
    return sum(a == b for a, b in paired) / max(len(paired), 1)


def _cluster_rows(rows: Sequence[str], threshold: float) -> tuple[list[str], int]:
    representatives: list[str] = []
    for row in rows:
        if not any(_identity(row, representative) >= threshold for representative in representatives):
            representatives.append(row)
    return representatives, len(representatives)


def build_msa_record(
    raw_alignment: str,
    *,
    msa_family_id: str,
    target_sequence_id: str,
    source: str,
    source_version: str,
    target_sequence: str | None = None,
    taxonomy_by_row: Mapping[int, str] | None = None,
    raw_sequence_records: Sequence[Mapping[str, Any]] = (),
    generation_method: str = "external-alignment-normalization",
    generation_version: str = "msa-normalization-v0.1",
    config: MSAConfig | None = None,
) -> MSARecord:
    """Create raw, normalized and padded model-ready views without losing input rows."""
    config = config or MSAConfig()
    raw_rows = _parse_fasta_rows(raw_alignment)
    normalized_rows = [_normalize_a3m_row(row) for row in raw_rows]
    lengths = {len(row) for row in normalized_rows}
    if len(lengths) != 1:
        raise ValueError("normalized MSA rows must have equal alignment length")
    alignment_length = lengths.pop()
    if alignment_length < 1:
        raise ValueError("alignment length must be positive")

    query = normalized_rows[0]
    if target_sequence is not None:
        target = _normalize_a3m_row(target_sequence.replace(" ", ""))
        if query.replace("-", "") != target.replace("-", ""):
            raise ValueError("first MSA row must preserve the target sequence")

    kept = [query]
    for row in normalized_rows[1:]:
        coverage = sum(letter != "-" for letter in row) / alignment_length
        gap_fraction = sum(letter == "-" for letter in row) / alignment_length
        if coverage >= config.minimum_coverage and gap_fraction <= config.maximum_gap_fraction:
            kept.append(row)
    representatives, neff = _cluster_rows(kept, config.identity_cluster_threshold)

    if len(representatives) > config.maximum_depth:
        randomizer = random.Random(config.seed)
        sampled = randomizer.sample(representatives[1:], config.maximum_depth - 1)
        model_rows = [representatives[0], *sampled]
    else:
        model_rows = representatives

    target_row = model_rows[0]
    query_mapping = [index for index, letter in enumerate(target_row) if letter != "-"]
    target_identity = [_identity(target_row, row) for row in model_rows]
    coverages = [sum(letter != "-" for letter in row) / alignment_length for row in model_rows]
    gap_fractions = [sum(letter == "-" for letter in row) / alignment_length for row in model_rows]

    depth = config.maximum_depth if config.pad_to_maximum_depth else len(model_rows)
    tokens = torch.full((depth, alignment_length), PAD_TOKEN, dtype=torch.long)
    row_mask = torch.zeros((depth, alignment_length), dtype=torch.bool)
    for row_index, row in enumerate(model_rows):
        tokens[row_index] = torch.tensor([AA_TO_TOKEN[letter] for letter in row], dtype=torch.long)
        row_mask[row_index] = True

    taxonomy_distribution: dict[str, int] = {}
    for row_index in range(len(raw_rows)):
        taxonomy = (taxonomy_by_row or {}).get(row_index)
        if taxonomy is not None:
            taxonomy_distribution[taxonomy] = taxonomy_distribution.get(taxonomy, 0) + 1

    return MSARecord(
        msa_family_id=msa_family_id,
        target_sequence_id=target_sequence_id,
        source=source,
        source_version=source_version,
        raw_alignment=raw_alignment,
        raw_rows=raw_rows,
        normalized_rows=normalized_rows,
        model_ready_tokens=tokens,
        model_ready_mask=row_mask,
        query_residue_to_msa_column=query_mapping,
        num_rows=len(raw_rows),
        alignment_length=alignment_length,
        effective_sequence_count=neff,
        neff=float(neff),
        identity_distribution={
            "mean_to_query": float(sum(target_identity) / len(target_identity)),
            "minimum_to_query": float(min(target_identity)),
            "maximum_to_query": float(max(target_identity)),
        },
        coverage_statistics={
            "mean": float(sum(coverages) / len(coverages)),
            "minimum": float(min(coverages)),
        },
        gap_statistics={
            "mean_fraction": float(sum(gap_fractions) / len(gap_fractions)),
            "maximum_fraction": float(max(gap_fractions)),
        },
        taxonomy_distribution=taxonomy_distribution,
        generation_method=generation_method,
        generation_version=generation_version,
        raw_sha256=hashlib.sha256(raw_alignment.encode("utf-8")).hexdigest(),
        model_ready_row_count=len(model_rows),
        dropped_row_count=len(normalized_rows) - len(model_rows),
        metadata={"query_preserved": "true", "padding_token": str(PAD_TOKEN)},
        raw_sequence_records=[dict(record) for record in raw_sequence_records],
    )