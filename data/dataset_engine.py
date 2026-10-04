"""Canonical dataset-engine foundation for CHIMERA.

This module provides the dataset and provenance contracts required to support
structural ingestion, MSA linking, NRPS annotations, split manifests, and
versioned dataset construction without redesigning the model stack itself.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Mapping, Sequence

UNKNOWN = "UNKNOWN"


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def stable_hash(value: Any) -> str:
    """Return a deterministic SHA-256 hash for an arbitrary JSON-serializable payload."""
    payload = _canonical_json(value).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class ProvenanceRecord:
    """Immutable provenance for any processed biological artifact."""

    source_database: str
    source_accession: str
    source_version: str
    download_timestamp: str
    source_file_checksum: str | None = None
    processed_file_checksum: str | None = None
    processing_pipeline_version: str = "CHIMERA-DATASET-v0.1"
    parser_version: str = "UNKNOWN"
    normalization_version: str = "UNKNOWN"
    qc_version: str = "UNKNOWN"
    source_release_date: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DataQualityResult:
    """Machine-readable QC verdict for a structure or annotation record."""

    status: str
    reasons: list[str] = field(default_factory=list)
    qc_version: str = "UNKNOWN"
    additional_metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status not in {"accepted", "rejected", "quarantined"}:
            raise ValueError("status must be accepted, rejected, or quarantined")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AcquisitionManifest:
    """Explicit acquisition manifest for raw external-source retrieval."""

    source: str
    query_filter: str
    source_release: str | None
    record_identifiers: Sequence[str]
    requested_timestamp: str
    download_status: str = "requested"
    checksum: str | None = None
    file_path: str | None = None
    processing_status: str = "requested"
    failure_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DatasetSplit:
    """Split manifest for safe train/validation/test/OOD assignment."""

    name: str
    record_ids: Sequence[str]
    split_policy: str
    leakage_control: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DatasetVersion:
    """Version metadata for a reproducible dataset release."""

    name: str
    source_versions: dict[str, str]
    processing_version: str
    annotation_versions: dict[str, str] = field(default_factory=dict)
    qc_config: dict[str, Any] = field(default_factory=dict)
    split_algorithm: str = "UNKNOWN"
    feature_generation_version: str = "UNKNOWN"
    manifest_checksums: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class StructuralManifest:
    """Canonical per-chain structural record used for downstream geometry and splits."""

    structure_id: str
    source: str
    source_version: str
    release_date: str
    experimental_method: str
    resolution: float | None
    entity_id: str | None
    chain_id: str | None
    assembly_id: str | None
    sequence_sha256: str
    coordinate_sha256: str
    processing_version: str
    quality_flags: Sequence[str]
    num_residues: int
    num_resolved_residues: int
    num_missing_residues: int
    num_missing_backbone_atoms: int
    ligand_count: int
    polymer_type: str
    organism: str | None
    taxonomy_id: str | None
    split: str
    provenance: ProvenanceRecord | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        if self.provenance is not None:
            payload["provenance"] = self.provenance.to_dict()
        return payload


@dataclass(frozen=True)
class ExternalSourceInventory:
    """Machine-readable inventory of authoritative external data sources."""

    sources: Sequence[Mapping[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {"sources": [dict(source) for source in self.sources]}


def build_acquisition_manifest(
    *,
    source: str,
    query_filter: str,
    source_release: str | None,
    record_identifiers: Iterable[str],
    requested_timestamp: str,
    download_status: str = "requested",
    checksum: str | None = None,
    file_path: str | None = None,
    processing_status: str = "requested",
    failure_reason: str | None = None,
) -> AcquisitionManifest:
    """Construct a deterministic acquisition manifest from a source query."""
    return AcquisitionManifest(
        source=source,
        query_filter=query_filter,
        source_release=source_release,
        record_identifiers=list(record_identifiers),
        requested_timestamp=requested_timestamp,
        download_status=download_status,
        checksum=checksum,
        file_path=file_path,
        processing_status=processing_status,
        failure_reason=failure_reason,
    )


def build_dataset_version(
    *,
    name: str,
    source_versions: Mapping[str, str],
    processing_version: str,
    annotation_versions: Mapping[str, str] | None = None,
    qc_config: Mapping[str, Any] | None = None,
    split_algorithm: str = "UNKNOWN",
    feature_generation_version: str = "UNKNOWN",
    manifest_checksums: Mapping[str, str] | None = None,
) -> DatasetVersion:
    """Factory for pinned dataset-version metadata."""
    return DatasetVersion(
        name=name,
        source_versions=dict(source_versions),
        processing_version=processing_version,
        annotation_versions=dict(annotation_versions or {}),
        qc_config=dict(qc_config or {}),
        split_algorithm=split_algorithm,
        feature_generation_version=feature_generation_version,
        manifest_checksums=dict(manifest_checksums or {}),
    )


def summarize_records(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Compute simple dataset statistics for manifest-driven dataset builds."""
    if not records:
        return {
            "record_count": 0,
            "unique_record_count": 0,
            "source_count": 0,
            "structure_count": 0,
            "chain_count": 0,
            "split_counts": {},
        }

    source_names = {str(record.get("source", "UNKNOWN")) for record in records}
    split_counts: dict[str, int] = {}
    for record in records:
        split_name = str(record.get("split", "UNKNOWN"))
        split_counts[split_name] = split_counts.get(split_name, 0) + 1

    return {
        "record_count": len(records),
        "unique_record_count": len({stable_hash(record) for record in records}),
        "source_count": len(source_names),
        "structure_count": sum(1 for record in records if "structure_id" in record),
        "chain_count": sum(1 for record in records if "chain_id" in record),
        "split_counts": split_counts,
    }
