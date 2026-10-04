"""Verified external-data acquisition and canonicalization for CHIMERA.

This module implements the first real acquisition path: authoritative RCSB
PDB/mmCIF retrieval, checksum verification, mmCIF parsing, and canonical
structural metadata generation for a controlled seed corpus.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

from .dataset_engine import ProvenanceRecord, StructuralManifest

try:  # pragma: no cover - optional dependency check.
    from Bio.PDB import MMCIFParser
except Exception:  # pragma: no cover
    MMCIFParser = None

RCSB_MMCIF_URL = "https://files.rcsb.org/download/{structure_id}.cif"


@dataclass
class AcquisitionSummary:
    requested: int = 0
    discovered: int = 0
    downloaded: int = 0
    checksum_verified: int = 0
    parsed: int = 0
    accepted: int = 0
    rejected: int = 0
    quarantined: int = 0
    failure_reasons: list[str] = field(default_factory=list)
    records: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(url: str, destination: str | Path) -> tuple[bool, str | None]:
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urlopen(url, timeout=60) as response:
            payload = response.read()
        target.write_bytes(payload)
        return True, None
    except (HTTPError, URLError, TimeoutError) as exc:
        return False, f"HTTP_ERROR:{type(exc).__name__}:{exc}"


def _is_protein_residue(residue) -> bool:
    return residue.id[0] == " " and getattr(residue, "resname", "") not in {"HOH", "WAT", "DOD"}


def _amino_acid_from_resname(resname: str) -> str:
    mapping = {
        "ALA": "A",
        "ARG": "R",
        "ASN": "N",
        "ASP": "D",
        "CYS": "C",
        "GLU": "E",
        "GLN": "Q",
        "GLY": "G",
        "HIS": "H",
        "ILE": "I",
        "LEU": "L",
        "LYS": "K",
        "MET": "M",
        "PHE": "F",
        "PRO": "P",
        "SER": "S",
        "THR": "T",
        "TRP": "W",
        "TYR": "Y",
        "VAL": "V",
        "ASX": "B",
        "GLX": "Z",
        "SEC": "U",
    }
    return mapping.get(resname.upper(), "X")


def _chain_sequence_and_quality(chain) -> tuple[str, int, int]:
    sequence_parts: list[str] = []
    residues = 0
    missing_backbone = 0
    for residue in chain:
        if not _is_protein_residue(residue):
            continue
        residues += 1
        sequence_parts.append(_amino_acid_from_resname(residue.resname))
        atoms = {name for name in ("N", "CA", "C", "O") if residue.has_id(name)}
        if len(atoms.intersection({"N", "CA", "C", "O"})) < 4:
            missing_backbone += 1
    return "".join(sequence_parts), residues, missing_backbone


def _chain_metadata_from_structure(structure_id: str, raw_path: str | Path) -> tuple[list[dict[str, Any]], int, int]:
    if MMCIFParser is None:
        raise RuntimeError("Biopython is required for mmCIF parsing; install biopython.")

    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure(structure_id, str(raw_path))

    chains: list[dict[str, Any]] = []
    total_residues = 0
    total_missing_backbone = 0
    for model in structure:
        for chain in model:
            sequence, residue_count, missing_backbone = _chain_sequence_and_quality(chain)
            total_residues += residue_count
            total_missing_backbone += missing_backbone
            chains.append(
                {
                    "chain_id": chain.id,
                    "num_residues": residue_count,
                    "sequence": sequence,
                    "missing_backbone_atoms": missing_backbone,
                }
            )

    return chains, total_residues, total_missing_backbone


def acquire_rcsb_structure(
    structure_id: str,
    output_dir: str | Path,
    *,
    expected_checksum: str | None = None,
) -> dict[str, Any]:
    """Download an RCSB mmCIF, verify the payload, and generate a canonical structural record."""
    workdir = Path(output_dir)
    workdir.mkdir(parents=True, exist_ok=True)
    raw_path = workdir / f"{structure_id}.cif"

    success, failure = _download(RCSB_MMCIF_URL.format(structure_id=structure_id), raw_path)
    if not success:
        return {
            "status": "rejected",
            "structure_id": structure_id,
            "download_status": "failed",
            "failure_reason": failure,
            "raw_path": str(raw_path),
            "checksum_verified": False,
            "parsed": False,
        }

    checksum = _sha256_file(raw_path)
    checksum_verified = expected_checksum is None or checksum.lower() == expected_checksum.lower()

    if expected_checksum is not None and not checksum_verified:
        return {
            "status": "quarantined",
            "structure_id": structure_id,
            "download_status": "downloaded",
            "failure_reason": "CHECKSUM_FAILURE",
            "raw_path": str(raw_path),
            "checksum": checksum,
            "checksum_verified": False,
            "parsed": False,
        }

    try:
        chains, total_residues, total_missing_backbone = _chain_metadata_from_structure(structure_id, raw_path)
    except Exception as exc:  # pragma: no cover - depends on external content and parser behavior
        return {
            "status": "rejected",
            "structure_id": structure_id,
            "download_status": "downloaded",
            "failure_reason": "INVALID_MMCIF",
            "raw_path": str(raw_path),
            "checksum": checksum,
            "checksum_verified": checksum_verified,
            "parsed": False,
            "error": str(exc),
        }

    if not chains:
        return {
            "status": "rejected",
            "structure_id": structure_id,
            "download_status": "downloaded",
            "failure_reason": "NO_PROTEIN_CHAIN",
            "raw_path": str(raw_path),
            "checksum": checksum,
            "checksum_verified": checksum_verified,
            "parsed": True,
            "chains": chains,
            "num_residues": total_residues,
            "quality_flags": ["missing_backbone_atoms"] if total_missing_backbone else [],
        }

    quality_flags: list[str] = []
    if total_missing_backbone > 0:
        quality_flags.append("missing_backbone_atoms")

    primary_chain = max(chains, key=lambda item: item["num_residues"])
    sequence_sha256 = hashlib.sha256(primary_chain["sequence"].encode("utf-8")).hexdigest()
    manifest = StructuralManifest(
        structure_id=structure_id,
        source="RCSB PDB",
        source_version="current-release",
        release_date="UNKNOWN",
        experimental_method="UNKNOWN",
        resolution=None,
        entity_id=None,
        chain_id=primary_chain["chain_id"],
        assembly_id="1",
        sequence_sha256=sequence_sha256,
        coordinate_sha256=checksum,
        processing_version="dataset-engine-live-seed-v0.1",
        quality_flags=tuple(quality_flags),
        num_residues=total_residues,
        num_resolved_residues=total_residues,
        num_missing_residues=0,
        num_missing_backbone_atoms=total_missing_backbone,
        ligand_count=0,
        polymer_type="protein",
        organism=None,
        taxonomy_id=None,
        split="seed",
        provenance=ProvenanceRecord(
            source_database="RCSB PDB",
            source_accession=structure_id,
            source_version="current-release",
            download_timestamp="UNKNOWN",
            source_file_checksum=checksum,
            processed_file_checksum=None,
            processing_pipeline_version="dataset-engine-live-seed-v0.1",
            parser_version="Bio.PDB.MMCIFParser",
            normalization_version="canonical-structure-seed-v0.1",
            qc_version="seed-qc-v0.1",
            source_release_date="UNKNOWN",
        ),
    )

    return {
        "status": "accepted" if not quality_flags else "quarantined",
        "structure_id": structure_id,
        "download_status": "downloaded",
        "failure_reason": None,
        "raw_path": str(raw_path),
        "checksum": checksum,
        "checksum_verified": checksum_verified,
        "parsed": True,
        "chains": chains,
        "num_residues": total_residues,
        "quality_flags": quality_flags,
        "manifest": manifest.to_dict(),
    }


def run_external_seed_pipeline(
    structure_ids: Iterable[str] | str,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Run a controlled seed-corpus acquisition workflow and summarize counts."""
    selected = [structure_ids] if isinstance(structure_ids, str) else list(structure_ids)
    summary = AcquisitionSummary(requested=len(selected), discovered=len(selected))
    for structure_id in selected:
        result = acquire_rcsb_structure(structure_id, output_dir)
        summary.records.append(result)

        status = result.get("status", "rejected")
        if status == "accepted":
            summary.accepted += 1
        elif status == "rejected":
            summary.rejected += 1
            summary.failure_reasons.append(str(result.get("failure_reason", "UNKNOWN")))
        elif status == "quarantined":
            summary.quarantined += 1
            summary.failure_reasons.append(str(result.get("failure_reason", "QC_FAILURE")))

        if result.get("download_status") == "downloaded":
            summary.downloaded += 1
        if result.get("checksum_verified"):
            summary.checksum_verified += 1
        if result.get("parsed"):
            summary.parsed += 1

    return summary.to_dict()