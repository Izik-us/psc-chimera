"""Canonical sample construction and immutable SQLite dataset access."""

from __future__ import annotations

import hashlib
import io
import json
import os
import sqlite3
import tempfile
from dataclasses import asdict, fields, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .annotations import LigandContext, NRPSClusterAnnotation, build_ligand_context
from .dataset_engine import stable_hash
from .geometry import derive_geometry
from .msa import MSAConfig, MSARecord, build_msa_record
from .sequence_linkage import SequenceCandidate, SequenceLinkage, link_structure_sequence
from .structures import CanonicalChain, CanonicalStructure


def _geometry_to_dict(geometry) -> dict[str, torch.Tensor]:
    return {field.name: getattr(geometry, field.name) for field in fields(geometry)}


def _mapped_domain_spans(
    annotation: NRPSClusterAnnotation | None,
    linkage: SequenceLinkage,
) -> list[tuple[str, int, int]]:
    if annotation is None or linkage.external_accession is None:
        return []
    spans: list[tuple[str, int, int]] = []
    for domain in annotation.domains:
        if domain.protein_accession != linkage.external_accession:
            continue
        target_positions = [
            target_index
            for target_index, candidate_position in enumerate(linkage.target_to_candidate_positions)
            if candidate_position is not None and domain.start_aa <= candidate_position < domain.end_aa
        ]
        if target_positions:
            spans.append((domain.domain_id, min(target_positions), max(target_positions) + 1))
    return spans


def build_dataset_sample(
    structure: CanonicalStructure,
    chain: CanonicalChain,
    *,
    msa: MSARecord | None = None,
    sequence_candidates: Sequence[SequenceCandidate] = (),
    sequence_linkage: SequenceLinkage | None = None,
    nrps_annotation: NRPSClusterAnnotation | None = None,
    related_bgc_annotations: Sequence[NRPSClusterAnnotation] = (),
    acquisition_record: Mapping[str, Any] | None = None,
    dataset_version: str = "CHIMERA-DATASET-v0.1",
    split: str = "UNASSIGNED",
    qc_configuration_version: str = "structural-qc-v0.1",
) -> dict[str, Any]:
    """Build all model-facing tensors and provenance for one canonical chain."""
    if chain.structure_id != structure.structure_id:
        raise ValueError("chain and structure identifiers do not match")
    if not any(candidate is chain for candidate in structure.chains):
        raise ValueError("chain must be part of the supplied canonical structure")

    linkage = sequence_linkage or link_structure_sequence(
        structure_id=structure.structure_id,
        chain_id=chain.label_asym_id,
        entity_id=chain.entity_id,
        sequence=chain.sequence,
        candidates=sequence_candidates,
        reported_accession=next(
            (item["accession"] for item in structure.source_accessions), None
        ),
    )
    if (
        linkage.structure_id != structure.structure_id
        or linkage.chain_id != chain.label_asym_id
        or linkage.entity_id != chain.entity_id
    ):
        raise ValueError("sequence linkage does not identify this canonical chain")

    if msa is None:
        raw_alignment = f">{structure.structure_id}:{chain.label_asym_id}\n{chain.sequence}\n"
        msa = build_msa_record(
            raw_alignment,
            msa_family_id=f"single:{chain.sequence_sha256[:16]}",
            target_sequence_id=f"{structure.structure_id}:{chain.label_asym_id}",
            source="RCSB PDB polymer entity sequence",
            source_version="entry_revision_metadata",
            target_sequence=chain.sequence,
            generation_method="single-sequence-fallback",
            generation_version="msa-normalization-v0.1",
            config=MSAConfig(maximum_depth=1, pad_to_maximum_depth=False),
        )
    if msa.alignment_length != len(chain.sequence):
        raise ValueError("model-ready MSA alignment length must match the canonical chain sequence")

    coordinate_tensor = torch.as_tensor(chain.coordinates, dtype=torch.float32)
    atom_mask_tensor = torch.as_tensor(chain.atom_mask, dtype=torch.bool)
    residue_mask_tensor = torch.as_tensor(chain.residue_mask, dtype=torch.bool)
    geometry = derive_geometry(
        coordinate_tensor,
        atom_mask_tensor,
        residue_mask_tensor,
        chain_ids=[chain.label_asym_id] * len(chain.sequence),
    )

    mapped_domains = _mapped_domain_spans(nrps_annotation, linkage)
    ligand_contexts: list[LigandContext] = []
    for ligand_index, ligand in enumerate(structure.ligands):
        if not any(
            reference["label_asym_id"] == chain.label_asym_id
            for reference in ligand.binding_residue_refs
        ):
            continue
        ligand_contexts.append(
            build_ligand_context(
                structure_id=structure.structure_id,
                ligand_id=f"{structure.structure_id}:{ligand.label_asym_id}:{ligand.chemical_component_id}:{ligand.auth_seq_id or ligand_index}",
                ligand=ligand,
                chain_entity_id=chain.entity_id,
                chain_id=chain.label_asym_id,
                canonical_chemical_ids=ligand.canonical_identifiers,
                domains=mapped_domains,
                experimental_method=structure.experimental_method,
                resolution_angstrom=structure.resolution_angstrom,
                verification_date=datetime.now(timezone.utc).date().isoformat(),
            )
        )

    query_map = torch.full((len(chain.sequence),), -1, dtype=torch.long)
    for sequence_index, msa_column in enumerate(msa.query_residue_to_msa_column):
        if sequence_index < len(query_map):
            query_map[sequence_index] = msa_column

    return {
        "sample_id": f"{structure.structure_id}:{chain.label_asym_id}:{chain.entity_id}",
        "dataset_version": dataset_version,
        "split": split.upper(),
        "structure_id": structure.structure_id,
        "assembly_id": chain.assembly_id,
        "model_id": chain.model_id,
        "label_asym_id": chain.label_asym_id,
        "auth_asym_id": chain.auth_asym_id,
        "entity_id": chain.entity_id,
        "sequence": chain.sequence,
        "sequence_sha256": chain.sequence_sha256,
        "coordinates": coordinate_tensor,
        "atom_mask": atom_mask_tensor,
        "residue_mask": residue_mask_tensor,
        "sequence_mask": torch.as_tensor(chain.sequence_mask, dtype=torch.bool),
        "residue_mappings": [asdict(mapping) for mapping in chain.mappings],
        "b_factors": torch.as_tensor(chain.b_factors, dtype=torch.float32),
        "occupancies": torch.as_tensor(chain.occupancies, dtype=torch.float32),
        "geometry": _geometry_to_dict(geometry),
        "msa_tokens": msa.model_ready_tokens,
        "msa_mask": msa.model_ready_mask,
        "sequence_to_msa_columns": query_map,
        "msa_metadata": msa.to_dict(),
        "sequence_linkage": asdict(linkage),
        "nrps_annotation": None if nrps_annotation is None else nrps_annotation.to_dict(),
        "related_bgc_annotations": [annotation.to_dict() for annotation in related_bgc_annotations],
        "ligand_contexts": [context.to_dict(include_coordinates=True) for context in ligand_contexts],
        "experimental_metadata": {
            "experimental_method": structure.experimental_method,
            "resolution_angstrom": structure.resolution_angstrom,
            "release_date": structure.release_date,
            "assembly_ids": structure.assembly_ids,
        },
        "quality": chain.quality.to_dict(),
        "provenance": asdict(chain.provenance),
        "source_file_sha256": structure.source_sha256,
        "acquisition": None if acquisition_record is None else dict(acquisition_record),
        "qc_configuration_version": qc_configuration_version,
    }


def _to_json_and_arrays(value: Any, arrays: dict[str, np.ndarray], prefix: str = "a") -> Any:
    if torch.is_tensor(value):
        name = f"{prefix}_{len(arrays)}"
        arrays[name] = value.detach().cpu().numpy()
        return {"__array__": name}
    if isinstance(value, np.ndarray):
        name = f"{prefix}_{len(arrays)}"
        arrays[name] = value
        return {"__array__": name}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        value = asdict(value)
    if isinstance(value, Mapping):
        return {str(key): _to_json_and_arrays(item, arrays, prefix) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_json_and_arrays(item, arrays, prefix) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"dataset sample contains unsupported value type {type(value).__name__}")


def _encode_sample(sample: Mapping[str, Any]) -> tuple[str, bytes, str]:
    arrays: dict[str, np.ndarray] = {}
    metadata = _to_json_and_arrays(sample, arrays)
    metadata_json = json.dumps(metadata, sort_keys=True, separators=(",", ":"))
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    array_blob = buffer.getvalue()
    sample_digest = hashlib.sha256(metadata_json.encode("utf-8") + array_blob).hexdigest()
    return metadata_json, array_blob, sample_digest


def _decode_sample(metadata_json: str, array_blob: bytes) -> dict[str, Any]:
    metadata = json.loads(metadata_json)
    with np.load(io.BytesIO(array_blob), allow_pickle=False) as arrays:
        def restore(value: Any) -> Any:
            if isinstance(value, dict) and set(value) == {"__array__"}:
                return torch.from_numpy(arrays[value["__array__"]].copy())
            if isinstance(value, dict):
                return {key: restore(item) for key, item in value.items()}
            if isinstance(value, list):
                return [restore(item) for item in value]
            return value
        return restore(metadata)


def write_immutable_dataset(
    path: str | Path,
    samples: Iterable[Mapping[str, Any]],
    *,
    dataset_version: str,
    split_manifest_sha256: str,
) -> dict[str, Any]:
    """Write a content-addressed SQLite dataset once; existing versions are immutable."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"immutable dataset already exists: {destination}")
    sorted_samples = sorted(samples, key=lambda sample: str(sample["sample_id"]))
    if not sorted_samples:
        raise ValueError("cannot write an empty dataset")
    sample_ids = [str(sample["sample_id"]) for sample in sorted_samples]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("sample_id values must be unique")

    manifest_body = {
        "dataset_version": dataset_version,
        "split_manifest_sha256": split_manifest_sha256,
        "sample_ids": sample_ids,
        "sample_count": len(sample_ids),
        "storage_schema": "sqlite-npz-v0.1",
    }
    fd, temporary_name = tempfile.mkstemp(prefix=destination.name, suffix=".tmp", dir=destination.parent)
    os.close(fd)
    temporary = Path(temporary_name)
    record_hashes: dict[str, str] = {}
    try:
        connection = sqlite3.connect(temporary)
        with connection:
            connection.execute("PRAGMA journal_mode=DELETE")
            connection.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
            connection.execute(
                "CREATE TABLE samples (ordinal INTEGER PRIMARY KEY, sample_id TEXT UNIQUE NOT NULL, "
                "split TEXT NOT NULL, metadata_json TEXT NOT NULL, arrays_npz BLOB NOT NULL, sample_sha256 TEXT NOT NULL)"
            )
            for key, value in manifest_body.items():
                connection.execute(
                    "INSERT INTO metadata(key,value) VALUES (?,?)",
                    (key, json.dumps(value, sort_keys=True, separators=(",", ":"))),
                )
            for ordinal, sample in enumerate(sorted_samples):
                metadata_json, array_blob, sample_digest = _encode_sample(sample)
                sample_id = str(sample["sample_id"])
                split = str(sample.get("split", "UNASSIGNED")).upper()
                connection.execute(
                    "INSERT INTO samples(ordinal,sample_id,split,metadata_json,arrays_npz,sample_sha256) VALUES (?,?,?,?,?,?)",
                    (ordinal, sample_id, split, metadata_json, array_blob, sample_digest),
                )
                record_hashes[sample_id] = sample_digest
        connection.close()
        manifest_body["record_sha256"] = record_hashes
        manifest_body["manifest_sha256"] = stable_hash(manifest_body)
        temporary.replace(destination)
        manifest_path = destination.with_suffix(destination.suffix + ".manifest.json")
        manifest_path.write_text(json.dumps(manifest_body, indent=2, sort_keys=True), encoding="utf-8")
        return manifest_body
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


class SQLiteChimeraDataset(Dataset):
    """Read-only random-access dataset with worker-safe per-read connections."""

    def __init__(self, path: str | Path, *, split: str | None = None):
        self.path = Path(path).resolve()
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        self.split = None if split is None else split.upper()
        connection = self._connect()
        try:
            query = "SELECT sample_id FROM samples"
            parameters: tuple[Any, ...] = ()
            if self.split is not None:
                query += " WHERE split=?"
                parameters = (self.split,)
            query += " ORDER BY ordinal"
            self.sample_ids = [row[0] for row in connection.execute(query, parameters)]
            self.metadata = {
                key: json.loads(value)
                for key, value in connection.execute("SELECT key,value FROM metadata")
            }
        finally:
            connection.close()

    def _connect(self):
        uri = f"file:{self.path.as_posix()}?mode=ro"
        connection = sqlite3.connect(uri, uri=True, timeout=30)
        connection.execute("PRAGMA query_only=ON")
        return connection

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample_id = self.sample_ids[index]
        connection = self._connect()
        try:
            row = connection.execute(
                "SELECT metadata_json,arrays_npz FROM samples WHERE sample_id=?",
                (sample_id,),
            ).fetchone()
        finally:
            connection.close()
        if row is None:
            raise IndexError(index)
        return _decode_sample(row[0], row[1])

    def iter_batches(self, batch_size: int = 128) -> Iterator[list[dict[str, Any]]]:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        connection = self._connect()
        try:
            query = "SELECT metadata_json,arrays_npz FROM samples"
            parameters: tuple[Any, ...] = ()
            if self.split is not None:
                query += " WHERE split=?"
                parameters = (self.split,)
            query += " ORDER BY ordinal"
            cursor = connection.execute(query, parameters)
            while rows := cursor.fetchmany(batch_size):
                yield [_decode_sample(metadata, arrays) for metadata, arrays in rows]
        finally:
            connection.close()


def _collate_values(values: Sequence[Any], key: str = "") -> Any:
    if all(torch.is_tensor(value) for value in values):
        tensors = list(values)
        if len({tensor.ndim for tensor in tensors}) != 1:
            return tensors
        max_shape = tuple(max(tensor.shape[axis] for tensor in tensors) for axis in range(tensors[0].ndim))
        pad_value = 22 if key.endswith("msa_tokens") else 0
        output = torch.full((len(tensors), *max_shape), pad_value, dtype=tensors[0].dtype)
        for batch_index, tensor in enumerate(tensors):
            slices = (batch_index,) + tuple(slice(0, size) for size in tensor.shape)
            output[slices] = tensor
        return output
    if all(isinstance(value, dict) for value in values):
        keys = set.intersection(*(set(value) for value in values))
        return {key_name: _collate_values([value[key_name] for value in values], key_name) for key_name in sorted(keys)}
    return list(values)


def collate_chimera_samples(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not samples:
        raise ValueError("cannot collate an empty batch")
    return _collate_values(samples)


def make_dataloader(
    dataset: SQLiteChimeraDataset,
    *,
    batch_size: int = 1,
    shuffle: bool = False,
    num_workers: int = 0,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_chimera_samples,
    )