"""Canonical PDBx/mmCIF structure records and deterministic structural QC."""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from Bio import __version__ as biopython_version
from Bio.PDB.MMCIF2Dict import MMCIF2Dict

from .dataset_engine import DataQualityResult, ProvenanceRecord

ATOM_NAMES = ("N", "CA", "C", "O", "CB")
_AA3_TO_1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "SEC": "U", "PYL": "O", "ASX": "B", "GLX": "Z",
}
_EXPECTED_SIDECHAIN_ATOMS = {
    "ALA": {"CB"}, "ARG": {"CB", "CG", "CD", "NE", "CZ", "NH1", "NH2"},
    "ASN": {"CB", "CG", "OD1", "ND2"}, "ASP": {"CB", "CG", "OD1", "OD2"},
    "CYS": {"CB", "SG"}, "GLN": {"CB", "CG", "CD", "OE1", "NE2"},
    "GLU": {"CB", "CG", "CD", "OE1", "OE2"}, "GLY": set(),
    "HIS": {"CB", "CG", "ND1", "CD2", "CE1", "NE2"},
    "ILE": {"CB", "CG1", "CG2", "CD1"}, "LEU": {"CB", "CG", "CD1", "CD2"},
    "LYS": {"CB", "CG", "CD", "CE", "NZ"}, "MET": {"CB", "CG", "SD", "CE"},
    "PHE": {"CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ"},
    "PRO": {"CB", "CG", "CD"}, "SER": {"CB", "OG"},
    "THR": {"CB", "OG1", "CG2"},
    "TRP": {"CB", "CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"},
    "TYR": {"CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH"},
    "VAL": {"CB", "CG1", "CG2"},
}


@dataclass(frozen=True)
class ResidueMapping:
    source_residue_index: int
    canonical_residue_index: int
    sequence_index: int
    label_asym_id: str
    auth_asym_id: str
    entity_id: str
    label_seq_id: int
    auth_seq_id: str | None
    insertion_code: str | None
    residue_name: str
    amino_acid: str


@dataclass
class CanonicalChain:
    structure_id: str
    model_id: str
    assembly_id: str
    entity_id: str
    label_asym_id: str
    auth_asym_id: str
    sequence: str
    coordinates: np.ndarray
    atom_mask: np.ndarray
    residue_mask: np.ndarray
    sequence_mask: np.ndarray
    b_factors: np.ndarray
    occupancies: np.ndarray
    alternate_locations: list[str | None]
    residue_names: list[str]
    mappings: list[ResidueMapping]
    quality: DataQualityResult
    provenance: ProvenanceRecord

    @property
    def observed_sequence(self) -> str:
        return "".join(
            amino_acid
            for amino_acid, valid in zip(self.sequence, self.residue_mask)
            if valid
        )

    @property
    def sequence_sha256(self) -> str:
        return hashlib.sha256(self.sequence.encode("ascii")).hexdigest()

    def to_dict(self, *, include_arrays: bool = False) -> dict[str, Any]:
        value = asdict(self)
        if not include_arrays:
            for key in ("coordinates", "atom_mask", "residue_mask", "sequence_mask", "b_factors", "occupancies"):
                value.pop(key)
        else:
            for key in ("coordinates", "atom_mask", "residue_mask", "sequence_mask", "b_factors", "occupancies"):
                value[key] = getattr(self, key).tolist()
        return value


@dataclass
class LigandRecord:
    chemical_component_id: str
    label_asym_id: str
    auth_asym_id: str
    auth_seq_id: str | None
    coordinates: np.ndarray
    binding_residue_indices: list[int]
    minimum_contact_distance: float | None
    binding_residue_refs: list[dict[str, Any]] = field(default_factory=list)
    name: str | None = None
    parent_component_id: str | None = None
    canonical_identifiers: dict[str, str] = field(default_factory=dict)
    evidence_class: str = "EXPERIMENTAL"

    def to_dict(self, *, include_coordinates: bool = False) -> dict[str, Any]:
        value = asdict(self)
        if include_coordinates:
            value["coordinates"] = self.coordinates.tolist()
        else:
            value.pop("coordinates")
        return value


@dataclass
class CanonicalStructure:
    structure_id: str
    source_path: str
    source_sha256: str
    experimental_method: str | None
    resolution_angstrom: float | None
    release_date: str | None
    assembly_ids: list[str]
    chains: list[CanonicalChain]
    ligands: list[LigandRecord]
    source_accessions: list[dict[str, str]] = field(default_factory=list)
    chemical_components: dict[str, dict[str, Any]] = field(default_factory=dict)

    def to_dict(self, *, include_arrays: bool = False) -> dict[str, Any]:
        return {
            "structure_id": self.structure_id,
            "source_path": self.source_path,
            "source_sha256": self.source_sha256,
            "experimental_method": self.experimental_method,
            "resolution_angstrom": self.resolution_angstrom,
            "release_date": self.release_date,
            "assembly_ids": self.assembly_ids,
            "chains": [chain.to_dict(include_arrays=include_arrays) for chain in self.chains],
            "ligands": [ligand.to_dict(include_coordinates=include_arrays) for ligand in self.ligands],
            "source_accessions": self.source_accessions,
            "chemical_components": self.chemical_components,
        }


@dataclass(frozen=True)
class StructuralQCConfig:
    minimum_chain_length: int = 10
    maximum_missing_sequence_fraction: float = 0.2
    maximum_missing_backbone_fraction: float = 0.05
    maximum_missing_sidechain_fraction: float = 0.25
    minimum_bond_length_angstrom: float = 1.0
    maximum_bond_length_angstrom: float = 2.0
    peptide_break_distance_angstrom: float = 2.0


def _column(cif: dict[str, Any], key: str) -> list[str]:
    value = cif.get(key, [])
    return value if isinstance(value, list) else [value]


def _first(cif: dict[str, Any], key: str) -> str | None:
    values = _column(cif, key)
    if not values or values[0] in {".", "?", ""}:
        return None
    return values[0]


def _parse_float(value: str, *, default: float | None = None) -> float | None:
    if value in {".", "?", ""}:
        return default
    parsed = float(value)
    if not np.isfinite(parsed):
        raise ValueError("non-finite coordinate or atom metadata")
    return parsed


def _entity_sequences(cif: dict[str, Any]) -> dict[str, str]:
    entities = _column(cif, "_entity_poly.entity_id")
    sequences = _column(cif, "_entity_poly.pdbx_seq_one_letter_code_can")
    output: dict[str, str] = {}
    for entity_id, raw_sequence in zip(entities, sequences):
        sequence = re.sub(r"\([^)]*\)", "X", raw_sequence)
        sequence = re.sub(r"\s+", "", sequence).upper()
        if sequence:
            output[entity_id] = "".join(letter if letter in "ACDEFGHIKLMNPQRSTVWYBXZUO" else "X" for letter in sequence)
    return output


def _row_values(cif: dict[str, Any], names: tuple[str, ...], index: int, default: str = "?") -> str:
    for name in names:
        values = _column(cif, name)
        if values:
            return values[index]
    return default


def _select_residue_alternates(rows: list[dict[str, str]]) -> tuple[dict[str, dict[str, str]], str | None, int]:
    occupancy_by_alt: dict[str, list[float]] = {}
    for row in rows:
        alt = row["alt"]
        alt = alt if alt not in {".", "?", ""} else None
        row["alt"] = alt
        if alt:
            occupancy_by_alt.setdefault(alt, []).append(float(row["occupancy"]))
    selected_alt = min(
        occupancy_by_alt,
        key=lambda alt: (-float(np.mean(occupancy_by_alt[alt])), alt != "A", alt),
    ) if occupancy_by_alt else None

    candidates: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        if row["alt"] in {None, selected_alt}:
            candidates.setdefault(row["atom"], []).append(row)

    selected: dict[str, dict[str, str]] = {}
    duplicate_count = 0
    for atom_name, atom_rows in candidates.items():
        if len(atom_rows) > 1:
            duplicate_count += len(atom_rows) - 1
        selected[atom_name] = max(
            atom_rows,
            key=lambda row: (row["alt"] is None, float(row["occupancy"])),
        )
    return selected, selected_alt, duplicate_count


def _make_quality(
    *,
    sequence_length: int,
    observed_count: int,
    atom_mask: np.ndarray,
    coordinates: np.ndarray,
    mappings: list[ResidueMapping],
    chain_breaks: list[int],
    duplicate_count: int,
    invalid_residues: list[int],
    sidechain_missing: dict[int, list[str]],
    pathological_geometry: list[str],
    config: StructuralQCConfig,
) -> DataQualityResult:
    reasons: list[str] = []
    if sequence_length < config.minimum_chain_length:
        reasons.append("CHAIN_TOO_SHORT")
    missing_sequence = max(sequence_length - observed_count, 0)
    if sequence_length and missing_sequence / sequence_length > config.maximum_missing_sequence_fraction:
        reasons.append("EXCESSIVE_UNRESOLVED_RESIDUES")
    observed_mask = atom_mask.any(axis=-1)
    backbone_present = atom_mask[:, :4].all(axis=-1)
    backbone_missing_fraction = (
        float((observed_mask & ~backbone_present).sum()) / max(int(observed_mask.sum()), 1)
    )
    if backbone_missing_fraction > config.maximum_missing_backbone_fraction:
        reasons.append("EXCESSIVE_BACKBONE_MISSINGNESS")
    sidechain_residue_count = sum(mapping.residue_name in _EXPECTED_SIDECHAIN_ATOMS for mapping in mappings)
    sidechain_missing_fraction = len(sidechain_missing) / max(sidechain_residue_count, 1)
    if sidechain_missing_fraction > config.maximum_missing_sidechain_fraction:
        reasons.append("EXCESSIVE_SIDECHAIN_MISSINGNESS")
    if chain_breaks:
        reasons.append("CHAIN_DISCONTINUITY")
    if duplicate_count:
        reasons.append("DUPLICATE_ATOMS_OR_RESIDUES")
    if invalid_residues:
        reasons.append("INVALID_OR_NONSTANDARD_RESIDUES")
    if pathological_geometry:
        reasons.append("PATHOLOGICAL_GEOMETRY")
    if not np.isfinite(coordinates).all():
        reasons.append("MALFORMED_COORDINATES")
    status = "accepted" if not reasons else "quarantined"
    return DataQualityResult(
        status=status,
        reasons=reasons,
        qc_version="structural-qc-v0.1",
        additional_metadata={
            "sequence_length": sequence_length,
            "observed_residues": observed_count,
            "missing_residues": missing_sequence,
            "missing_backbone_residues": int((observed_mask & ~backbone_present).sum()),
            "missing_sidechain_atom_residues": len(sidechain_missing),
            "missing_sidechain_fraction": sidechain_missing_fraction,
            "missing_sidechain_atoms": sidechain_missing,
            "chain_break_indices": chain_breaks,
            "duplicate_count": duplicate_count,
            "invalid_residue_indices": invalid_residues,
            "pathological_geometry": pathological_geometry,
            "alternate_location_policy": "highest_mean_occupancy_residue_conformer; blank atoms preferred",
        },
    )


def parse_mmcif_structure(
    path: str | Path,
    *,
    structure_id: str | None = None,
    model_id: str = "1",
    assembly_id: str = "asymmetric_unit",
    config: StructuralQCConfig | None = None,
) -> CanonicalStructure:
    """Parse a deposited mmCIF into entity-aware chain records without renumbering source residues."""
    source_path = Path(path)
    cif = MMCIF2Dict(str(source_path))
    structure_id = (structure_id or _first(cif, "_entry.id") or source_path.stem).upper()
    source_bytes = source_path.read_bytes()
    source_sha256 = hashlib.sha256(source_bytes).hexdigest()
    config = config or StructuralQCConfig()

    required_columns = (
        "_atom_site.label_asym_id", "_atom_site.label_entity_id", "_atom_site.label_seq_id",
        "_atom_site.auth_asym_id", "_atom_site.auth_seq_id", "_atom_site.label_atom_id",
        "_atom_site.label_comp_id", "_atom_site.Cartn_x", "_atom_site.Cartn_y", "_atom_site.Cartn_z",
    )
    if any(not _column(cif, key) for key in required_columns):
        raise ValueError("mmCIF lacks required polymer atom_site categories")

    row_count = len(_column(cif, "_atom_site.label_asym_id"))
    rows_by_chain: dict[tuple[str, str, str], list[dict[str, str]]] = {}
    ligand_atoms: dict[tuple[str, str, str, str], list[list[float]]] = {}
    entity_sequence = _entity_sequences(cif)
    for index in range(row_count):
        row_model = _row_values(cif, ("_atom_site.pdbx_PDB_model_num",), index, "1")
        if row_model != model_id:
            continue
        group = _row_values(cif, ("_atom_site.group_PDB",), index, "ATOM")
        label_asym = _row_values(cif, ("_atom_site.label_asym_id",), index)
        auth_asym = _row_values(cif, ("_atom_site.auth_asym_id",), index)
        entity_id = _row_values(cif, ("_atom_site.label_entity_id",), index)
        label_seq = _row_values(cif, ("_atom_site.label_seq_id",), index)
        comp = _row_values(cif, ("_atom_site.label_comp_id",), index).upper()
        atom_name = _row_values(cif, ("_atom_site.label_atom_id",), index).upper()
        x = _parse_float(_row_values(cif, ("_atom_site.Cartn_x",), index))
        y = _parse_float(_row_values(cif, ("_atom_site.Cartn_y",), index))
        z = _parse_float(_row_values(cif, ("_atom_site.Cartn_z",), index))
        xyz = [float(x), float(y), float(z)]

        if group == "ATOM" and label_seq not in {".", "?", ""} and entity_id in entity_sequence:
            key = (label_asym, auth_asym, entity_id)
            rows_by_chain.setdefault(key, []).append(
                {
                    "label_seq": label_seq,
                    "auth_seq": _row_values(cif, ("_atom_site.auth_seq_id",), index),
                    "ins": _row_values(cif, ("_atom_site.pdbx_PDB_ins_code",), index),
                    "comp": comp,
                    "atom": atom_name,
                    "alt": _row_values(cif, ("_atom_site.label_alt_id",), index),
                    "x": str(x), "y": str(y), "z": str(z),
                    "occupancy": str(_parse_float(_row_values(cif, ("_atom_site.occupancy",), index), default=1.0)),
                    "bfactor": str(_parse_float(_row_values(cif, ("_atom_site.B_iso_or_equiv",), index), default=0.0)),
                }
            )
        elif group == "HETATM" and comp not in {"HOH", "WAT", "DOD"}:
            auth_seq = _row_values(cif, ("_atom_site.auth_seq_id",), index)
            ligand_atoms.setdefault((comp, label_asym, auth_asym, auth_seq), []).append(xyz)

    now = datetime.now(timezone.utc).isoformat()
    provenance = ProvenanceRecord(
        source_database="RCSB PDB / wwPDB",
        source_accession=structure_id,
        source_version="entry_revision_metadata",
        download_timestamp=now,
        source_file_checksum=source_sha256,
        processing_pipeline_version="CHIMERA-data-v0.2",
        parser_version=f"Bio.PDB.MMCIF2Dict-{biopython_version}",
        normalization_version="pdbx-entity-chain-map-v0.1",
        qc_version="structural-qc-v0.1",
        source_release_date=_first(cif, "_pdbx_database_status.recvd_initial_deposition_date"),
    )

    chains: list[CanonicalChain] = []
    for (label_asym, auth_asym, entity_id), rows in sorted(rows_by_chain.items()):
        sequence = entity_sequence[entity_id]
        length = len(sequence)
        coordinates = np.zeros((length, len(ATOM_NAMES), 3), dtype=np.float32)
        atom_mask = np.zeros((length, len(ATOM_NAMES)), dtype=bool)
        b_factors = np.zeros((length, len(ATOM_NAMES)), dtype=np.float32)
        occupancies = np.zeros((length, len(ATOM_NAMES)), dtype=np.float32)
        alternate_locations: list[str | None] = [None] * length
        residue_names = ["UNK"] * length
        mappings = [
            ResidueMapping(
                source_residue_index=index,
                canonical_residue_index=index,
                sequence_index=index,
                label_asym_id=label_asym,
                auth_asym_id=auth_asym,
                entity_id=entity_id,
                label_seq_id=index + 1,
                auth_seq_id=None,
                insertion_code=None,
                residue_name="UNK",
                amino_acid=sequence[index],
            )
            for index in range(length)
        ]
        grouped: dict[int, list[dict[str, str]]] = {}
        for row in rows:
            source_index = int(row["label_seq"]) - 1
            if not 0 <= source_index < length:
                continue
            grouped.setdefault(source_index, []).append(row)

        duplicate_count = 0
        sidechain_missing: dict[int, list[str]] = {}
        for source_index, residue_rows in sorted(grouped.items()):
            selected, chosen_alt, duplicate_atoms = _select_residue_alternates(residue_rows)
            duplicate_count += duplicate_atoms
            residue_row = min(residue_rows, key=lambda item: item["alt"] not in {".", "?", ""})
            residue_name = residue_row["comp"]
            amino_acid = _AA3_TO_1.get(residue_name, "X")
            residue_names[source_index] = residue_name
            alternate_locations[source_index] = chosen_alt
            auth_seq = residue_row["auth_seq"]
            insertion = residue_row["ins"]
            mapping = ResidueMapping(
                source_residue_index=source_index,
                canonical_residue_index=source_index,
                sequence_index=source_index,
                label_asym_id=label_asym,
                auth_asym_id=auth_asym,
                entity_id=entity_id,
                label_seq_id=source_index + 1,
                auth_seq_id=None if auth_seq in {".", "?", ""} else auth_seq,
                insertion_code=None if insertion in {".", "?", ""} else insertion,
                residue_name=residue_name,
                amino_acid=sequence[source_index],
            )
            mappings[source_index] = mapping
            present_atom_names = set(selected)
            expected_sidechain = _EXPECTED_SIDECHAIN_ATOMS.get(residue_name)
            if expected_sidechain is not None:
                missing_atoms = sorted(expected_sidechain - present_atom_names)
                if missing_atoms:
                    sidechain_missing[source_index] = missing_atoms
            for atom_index, atom_name in enumerate(ATOM_NAMES):
                atom = selected.get(atom_name)
                if atom is None:
                    continue
                coordinates[source_index, atom_index] = [float(atom[key]) for key in ("x", "y", "z")]
                atom_mask[source_index, atom_index] = True
                b_factors[source_index, atom_index] = float(atom["bfactor"])
                occupancies[source_index, atom_index] = float(atom["occupancy"])

        residue_mask = atom_mask[:, 1].copy()
        sequence_mask = np.ones(length, dtype=bool)
        observed_indices = np.flatnonzero(residue_mask)
        chain_breaks: list[int] = []
        pathological_geometry: list[str] = []
        for index in observed_indices:
            if atom_mask[index, 0] and atom_mask[index, 1]:
                n_ca = float(np.linalg.norm(coordinates[index, 0] - coordinates[index, 1]))
                if not config.minimum_bond_length_angstrom <= n_ca <= config.maximum_bond_length_angstrom:
                    pathological_geometry.append(f"N_CA_BOND:{int(index)}")
            if atom_mask[index, 1] and atom_mask[index, 2]:
                ca_c = float(np.linalg.norm(coordinates[index, 1] - coordinates[index, 2]))
                if not config.minimum_bond_length_angstrom <= ca_c <= config.maximum_bond_length_angstrom:
                    pathological_geometry.append(f"CA_C_BOND:{int(index)}")
        auth_residue_ids = [
            (mapping.auth_seq_id, mapping.insertion_code)
            for mapping in mappings
            if mapping.auth_seq_id is not None
        ]
        if len(auth_residue_ids) != len(set(auth_residue_ids)):
            pathological_geometry.append("DUPLICATE_AUTH_RESIDUE_NUMBERING")
        for left, right in zip(observed_indices, observed_indices[1:]):
            if right != left + 1:
                continue
            c_xyz = coordinates[left, 2]
            n_xyz = coordinates[right, 0]
            if atom_mask[left, 2] and atom_mask[right, 0]:
                distance = float(np.linalg.norm(c_xyz - n_xyz))
                if distance > config.peptide_break_distance_angstrom or distance < config.minimum_bond_length_angstrom:
                    chain_breaks.append(int(left))
        nonstandard_names = {"MSE", "SEC", "PYL"}
        invalid_residues = [
            index for index, residue_name in enumerate(residue_names)
            if residue_name != "UNK" and (residue_name not in _AA3_TO_1 or residue_name in nonstandard_names)
        ]
        quality = _make_quality(
            sequence_length=length,
            observed_count=int(residue_mask.sum()),
            atom_mask=atom_mask,
            coordinates=coordinates,
            mappings=mappings,
            chain_breaks=chain_breaks,
            duplicate_count=duplicate_count,
            invalid_residues=invalid_residues,
            sidechain_missing=sidechain_missing,
            pathological_geometry=pathological_geometry,
            config=config,
        )
        chains.append(
            CanonicalChain(
                structure_id=structure_id,
                model_id=model_id,
                assembly_id=assembly_id,
                entity_id=entity_id,
                label_asym_id=label_asym,
                auth_asym_id=auth_asym,
                sequence=sequence,
                coordinates=coordinates,
                atom_mask=atom_mask,
                residue_mask=residue_mask,
                sequence_mask=sequence_mask,
                b_factors=b_factors,
                occupancies=occupancies,
                alternate_locations=alternate_locations,
                residue_names=residue_names,
                mappings=mappings,
                quality=quality,
                provenance=provenance,
            )
        )

    component_ids = _column(cif, "_chem_comp.id")
    component_names = _column(cif, "_chem_comp.name") or ["?"] * len(component_ids)
    component_formulas = _column(cif, "_chem_comp.formula") or ["?"] * len(component_ids)
    component_parents = _column(cif, "_chem_comp.mon_nstd_parent_comp_id") or ["?"] * len(component_ids)
    chemical_components: dict[str, dict[str, Any]] = {}
    for component_id, name, formula, parent in zip(
        component_ids, component_names, component_formulas, component_parents
    ):
        chemical_components[component_id] = {
            "name": name.strip("'\""),
            "formula": formula.strip("'\""),
            "parent_component_id": None if parent in {".", "?", ""} else parent,
            "identifiers": {},
        }
    descriptor_components = _column(cif, "_pdbx_chem_comp_descriptor.comp_id")
    descriptor_types = _column(cif, "_pdbx_chem_comp_descriptor.type")
    descriptors = _column(cif, "_pdbx_chem_comp_descriptor.descriptor")
    for component_id, descriptor_type, descriptor in zip(descriptor_components, descriptor_types, descriptors):
        entry = chemical_components.setdefault(component_id, {"name": None, "formula": None, "parent_component_id": None, "identifiers": {}})
        normalized_type = descriptor_type.strip("'\"").upper()
        if normalized_type in {"INCHI", "INCHIKEY", "SMILES", "SMILES_CANONICAL"}:
            entry["identifiers"].setdefault(normalized_type, descriptor.strip("'\""))
    identifier_components = _column(cif, "_pdbx_chem_comp_identifier.comp_id")
    identifier_types = _column(cif, "_pdbx_chem_comp_identifier.type")
    identifiers = _column(cif, "_pdbx_chem_comp_identifier.identifier")
    for component_id, identifier_type, identifier in zip(identifier_components, identifier_types, identifiers):
        entry = chemical_components.setdefault(component_id, {"name": None, "formula": None, "parent_component_id": None, "identifiers": {}})
        entry["identifiers"][identifier_type.strip("'\"")] = identifier.strip("'\"")

    ligands: list[LigandRecord] = []
    for (component_id, label_asym, auth_asym, auth_seq), atom_coordinates in sorted(ligand_atoms.items()):
        ligand_xyz = np.asarray(atom_coordinates, dtype=np.float32)
        contact_indices: list[int] = []
        contact_references: list[dict[str, Any]] = []
        minimum_distance: float | None = None
        for chain in chains:
            valid = np.flatnonzero(chain.residue_mask)
            if not len(valid):
                continue
            ca = chain.coordinates[valid, 1]
            distances = np.linalg.norm(ca[:, None, :] - ligand_xyz[None, :, :], axis=-1)
            min_per_residue = distances.min(axis=1)
            local = np.flatnonzero(min_per_residue <= 5.0)
            contact_indices.extend(int(valid[index]) for index in local)
            for local_index in local:
                residue_index = int(valid[local_index])
                mapping = chain.mappings[residue_index]
                contact_references.append(
                    {
                        "label_asym_id": chain.label_asym_id,
                        "auth_asym_id": chain.auth_asym_id,
                        "entity_id": chain.entity_id,
                        "canonical_residue_index": residue_index,
                        "label_seq_id": mapping.label_seq_id,
                        "auth_seq_id": mapping.auth_seq_id,
                        "insertion_code": mapping.insertion_code,
                        "residue_name": mapping.residue_name,
                        "amino_acid": mapping.amino_acid,
                        "minimum_distance_angstrom": float(min_per_residue[local_index]),
                    }
                )
            current_min = float(min_per_residue.min())
            minimum_distance = current_min if minimum_distance is None else min(minimum_distance, current_min)
        component_metadata = chemical_components.get(component_id, {})
        ligands.append(
            LigandRecord(
                chemical_component_id=component_id,
                label_asym_id=label_asym,
                auth_asym_id=auth_asym,
                auth_seq_id=None if auth_seq in {".", "?", ""} else auth_seq,
                coordinates=ligand_xyz,
                binding_residue_indices=sorted(set(contact_indices)),
                minimum_contact_distance=minimum_distance,
                binding_residue_refs=contact_references,
                name=component_metadata.get("name"),
                parent_component_id=component_metadata.get("parent_component_id"),
                canonical_identifiers=dict(component_metadata.get("identifiers", {})),
            )
        )

    accessions: list[dict[str, str]] = []
    ref_ids = _column(cif, "_struct_ref.id")
    ref_databases = _column(cif, "_struct_ref.db_name")
    ref_accessions = _column(cif, "_struct_ref.pdbx_db_accession")
    for ref_id, database, accession in zip(ref_ids, ref_databases, ref_accessions):
        if database.upper() in {"UNP", "UNIPROT"} and accession not in {".", "?", ""}:
            accessions.append({"database": "UniProt", "accession": accession, "reference_id": ref_id})

    methods = _column(cif, "_exptl.method")
    method = "; ".join(dict.fromkeys(value.strip("'") for value in methods if value not in {".", "?"})) or None
    resolution_values = _column(cif, "_refine.ls_d_res_high") or _column(cif, "_em_3d_reconstruction.resolution")
    resolution = None
    for value in resolution_values:
        try:
            resolution = _parse_float(value)
            if resolution is not None:
                break
        except ValueError:
            continue
    release_date = _first(cif, "_pdbx_database_status.recvd_initial_deposition_date")
    assembly_ids = [value for value in _column(cif, "_pdbx_struct_assembly.id") if value not in {".", "?"}]

    if not chains:
        raise ValueError(f"No polymer entity chains with coordinates found in {source_path}")
    return CanonicalStructure(
        structure_id=structure_id,
        source_path=str(source_path),
        source_sha256=source_sha256,
        experimental_method=method,
        resolution_angstrom=resolution,
        release_date=release_date,
        assembly_ids=assembly_ids,
        chains=chains,
        ligands=ligands,
        source_accessions=accessions,
        chemical_components=chemical_components,
    )