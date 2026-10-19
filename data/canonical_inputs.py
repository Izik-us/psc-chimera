"""Versioned adapter from canonical dataset samples to CHIMERA model tensors.

The adapter requires an ingested sample with an MSA and valid structure geometry.
It does not retrieve homologues or invent a missing structure. Pair features are
deterministic geometric descriptors, not pretrained EvoFormer embeddings.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

import torch

from chimera.input_contract import (
    PreparedInferenceInputs,
    PreparedInputManifest,
    prepare_inference_inputs,
    validate_rotation_frames,
)
from data.msa import AA_ALPHABET, AA_TO_TOKEN, PAD_TOKEN

PAIR_FEATURE_SCHEMA = "geometry-pair-v1"
PREPROCESSING_ID = "canonical-dataset-sample-v1"


def preprocessing_id_for(pair_channels: int) -> str:
    """Identify the preprocessing algorithm and exact pair-feature width."""
    if isinstance(pair_channels, bool) or not isinstance(pair_channels, int):
        raise TypeError("pair_channels must be an integer")
    if pair_channels < _MIN_PAIR_CHANNELS:
        raise ValueError(f"pair_channels must be >= {_MIN_PAIR_CHANNELS}")
    return f"{PREPROCESSING_ID}/{PAIR_FEATURE_SCHEMA}/channels-{pair_channels}"
_DISTANCE_BINS = 32
_SEQUENCE_FREQUENCIES = 8
_MIN_PAIR_CHANNELS = _DISTANCE_BINS + 3 + 9 + 16 + 12 + 4


def _field(record: Any, name: str) -> Any:
    if isinstance(record, Mapping):
        if name not in record:
            raise ValueError(f"geometry/sample is missing required field {name!r}")
        return record[name]
    try:
        return getattr(record, name)
    except AttributeError as exc:
        raise ValueError(f"geometry is missing required field {name!r}") from exc


def _tensor(value: Any, dtype: torch.dtype | None = None) -> torch.Tensor:
    result = value if torch.is_tensor(value) else torch.as_tensor(value)
    return result.to(dtype=dtype) if dtype is not None else result


def _stable_identity(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def pair_features_from_geometry(
    geometry: Any,
    *,
    channels: int = 128,
    residue_index: torch.Tensor | None = None,
) -> torch.Tensor:
    """Build deterministic (L,L,C) descriptors from a data.GeometryRecord."""
    if isinstance(channels, bool) or not isinstance(channels, int) or channels < _MIN_PAIR_CHANNELS:
        raise ValueError(f"channels must be an integer >= {_MIN_PAIR_CHANNELS}")

    distances = _tensor(_field(geometry, "distances"), torch.float32)
    directions = _tensor(_field(geometry, "relative_positions"), torch.float32)
    rotations = _tensor(_field(geometry, "relative_rotations"), torch.float32)
    torsions = _tensor(_field(geometry, "torsions"), torch.float32)
    torsion_mask = _tensor(_field(geometry, "torsion_mask"), torch.bool)
    contacts = _tensor(_field(geometry, "contacts"), torch.bool)
    pair_mask = _tensor(_field(geometry, "pair_mask"), torch.bool)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("geometry.distances must have shape (L,L)")
    length = distances.shape[0]
    expected = {
        "relative_positions": (length, length, 3),
        "relative_rotations": (length, length, 3, 3),
        "torsions": (length, 3),
        "torsion_mask": (length, 3),
        "contacts": (length, length),
        "pair_mask": (length, length),
    }
    actual = {
        "relative_positions": tuple(directions.shape),
        "relative_rotations": tuple(rotations.shape),
        "torsions": tuple(torsions.shape),
        "torsion_mask": tuple(torsion_mask.shape),
        "contacts": tuple(contacts.shape),
        "pair_mask": tuple(pair_mask.shape),
    }
    for name, shape in expected.items():
        if actual[name] != shape:
            raise ValueError(f"geometry.{name} must have shape {shape}, got {actual[name]}")
    for name, value in (
        ("distances", distances),
        ("relative_positions", directions),
        ("relative_rotations", rotations),
        ("torsions", torsions),
    ):
        if not torch.isfinite(value).all():
            raise ValueError(f"geometry.{name} contains non-finite values")

    device = distances.device
    valid_pairs = pair_mask.to(device)
    centers = torch.linspace(0.0, 20.0, _DISTANCE_BINS, device=device)
    sigma = 20.0 / (_DISTANCE_BINS - 1)
    distance_rbf = torch.exp(
        -0.5 * ((distances.unsqueeze(-1) - centers) / sigma).square()
    )

    if residue_index is None:
        indices = torch.arange(length, dtype=distances.dtype, device=device)
    else:
        indices = _tensor(residue_index).to(device=device, dtype=distances.dtype)
        if indices.shape != (length,):
            raise ValueError("residue_index must have shape (L,)")
    offset = indices[:, None] - indices[None, :]
    frequencies = torch.exp(
        torch.arange(_SEQUENCE_FREQUENCIES, dtype=distances.dtype, device=device)
        * (-torch.log(distances.new_tensor(10000.0)) / (_SEQUENCE_FREQUENCIES - 1))
    )
    phase = offset.unsqueeze(-1) * frequencies
    sequence_features = torch.cat((phase.sin(), phase.cos()), dim=-1)

    valid_torsions = torsion_mask.to(device)
    torsion_features = torch.cat(
        (
            torsions.sin() * valid_torsions.to(torsions.dtype),
            torsions.cos() * valid_torsions.to(torsions.dtype),
        ),
        dim=-1,
    )
    left_torsions = torsion_features[:, None, :].expand(length, length, -1)
    right_torsions = torsion_features[None, :, :].expand(length, length, -1)
    features = torch.cat(
        (
            distance_rbf,
            directions,
            rotations.reshape(length, length, 9),
            sequence_features,
            left_torsions,
            right_torsions,
            contacts.to(distances.dtype).unsqueeze(-1),
            valid_pairs.to(distances.dtype).unsqueeze(-1),
            (distances.clamp(0.0, 20.0) / 20.0).unsqueeze(-1),
            (offset / max(length - 1, 1)).unsqueeze(-1),
        ),
        dim=-1,
    )
    features = features * valid_pairs.to(features.dtype).unsqueeze(-1)
    if features.shape[-1] > channels:
        raise RuntimeError("pair-feature schema exceeds the requested channel count")
    if features.shape[-1] < channels:
        features = torch.cat(
            (features, features.new_zeros(length, length, channels - features.shape[-1])),
            dim=-1,
        )
    return features.contiguous()


def _sample_tensors(sample: Mapping[str, Any], pair_channels: int):
    for name in ("sample_id", "structure_id", "sequence", "msa_tokens", "geometry", "provenance", "msa_metadata"):
        if name not in sample:
            raise ValueError(f"dataset sample is missing required field {name!r}")
    sequence = str(sample["sequence"]).strip().upper()
    if not sequence:
        raise ValueError("dataset sample sequence must not be empty")
    msa = _tensor(sample["msa_tokens"], torch.long)
    if msa.ndim != 2 or msa.shape[0] < 1 or msa.shape[-1] != len(sequence):
        raise ValueError("msa_tokens must have shape (N,L) aligned to the canonical sequence")
    if torch.any((msa < 0) | (msa > PAD_TOKEN)):
        raise ValueError(f"msa_tokens must contain token IDs in [0,{PAD_TOKEN}]")
    expected_query = torch.tensor(
        [AA_TO_TOKEN.get(residue, 20) for residue in sequence],
        dtype=torch.long,
        device=msa.device,
    )
    if not torch.equal(msa[0], expected_query):
        raise ValueError(
            "the first MSA row must match the canonical sample sequence residue-by-residue"
        )
    computed_sequence_hash = hashlib.sha256(sequence.encode("ascii")).hexdigest()
    declared_sequence_hash = sample.get("sequence_sha256")
    if declared_sequence_hash is not None and str(declared_sequence_hash).lower() != computed_sequence_hash:
        raise ValueError("sequence_sha256 does not match the canonical sample sequence")

    geometry = sample["geometry"]
    rotations = _tensor(_field(geometry, "residue_frames"), torch.float32)
    translations = _tensor(_field(geometry, "ca_coordinates"), torch.float32)
    geometry_mask = _tensor(_field(geometry, "residue_mask"), torch.bool)
    frame_mask = _tensor(_field(geometry, "frame_mask"), torch.bool)
    length = len(sequence)
    if rotations.shape != (length, 3, 3) or translations.shape != (length, 3):
        raise ValueError("geometry frames and translations must align to the sequence")
    if geometry_mask.shape != (length,) or frame_mask.shape != (length,):
        raise ValueError("geometry residue/frame masks must have shape (L,)")
    sample_mask = _tensor(sample.get("residue_mask", geometry_mask), torch.bool)
    if sample_mask.shape != (length,):
        raise ValueError("sample residue_mask must have shape (L,)")
    residue_mask = sample_mask & geometry_mask & frame_mask
    if not torch.isfinite(translations).all():
        raise ValueError("source translations contain non-finite values")
    validate_rotation_frames(rotations.unsqueeze(0))

    index = torch.arange(length, dtype=torch.long, device=msa.device)
    pair = pair_features_from_geometry(
        geometry,
        channels=pair_channels,
        residue_index=index.to(_field(geometry, "distances").device),
    )
    device = msa.device
    return (
        msa.unsqueeze(0),
        pair.unsqueeze(0).to(device=device, dtype=torch.float32),
        rotations.unsqueeze(0).to(device=device),
        translations.unsqueeze(0).to(device=device),
        msa.unsqueeze(0).eq(PAD_TOKEN),
        residue_mask.unsqueeze(0).to(device=device),
        index.unsqueeze(0),
    )


def dataset_sample_to_prepared_inputs(
    sample: Mapping[str, Any],
    *,
    pair_channels: int = 128,
) -> PreparedInferenceInputs:
    """Convert one canonical dataset sample into provenance-bound inference tensors."""
    msa, pair, rotations, translations, msa_padding, residue_mask, residue_index = (
        _sample_tensors(sample, pair_channels)
    )
    msa_metadata = sample["msa_metadata"]
    provenance = sample["provenance"]
    if not isinstance(msa_metadata, Mapping) or not isinstance(provenance, Mapping):
        raise ValueError("msa_metadata and provenance must be mappings")
    msa_source = str(msa_metadata.get("source") or "")
    structure_source = str(provenance.get("source_database") or "")
    structure_id = str(sample["structure_id"])
    if not msa_source or not structure_source or not structure_id:
        raise ValueError("MSA source, structure source, and structure ID are required")
    sequence_hash = str(
        sample.get("sequence_sha256")
        or hashlib.sha256(str(sample["sequence"]).upper().encode("ascii")).hexdigest()
    )
    manifest = PreparedInputManifest(
        preprocessing_id=preprocessing_id_for(pair_channels),
        sequence_id=sequence_hash,
        msa_source=msa_source,
        pair_feature_source=f"{PAIR_FEATURE_SCHEMA}/channels-{pair_channels}",
        structure_source=structure_source,
        structure_id=structure_id,
        residue_mapping_id=_stable_identity(sample.get("residue_mappings", [])),
    )
    return prepare_inference_inputs(
        msa_tokens=msa,
        pair_features=pair,
        source_rotations=rotations,
        source_translations=translations,
        manifest=manifest,
        msa_padding_mask=msa_padding,
        residue_mask=residue_mask,
        residue_index=residue_index,
    )


def dataset_sample_to_training_batch(
    sample: Mapping[str, Any],
    *,
    pair_channels: int = 128,
    source_frames: tuple[torch.Tensor, torch.Tensor] | None = None,
    source_pair_features: torch.Tensor | None = None,
    source_geometry: Any | None = None,
    constraints: Any | None = None,
    substrate_id: torch.Tensor | int | None = None,
    substrate_coords: torch.Tensor | None = None,
    substrate_types: torch.Tensor | None = None,
    structure_source: str | None = None,
):
    """Build a CanonicalTrainingBatch from a sample's MSA and target structure.

    Flow training requires distinct source frames plus pair features derived
    from that same source (source_pair_features or source_geometry). This avoids
    leaking target-structure geometry into the flow-conditioning input. The
    adapter never fabricates noisy frames. Unsupported target residues are masked
    from sequence-recovery supervision.
    """
    from chimera.training import CanonicalTrainingBatch

    prepared = dataset_sample_to_prepared_inputs(sample, pair_channels=pair_channels)
    sequence = str(sample["sequence"]).strip().upper()
    target = torch.full((len(sequence),), 20, dtype=torch.long, device=prepared.msa_tokens.device)
    unsupported = torch.zeros((len(sequence),), dtype=torch.bool, device=target.device)
    for index, residue in enumerate(sequence):
        token = AA_TO_TOKEN.get(residue)
        if residue not in AA_ALPHABET:
            unsupported[index] = True
        else:
            target[index] = token
    sequence_mask = unsupported.unsqueeze(0) | ~prepared.residue_mask

    source_rotations = None
    source_translations = None
    training_pair_features = prepared.pair_features
    if source_frames is not None:
        if not isinstance(source_frames, (tuple, list)) or len(source_frames) != 2:
            raise ValueError("source_frames must be a (rotations, translations) pair")
        source_rotations = _tensor(source_frames[0], torch.float32).to(prepared.source_rotations.device)
        source_translations = _tensor(source_frames[1], torch.float32).to(prepared.source_translations.device)
        if source_rotations.shape == (prepared.length, 3, 3):
            source_rotations = source_rotations.unsqueeze(0)
        if source_translations.shape == (prepared.length, 3):
            source_translations = source_translations.unsqueeze(0)
        if source_rotations.shape != prepared.source_rotations.shape:
            raise ValueError("source rotations must have shape (L,3,3) or (1,L,3,3)")
        if source_translations.shape != prepared.source_translations.shape:
            raise ValueError("source translations must have shape (L,3) or (1,L,3)")
        validate_rotation_frames(source_rotations)
        if not torch.isfinite(source_translations).all():
            raise ValueError("source translations contain non-finite values")
        if (source_pair_features is None) == (source_geometry is None):
            raise ValueError(
                "source_frames require exactly one matching source_pair_features or source_geometry; "
                "target-derived pair features must not be reused as source conditioning"
            )
        if source_geometry is not None:
            source_ca = _tensor(_field(source_geometry, "ca_coordinates"), torch.float32)
            source_geom_mask = _tensor(_field(source_geometry, "frame_mask"), torch.bool)
            if source_ca.shape != (prepared.length, 3) or source_geom_mask.shape != (prepared.length,):
                raise ValueError("source_geometry must describe the same (L,3) source frame set")
            if not torch.allclose(
                source_ca.to(source_translations.device),
                source_translations[0],
                atol=5e-3,
                rtol=5e-3,
            ):
                raise ValueError("source_geometry CA coordinates do not match source_translations")
            source_geom_R = _tensor(_field(source_geometry, "residue_frames"), torch.float32)
            if source_geom_R.shape != (prepared.length, 3, 3):
                raise ValueError("source_geometry residue_frames must have shape (L,3,3)")
            valid_frames = source_geom_mask.to(source_rotations.device)
            if valid_frames.any() and not torch.allclose(
                source_geom_R.to(source_rotations.device)[valid_frames],
                source_rotations[0, valid_frames],
                atol=5e-3,
                rtol=5e-3,
            ):
                raise ValueError("source_geometry frames do not match source_rotations")
            training_pair_features = pair_features_from_geometry(
                source_geometry,
                channels=pair_channels,
                residue_index=prepared.residue_index[0].to(
                    _field(source_geometry, "distances").device
                ),
            ).unsqueeze(0).to(
                device=prepared.pair_features.device,
                dtype=prepared.pair_features.dtype,
            )
        else:
            source_pair_features = _tensor(source_pair_features, torch.float32)
            if source_pair_features.shape == (
                prepared.length, prepared.length, prepared.pair_features.shape[-1]
            ):
                source_pair_features = source_pair_features.unsqueeze(0)
            if source_pair_features.shape != prepared.pair_features.shape:
                raise ValueError(
                    "source_pair_features must have shape (L,L,C) or (1,L,L,C) matching prepared inputs"
                )
            if not torch.isfinite(source_pair_features).all():
                raise ValueError("source_pair_features contains non-finite values")
            training_pair_features = source_pair_features.to(
                device=prepared.pair_features.device,
                dtype=prepared.pair_features.dtype,
            )
    elif source_pair_features is not None or source_geometry is not None:
        raise ValueError("source pair features/geometry require an explicit source_frames pair")

    if substrate_id is not None:
        substrate_id = _tensor(substrate_id, torch.long)
        if substrate_id.ndim == 0:
            substrate_id = substrate_id.unsqueeze(0)
    if substrate_coords is not None:
        substrate_coords = _tensor(substrate_coords, torch.float32)
    if substrate_types is not None:
        substrate_types = _tensor(substrate_types, torch.long)

    return CanonicalTrainingBatch(
        msa_tokens=prepared.msa_tokens,
        pair_features=training_pair_features,
        source_R=source_rotations,
        source_t=source_translations,
        target_R=prepared.source_rotations,
        target_t=prepared.source_translations,
        target_sequence=target.unsqueeze(0),
        msa_padding_mask=prepared.msa_padding_mask,
        sequence_padding_mask=sequence_mask,
        constraints=constraints,
        structure_source=structure_source,
        substrate_id=substrate_id,
        substrate_coords=substrate_coords,
        substrate_types=substrate_types,
        residue_index=prepared.residue_index,
        residue_mask=prepared.residue_mask,
    )