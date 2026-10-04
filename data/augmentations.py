"""Seeded, recordable structural and MSA input augmentations."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from typing import Any, Mapping

import torch

from .geometry import derive_geometry


@dataclass(frozen=True)
class AugmentationConfig:
    seed: int = 0
    rigid_rotation: bool = True
    translation_std_angstrom: float = 0.0
    coordinate_noise_std_angstrom: float = 0.0
    crop_length: int | None = None
    crop_interval: tuple[int, int] | None = None
    crop_kind: str = "sequence"
    msa_row_limit: int | None = None
    residue_mask_probability: float = 0.0

    def __post_init__(self) -> None:
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if self.translation_std_angstrom < 0 or self.coordinate_noise_std_angstrom < 0:
            raise ValueError("coordinate standard deviations must be non-negative")
        if self.crop_length is not None and self.crop_length < 1:
            raise ValueError("crop_length must be positive")
        if self.crop_interval is not None and not 0 <= self.crop_interval[0] < self.crop_interval[1]:
            raise ValueError("crop_interval must be a non-empty half-open interval")
        if self.crop_kind not in {"sequence", "domain", "module", "pocket"}:
            raise ValueError("crop_kind must be sequence, domain, module or pocket")
        if self.msa_row_limit is not None and self.msa_row_limit < 1:
            raise ValueError("msa_row_limit must be positive")
        if not 0.0 <= self.residue_mask_probability <= 1.0:
            raise ValueError("residue_mask_probability must be in [0, 1]")
        if self.crop_interval is not None and self.crop_length is not None:
            raise ValueError("provide crop_interval or crop_length, not both")


def _generator(sample_id: str, config: AugmentationConfig, epoch: int, draw: int) -> tuple[torch.Generator, int]:
    material = f"{config.seed}:{epoch}:{draw}:{sample_id}".encode("utf-8")
    seed = int.from_bytes(hashlib.sha256(material).digest()[:8], "big") % (2**63 - 1)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return generator, seed


def _rotation_matrix(generator: torch.Generator, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    axis = torch.randn(3, generator=generator, dtype=torch.float64)
    axis = axis / axis.norm().clamp_min(1e-12)
    angle = (torch.rand((), generator=generator, dtype=torch.float64) * 2.0 - 1.0) * torch.pi
    x, y, z = axis
    skew = torch.tensor(
        [[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]],
        dtype=torch.float64,
    )
    identity = torch.eye(3, dtype=torch.float64)
    rotation = identity + torch.sin(angle) * skew + (1.0 - torch.cos(angle)) * (skew @ skew)
    return rotation.to(device=device, dtype=dtype)


def _crop_sample(sample: dict[str, Any], start: int, end: int, kind: str) -> dict[str, Any]:
    result = dict(sample)
    original_sequence = str(sample["sequence"])
    if not 0 <= start < end <= len(original_sequence):
        raise ValueError("crop interval must lie within the canonical sequence")
    result["sequence"] = original_sequence[start:end]
    residue_keys = (
        "coordinates", "atom_mask", "residue_mask", "sequence_mask", "fixed_positions",
        "b_factors", "occupancies", "sequence_to_msa_columns",
    )
    for key in residue_keys:
        value = sample.get(key)
        if torch.is_tensor(value) and value.ndim > 0 and value.shape[0] == len(original_sequence):
            result[key] = value[start:end].clone()
    for key in ("residue_mappings",):
        if key in sample:
            result[key] = sample[key][start:end]
    if torch.is_tensor(sample.get("msa_tokens")):
        result["msa_tokens"] = sample["msa_tokens"][:, start:end].clone()
    if torch.is_tensor(sample.get("msa_mask")):
        result["msa_mask"] = sample["msa_mask"][:, start:end].clone()
    contexts = []
    for context in sample.get("ligand_contexts", []):
        updated = dict(context)
        updated["binding_residue_refs"] = [
            {**reference, "canonical_residue_index": reference["canonical_residue_index"] - start}
            for reference in context.get("binding_residue_refs", [])
            if start <= reference["canonical_residue_index"] < end
        ]
        updated["binding_residue_indices"] = [
            residue_index - start
            for residue_index in context.get("binding_residue_indices", [])
            if start <= residue_index < end
        ]
        contexts.append(updated)
    if "ligand_contexts" in sample:
        result["ligand_contexts"] = contexts
    result["augmentation_crop"] = {"kind": kind, "start": start, "end": end}
    return result


def augment_sample(
    sample: Mapping[str, Any],
    config: AugmentationConfig,
    *,
    epoch: int = 0,
    draw: int = 0,
) -> dict[str, Any]:
    """Apply deterministic, mask-aware geometric/MSA input transforms to one sample."""
    if "coordinates" not in sample or "atom_mask" not in sample or "sequence" not in sample:
        raise ValueError("sample requires sequence, coordinates and atom_mask")
    sample_id = str(sample.get("sample_id", "sample"))
    generator, actual_seed = _generator(sample_id, config, epoch, draw)
    result = dict(sample)
    coordinates = sample["coordinates"].clone()
    atom_mask = sample["atom_mask"].bool()
    length = len(sample["sequence"])
    valid_atom_mask = atom_mask

    rotation = torch.eye(3, dtype=coordinates.dtype, device=coordinates.device)
    if config.rigid_rotation:
        rotation = _rotation_matrix(generator, coordinates.dtype, coordinates.device)
        coordinates = torch.einsum("ij,laj->lai", rotation, coordinates)
    translation = torch.zeros(3, dtype=coordinates.dtype, device=coordinates.device)
    if config.translation_std_angstrom:
        translation = torch.randn(3, generator=generator, dtype=coordinates.dtype).to(coordinates.device)
        translation = translation * config.translation_std_angstrom
        coordinates = coordinates + translation
    if config.coordinate_noise_std_angstrom:
        noise = torch.randn(coordinates.shape, generator=generator, dtype=coordinates.dtype).to(coordinates.device)
        coordinates = coordinates + noise * config.coordinate_noise_std_angstrom * valid_atom_mask.unsqueeze(-1)
    coordinates = coordinates * valid_atom_mask.unsqueeze(-1).to(coordinates.dtype)
    result["coordinates"] = coordinates

    crop_start, crop_end = 0, length
    if config.crop_interval is not None:
        crop_start, crop_end = config.crop_interval
    elif config.crop_length is not None and config.crop_length < length:
        crop_start = int(torch.randint(length - config.crop_length + 1, (1,), generator=generator).item())
        crop_end = crop_start + config.crop_length
    if crop_start != 0 or crop_end != length:
        result = _crop_sample(result, crop_start, crop_end, config.crop_kind)
        coordinates = result["coordinates"]
        atom_mask = result["atom_mask"].bool()
        length = len(result["sequence"])

    if config.msa_row_limit is not None and torch.is_tensor(result.get("msa_tokens")):
        tokens = result["msa_tokens"]
        mask = result.get("msa_mask")
        row_count = tokens.shape[0]
        keep_count = min(config.msa_row_limit, row_count)
        if keep_count < row_count:
            selected = torch.randperm(max(row_count - 1, 0), generator=generator)[:keep_count - 1] + 1
            rows = torch.cat([torch.zeros(1, dtype=torch.long), selected]).to(tokens.device)
            result["msa_tokens"] = tokens.index_select(0, rows)
            if torch.is_tensor(mask):
                result["msa_mask"] = mask.index_select(0, rows)
            result["augmentation_msa_rows"] = rows.tolist()
        else:
            result["augmentation_msa_rows"] = list(range(row_count))

    residue_mask = result.get("residue_mask", atom_mask[:, 1]).bool()
    corruption_mask = torch.zeros(length, dtype=torch.bool, device=coordinates.device)
    if config.residue_mask_probability:
        draws = torch.rand(length, generator=generator).to(coordinates.device)
        corruption_mask = (draws < config.residue_mask_probability) & residue_mask
        fixed_positions = result.get("fixed_positions")
        if torch.is_tensor(fixed_positions):
            corruption_mask &= ~fixed_positions.bool()
    result["residue_corruption_mask"] = corruption_mask

    result["geometry"] = {
        field_name: getattr(
            derive_geometry(
                coordinates,
                atom_mask,
                residue_mask,
                chain_ids=[str(sample.get("label_asym_id", "A"))] * length,
            ),
            field_name,
        )
        for field_name in (
            "ca_coordinates", "backbone_coordinates", "residue_frames", "frame_mask",
            "residue_mask", "pair_mask", "distances", "relative_positions",
            "relative_rotations", "torsions", "torsion_mask", "contacts",
            "edge_index", "edge_features", "edge_mask",
        )
    }
    result["augmentation"] = {
        "config": asdict(config),
        "seed": actual_seed,
        "epoch": epoch,
        "draw": draw,
        "rotation_matrix": rotation.detach().cpu().tolist(),
        "translation_angstrom": translation.detach().cpu().tolist(),
        "crop": result.get("augmentation_crop"),
        "masked_residues": torch.nonzero(corruption_mask, as_tuple=False).flatten().tolist(),
        "msa_rows": result.get("augmentation_msa_rows"),
    }
    return result