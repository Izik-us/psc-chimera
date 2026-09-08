"""Executable JSONL dataset for CHIMERA training examples.

Each record contains integer MSA tokens and paired source/target frames. The
loader deliberately requires prepared tensors; it does not invent biological
labels or random structures for production training.
"""

import json
from pathlib import Path
from typing import Any, Dict

import torch
from torch.utils.data import Dataset


class ChimeraJSONLDataset(Dataset):
    """Load validated, tensorized training records from a JSONL manifest."""

    required = ("msa_tokens", "source_R", "source_t", "target_R", "target_t")

    def __init__(self, path: str | Path):
        self.path = Path(path)
        with self.path.open(encoding="utf-8") as handle:
            self.records = [json.loads(line) for line in handle if line.strip()]
        if not self.records:
            raise ValueError(f"Training manifest is empty: {self.path}")
        for index, record in enumerate(self.records):
            missing = [key for key in self.required if key not in record]
            if missing:
                raise ValueError(f"Record {index} missing required fields: {missing}")

    def __len__(self) -> int:
        return len(self.records)

    @staticmethod
    def _validate_rotation(name: str, rotation: torch.Tensor) -> None:
        if not torch.isfinite(rotation).all():
            raise ValueError(f"{name} contains non-finite values")
        identity = torch.eye(3, dtype=rotation.dtype, device=rotation.device)
        gram = rotation.transpose(-1, -2) @ rotation
        if not torch.allclose(gram, identity.expand_as(gram), atol=2e-3, rtol=2e-3):
            raise ValueError(f"{name} contains non-orthonormal frames")
        det = torch.linalg.det(rotation)
        if not torch.allclose(det, torch.ones_like(det), atol=2e-3, rtol=2e-3):
            raise ValueError(f"{name} contains reflections; expected proper rotations in SO(3)")

    def __getitem__(self, index: int) -> Dict[str, Any]:
        record = self.records[index]
        item = {key: torch.tensor(record[key]) for key in self.required}
        item["msa_tokens"] = item["msa_tokens"].long()
        for key in ("source_R", "source_t", "target_R", "target_t"):
            item[key] = item[key].float()

        if item["msa_tokens"].ndim != 2:
            raise ValueError("msa_tokens must have shape (N_seq, L)")
        if item["msa_tokens"].shape[0] < 1 or item["msa_tokens"].shape[1] < 1:
            raise ValueError("msa_tokens must contain at least one sequence and one residue")
        if torch.any((item["msa_tokens"] < 0) | (item["msa_tokens"] >= 23)):
            raise ValueError("msa_tokens must contain integer token IDs in [0,22]")

        for key in ("source_R", "target_R"):
            if item[key].shape[-2:] != (3, 3):
                raise ValueError(f"{key} must end in (3, 3)")
        for key in ("source_t", "target_t"):
            if item[key].shape[-1] != 3:
                raise ValueError(f"{key} must end in 3")

        lengths = {
            item["source_R"].shape[-3],
            item["source_t"].shape[-2],
            item["target_R"].shape[-3],
            item["target_t"].shape[-2],
            item["msa_tokens"].shape[-1],
        }
        if len(lengths) != 1:
            raise ValueError(
                "MSA, source frames, and target frames must describe the same residue length"
            )

        for key in ("source_R", "target_R"):
            self._validate_rotation(key, item[key])
        for key in ("source_t", "target_t"):
            if not torch.isfinite(item[key]).all():
                raise ValueError(f"{key} contains non-finite coordinates")

        return item
