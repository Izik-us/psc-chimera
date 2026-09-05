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

    def __getitem__(self, index: int) -> Dict[str, Any]:
        record = self.records[index]
        item = {key: torch.tensor(record[key]) for key in self.required}
        item["msa_tokens"] = item["msa_tokens"].long()
        for key in ("source_R", "source_t", "target_R", "target_t"):
            item[key] = item[key].float()
        if item["msa_tokens"].ndim != 2:
            raise ValueError("msa_tokens must have shape (N_seq, L)")
        for key in ("source_R", "target_R"):
            if item[key].shape[-2:] != (3, 3):
                raise ValueError(f"{key} must end in (3, 3)")
        for key in ("source_t", "target_t"):
            if item[key].shape[-1] != 3:
                raise ValueError(f"{key} must end in 3")
        return item
