"""Validated JSONL dataset for codon-optimizer training."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from chimera.codon_optimizer import tokenize_dna, tokenize_protein, translate_dna


class CodonJSONLDataset(Dataset):
    """Load records with aa_sequence, codon_sequence, and expression fields."""

    required = ("aa_sequence", "codon_sequence", "expression")

    def __init__(self, path: str | Path, max_records: int | None = None) -> None:
        self.path = Path(path)
        self.records = [
            json.loads(line)
            for line in self.path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if max_records is not None:
            self.records = self.records[:max_records]
        if not self.records:
            raise ValueError(f"Dataset is empty: {self.path}")

    @staticmethod
    def _normalize(record: dict[str, Any], index: int) -> tuple[str, str, float]:
        missing = [key for key in CodonJSONLDataset.required if key not in record]
        if missing:
            raise ValueError(f"Record {index} missing fields: {missing}")
        aa_sequence = "".join(str(record["aa_sequence"]).split()).upper()
        codon_sequence = str(record["codon_sequence"]).replace(" ", "").upper()
        if not aa_sequence:
            raise ValueError(f"Record {index} has an empty aa_sequence")
        if len(codon_sequence) != len(aa_sequence) * 3:
            raise ValueError(f"Record {index} DNA length does not match protein length")
        tokenize_dna(codon_sequence)
        if translate_dna(codon_sequence) != aa_sequence:
            raise ValueError(f"Record {index} DNA does not translate to aa_sequence")
        expression = float(record["expression"])
        if not 0.0 <= expression <= 1.0:
            raise ValueError(f"Record {index} expression must be between 0 and 1")
        return aa_sequence, codon_sequence, expression

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | list[str]]:
        aa_sequence, codon_sequence, expression = self._normalize(self.records[index], index)
        return {
            "protein_tokens": tokenize_protein(aa_sequence),
            "codon_tokens": tokenize_dna(codon_sequence),
            "aa_sequence": [aa_sequence],
            "expression": torch.tensor(expression, dtype=torch.float32),
        }

    @classmethod
    def write_example(cls, path: str | Path) -> Path:
        """Write a tiny schema-valid example dataset for pipeline checks."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "aa_sequence": "MTEYKLVV",
            "codon_sequence": "ATGACCGAGTATAAGCTGGTGGTT",
            "expression": 0.5,
        }
        destination.write_text(json.dumps(record) + "\n", encoding="utf-8")
        return destination
