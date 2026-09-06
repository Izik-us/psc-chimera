"""Merge codon-optimizer JSONL sources with provenance and leakage checks.

Each input record must contain ``aa_sequence``, ``codon_sequence``, and
``expression``. Optional fields such as ``accession``, ``source``, and
``label_type`` are preserved. The output is split by accession when possible,
so wildtype/optimized records cannot leak across train and validation sets.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from data.codon_dataset import CodonJSONLDataset


LABEL_PRIORITY = {"measured": 3, "weak": 2, "synthetic": 1, "unknown": 0}


def load_records(path: Path, source_name: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    inferred_label_type = (
        "synthetic"
        if "synthetic" in path.stem.lower()
        else "weak"
        if "proxy" in path.stem.lower() or "weak" in path.stem.lower()
        else "unknown"
    )
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        record = json.loads(line)
        record["source"] = str(record.get("source", source_name))
        record["label_type"] = str(
            record.get("label_type", inferred_label_type)
        ).lower()
        record["accession"] = str(record.get("accession", ""))
        CodonJSONLDataset._normalize(record, line_number)
        records.append(record)
    return records


def record_key(record: dict[str, Any]) -> str:
    return hashlib.sha256(
        f"{record['aa_sequence']}|{record['codon_sequence']}".encode("utf-8")
    ).hexdigest()


def merge_records(paths: list[Path]) -> list[dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    for path in paths:
        for record in load_records(path, path.stem):
            key = record_key(record)
            previous = selected.get(key)
            if previous is None or LABEL_PRIORITY.get(record["label_type"], 0) > LABEL_PRIORITY.get(
                previous["label_type"], 0
            ):
                selected[key] = record
    return sorted(
        selected.values(),
        key=lambda record: (
            record.get("accession", ""),
            record.get("source", ""),
            record["codon_sequence"],
        ),
    )


def split_by_accession(
    records: list[dict[str, Any]], validation_fraction: float
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")
    groups: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        accession = record.get("accession") or record_key(record)
        groups.setdefault(accession, []).append(record)

    ordered_groups = sorted(groups.items())
    validation_count = max(1, round(len(ordered_groups) * validation_fraction))
    validation_accessions = {
        accession for accession, _ in ordered_groups[:validation_count]
    }
    train = [record for accession, group in ordered_groups if accession not in validation_accessions for record in group]
    validation = [record for accession, group in ordered_groups if accession in validation_accessions for record in group]
    return train, validation


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("data/merged"))
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    args = parser.parse_args()

    records = merge_records(args.inputs)
    train, validation = split_by_accession(records, args.validation_fraction)
    write_jsonl(args.output_dir / "train.jsonl", train)
    write_jsonl(args.output_dir / "validation.jsonl", validation)
    summary = {
        "total_records": len(records),
        "train_records": len(train),
        "validation_records": len(validation),
        "sources": sorted({record["source"] for record in records}),
        "label_types": sorted({record["label_type"] for record in records}),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
