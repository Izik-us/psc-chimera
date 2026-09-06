"""Convert Fath et al. 2011 FASTA pairs and Table 1 ratios into JSONL.

Table 1 reports optimized:wildtype expression ratios for 44 pairs. Six genes
are marked ``only opt`` and are excluded because no wildtype comparison is
reported. Labels are normalized by the largest reported ratio.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

from chimera.codon_optimizer import translate_dna


HEADER_PATTERN = re.compile(r"^>(?P<accession>\S+)\s+.*\s+(?P<kind>wildtype|optimized)$", re.IGNORECASE)
STOP_CODONS = {"TAA", "TAG", "TGA"}


def read_fasta(path: Path) -> dict[tuple[str, str], str]:
    records: dict[tuple[str, str], str] = {}
    accession: str | None = None
    kind: str | None = None
    sequence_parts: list[str] = []

    def flush() -> None:
        if accession is not None and kind is not None:
            sequence = "".join(sequence_parts).upper().replace("U", "T")
            records[(accession, kind)] = sequence

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(">"):
            flush()
            match = HEADER_PATTERN.match(line)
            if match is None:
                raise ValueError(f"Unsupported FASTA header: {line}")
            accession = match.group("accession")
            kind = match.group("kind").lower()
            sequence_parts = []
        else:
            sequence_parts.append(line)
    flush()
    return records


def read_expression_table(path: Path) -> dict[str, float | None]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return {
            row["accession"]: (
                float(row["expression_ratio_opt_over_wt"])
                if row["expression_ratio_opt_over_wt"]
                else None
            )
            for row in csv.DictReader(handle)
        }


def build_records(
    records: dict[tuple[str, str], str],
    expression_ratios: dict[str, float | None],
) -> list[dict[str, object]]:
    accessions = sorted({accession for accession, _ in records})
    numeric_ratios = [
        ratio for accession, ratio in expression_ratios.items()
        if accession in accessions and ratio is not None
    ]
    if not numeric_ratios:
        raise ValueError("Expression table contains no ratios matching the FASTA")
    scale = max(numeric_ratios)
    output: list[dict[str, object]] = []
    for accession in accessions:
        ratio = expression_ratios.get(accession)
        if ratio is None:
            continue
        wildtype = records.get((accession, "wildtype"))
        optimized = records.get((accession, "optimized"))
        if wildtype is None or optimized is None:
            raise ValueError(f"Missing wildtype/optimized pair for {accession}")
        while wildtype[-3:] in STOP_CODONS:
            wildtype = wildtype[:-3]
        while optimized[-3:] in STOP_CODONS:
            optimized = optimized[:-3]
        if len(wildtype) != len(optimized):
            raise ValueError(f"Unequal sequence lengths for {accession}")
        if len(wildtype) % 3 != 0:
            raise ValueError(f"Sequence length is not divisible by three for {accession}")

        wildtype_protein = translate_dna(wildtype)
        optimized_protein = translate_dna(optimized)
        if wildtype_protein != optimized_protein:
            raise ValueError(f"Pair is not synonymous for {accession}")

        output.extend(
            [
                {
                    "aa_sequence": wildtype_protein,
                    "codon_sequence": wildtype,
                    "expression": 1.0 / scale,
                    "expression_ratio": 1.0,
                    "accession": accession,
                    "source": "Fath2011_Table1",
                    "label_type": "measured",
                },
                {
                    "aa_sequence": optimized_protein,
                    "codon_sequence": optimized,
                    "expression": ratio / scale,
                    "expression_ratio": ratio,
                    "accession": accession,
                    "source": "Fath2011_Table1",
                    "label_type": "measured",
                },
            ]
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_fasta", type=Path)
    parser.add_argument("output_jsonl", type=Path)
    parser.add_argument(
        "--expression-table",
        type=Path,
        default=Path("data/Fath2011_Table1_expression.csv"),
    )
    args = parser.parse_args()

    records = build_records(
        read_fasta(args.input_fasta),
        read_expression_table(args.expression_table),
    )
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w", encoding="utf-8", newline="\n") as output_file:
        for record in records:
            output_file.write(json.dumps(record) + "\n")
    print(f"records={len(records)} pairs={len(records) // 2} saved={args.output_jsonl}")


if __name__ == "__main__":
    main()
