"""Build a human CDS/expression proxy dataset from the Pouyet archive.

The archive supplies Ensembl transcript IDs and endogenous human FPKM values;
Ensembl REST supplies the corresponding CDS. These are proxy labels for native
human expression, not controlled synonymous-variant experiments.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from chimera.codon_optimizer import translate_dna


SOURCE_URL = "https://zenodo.org/api/records/835063/files/data_HumanCodonUsage.zip/content"
ENSEMBL_URL = "https://rest.ensembl.org/sequence/id/{transcript}?type=cds"


def read_candidates(path: Path, expression_column: str) -> list[dict[str, Any]]:
    lines = [line.split() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"Empty summary file: {path}")
    header = lines[0]
    try:
        transcript_index = header.index("Transcript")
        expression_index = header.index(expression_column)
        gene_index = header.index("Ensembl.Gene.ID")
        symbol_index = header.index("gene.symbol")
    except ValueError as exc:
        raise ValueError(f"Missing expected column in {path}: {exc}") from exc

    candidates: list[dict[str, Any]] = []
    for fields in lines[1:]:
        if len(fields) <= max(transcript_index, expression_index, gene_index, symbol_index):
            continue
        try:
            expression = float(fields[expression_index])
        except ValueError:
            continue
        transcript = fields[transcript_index]
        if expression <= 0 or not transcript.startswith("ENST"):
            continue
        candidates.append(
            {
                "transcript": transcript,
                "gene_id": fields[gene_index],
                "gene_symbol": fields[symbol_index],
                "expression_raw": expression,
            }
        )
    candidates.sort(key=lambda item: item["expression_raw"], reverse=True)
    return candidates


def fetch_cds(transcript: str, cache_dir: Path) -> str | None:
    cache_path = cache_dir / f"{transcript}.txt"
    if cache_path.is_file():
        return cache_path.read_text(encoding="utf-8").strip().upper()
    request = Request(
        ENSEMBL_URL.format(transcript=transcript),
        headers={"Content-Type": "text/plain", "User-Agent": "psc-chimera-dataset-builder/1.0"},
    )
    try:
        with urlopen(request, timeout=30) as response:
            cds = response.read().decode("utf-8").strip().upper()
    except (HTTPError, URLError, TimeoutError):
        return None
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(cds + "\n", encoding="utf-8")
    return cds


def build_records(summary_path: Path, cache_dir: Path, max_records: int, expression_column: str) -> list[dict[str, Any]]:
    candidates = read_candidates(summary_path, expression_column)
    if not candidates:
        raise ValueError("No positive expression candidates found")
    scale = max(candidate["expression_raw"] for candidate in candidates)
    records: list[dict[str, Any]] = []
    seen_cds: set[str] = set()
    for candidate in candidates:
        if len(records) >= max_records:
            break
        cds = fetch_cds(candidate["transcript"], cache_dir)
        time.sleep(0.1)
        if cds is None or len(cds) < 90 or len(cds) % 3:
            continue
        if cds[-3:] in {"TAA", "TAG", "TGA"}:
            cds = cds[:-3]
        if len(cds) % 3 or cds in seen_cds:
            continue
        try:
            protein = translate_dna(cds)
        except (AssertionError, ValueError):
            continue
        if not protein or "*" in protein or any(amino_acid not in "ACDEFGHIKLMNPQRSTVWY" for amino_acid in protein):
            continue
        seen_cds.add(cds)
        records.append(
            {
                "aa_sequence": protein,
                "codon_sequence": cds,
                "expression": min(1.0, candidate["expression_raw"] / scale),
                "expression_raw": candidate["expression_raw"],
                "expression_metric": expression_column,
                "accession": candidate["transcript"],
                "gene_id": candidate["gene_id"],
                "gene_symbol": candidate["gene_symbol"],
                "source": "Pouyet_HumanCodonUsage",
                "label_type": "proxy",
                "host": "human",
                "assay": "endogenous_FPKM",
            }
        )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    parser.add_argument("--cache-dir", type=Path, default=Path("data/external/ensembl_cds_cache"))
    parser.add_argument("--output", type=Path, default=Path("data/human_expression_cds_proxy.jsonl"))
    parser.add_argument("--max-records", type=int, default=500)
    parser.add_argument("--expression-column", default="Exp_Meiosis")
    args = parser.parse_args()
    if args.max_records <= 0:
        raise ValueError("--max-records must be positive")
    records = build_records(args.summary, args.cache_dir, args.max_records, args.expression_column)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    print(f"records={len(records)} saved={args.output}")


if __name__ == "__main__":
    main()
