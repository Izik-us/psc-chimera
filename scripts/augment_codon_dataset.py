"""Generate reproducible synonymous-codon augmentations for the CHIMERA codon dataset.

Synthetic records use an in-silico codon-adaptiveness score. They are not new
biological observations and their scores are not experimentally measured expression.
This script uses only the Python standard library.
"""
from __future__ import annotations

import argparse
import ast
import json
import math
import random
from pathlib import Path
from typing import Any

AUGMENTATION_SOURCE = "CHIMERA_SynonymousAugmentation_v1"
VALID_AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
TEMPERATURES = (0.55, 0.7, 0.85, 1.0, 1.15, 1.35, 1.6, 2.0)


def load_codon_resources(project_root: Path) -> tuple[dict[str, list[str]], dict[str, float]]:
    """Read the canonical literal codon tables without importing PyTorch."""
    source = project_root / "chimera" / "codon_optimizer.py"
    module = ast.parse(source.read_text(encoding="utf-8"))
    values: dict[str, Any] = {}
    for node in module.body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.target.id in {"CODON_TABLE", "HUMAN_CODON_FREQ"}:
                values[node.target.id] = ast.literal_eval(node.value)
    if "CODON_TABLE" not in values or "HUMAN_CODON_FREQ" not in values:
        raise RuntimeError(f"Could not read canonical codon constants from {source}")
    return values["CODON_TABLE"], values["HUMAN_CODON_FREQ"]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_atomic(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    temporary.replace(path)


def translate_dna(dna: str, codon_to_aa: dict[str, str]) -> str:
    dna = "".join(dna.split()).upper()
    if len(dna) % 3:
        raise ValueError("Coding sequence length must be divisible by three")
    codons = [dna[index:index + 3] for index in range(0, len(dna), 3)]
    try:
        return "".join(codon_to_aa[codon] for codon in codons)
    except KeyError as exc:
        raise ValueError(f"Unknown codon: {exc.args[0]}") from exc


def validate_record(
    record: dict[str, Any],
    context: str,
    codon_to_aa: dict[str, str],
) -> None:
    protein = str(record.get("aa_sequence", "")).strip().upper()
    dna = str(record.get("codon_sequence", "")).replace(" ", "").upper()
    if not protein or len(dna) != 3 * len(protein):
        raise ValueError(f"{context}: protein and CDS lengths do not match")
    if any(residue not in VALID_AMINO_ACIDS for residue in protein):
        raise ValueError(f"{context}: noncanonical amino-acid symbol")
    if translate_dna(dna, codon_to_aa) != protein:
        raise ValueError(f"{context}: codon sequence does not translate to protein")
    expression = float(record.get("expression", -1.0))
    if not 0.0 <= expression <= 1.0:
        raise ValueError(f"{context}: expression/score must be in [0, 1]")


def codon_adaptiveness_score(
    protein: str,
    dna: str,
    codon_table: dict[str, list[str]],
    codon_frequency: dict[str, float],
) -> float:
    """Compute a CAI-like score, explicitly not an expression measurement."""
    log_scores: list[float] = []
    for index, amino_acid in enumerate(protein):
        codon = dna[index * 3:index * 3 + 3]
        synonyms = codon_table[amino_acid]
        maximum = max(codon_frequency.get(candidate, 0.1) for candidate in synonyms)
        relative = codon_frequency.get(codon, 0.1) / maximum
        log_scores.append(math.log(max(relative, 1e-12)))
    return math.exp(sum(log_scores) / max(len(log_scores), 1))


def draw_synonym(
    rng: random.Random,
    alternatives: list[str],
    temperature: float,
    codon_frequency: dict[str, float],
) -> str:
    weights = [
        max(codon_frequency.get(codon, 0.1), 1e-6) ** (1.0 / temperature)
        for codon in alternatives
    ]
    return rng.choices(alternatives, weights=weights, k=1)[0]


def make_variant(
    parent: dict[str, Any],
    variant_index: int,
    rng: random.Random,
    seen: set[tuple[str, str]],
    seed: int,
    change_probability: float,
    codon_table: dict[str, list[str]],
    codon_frequency: dict[str, float],
    codon_to_aa: dict[str, str],
) -> dict[str, Any]:
    protein = str(parent["aa_sequence"]).upper()
    parent_dna = str(parent["codon_sequence"]).upper()
    parent_codons = [parent_dna[i:i + 3] for i in range(0, len(parent_dna), 3)]
    temperature = TEMPERATURES[(variant_index - 1) % len(TEMPERATURES)]
    parent_accession = str(parent.get("accession", "UNKNOWN"))

    for _attempt in range(250):
        new_codons: list[str] = []
        changed = False
        for amino_acid, original_codon in zip(protein, parent_codons):
            synonyms = codon_table[amino_acid]
            if len(synonyms) < 2 or rng.random() > change_probability:
                new_codons.append(original_codon)
                continue
            alternatives = [codon for codon in synonyms if codon != original_codon]
            if not alternatives:
                new_codons.append(original_codon)
                continue
            new_codons.append(
                draw_synonym(rng, alternatives, temperature, codon_frequency)
            )
            changed = True

        dna = "".join(new_codons)
        key = (protein, dna)
        if not changed or key in seen:
            continue
        if translate_dna(dna, codon_to_aa) != protein:
            raise ValueError(f"Generated non-synonymous record for {parent_accession}")
        score = codon_adaptiveness_score(protein, dna, codon_table, codon_frequency)
        record = {
            "aa_sequence": protein,
            "accession": f"CHIMERA-SYN:{parent_accession}:v{variant_index:03d}",
            "augmentation_change_probability": change_probability,
            "augmentation_seed": seed,
            "augmentation_temperature": temperature,
            "augmentation_variant_index": variant_index,
            "codon_sequence": dna,
            "expression": score,
            "expression_metric": "synthetic_CAI_like_score",
            "expression_target_kind": "synthetic_codon_adaptiveness_not_measured_expression",
            "gene_id": parent.get("gene_id"),
            "gene_symbol": parent.get("gene_symbol"),
            "host": "human",
            "label_type": "synthetic",
            "parent_accession": parent_accession,
            "source": AUGMENTATION_SOURCE,
        }
        validate_record(record, f"synthetic {record['accession']}", codon_to_aa)
        seen.add(key)
        return record

    raise RuntimeError(f"Unable to make a unique variant for {parent_accession}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--train", type=Path, default=root / "data/merged_codon_v2/train.jsonl")
    parser.add_argument("--validation", type=Path, default=root / "data/merged_codon_v2/validation.jsonl")
    parser.add_argument("--manifest", type=Path, default=root / "data/merged_codon_v2/manifest.json")
    parser.add_argument("--variants-per-parent", type=int, default=23)
    parser.add_argument("--seed", type=int, default=20261025)
    parser.add_argument("--minimum-total-records", type=int, default=10000)
    parser.add_argument("--change-probability", type=float, default=0.35)
    args = parser.parse_args()

    if args.variants_per_parent < 1:
        raise ValueError("variants-per-parent must be positive")
    if args.seed < 0:
        raise ValueError("seed must be non-negative")
    if not 0.0 < args.change_probability <= 1.0:
        raise ValueError("change-probability must be in (0, 1]")

    codon_table, codon_frequency = load_codon_resources(root)
    codon_to_aa = {
        codon: amino_acid
        for amino_acid, codons in codon_table.items()
        for codon in codons
    }

    # Rebuild from original sources, not from generated variants, so reruns
    # remain deterministic and do not recursively expand the dataset.
    loaded = read_jsonl(args.train)
    original = [record for record in loaded if record.get("source") != AUGMENTATION_SOURCE]
    validation = read_jsonl(args.validation)
    for index, record in enumerate(original):
        validate_record(record, f"original train record {index}", codon_to_aa)
    for index, record in enumerate(validation):
        validate_record(record, f"validation record {index}", codon_to_aa)

    validation_proteins = {record["aa_sequence"] for record in validation}
    if any(record["aa_sequence"] in validation_proteins for record in original):
        raise ValueError("Training and validation share an amino-acid sequence; refusing augmentation")

    parents = [
        record for record in original
        if record.get("source") == "Pouyet_HumanCodonUsage"
        and str(record.get("label_type", "")).lower() == "proxy"
    ]
    if not parents:
        raise ValueError("No Pouyet human-expression proxy records found to augment")

    seen: set[tuple[str, str]] = set()
    for record in original:
        key = (record["aa_sequence"], record["codon_sequence"])
        if key in seen:
            raise ValueError("Original training records contain duplicate protein/CDS pairs")
        seen.add(key)

    rng = random.Random(args.seed)
    synthetic = [
        make_variant(
            parent, variant_index, rng, seen, args.seed, args.change_probability,
            codon_table, codon_frequency, codon_to_aa,
        )
        for parent in parents
        for variant_index in range(1, args.variants_per_parent + 1)
    ]
    output_records = original + synthetic
    if len(output_records) < args.minimum_total_records:
        raise ValueError(
            f"Only {len(output_records)} training records; need at least {args.minimum_total_records}"
        )
    if len(seen) != len(output_records):
        raise ValueError("Duplicate protein/CDS pair detected after augmentation")

    all_sources = sorted({str(record.get("source", "UNKNOWN")) for record in output_records + validation})
    all_labels = sorted({str(record.get("label_type", "unknown")).lower() for record in output_records + validation})
    old_manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    measured_count = sum(record.get("label_type") == "measured" for record in original)
    proxy_count = sum(record.get("label_type") == "proxy" for record in original)
    manifest = {
        **old_manifest,
        "total_records": len(output_records) + len(validation),
        "train_records": len(output_records),
        "validation_records": len(validation),
        "sources": all_sources,
        "label_types": all_labels,
        "record_counts": {
            "original_train_records": len(original),
            "original_measured_train_records": measured_count,
            "original_proxy_train_records": proxy_count,
            "synthetic_train_records": len(synthetic),
            "validation_records": len(validation),
        },
        "synthetic_augmentation": {
            "version": 1,
            "source": AUGMENTATION_SOURCE,
            "seed": args.seed,
            "parent_source": "Pouyet_HumanCodonUsage",
            "parent_records": len(parents),
            "variants_per_parent": args.variants_per_parent,
            "synthetic_records": len(synthetic),
            "scoring_method": "Geometric mean of per-codon relative adaptiveness from HUMAN_CODON_FREQ in chimera/codon_optimizer.py",
            "score_is_measured_expression": False,
            "records_are_new_independent_biological_observations": False,
            "validation_augmented": False,
        },
    }

    train_text = "".join(
        json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
        for record in output_records
    )
    manifest_text = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    write_atomic(args.train, train_text)
    write_atomic(args.manifest, manifest_text)

    scores = [float(record["expression"]) for record in synthetic]
    print(json.dumps({
        "train_records": len(output_records),
        "original_train_records": len(original),
        "synthetic_records": len(synthetic),
        "proxy_parents": len(parents),
        "validation_records_unchanged": len(validation),
        "total_records_across_splits": len(output_records) + len(validation),
        "synthetic_score_min": min(scores),
        "synthetic_score_max": max(scores),
        "synthetic_labels_are_measured_expression": False,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
