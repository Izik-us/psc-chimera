"""Compare a trained codon checkpoint with the sliding-window baseline."""

from __future__ import annotations

import argparse

from chimera.codon_optimizer import (
    ALL_CODONS,
    CodonOptimizer,
    compute_cai,
    count_bad_motifs,
    detokenize_dna,
    sliding_window_optimize,
    tokenize_dna,
    tokenize_protein,
    translate_dna,
)


def print_metrics(
    label: str,
    dna: str,
    protein: str,
    cai: float,
    bad_motifs: int,
) -> None:
    translated = translate_dna(dna)
    gc_fraction = (dna.count("G") + dna.count("C")) / len(dna)
    print(f"\n[{label}]")
    print(f"Optimizing {len(protein)}-AA protein...")
    print(f"  Input:   {protein}")
    print(f"  Codons:  {len(dna) // 3} ({len(dna)} bp)")
    print(
        f"  CAI:     {cai:.3f} "
        f"{'PASS' if cai >= 0.96 else 'FAIL (target >= 0.96)'}"
    )
    print(
        f"  GC content: {gc_fraction:.3f} "
        f"{'PASS' if 0.58 <= gc_fraction <= 0.65 else 'FAIL (target 0.58-0.65)'}"
    )
    print(f"  Bad motifs: {bad_motifs} {'PASS' if bad_motifs == 0 else 'FAIL'}")
    print(f"  Translation check: {'PASS' if translated == protein else 'FAIL'}")
    print(f"  DNA:     {dna}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="checkpoints/codon_fath2011_measured_v2.pt")
    parser.add_argument(
        "--protein",
        default="MTEYKLVVVGAGGVGKSALTIQLIQNHFVDEYDPTIEDSY",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--compare-sliding", action="store_true")
    args = parser.parse_args()

    protein = "".join(args.protein.split()).upper()
    model = CodonOptimizer.from_checkpoint(args.checkpoint)
    model.eval()
    protein_tokens = tokenize_protein(protein).unsqueeze(0)
    generated_tokens, estimated_cai = model.generate(
        protein_tokens=protein_tokens,
        aa_sequence=[protein],
        temperature=args.temperature,
    )
    dna = detokenize_dna(generated_tokens[0])

    print("=" * 65)
    print("PSC CodonOptimizer - NRPS Module Codon Optimization")
    print("=" * 65)

    if args.compare_sliding:
        sliding_dna, sliding_metrics = sliding_window_optimize(protein)
        print_metrics(
            "Rule-based only: Fath et al. sliding window",
            sliding_dna,
            protein,
            sliding_metrics["cai"],
            sliding_metrics["n_bad_motifs"],
        )

    print_metrics(
        f"AI model: CodonOptimizer (trained checkpoint: {args.checkpoint})",
        dna,
        protein,
        estimated_cai,
        count_bad_motifs(dna),
    )
    print(f"\nVocabulary: {len(ALL_CODONS)} codons")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(
        "Trainable parameters: "
        f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
    )
    print(f"Actual CAI: {compute_cai(tokenize_dna(dna)).item():.4f}")


if __name__ == "__main__":
    main()
