#!/usr/bin/env python3
"""
PSC-CHIMERA: Run a design session from the command line.

Usage:
    python scripts/run_design.py \
        --substrate PHE \
        --n-designs 500 \
        --n-pareto 50 \
        --flow-ckpt weights/rfdiffusion_base.pt \
        --mpnn-ckpt weights/proteinmpnn_v48_020.pt \
        --source-pdb data/1AMU.pdb \
        --output-dir results/phe_designs/
"""

import argparse
import torch
import os
import json
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="CHIMERA v2 NRPS Design")
    p.add_argument("--substrate", default="PHE", help="Target substrate (3-letter code)")
    p.add_argument("--n-designs", type=int, default=500)
    p.add_argument("--n-pareto", type=int, default=50)
    p.add_argument("--flow-ckpt", default=None, help="Compatible local flow/SB checkpoint path")
    p.add_argument("--mpnn-ckpt", default=None, help="Compatible local ProteinMPNN checkpoint path")
    p.add_argument(
        "--evof-ckpt", default=None, help="Compatible local EvoFormer checkpoint path"
    )
    p.add_argument("--source-pdb", default=None, help="Source bacterial NRPS PDB file")
    p.add_argument(
        "--msa-file", default=None, help="Animal NRPS MSA file (.a3m or .fasta)"
    )
    p.add_argument("--output-dir", default="results/", help="Output directory")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument(
        "--flow-steps", type=int, default=20, help="Stochastic Schrödinger-bridge integration steps (default 20)"
    )
    p.add_argument(
        "--seed", type=int, default=0, help="Random seed for reproducible demo/model initialization"
    )
    p.add_argument(
        "--no-rag", action="store_true", help="Disable structural retrieval RAG"
    )
    p.add_argument(
        "--demo", action="store_true", help="Use synthetic inputs; do not use output as a biological design"
    )
    return p.parse_args()


def main():
    args = parse_args()
    if args.n_designs < 1 or args.n_pareto < 1:
        raise ValueError("--n-designs and --n-pareto must both be positive")
    if args.flow_steps < 1:
        raise ValueError("--flow-steps must be positive")
    if args.n_pareto > args.n_designs:
        raise ValueError("--n-pareto cannot exceed --n-designs")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device)

    print(f"PSC-CHIMERA Design Run")
    print(f"  Target substrate: {args.substrate}")
    print(f"  Designs to generate: {args.n_designs}")
    print(f"  Pareto samples to return: {args.n_pareto}")
    print(f"  Device: {args.device}")
    print(f"  SB integration steps: {args.flow_steps}")
    print(f"  Seed: {args.seed}")
    print()

    import sys

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from chimera.chimera_v2 import CHIMERAv2, NRPSConstraints
    from chimera.structure_utils import load_backbone_pdb, load_msa

    model = CHIMERAv2.from_pretrained(
        evoformer_ckpt=args.evof_ckpt,
        flow_ckpt=args.flow_ckpt,
        mpnn_ckpt=args.mpnn_ckpt,
    ).to(device)

    if args.source_pdb:
        print(f"Loading source backbone from {args.source_pdb}")
        source_R, source_t = load_backbone_pdb(args.source_pdb)
        source_R, source_t = source_R.to(device), source_t.to(device)
    else:
        if not args.demo:
            raise ValueError("--source-pdb is required outside --demo mode")
        print("No source PDB provided — using identity frames (for testing)")
        L = 600
        source_R = torch.eye(3, device=device).view(1, 1, 3, 3).expand(1, L, -1, -1)
        source_t = torch.zeros(1, L, 3, device=device)

    if args.msa_file:
        print(f"Loading MSA from {args.msa_file}")
        msa_tokens = load_msa(args.msa_file).to(device)
        L = msa_tokens.shape[-1]
        if source_t.shape[-2] != L:
            raise ValueError("PDB residue count and MSA alignment length must match")
    else:
        if not args.demo:
            raise ValueError("--msa-file is required outside --demo mode")
        print("No MSA file provided — using random tokens (for testing)")
        L, N_seq = 600, 32
        msa_tokens = torch.randint(0, 23, (1, N_seq, L), device=device)

    print(f"\nGenerating {args.n_designs} designs...")
    results = model.design(
        nrps_msa=msa_tokens,
        source_backbone=(source_R, source_t),
        initial_pair_features=torch.zeros(1, L, L, model.evoformer.d_pair, device=device),
        target_substrate=args.substrate,
        n_designs=args.n_designs,
        n_pareto_samples=args.n_pareto,
        device=str(device),
        flow_steps=args.flow_steps,
        use_rag=not args.no_rag,
    )

    os.makedirs(args.output_dir, exist_ok=True)

    fasta_path = os.path.join(args.output_dir, "pareto_sequences.fasta")
    AA = "ACDEFGHIKLMNPQRSTVWY"
    with open(fasta_path, "w") as f:
        for i, seq in enumerate(results["pareto_sequences"]):
            scores = results["pareto_scores"][i]
            f.write(
                f">design_{i:04d} | evol={scores[0]:.3f} stab={scores[1]:.3f} "
                f"expr={scores[2]:.3f} sel={scores[3]:.3f} asm={scores[4]:.3f}\n"
            )
            aa_str = "".join(AA[t] if t < 20 else "X" for t in seq.tolist())
            f.write(aa_str + "\n")

    meta_path = os.path.join(args.output_dir, "design_metadata.json")
    meta = {
        "substrate": args.substrate,
        "n_generated": results["total_generated"],
        "pareto_count": results["pareto_count"],
        "n_returned": len(results["pareto_sequences"]),
        "device": str(device),
        "sb_integration_steps": args.flow_steps,
        "seed": args.seed,
        "rag_enabled": not args.no_rag,
        "demo": args.demo,
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nResults saved to {args.output_dir}")
    print(
        f"  {len(results['pareto_sequences'])} Pareto-optimal sequences → {fasta_path}"
    )
    print(f"  Metadata → {meta_path}")
    print(f"\nNext step: send sequences to PROTEUS for mammalian cell screening.")


if __name__ == "__main__":
    main()
