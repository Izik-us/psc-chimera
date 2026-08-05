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
    p.add_argument(
        "--substrate", default="PHE", help="Target substrate (3-letter code)"
    )
    p.add_argument("--n-designs", type=int, default=500)
    p.add_argument("--n-pareto", type=int, default=50)
    p.add_argument("--flow-ckpt", default=None, help="RFdiffusion checkpoint path")
    p.add_argument("--mpnn-ckpt", default=None, help="ProteinMPNN checkpoint path")
    p.add_argument(
        "--evof-ckpt", default=None, help="EvoFormer/OpenFold checkpoint path"
    )
    p.add_argument("--source-pdb", default=None, help="Source bacterial NRPS PDB file")
    p.add_argument(
        "--msa-file", default=None, help="Animal NRPS MSA file (.a3m or .fasta)"
    )
    p.add_argument("--output-dir", default="results/", help="Output directory")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument(
        "--flow-steps", type=int, default=20, help="OT-Flow Matching NFE (default 20)"
    )
    p.add_argument(
        "--no-rag", action="store_true", help="Disable structural retrieval RAG"
    )
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    print(f"PSC-CHIMERA Design Run")
    print(f"  Target substrate: {args.substrate}")
    print(f"  Designs to generate: {args.n_designs}")
    print(f"  Pareto samples to return: {args.n_pareto}")
    print(f"  Device: {args.device}")
    print(f"  Flow steps: {args.flow_steps}")
    print()

    # Import here so CLI works even if some deps missing
    import sys

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from chimera.chimera_v2 import CHIMERAv2, NRPSConstraints

    # Load model
    model = CHIMERAv2.from_pretrained(
        evoformer_ckpt=args.evof_ckpt,
        flow_ckpt=args.flow_ckpt,
        mpnn_ckpt=args.mpnn_ckpt,
    ).to(device)

    # Load source backbone (bacterial NRPS)
    if args.source_pdb:
        print(f"Loading source backbone from {args.source_pdb}")
        # Parse PDB → extract N/CA/C/O → convert to SE(3) frames
        # (Full implementation: use biotite or biopython)
        raise NotImplementedError(
            "PDB parsing not yet implemented. "
            "Provide source_R and source_t tensors directly for now."
        )
    else:
        # Use random initialization (for testing without a PDB)
        print("No source PDB provided — using identity frames (for testing)")
        L = 600  # default A-domain length
        source_R = torch.eye(3).view(1, 1, 3, 3).expand(1, L, -1, -1).to(device)
        source_t = torch.zeros(1, L, 3, device=device)

    # Load or create MSA
    if args.msa_file:
        print(f"Loading MSA from {args.msa_file}")
        # Parse MSA file → tokenize
        raise NotImplementedError("MSA loading: implement based on your file format")
    else:
        print("No MSA file provided — using random tokens (for testing)")
        L, N_seq = 600, 32
        msa_tokens = torch.randint(0, 23, (1, N_seq, L), device=device)

    # Run design
    print(f"\nGenerating {args.n_designs} designs...")
    results = model.design(
        nrps_msa=msa_tokens,
        source_backbone=(source_R, source_t),
        initial_pair_features=torch.zeros(1, L, L, 32, device=device),
        target_substrate=args.substrate,
        n_designs=args.n_designs,
        n_pareto_samples=args.n_pareto,
        device=str(device),
    )

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)

    # Save Pareto sequences as FASTA
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

    # Save metadata
    meta_path = os.path.join(args.output_dir, "design_metadata.json")
    meta = {
        "substrate": args.substrate,
        "n_generated": results["total_generated"],
        "pareto_count": results["pareto_count"],
        "n_returned": len(results["pareto_sequences"]),
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
