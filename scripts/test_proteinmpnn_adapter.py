"""Smoke-test the native ProteinMPNN adapter with a synthetic backbone.

This verifies checkpoint loading and upstream sampling only. The generated
coordinates are random and must not be interpreted as a biological design.

Example:
    python scripts/test_proteinmpnn_adapter.py
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_ROOT = REPO_ROOT.parent / "external" / "ProteinMPNN"
DEFAULT_CHECKPOINT = EXTERNAL_ROOT / "vanilla_model_weights" / "v_48_020.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="Native ProteinMPNN checkpoint path",
    )
    parser.add_argument(
        "--pdb",
        type=Path,
        help="Optional PDB containing complete N/CA/C/O atoms",
    )
    parser.add_argument("--length", type=int, default=12)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.length < 2:
        raise ValueError("--length must be at least 2")
    if args.pdb and not args.pdb.is_file():
        raise SystemExit(
            f"PDB file not found: {args.pdb}\n"
            "Pass the path to an existing .pdb file, or omit --pdb for the "
            "synthetic smoke test."
        )

    sys.path.insert(0, str(REPO_ROOT))
    from chimera import ProteinMPNNAdapter

    torch.manual_seed(args.seed)
    if args.pdb:
        from chimera.structure_utils import load_backbone_coords_pdb

        backbone_coords = load_backbone_coords_pdb(args.pdb)
    else:
        backbone_coords = torch.randn(1, args.length, 4, 3)
    adapter = ProteinMPNNAdapter(
        checkpoint=args.checkpoint,
        device=args.device,
        temperature=args.temperature,
    )
    sequences, log_probs = adapter.design(backbone_coords)

    alphabet = "ACDEFGHIKLMNPQRSTVWYX"
    sequence_text = "".join(alphabet[token] for token in sequences[0].tolist())
    print("Native ProteinMPNN adapter smoke test")
    print(f"  checkpoint: {args.checkpoint}")
    print(f"  backbone shape: {tuple(backbone_coords.shape)}")
    print(f"  sequences shape: {tuple(sequences.shape)}")
    print(f"  log_probs shape: {tuple(log_probs.shape)}")
    print(f"  sampled sequence: {sequence_text}")
    print(f"  log_probs finite: {bool(torch.isfinite(log_probs).all())}")
    print("  note: random coordinates are for integration testing only")


if __name__ == "__main__":
    main()
