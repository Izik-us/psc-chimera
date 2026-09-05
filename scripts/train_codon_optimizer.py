"""Train the codon optimizer on a prepared JSONL dataset.

Each record must contain:
    {"aa_sequence": "MTEYK...", "codon_sequence": "ATG...", "expression": 0.73}

This script does not invent biological labels. Use ``--smoke`` only to verify
that the training loop and checkpointing work; its synthetic data is not useful
for model training.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from chimera.codon_optimizer import (
    CodonOptimizer,
    codon_optimizer_loss,
)
from data.codon_dataset import CodonJSONLDataset


def collate_records(batch: list[dict]) -> dict[str, torch.Tensor | list[str]]:
    protein_lengths = {item["protein_tokens"].shape[0] for item in batch}
    if len(protein_lengths) != 1:
        raise ValueError(
            "Mixed-length batches are unsupported; use --batch-size 1 "
            "or group records by sequence length"
        )
    return {
        "protein_tokens": torch.stack([item["protein_tokens"] for item in batch]),
        "codon_tokens": torch.stack([item["codon_tokens"] for item in batch]),
        "aa_sequence": [item["aa_sequence"][0] for item in batch],
        "expression": torch.stack([item["expression"] for item in batch]),
    }


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_epoch(model, loader, optimizer, device) -> dict[str, float]:
    model.train()
    totals: dict[str, float] = {}
    for batch in loader:
        protein_tokens = batch["protein_tokens"].to(device)
        codon_tokens = batch["codon_tokens"].to(device)
        expression = batch["expression"].to(device)
        output = model(
            protein_tokens,
            codon_tokens,
            batch["aa_sequence"],
        )
        loss, metrics = codon_optimizer_loss(
            output["logits"],
            codon_tokens,
            output["expression"],
            target_expression=expression,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        for name, value in metrics.items():
            totals[name] = totals.get(name, 0.0) + value
    count = max(1, len(loader))
    return {name: value / count for name, value in totals.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/codon_optimizer.pt"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--esm-model-path", type=Path)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.epochs <= 0 or args.batch_size <= 0:
        raise ValueError("epochs and batch-size must be positive")
    if not args.smoke and args.dataset is None:
        raise ValueError("--dataset is required unless --smoke is used")

    set_seed(args.seed)
    device = torch.device(args.device)
    if args.smoke:
        records = [
            {"aa_sequence": "MTEYKLVV", "codon_sequence": "ATGACCGAGTATAAGCTGGTGGTT", "expression": 0.5},
            {"aa_sequence": "MTEYKLVV", "codon_sequence": "ATGACCGAGTATAAGCTGGTGGTT", "expression": 0.7},
        ]
        dataset = CodonJSONLDataset.from_records(records) if hasattr(CodonJSONLDataset, "from_records") else None
        if dataset is None:
            dataset = _InMemoryDataset(records)
    else:
        dataset = CodonJSONLDataset(args.dataset)

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_records)
    model_config = {
        "d_model": 64,
        "n_heads": 4,
        "n_dec_layers": 2,
        "dim_ff": 256,
        "esm_model_path": str(args.esm_model_path) if args.esm_model_path else None,
    }
    model = CodonOptimizer(**model_config).to(device)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.lr,
    )
    if args.resume:
        if not args.checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
        state = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(state["model"])
        try:
            optimizer.load_state_dict(state["optimizer"])
        except ValueError:
            print(
                "optimizer_state=reset (checkpoint parameter groups differ)",
                flush=True,
            )
        print(f"resumed={args.checkpoint}", flush=True)
    for epoch in range(1, args.epochs + 1):
        metrics = train_epoch(model, loader, optimizer, device)
        print(
            f"epoch={epoch} loss={metrics['total']:.6f} "
            f"ce={metrics['ce']:.6f} cai={metrics['cai']:.4f} "
            f"gc={metrics['gc']:.4f} motif={metrics['motif']:.4f} "
            f"expr={metrics['expression']:.4f}",
            flush=True,
        )

    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": model_config,
            "seed": args.seed,
        },
        args.checkpoint,
    )
    print(f"saved={args.checkpoint}", flush=True)


class _InMemoryDataset(CodonJSONLDataset):
    def __init__(self, records: list[dict]) -> None:
        self.records = records


if __name__ == "__main__":
    main()
