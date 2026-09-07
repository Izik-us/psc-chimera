"""Train the PSC codon optimizer with optional live telemetry visualization."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import time
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from chimera.codon_optimizer import CodonOptimizer, codon_optimizer_loss, AA_VOCAB, PAD_TOKEN, ALL_CODONS, CODON_TABLE
from data.codon_dataset import CodonJSONLDataset
from codon_observatory import CodonObservatory, TrainingTelemetry, read_system_stats

def collate_records(batch: list[dict]) -> dict[str, torch.Tensor | list[str]]:
    protein_lengths = torch.tensor([item["protein_tokens"].shape[0] for item in batch], dtype=torch.long)
    codon_lengths = torch.tensor([item["codon_tokens"].shape[0] for item in batch], dtype=torch.long)
    max_protein_len = int(protein_lengths.max())
    max_codon_len = int(codon_lengths.max())
    protein_pad_token = len(AA_VOCAB)
    protein_tokens = torch.full((len(batch), max_protein_len), protein_pad_token, dtype=torch.long)
    codon_tokens = torch.full((len(batch), max_codon_len), PAD_TOKEN, dtype=torch.long)
    protein_padding_mask = torch.ones((len(batch), max_protein_len), dtype=torch.bool)
    codon_padding_mask = torch.ones((len(batch), max_codon_len), dtype=torch.bool)
    for i, item in enumerate(batch):
        p, c = item["protein_tokens"], item["codon_tokens"]
        protein_tokens[i, :p.shape[0]] = p
        codon_tokens[i, :c.shape[0]] = c
        protein_padding_mask[i, :p.shape[0]] = False
        codon_padding_mask[i, :c.shape[0]] = False
    return {
        "protein_tokens": protein_tokens,
        "codon_tokens": codon_tokens,
        "protein_padding_mask": protein_padding_mask,
        "codon_padding_mask": codon_padding_mask,
        "aa_sequence": [item["aa_sequence"][0] for item in batch],
        "expression": torch.stack([item["expression"] for item in batch]),
    }


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


DEFAULT_MODEL_CONFIG = {"d_model": 72, "n_heads": 4, "n_dec_layers": 2, "dim_ff": 288}


def validate_model_config(config: dict[str, Any]) -> None:
    for key in ("d_model", "n_heads", "n_dec_layers", "dim_ff"):
        if int(config[key]) <= 0:
            raise ValueError(f"{key} must be positive")
    if int(config["d_model"]) % int(config["n_heads"]) != 0:
        raise ValueError("d_model must be divisible by n_heads")


def build_model_config(args: argparse.Namespace, checkpoint_config: dict[str, Any] | None = None) -> dict[str, Any]:
    config = dict(DEFAULT_MODEL_CONFIG)
    if checkpoint_config:
        config.update({k: checkpoint_config[k] for k in DEFAULT_MODEL_CONFIG if k in checkpoint_config})
    for key in ("d_model", "n_heads", "n_dec_layers", "dim_ff"):
        value = getattr(args, key)
        if value is not None:
            config[key] = value
    if args.esm_model_path is not None:
        config["esm_model_path"] = str(args.esm_model_path)
    elif checkpoint_config is not None:
        config["esm_model_path"] = checkpoint_config.get("esm_model_path")
    else:
        config["esm_model_path"] = None
    validate_model_config(config)
    return config


def _grad_norm(model: torch.nn.Module) -> float:
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            value = float(p.grad.detach().norm(2).item())
            total += value * value
    return total ** 0.5


def train_epoch(model, loader, optimizer, device, epoch, total_epochs, global_step, total_steps, observatory=None):
    model.train()
    totals: dict[str, float] = {}
    for batch in loader:
        batch_start = time.perf_counter()
        protein_tokens = batch["protein_tokens"].to(device)
        codon_tokens = batch["codon_tokens"].to(device)
        protein_mask = batch["protein_padding_mask"].to(device)
        codon_mask = batch["codon_padding_mask"].to(device)
        expression = batch["expression"].to(device)
        output = model(protein_tokens, codon_tokens, batch["aa_sequence"], protein_padding_mask=protein_mask, codon_padding_mask=codon_mask)
        loss, metrics = codon_optimizer_loss(output["logits"], codon_tokens, output["expression"], target_expression=expression, codon_padding_mask=codon_mask)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = _grad_norm(model)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        global_step += 1
        for name, value in metrics.items():
            totals[name] = totals.get(name, 0.0) + float(value)

        if observatory is not None:
            try:
                from codon_observatory import TrainingTelemetry, read_system_stats
                cpu, ram = read_system_stats()
                sample_idx = 0
                valid_len = int((~codon_mask[sample_idx]).sum().item())
                pos = max(0, valid_len - 1)
                probs = torch.softmax(output["logits"][sample_idx, pos].detach(), dim=-1)
                aa = batch["aa_sequence"][sample_idx][pos]
                candidates = CODON_TABLE[aa]
                candidate_ids = [ALL_CODONS.index(c) for c in candidates]
                candidate_probs = [float(probs[i].item()) for i in candidate_ids]
                lr = float(optimizer.param_groups[0]["lr"])
                elapsed = max(time.perf_counter() - batch_start, 1e-9)
                telemetry = TrainingTelemetry(
                    epoch=epoch, total_epochs=total_epochs, global_step=global_step, total_steps=total_steps,
                    loss=float(loss.detach().item()), ce=float(metrics.get("ce", 0.0)), cai=float(metrics.get("cai", 0.0)),
                    gc=float(metrics.get("gc", 0.0)), motif=float(metrics.get("motif", 0.0)), upa=float(metrics.get("upa", 0.0)),
                    expression_loss=float(metrics.get("expression", 0.0)), gradient_norm=grad_norm, learning_rate=lr,
                    cpu_percent=cpu, memory_gb=ram, batch_time=elapsed, samples_per_second=len(batch["aa_sequence"]) / elapsed,
                    protein_length=len(batch["aa_sequence"][sample_idx]), codon_length=valid_len,
                    predicted_expression=float(output["expression"][sample_idx].detach().item()), target_expression=float(expression[sample_idx].detach().item()),
                    architecture=(f"d_model={model.d_model}, heads={model.n_heads}, decoder_layers={model.n_dec_layers}, dim_ff={model.dim_ff}"),
                    sample_protein=batch["aa_sequence"][sample_idx], sample_probabilities=candidate_probs,
                    sample_candidate_codons=candidates,
                )
                observatory.update(telemetry)
            except Exception as exc:
                print(f"visualizer_warning={type(exc).__name__}: {exc}", flush=True)
    count = max(1, len(loader))
    return {name: value / count for name, value in totals.items()}, global_step


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
    parser.add_argument("--d-model", type=int, default=None)
    parser.add_argument("--n-heads", type=int, default=None)
    parser.add_argument("--n-dec-layers", type=int, default=None)
    parser.add_argument("--dim-ff", type=int, default=None)
    parser.add_argument("--visualize", action="store_true", help="Enable live matplotlib training observatory.")
    parser.add_argument("--visualize-interval", type=int, default=5, help="Render dashboard every N optimizer steps.")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.epochs <= 0 or args.batch_size <= 0 or args.lr <= 0:
        raise ValueError("epochs, batch-size and lr must be positive")
    if not args.smoke and args.dataset is None:
        raise ValueError("--dataset is required unless --smoke is used")
    set_seed(args.seed)
    device = torch.device(args.device)

    if args.smoke:
        records = [
            {"aa_sequence": "MTEYKLVV", "codon_sequence": "ATGACCGAGTATAAGCTGGTGGTT", "expression": 0.5},
            {"aa_sequence": "MTEYKLVV", "codon_sequence": "ATGACCGAGTATAAGCTGGTGGTT", "expression": 0.7},
        ]
        dataset = CodonJSONLDataset.from_records(records) if hasattr(CodonJSONLDataset, "from_records") else _InMemoryDataset(records)
    else:
        dataset = CodonJSONLDataset(args.dataset)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_records)

    checkpoint_state = None
    checkpoint_config = None
    if args.resume:
        if not args.checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
        checkpoint_state = torch.load(args.checkpoint, map_location=device, weights_only=False)
        checkpoint_config = checkpoint_state.get("config")
        if checkpoint_config is None:
            raise ValueError("Checkpoint does not contain saved model config")

    model_config = build_model_config(args, checkpoint_config)
    print("\n" + "=" * 72 + "\nPSC CODON OPTIMIZER TRAINING\n" + "=" * 72, flush=True)
    print(f"dataset       : {args.dataset if args.dataset else 'SMOKE'}", flush=True)
    print(f"device        : {device}", flush=True)
    print(f"batch_size    : {args.batch_size}", flush=True)
    print(f"epochs        : {args.epochs}", flush=True)
    print(f"learning_rate : {args.lr}", flush=True)
    print("\nMODEL CONFIG", flush=True)
    for key in ("d_model", "n_heads", "n_dec_layers", "dim_ff"):
        print(f"  {key:<12}= {model_config[key]}", flush=True)
    print(f"  ESM         = {model_config.get('esm_model_path')}", flush=True)

    model = CodonOptimizer(**model_config).to(device)
    total_parameters = sum(p.numel() for p in model.parameters())
    trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("\nMODEL PARAMETERS", flush=True)
    print(f"  total       = {total_parameters:,}", flush=True)
    print(f"  trainable   = {trainable_parameters:,}", flush=True)

    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr)
    if args.resume:
        model.load_state_dict(checkpoint_state["model"], strict=True)
        try:
            optimizer.load_state_dict(checkpoint_state["optimizer"])
        except ValueError:
            print("optimizer_state=reset (checkpoint parameter groups differ)", flush=True)
        print(f"resumed={args.checkpoint}", flush=True)

    observatory = None
    if args.visualize:
        try:
            from codon_observatory import CodonObservatory
            observatory = CodonObservatory(update_every=args.visualize_interval)
        except Exception as exc:
            print(f"visualizer_disabled={type(exc).__name__}: {exc}", flush=True)

    total_steps = args.epochs * len(loader)
    global_step = 0
    try:
        for epoch in range(1, args.epochs + 1):
            metrics, global_step = train_epoch(model, loader, optimizer, device, epoch, args.epochs, global_step, total_steps, observatory)
            print(
                f"epoch={epoch} loss={metrics.get('total', 0.0):.6f} ce={metrics.get('ce', 0.0):.6f} "
                f"cai={metrics.get('cai', 0.0):.4f} gc={metrics.get('gc', 0.0):.4f} "
                f"motif={metrics.get('motif', 0.0):.4f} expr={metrics.get('expression', 0.0):.4f}", flush=True)
    finally:
        if observatory is not None:
            observatory.close()

    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "config": model_config, "seed": args.seed,
                "training": {"epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr}}, args.checkpoint)
    print(f"saved={args.checkpoint}", flush=True)


class _InMemoryDataset(CodonJSONLDataset):
    def __init__(self, records: list[dict]) -> None:
        self.records = records


if __name__ == "__main__":
    main()
