"""Train the PSC codon optimizer with optional live telemetry visualization."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import random
import time
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from chimera.codon_optimizer import (
    CodonOptimizer,
    codon_optimizer_loss,
    AA_VOCAB,
    PAD_TOKEN,
    ALL_CODONS,
    CODON_TABLE,
)
from data.codon_dataset import CodonJSONLDataset
from codon_observatory import CodonObservatory, TrainingTelemetry, read_system_stats


def collate_records(batch: list[dict]) -> dict[str, torch.Tensor | list[str]]:
    protein_lengths = torch.tensor(
        [item["protein_tokens"].shape[0] for item in batch], dtype=torch.long
    )
    codon_lengths = torch.tensor(
        [item["codon_tokens"].shape[0] for item in batch], dtype=torch.long
    )
    max_protein_len = int(protein_lengths.max())
    max_codon_len = int(codon_lengths.max())
    protein_pad_token = len(AA_VOCAB)
    protein_tokens = torch.full(
        (len(batch), max_protein_len), protein_pad_token, dtype=torch.long
    )
    codon_tokens = torch.full(
        (len(batch), max_codon_len), PAD_TOKEN, dtype=torch.long
    )
    protein_padding_mask = torch.ones(
        (len(batch), max_protein_len), dtype=torch.bool
    )
    codon_padding_mask = torch.ones(
        (len(batch), max_codon_len), dtype=torch.bool
    )
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


def _record_key(item: dict) -> str:
    """Stable grouping key so identical proteins never cross train/validation."""
    sequence = str(item["aa_sequence"][0]).strip().upper()
    return hashlib.sha256(sequence.encode("utf-8")).hexdigest()


def split_dataset(dataset, val_fraction: float, seed: int):
    """Split by protein identity rather than individual rows to prevent leakage."""
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError("val-fraction must be in [0, 1).")

    groups: dict[str, list[int]] = {}
    for index in range(len(dataset)):
        groups.setdefault(_record_key(dataset[index]), []).append(index)

    keys = sorted(groups)
    rng = random.Random(seed)
    rng.shuffle(keys)

    if len(keys) < 2 or val_fraction == 0.0:
        return Subset(dataset, list(range(len(dataset)))), None

    n_val = max(1, int(round(len(keys) * val_fraction)))
    n_val = min(n_val, len(keys) - 1)
    val_keys = set(keys[:n_val])

    train_indices = [i for key in keys if key not in val_keys for i in groups[key]]
    val_indices = [i for key in keys if key in val_keys for i in groups[key]]

    if not train_indices or not val_indices:
        raise RuntimeError("Dataset split produced an empty train or validation set.")

    return Subset(dataset, train_indices), Subset(dataset, val_indices)


def _grad_norm(model: torch.nn.Module) -> float:
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            value = float(p.grad.detach().norm(2).item())
            total += value * value
    return total ** 0.5


def _named_grad_norms(model: torch.nn.Module) -> dict[str, float]:
    """Compact layer-level gradient telemetry for diagnosing stalled adaptation."""
    groups = {
        "esm": [],
        "esm_projection": [],
        "decoder": [],
        "codon_head": [],
        "expression_predictor": [],
    }
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        value = float(parameter.grad.detach().norm(2).item())
        if name.startswith("esm_model."):
            groups["esm"].append(value)
        elif name.startswith("esm_projection."):
            groups["esm_projection"].append(value)
        elif name.startswith("decoder."):
            groups["decoder"].append(value)
        elif name.startswith("codon_head."):
            groups["codon_head"].append(value)
        elif name.startswith("expression_predictor."):
            groups["expression_predictor"].append(value)
    return {
        key: float(np.sqrt(np.sum(np.square(values)))) if values else 0.0
        for key, values in groups.items()
    }


def _run_eval(model, loader, device) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = {}
    with torch.no_grad():
        for batch in loader:
            protein_tokens = batch["protein_tokens"].to(device)
            codon_tokens = batch["codon_tokens"].to(device)
            protein_mask = batch["protein_padding_mask"].to(device)
            codon_mask = batch["codon_padding_mask"].to(device)
            expression = batch["expression"].to(device)
            output = model(
                protein_tokens,
                codon_tokens,
                batch["aa_sequence"],
                protein_padding_mask=protein_mask,
                codon_padding_mask=codon_mask,
            )
            _, metrics = codon_optimizer_loss(
                output["logits"],
                codon_tokens,
                output["expression"],
                target_expression=expression,
                codon_padding_mask=codon_mask,
            )
            for name, value in metrics.items():
                totals[name] = totals.get(name, 0.0) + float(value)
    count = max(1, len(loader))
    return {name: value / count for name, value in totals.items()}


def train_epoch(
    model,
    loader,
    optimizer,
    device,
    epoch,
    total_epochs,
    global_step,
    total_steps,
    observatory=None,
):
    model.train()
    totals: dict[str, float] = {}
    for batch in loader:
        batch_start = time.perf_counter()
        protein_tokens = batch["protein_tokens"].to(device)
        codon_tokens = batch["codon_tokens"].to(device)
        protein_mask = batch["protein_padding_mask"].to(device)
        codon_mask = batch["codon_padding_mask"].to(device)
        expression = batch["expression"].to(device)
        capture_now = observatory is not None and (
            global_step % observatory.update_every == 0 or global_step == 0
        )
        model.set_attention_capture(capture_now, final_layer_only=True)
        optimizer.zero_grad(set_to_none=True)
        output = model(
            protein_tokens,
            codon_tokens,
            batch["aa_sequence"],
            protein_padding_mask=protein_mask,
            codon_padding_mask=codon_mask,
        )
        loss, metrics = codon_optimizer_loss(
            output["logits"],
            codon_tokens,
            output["expression"],
            target_expression=expression,
            codon_padding_mask=codon_mask,
        )
        loss.backward()
        grad_norm = _grad_norm(model)
        grad_groups = _named_grad_norms(model)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        global_step += 1
        for name, value in metrics.items():
            totals[name] = totals.get(name, 0.0) + float(value)

        if observatory is not None:
            try:
                if capture_now:
                    cpu, ram = read_system_stats()
                    sample_idx = 0
                    valid_len = int((~codon_mask[sample_idx]).sum().item())
                    if valid_len > 0:
                        selected_position = (global_step // observatory.update_every) % valid_len
                        aa = batch["aa_sequence"][sample_idx][selected_position]
                        candidates = CODON_TABLE[aa]
                        logits_at_position = output["logits"][sample_idx, selected_position].detach()
                        probs = torch.softmax(logits_at_position, dim=-1)
                        candidate_ids = [ALL_CODONS.index(codon) for codon in candidates]
                        candidate_probs = [float(probs[idx].item()) for idx in candidate_ids]
                        predicted_ids = output["logits"][sample_idx, :valid_len].argmax(dim=-1).tolist()
                        predicted_codons = [ALL_CODONS[idx] for idx in predicted_ids]
                        attention = model.get_cross_attention()
                        attention_sample = (
                            attention[sample_idx].cpu().numpy()
                            if attention is not None
                            else None
                        )
                        lr = float(optimizer.param_groups[0]["lr"])
                        elapsed = max(time.perf_counter() - batch_start, 1e-9)
                        telemetry = TrainingTelemetry(
                            epoch=epoch,
                            total_epochs=total_epochs,
                            global_step=global_step,
                            total_steps=total_steps,
                            loss=float(loss.detach().item()),
                            ce=float(metrics.get("ce", 0.0)),
                            cai=float(metrics.get("cai", 0.0)),
                            gc=float(metrics.get("gc", 0.0)),
                            motif=float(metrics.get("motif", 0.0)),
                            upa=float(metrics.get("upa", 0.0)),
                            expression_loss=float(metrics.get("expression", 0.0)),
                            gradient_norm=grad_norm,
                            learning_rate=lr,
                            cpu_percent=cpu,
                            memory_gb=ram,
                            batch_time=elapsed,
                            samples_per_second=len(batch["aa_sequence"]) / elapsed,
                            protein_length=len(batch["aa_sequence"][sample_idx]),
                            codon_length=valid_len,
                            predicted_expression=float(output["expression"][sample_idx].detach().item()),
                            target_expression=float(expression[sample_idx].detach().item()),
                            architecture=(
                                f"d_model={model.d_model}, heads={model.n_heads}, "
                                f"decoder_layers={model.n_dec_layers}, dim_ff={model.dim_ff}"
                            ),
                            sample_protein=batch["aa_sequence"][sample_idx],
                            selected_position=selected_position,
                            selected_amino_acid=aa,
                            candidate_codons=candidates,
                            candidate_probabilities=candidate_probs,
                            predicted_codons=predicted_codons,
                            cross_attention=attention_sample,
                        )
                        observatory.update(telemetry)
                    print(
                        "gradients="
                        + ",".join(f"{k}:{v:.3e}" for k, v in grad_groups.items()),
                        flush=True,
                    )
            except Exception as exc:
                print(f"visualizer_warning={type(exc).__name__}: {exc}", flush=True)
            finally:
                model.set_attention_capture(False)

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
    parser.add_argument("--val-fraction", type=float, default=0.1)
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

    train_dataset, val_dataset = split_dataset(dataset, args.val_fraction, args.seed)
    loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_records)
    val_loader = (
        DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_records)
        if val_dataset is not None
        else None
    )

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
    print(f"train_samples  : {len(train_dataset)}", flush=True)
    print(f"val_samples    : {len(val_dataset) if val_dataset is not None else 0}", flush=True)
    print(f"device        : {device}", flush=True)
    print(f"batch_size    : {args.batch_size}", flush=True)
    print(f"epochs        : {args.epochs}", flush=True)
    print(f"learning_rate : {args.lr}", flush=True)
    print(f"val_fraction  : {args.val_fraction}", flush=True)
    print("\nMODEL CONFIG", flush=True)
    for key in ("d_model", "n_heads", "n_dec_layers", "dim_ff"):
        print(f"  {key:<12}= {model_config[key]}", flush=True)
    print(f"  ESM         = {model_config.get('esm_model_path')}", flush=True)

    model = CodonOptimizer(**model_config).to(device)
    total_parameters = sum(p.numel() for p in model.parameters())
    trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    esm_trainable = sum(
        p.numel() for p in model.esm_model.parameters() if p.requires_grad
    ) if getattr(model, "esm_model", None) is not None else 0
    print("\nMODEL PARAMETERS", flush=True)
    print(f"  total       = {total_parameters:,}", flush=True)
    print(f"  trainable   = {trainable_parameters:,}", flush=True)
    print(f"  ESM trainable = {esm_trainable:,}", flush=True)
    if model.esm_model is not None and esm_trainable == 0:
        print("  warning     = ESM is fully frozen; no ESM fine-tuning gradients will be observed", flush=True)

    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=args.lr
    )
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
            observatory = CodonObservatory(update_every=args.visualize_interval)
        except Exception as exc:
            print(f"visualizer_disabled={type(exc).__name__}: {exc}", flush=True)

    total_steps = args.epochs * len(loader)
    global_step = 0
    best_val = float("inf")
    best_state = None
    try:
        for epoch in range(1, args.epochs + 1):
            metrics, global_step = train_epoch(
                model, loader, optimizer, device, epoch, args.epochs,
                global_step, total_steps, observatory
            )
            print(
                f"epoch={epoch} loss={metrics.get('total', 0.0):.6f} "
                f"ce={metrics.get('ce', 0.0):.6f} cai={metrics.get('cai', 0.0):.4f} "
                f"gc={metrics.get('gc', 0.0):.4f} motif={metrics.get('motif', 0.0):.4f} "
                f"expr={metrics.get('expression', 0.0):.4f}",
                flush=True,
            )
            if val_loader is not None:
                val_metrics = _run_eval(model, val_loader, device)
                print(
                    f"validation epoch={epoch} loss={val_metrics.get('total', 0.0):.6f} "
                    f"ce={val_metrics.get('ce', 0.0):.6f} cai={val_metrics.get('cai', 0.0):.4f} "
                    f"gc={val_metrics.get('gc', 0.0):.4f} motif={val_metrics.get('motif', 0.0):.4f} "
                    f"expr={val_metrics.get('expression', 0.0):.4f}",
                    flush=True,
                )
                if val_metrics.get("ce", float("inf")) < best_val:
                    best_val = val_metrics["ce"]
                    best_state = {
                        "model": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                        "epoch": epoch,
                        "validation": val_metrics,
                    }
    finally:
        if observatory is not None:
            observatory.close()

    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": model_config,
        "seed": args.seed,
        "training": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "val_fraction": args.val_fraction,
            "best_validation_ce": best_val if val_loader is not None else None,
        },
    }
    torch.save(payload, args.checkpoint)
    print(f"saved={args.checkpoint}", flush=True)

    if best_state is not None:
        best_path = args.checkpoint.with_name(args.checkpoint.stem + ".best.pt")
        torch.save(
            {
                **payload,
                "model": best_state["model"],
                "best_epoch": best_state["epoch"],
                "validation": best_state["validation"],
            },
            best_path,
        )
        print(f"saved_best={best_path}", flush=True)


class _InMemoryDataset(CodonJSONLDataset):
    def __init__(self, records: list[dict]) -> None:
        self.records = records


if __name__ == "__main__":
    main()
