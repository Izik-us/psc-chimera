#!/usr/bin/env python3
"""Reproducible single-GPU CHIMERA regime microbenchmark.

Uses synthetic tensors only for systems profiling. It does not measure biological
learning or validate dataset quality. Run with the same architecture JSON,
precision, dimensions and regime on each target GPU.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chimera.architecture import CanonicalCHIMERAv2
from chimera.checkpoint_io import (
    canonical_model_from_configuration,
    load_canonical_model_from_checkpoint,
)
from chimera.lie import so3_exp
from chimera.training import CanonicalTrainingBatch, CanonicalTrainer, TrainingRegime


def make_batch(model, regime: TrainingRegime, batch: int, depth: int, length: int,
               device: torch.device) -> CanonicalTrainingBatch:
    msa = torch.randint(0, 20, (batch, depth, length), device=device)
    pair = torch.zeros(batch, length, length, model.d_evo_pair, device=device)
    eye = torch.eye(3, device=device).expand(batch, length, 3, 3).clone()
    source_t = torch.randn(batch, length, 3, device=device)
    source_R = eye
    values = {
        "msa_tokens": msa,
        "pair_features": pair,
        "residue_mask": torch.ones(batch, length, dtype=torch.bool, device=device),
    }
    if regime in (TrainingRegime.FLOW, TrainingRegime.CONSTRAINT):
        values.update({
            "source_R": source_R,
            "source_t": source_t,
            "target_R": so3_exp(torch.randn(batch, length, 3, device=device) * 0.05),
            "target_t": source_t + torch.randn_like(source_t) * 0.2,
        })
    elif regime == TrainingRegime.SEQUENCE:
        values.update({
            "source_R": source_R,
            "source_t": source_t,
            "target_sequence": torch.randint(0, 20, (batch, length), device=device),
        })
    else:
        values.update({
            "msa_padding_mask": torch.zeros(batch, depth, length, dtype=torch.bool, device=device),
        })
    return CanonicalTrainingBatch(**values)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--architecture-json", type=Path)
    source.add_argument("--checkpoint", type=Path,
                        help="optional canonical checkpoint; defaults to CanonicalCHIMERAv2()")
    parser.add_argument("--regime", choices=("representation", "flow", "sequence"),
                        default="representation")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--msa-depth", type=int, default=128)
    parser.add_argument("--length", type=int, default=600)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--evoformer-checkpoint-group-size", type=int, default=1)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(args.batch_size, args.msa_depth, args.length, args.steps) < 1:
        parser.error("batch size, MSA depth, length and steps must be positive")
    if args.evoformer_checkpoint_group_size < 1:\n        parser.error("EvoFormer checkpoint group size must be positive")\n    if args.warmup_steps < 0:
        parser.error("warmup steps cannot be negative")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if args.precision == "bf16" and device.type == "cuda" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("selected CUDA device does not support BF16")

    if args.checkpoint is not None:
        model = load_canonical_model_from_checkpoint(args.checkpoint)
    elif args.architecture_json is not None:
        configuration = json.loads(args.architecture_json.read_text(encoding="utf-8"))
        model = canonical_model_from_configuration(configuration)
    else:
        model = CanonicalCHIMERAv2()
    model = model.to(device)
    model.evoformer.gradient_checkpointing = args.gradient_checkpointing
    model.evoformer.gradient_checkpointing_group_size = args.evoformer_checkpoint_group_size
    model.evoformer.configuration["gradient_checkpointing_group_size"] = args.evoformer_checkpoint_group_size
    model.flow_model.set_gradient_checkpointing(args.gradient_checkpointing)
    regime = TrainingRegime(args.regime)
    trainer = CanonicalTrainer(
        model, regime, learning_rate=1e-4, dataset_manifest="synthetic-benchmark-only",
        min_validation_examples=1, precision=args.precision,
    )
    batch = make_batch(model, regime, args.batch_size, args.msa_depth, args.length, device)

    def step():
        return trainer.train_step(batch)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for _ in range(args.warmup_steps):
        step()
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    durations = []
    losses = []
    for _ in range(args.steps):
        started = time.perf_counter()
        result = step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        durations.append(time.perf_counter() - started)
        losses.append(result["loss"])

    report = {
        "status": "measured_synthetic_training_microbenchmark",
        "warning": "Synthetic data measures systems throughput only; it is not evidence of convergence, biological validity or full-dataset training time.",
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "torch_version": torch.__version__,
        "regime": regime.value,
        "batch_size": args.batch_size,
        "msa_depth": args.msa_depth,
        "length": args.length,
        "precision": args.precision,
        "gradient_checkpointing": args.gradient_checkpointing,
        "evoformer_checkpoint_group_size": args.evoformer_checkpoint_group_size,
        "warmup_steps": args.warmup_steps,
        "measured_steps": args.steps,
        "step_seconds_mean": statistics.mean(durations),
        "step_seconds_median": statistics.median(durations),
        "examples_per_second": args.batch_size / statistics.mean(durations),
        "mean_loss": statistics.mean(losses),
        "peak_allocated_GiB": (
            torch.cuda.max_memory_allocated(device) / (1024 ** 3)
            if device.type == "cuda" else None
        ),
        "peak_reserved_GiB": (
            torch.cuda.max_memory_reserved(device) / (1024 ** 3)
            if device.type == "cuda" else None
        ),
    }
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
