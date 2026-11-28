#!/usr/bin/env python3
"""Train one canonical CHIMERA regime from pre-tensorized JSONL batches.

Each JSONL record is a mapping whose keys match CanonicalTrainingBatch fields.
Tensor fields are JSON arrays and are converted to torch tensors. This runner
does not fabricate MSA, pair features, structures, labels, or constraints.
"""
from __future__ import annotations

import argparse
from dataclasses import fields, is_dataclass
import hashlib
import json
import os
from pathlib import Path
import random
import sys

import numpy as np
import torch
import torch.distributed as dist

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chimera.architecture import CanonicalCHIMERAv2
from chimera.checkpoint_io import canonical_model_from_configuration
from chimera.training import CanonicalTrainingBatch, CanonicalTrainer, TrainingRegime


_TENSOR_FIELDS = {
    "msa_tokens", "pair_features", "source_R", "source_t", "target_R", "target_t",
    "target_sequence", "msa_padding_mask", "sequence_padding_mask", "substrate_id",
    "substrate_coords", "substrate_types", "residue_index", "residue_mask",
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-jsonl", "--dataset", dest="training_jsonl", required=True, type=Path)
    parser.add_argument("--architecture-json", type=Path)
    parser.add_argument("--default-architecture", action="store_true",
                        help="instantiate CanonicalCHIMERAv2() when no architecture JSON is supplied")
    parser.add_argument(
        "--regime", choices=[item.value for item in TrainingRegime], default="sequence"
    )
    parser.add_argument("--dataset-manifest", default=None)
    parser.add_argument("--dataset-version", default="canonical-training-batch-jsonl-v1")
    parser.add_argument("--preprocessing-hash", default=None)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--epochs", "--epoch", "-epoch", dest="epochs", type=int, default=1)
    parser.add_argument("--checkpoint-every-steps", type=int, default=100)
    parser.add_argument("--learning-rate", "--lr", "-lr", dest="learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="number of JSONL batch-records to concatenate per optimizer update")
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--gradient-checkpointing", action="store_true",
                        help="enable EvoFormer and flow IPA activation checkpointing")
    return parser


def _iter_records(path: Path):
    """Stream JSONL records; never retain the complete dataset in host memory."""
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at line {line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"training record at line {line_number} must be a JSON object")
            yield row


def _count_records(path: Path) -> int:
    with path.open("r", encoding="utf-8") as stream:
        return sum(1 for line in stream if line.strip())


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _concat_values(values, field_name: str):
    if all(value is None for value in values):
        return None
    if any(value is None for value in values):
        raise ValueError(f"cannot batch mixed None/non-None values for {field_name}")
    first = values[0]
    if torch.is_tensor(first):
        if not all(torch.is_tensor(value) and value.ndim == first.ndim for value in values):
            raise ValueError(f"incompatible tensor records for {field_name}")
        return torch.cat(values, dim=0)
    if isinstance(first, dict):
        keys = set(first)
        if any(not isinstance(value, dict) or set(value) != keys for value in values):
            raise ValueError(f"inconsistent mapping keys for {field_name}")
        return {
            key: _concat_values([value[key] for value in values], f"{field_name}.{key}")
            for key in sorted(keys)
        }
    if is_dataclass(first) and not isinstance(first, type):
        if any(type(value) is not type(first) for value in values):
            raise ValueError(f"incompatible dataclass records for {field_name}")
        return type(first)(**{
            item.name: _concat_values(
                [getattr(value, item.name) for value in values],
                f"{field_name}.{item.name}",
            )
            for item in fields(first)
        })
    if all(value == first for value in values):
        return first
    raise ValueError(
        f"field {field_name} differs between records and cannot be represented "
        "as one batched CanonicalTrainingBatch; group homogeneous records"
    )


def _collate_batches(batches: list[CanonicalTrainingBatch]) -> CanonicalTrainingBatch:
    if not batches:
        raise ValueError("cannot collate an empty batch")
    return CanonicalTrainingBatch(**{
        item.name: _concat_values(
            [getattr(batch, item.name) for batch in batches], item.name
        )
        for item in fields(CanonicalTrainingBatch)
    })


def _iter_batches(path: Path, batch_size: int, *, rank: int = 0, world_size: int = 1):
    """Yield streamed batches, sharding records evenly across distributed ranks.

    Each JSONL row is a pre-tensorized CanonicalTrainingBatch. --batch-size
    concatenates that many such rows along their leading batch dimension.
    """
    total = _count_records(path)
    if total == 0:
        raise ValueError("training JSONL contains no records")
    usable = total if world_size == 1 else (total // (world_size * batch_size)) * world_size * batch_size
    pending = []
    selected = 0
    for index, record in enumerate(_iter_records(path)):
        if index >= usable:
            break
        if index % world_size != rank:
            continue
        pending.append(_batch(record))
        selected += 1
        if len(pending) == batch_size:
            yield pending[0] if batch_size == 1 else _collate_batches(pending)
            pending = []
    if pending:
        if world_size > 1:
            raise RuntimeError("distributed sharding produced an incomplete rank-local batch")
        yield pending[0] if len(pending) == 1 else _collate_batches(pending)


def _batch(record: dict) -> CanonicalTrainingBatch:
    allowed = set(CanonicalTrainingBatch.__dataclass_fields__)
    unknown = set(record) - allowed
    if unknown:
        raise ValueError(f"unknown CanonicalTrainingBatch fields: {sorted(unknown)}")
    values = dict(record)
    for name in _TENSOR_FIELDS:
        if name in values and values[name] is not None:
            values[name] = torch.as_tensor(values[name])
    for name in ("objective_labels", "objective_features"):
        if values.get(name) is not None:
            values[name] = {
                key: torch.as_tensor(value) for key, value in values[name].items()
            }
    if values.get("constraints") is not None:
        from chimera.domain_schema import NRPSConstraints

        constraints = values["constraints"]
        if not isinstance(constraints, dict):
            raise ValueError("constraints must be a JSON object")
        constraint_values = dict(constraints)
        for name, value in tuple(constraint_values.items()):
            if name in {
                "fixed_mask", "stachelhaus_positions", "domain_boundaries",
                "module_boundaries", "icosahedral_face", "hotspot_coords",
                "hotspot_indices", "fixed_sequence", "domain_types",
            } and value is not None:
                constraint_values[name] = torch.as_tensor(value)
        values["constraints"] = NRPSConstraints(**constraint_values)
    if values.get("preference_batch") is not None:
        from chimera.dpo import DPOBatch

        preference = values["preference_batch"]
        if not isinstance(preference, dict):
            raise ValueError("preference_batch must be a JSON object")
        values["preference_batch"] = DPOBatch(**{
            name: torch.as_tensor(value) if value is not None else None
            for name, value in preference.items()
        })
    if "objective_label_sources" in values and values["objective_label_sources"] is not None:
        if not isinstance(values["objective_label_sources"], dict):
            raise ValueError("objective_label_sources must be a JSON object")
    if "objective_label_kinds" in values and values["objective_label_kinds"] is not None:
        if not isinstance(values["objective_label_kinds"], dict):
            raise ValueError("objective_label_kinds must be a JSON object")
    return CanonicalTrainingBatch(**values)


def _dataset_manifest_identity(value: str) -> str:
    candidate = Path(value).expanduser()
    if candidate.is_file():
        document = json.loads(candidate.read_text(encoding="utf-8"))
        return json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False)
    if not value.strip():
        raise ValueError("--dataset-manifest must be a non-empty identity or JSON manifest path")
    return value.strip()

def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.epochs < 1 or args.checkpoint_every_steps < 1:
        raise ValueError("--epochs and --checkpoint-every-steps must be positive")
    if args.seed < 0 or args.batch_size < 1:
        raise ValueError("--seed must be non-negative and --batch-size must be positive")
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    if world_size > 1:
        if not dist.is_available():
            raise RuntimeError("torch.distributed is unavailable in this PyTorch build")
        if not dist.is_initialized():
            backend = "nccl" if torch.cuda.is_available() and args.device.startswith("cuda") else "gloo"
            dist.init_process_group(backend=backend)
        if args.device.startswith("cuda"):
            torch.cuda.set_device(local_rank)
            args.device = f"cuda:{local_rank}"
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    data_path = args.training_jsonl.expanduser().resolve()
    if not data_path.is_file():
        raise FileNotFoundError(data_path)
    record_count = _count_records(data_path)
    if record_count == 0:
        raise ValueError("training JSONL contains no records")
    dataset_hash = _file_sha256(data_path)
    dataset_manifest_identity = (
        _dataset_manifest_identity(args.dataset_manifest)
        if args.dataset_manifest
        else f"canonical-jsonl:{dataset_hash}"
    )
    preprocessing_hash = args.preprocessing_hash or hashlib.sha256(
        b"psc-chimera:CanonicalTrainingBatch-jsonl:v1"
    ).hexdigest()
    if len(preprocessing_hash) != 64 or any(
        character not in "0123456789abcdef" for character in preprocessing_hash
    ):
        raise ValueError("--preprocessing-hash must be a lowercase SHA-256 digest")
    _seed(args.seed)

    if args.resume:
        from chimera.checkpoint_io import load_canonical_model_from_checkpoint
        from chimera.inference import _safe_torch_load

        payload = _safe_torch_load(args.resume.expanduser().resolve())
        manifest = payload.get("manifest", {})
        model = load_canonical_model_from_checkpoint(args.resume)
        model.to(device)
        model.evoformer.gradient_checkpointing = args.gradient_checkpointing
        model.flow_model.set_gradient_checkpointing(args.gradient_checkpointing)
        saved_regime = payload.get("training", {}).get("regime")
        if saved_regime == args.regime:
            if manifest.get("dataset_hash") != dataset_hash:
                raise ValueError("resume dataset bytes do not match the checkpoint dataset_hash")
            if manifest.get("dataset_version") != args.dataset_version:
                raise ValueError("resume dataset version does not match")
            if manifest.get("dataset_manifest") != dataset_manifest_identity:
                raise ValueError("resume dataset manifest identity does not match")
            if manifest.get("preprocessing_hash") != preprocessing_hash:
                raise ValueError("resume preprocessing hash does not match")
            trainer = CanonicalTrainer.from_checkpoint(
                model,
                args.resume,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
            )
            trainer.precision = args.precision
        else:
            previous_trainer = CanonicalTrainer.from_checkpoint(
                model,
                args.resume,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
            )
            previous_trainer.precision = args.precision
            trainer = previous_trainer.transition(
                args.regime,
                dataset_manifest=dataset_manifest_identity,
                dataset_path=data_path,
                dataset_version=args.dataset_version,
                preprocessing_hash=preprocessing_hash,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
            )
    else:
        if args.architecture_json is not None and args.default_architecture:
            raise ValueError("--architecture-json and --default-architecture are mutually exclusive")
        if args.architecture_json is None and not args.default_architecture:
            raise ValueError("supply --architecture-json or explicitly select --default-architecture")
        if args.default_architecture:
            model = CanonicalCHIMERAv2()
        else:
            architecture = json.loads(args.architecture_json.expanduser().read_text(encoding="utf-8"))
            if not isinstance(architecture, dict):
                raise ValueError("architecture JSON must contain a model_configuration object")
            model = canonical_model_from_configuration(architecture)
        model.to(device)
        model.evoformer.gradient_checkpointing = args.gradient_checkpointing
        model.flow_model.set_gradient_checkpointing(args.gradient_checkpointing)
        trainer = CanonicalTrainer(
            model,
            args.regime,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            dataset_manifest=dataset_manifest_identity,
            dataset_path=data_path,
            dataset_version=args.dataset_version,
            preprocessing_hash=preprocessing_hash,
            random_seed=args.seed,
            precision=args.precision,
        )

    args.checkpoint.expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
    # Preserve identical model initialization across ranks, then decorrelate
    # stochastic training operations such as dropout.
    _seed(args.seed + rank)
    try:
        for _ in range(args.epochs):
            model.train()
            epoch_loss = 0.0
            batch_count = 0
            for batch in _iter_batches(
                data_path, args.batch_size, rank=rank, world_size=world_size
            ):
                result = trainer.train_step(batch.to(device))
                epoch_loss += float(result["loss"])
                batch_count += 1
                if trainer.global_step % args.checkpoint_every_steps == 0:
                    if rank == 0:
                        trainer.save_checkpoint(args.checkpoint)
                    if world_size > 1:
                        dist.barrier()
            if batch_count == 0:
                raise ValueError("training dataset produced no complete batches for this rank")
            if world_size > 1:
                epoch_stats = torch.tensor([epoch_loss, float(batch_count)], device=device, dtype=torch.float64)
                dist.all_reduce(epoch_stats, op=dist.ReduceOp.SUM)
                epoch_loss = float(epoch_stats[0].item())
                batch_count = int(epoch_stats[1].item())
            trainer.epoch += 1
            if rank == 0:
                trainer.save_checkpoint(args.checkpoint)
                print(json.dumps({
                    "epoch": trainer.epoch,
                    "global_step": trainer.global_step,
                    "mean_loss": epoch_loss / batch_count,
                    "batches": batch_count,
                    "world_size": world_size,
                    "batch_records": args.batch_size,
                    "precision": trainer.precision,
                    "checkpoint": str(args.checkpoint.resolve()),
                    "checkpoint_sha256": _file_sha256(args.checkpoint),
                }, sort_keys=True), flush=True)
            if world_size > 1:
                dist.barrier()
    finally:
        if world_size > 1 and dist.is_initialized():
            dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
