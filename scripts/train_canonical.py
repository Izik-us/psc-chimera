#!/usr/bin/env python3
"""Train one canonical CHIMERA regime from pre-tensorized JSONL batches.

Each JSONL record is a mapping whose keys match CanonicalTrainingBatch fields.
Tensor fields are JSON arrays and are converted to torch tensors. This runner
does not fabricate MSA, pair features, structures, labels, or constraints.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chimera.checkpoint_io import canonical_model_from_configuration
from chimera.training import CanonicalTrainingBatch, CanonicalTrainer, TrainingRegime


_TENSOR_FIELDS = {
    "msa_tokens", "pair_features", "source_R", "source_t", "target_R", "target_t",
    "target_sequence", "msa_padding_mask", "sequence_padding_mask", "substrate_id",
    "substrate_coords", "substrate_types", "residue_index", "residue_mask",
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-jsonl", required=True, type=Path)
    parser.add_argument("--architecture-json", type=Path)
    parser.add_argument("--regime", choices=[item.value for item in TrainingRegime], default="sequence")
    parser.add_argument("--dataset-manifest", required=True)
    parser.add_argument("--dataset-version", required=True)
    parser.add_argument("--preprocessing-hash", required=True)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--checkpoint-every-steps", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser


def _load_records(path: Path) -> list[dict]:
    rows = []
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
            rows.append(row)
    if not rows:
        raise ValueError("training JSONL contains no records")
    return rows


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
    if args.seed < 0:
        raise ValueError("--seed must be non-negative")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    data_path = args.training_jsonl.expanduser().resolve()
    if not data_path.is_file():
        raise FileNotFoundError(data_path)
    records = _load_records(data_path)
    dataset_hash = hashlib.sha256(data_path.read_bytes()).hexdigest()
    _seed(args.seed)

    if args.resume:
        from chimera.checkpoint_io import load_canonical_model_from_checkpoint
        from chimera.inference import _safe_torch_load

        payload = _safe_torch_load(args.resume.expanduser().resolve())
        manifest = payload.get("manifest", {})
        if manifest.get("dataset_hash") != dataset_hash:
            raise ValueError("resume dataset bytes do not match the checkpoint dataset_hash")
        if manifest.get("dataset_version") != args.dataset_version:
            raise ValueError("resume dataset version does not match")
        if manifest.get("preprocessing_hash") != args.preprocessing_hash:
            raise ValueError("resume preprocessing hash does not match")
        model = load_canonical_model_from_checkpoint(args.resume)
        model.to(device)
        trainer = CanonicalTrainer.from_checkpoint(
            model,
            args.resume,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
        )
        if trainer.regime.value != args.regime:
            raise ValueError("requested regime does not match resumed checkpoint")
    else:
        if args.architecture_json is None:
            raise ValueError("--architecture-json is required unless --resume is supplied")
        architecture = json.loads(args.architecture_json.expanduser().read_text(encoding="utf-8"))
        if not isinstance(architecture, dict):
            raise ValueError("architecture JSON must contain a model_configuration object")
        model = canonical_model_from_configuration(architecture)
        model.to(device)
        trainer = CanonicalTrainer(
            model,
            args.regime,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            dataset_manifest=args.dataset_manifest,
            dataset_path=data_path,
            dataset_version=args.dataset_version,
            preprocessing_hash=args.preprocessing_hash,
            random_seed=args.seed,
        )

    args.checkpoint.expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
    for _ in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        for record in records:
            result = trainer.train_step(_batch(record).to(device))
            epoch_loss += float(result["loss"])
            if trainer.global_step % args.checkpoint_every_steps == 0:
                trainer.save_checkpoint(args.checkpoint)
        trainer.epoch += 1
        trainer.save_checkpoint(args.checkpoint)
        print(json.dumps({
            "epoch": trainer.epoch,
            "global_step": trainer.global_step,
            "mean_loss": epoch_loss / len(records),
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
