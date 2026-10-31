# Canonical training checkpoints and checkpoint-driven inference

## Checkpoint contract

New `CanonicalTrainer.save_checkpoint()` artifacts use:

- `artifact_type = canonical_training_checkpoint`
- `checkpoint_schema_version = 2`
- manifest `format_version = 5`
- complete `architecture_config` from `CanonicalCHIMERAv2.model_configuration()`
- architecture/configuration SHA-256 in the manifest
- model state schema SHA-256 and model-state content SHA-256
- optimizer and optional scheduler state
- training regime, epoch, global step, component status and objective status
- Python, NumPy, PyTorch, CUDA and explicit-generator RNG states
- dataset, preprocessing, source-tree and runtime provenance

Checkpoint files are written to a temporary file in the destination directory, flushed, and atomically replaced. A checkpoint is only considered complete after the replacement succeeds.

Formats 1 through 4 remain supported by the existing caller-instantiated loading paths. Automatic architecture reconstruction requires the new v5/schema-v2 artifact because older artifacts did not persist the complete constructor configuration. A v4 training resume can be explicitly migrated with `migration_id="v4-to-v5-training-checkpoint"` when all other manifest fields match.

## Canonical training runner

The runner consumes JSONL where each record is a JSON object whose keys match `CanonicalTrainingBatch`. Tensor fields are represented as JSON arrays, including batch dimensions. It does not synthesize missing features, labels, source frames, constraints, or structures.

A fresh run requires a JSON file containing the exact `model_configuration()` mapping:

```bash
python scripts/train_canonical.py \
  --training-jsonl data/prepared/canonical_batches.jsonl \
  --architecture-json configs/canonical_architecture.json \
  --regime sequence \
  --dataset-manifest data/prepared/manifest.json \
  --dataset-version dataset-v1 \
  --preprocessing-hash <preprocessing-sha256> \
  --checkpoint runs/checkpoints/sequence-stage.pt \
  --epochs 2 \
  --checkpoint-every-steps 100 \
  --device cuda
```

The dataset file bytes are hashed by the runner. The manifest/version and preprocessing hash must describe the actual records and transformations. Use a distinct checkpoint path per experiment. The default regime is sequence; choose the appropriate regime and provide the batch fields required by that regime.

Resume from the most recent complete checkpoint:

```bash
python scripts/train_canonical.py \
  --training-jsonl data/prepared/canonical_batches.jsonl \
  --dataset-manifest data/prepared/manifest.json \
  --dataset-version dataset-v1 \
  --preprocessing-hash <preprocessing-sha256> \
  --checkpoint runs/checkpoints/sequence-stage.pt \
  --resume runs/checkpoints/sequence-stage.pt \
  --regime sequence \
  --epochs 1 \
  --device cuda
```

The runner verifies the dataset bytes, dataset version and preprocessing identity before restoring the trainer. It writes periodic and end-of-epoch checkpoints. An interruption can therefore resume from the last successfully replaced checkpoint. Training does not imply held-out validation: components remain unvalidated until the existing validation workflow records that evidence.

## Reconstruct and run inference

The public reconstruction helper rebuilds the architecture from the checkpoint without caller-supplied model dimensions:

```python
from chimera.checkpoint_io import load_canonical_model_from_checkpoint

model = load_canonical_model_from_checkpoint(
    "runs/checkpoints/sequence-stage.pt",
    map_location="cpu",
)
```

For direct candidate generation, prepare an `InferenceRequest` using validated MSA, pair features, source frames and a matching `InferenceConfig`, then call:

```python
from chimera.inference import run_inference_from_checkpoint

result = run_inference_from_checkpoint(
    "runs/checkpoints/sequence-stage.pt",
    request,
)
result.write_json("runs/results/inference-result.json")
```

The inference loader independently verifies checkpoint compatibility and weight-content identity. Candidate artifacts retain their computational validation and provenance. They do not establish biological activity, protein stability, expression, substrate specificity, or experimental validation.
