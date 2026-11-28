# CHIMERA compute-efficiency and benchmarking guide

This guide describes the compute tools introduced by the compute-efficiency branch.
The analytical profiler and synthetic benchmark are systems tools, not evidence of
biological learning or model quality.

## 1. Source-derived compute estimate

By default the tools instantiate `CanonicalCHIMERAv2()` with canonical defaults. Supply `--architecture-json` to profile a custom architecture or `--checkpoint` to reconstruct a saved canonical model.

```bash
python scripts/profile_chimera_compute.py \
  --batch-size 1 --msa-depth 128 --length 600 --dtype fp32
```

The report includes:
- Exact instantiated model parameter count.
- Estimated parameter storage for the selected dtype.
- MSA and pair representation tensor sizes.
- The approximate per-block checkpoint input-state storage, OPM outer-product chunk, triangle-product tensor, and IPA point-difference tensor.
- OPM and outgoing/incoming triangle-multiplication MAC estimates.
- The IPA point-difference and attention-logit intermediate sizes.

The operation estimate deliberately excludes attention, transitions, projections,
the rest of the flow network, backward-pass work, optimizer updates, data movement,
and allocator overhead. It must not be interpreted as total model FLOPs or total
training memory.

## 2. Reproducible synthetic GPU benchmark

Run the same command on each GPU and preserve the JSON report:

```bash
python scripts/benchmark_chimera_training.py \
  --regime representation \
  --batch-size 1 --msa-depth 128 --length 600 \
  --precision fp32 --gradient-checkpointing \
  --warmup-steps 2 --steps 10 \
  --output benchmark-representation.json
```

Repeat for `flow` and `sequence`. Compare FP32/BF16 and checkpointing on/off
as separate runs. The benchmark reports mean/median optimizer-step time,
examples/second, and peak allocated/reserved CUDA memory.

The benchmark uses synthetic inputs. It measures execution cost and numerical
finiteness on a synthetic batch, not convergence, biological validity, or the
quality of generated NRPS designs. Use real, representative batches before
making production claims.

## 3. Canonical training data format and streaming

The canonical runner consumes JSONL where each line is one pre-tensorized
`CanonicalTrainingBatch` record. Tensor fields must include a leading batch
dimension. The loader now streams rows and incrementally hashes the file rather
than loading the entire JSONL dataset into a Python list or a single bytes object.

`--batch-size` means the number of JSONL batch-records concatenated along their
leading batch dimension for one optimizer update. For example, if every record
contains one example, `--batch-size 8` creates a local batch of eight examples.
If every record already contains eight examples, the same setting creates a
local batch of 64. Keep records homogeneous: optional fields, constraint fields,
and metadata must be compatible across records being collated.

Example:

```bash
python scripts/train_canonical.py \
  --training-jsonl data/train.jsonl \
  --default-architecture \
  --regime representation \
  --checkpoint Models/representation.pt \
  --epochs 1 --batch-size 1 \
  --precision fp32 --gradient-checkpointing
```

BF16 is opt-in. The trainer checks CUDA BF16 support and leaves explicit
FP32-sensitive operations in the model's existing implementation. Validate
losses, gradient norms, and held-out metrics against an FP32 baseline before
using BF16 for long runs.

## 4. Multi-GPU replicated data parallelism

The canonical runner supports `torchrun` with manual, bucketed gradient
all-reduction:

```bash
torchrun --standalone --nproc_per_node=4 scripts/train_canonical.py \
  --training-jsonl data/train.jsonl \
  --architecture-json config/chimera_architecture.json \
  --regime representation \
  --checkpoint Models/representation.pt \
  --epochs 1 --batch-size 1 \
  --precision bf16 --gradient-checkpointing
```

This is **replicated data parallelism**, not FSDP or tensor parallelism. Every
GPU holds a complete model replica and its local activations. Multiple GPUs
increase aggregate throughput when communication overhead is justified; they
do not pool VRAM and do not make a model fit if one replica cannot fit on one
GPU. FSDP/model sharding remains a separate engineering task.

For distributed runs, the loader shards records evenly and drops the final
remainder that cannot form equal rank-local batches. Keep all ranks on
structurally compatible batches. Gradient presence is checked across ranks
before bucketed reductions; inconsistent optional inputs fail rather than
silently producing divergent updates.

The current checkpoint contains rank-0 RNG state. A multi-rank restart is
therefore not guaranteed to be bitwise-identical to an uninterrupted run.
Do not claim exact distributed reproducibility until per-rank RNG state and
sampler position are included in checkpoint state.

## 5. Recommended benchmark matrix

For each regime, benchmark progressively:

| MSA depth | Length | Local batch |
|---:|---:|---:|
| 32 | 256 | 1 |
| 64 | 400 | 1 |
| 128 | 400 | 1 |
| 128 | 600 | 1 |
| 128 | 600 | 2 |
| 128 | 600 | 4 |

Record GPU model, driver/CUDA/PyTorch versions, precision, checkpointing mode,
peak VRAM, optimizer-step time, examples/second, loss finiteness and gradient
finiteness. Do not extrapolate to a different architecture configuration,
recycling count, or data pipeline without a new measurement.

## 6. Current scope and remaining work

Implemented here:
- Analytical parameter/activation/selected-operation estimator.
- Synthetic per-regime GPU microbenchmark.
- Streaming JSONL ingestion and configurable collation.
- Incremental dataset hashing.
- BF16 autocast.
- Existing EvoFormer and IPA checkpointing exposed through the runner.
- Replicated multi-GPU training with bucketed gradient reduction.
- Faster per-step gradient auditing.

Still required before a production hardware decision:
- CUDA measurements on actual target GPUs.
- Real-data end-to-end throughput and validation comparisons.
- Per-rank RNG/sampler checkpoint restoration.
- FSDP/model-sharded training if per-GPU memory is insufficient.
- A robust cache/artifact format for reusable frozen EvoFormer representations.
- End-to-end profiling of attention, flow, optimizer, I/O and recycling costs.
