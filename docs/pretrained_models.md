# Pretrained model setup

PSC-CHIMERA separates upstream model assets from local approximation classes.
This prevents an upstream checkpoint from being loaded into a structurally incompatible local implementation.

## Fast setup

```bash
bash scripts/download_weights.sh ./weights --core
```

This downloads the public assets used by the project ecosystem:

- ESM-2 `esm2_t30_150M_UR50D.pt` for the CodonOptimizer protein encoder.
- ESMFold v1 for optional structure inference/evaluation.
- ProteinMPNN `v_48_020.pt` for the native ProteinMPNN adapter.
- RFdiffusion `Base_ckpt.pt` for the native RFdiffusion adapter.

For the AlphaFold2/OpenFold Evoformer parameter archive:

```bash
bash scripts/download_weights.sh ./weights --full
```

The `--full` profile additionally downloads and extracts the AlphaFold2 v2.3 parameter archive into `weights/alphafold2/params/`.

**Important:** AlphaFold2 has an Evoformer. AlphaFold3 does not use the AlphaFold2 Evoformer; its architecture uses a Pairformer. The AlphaFold2 archive is therefore the relevant asset for the repository's native Evoformer/OpenFold direction. The repository does not silently feed these parameters into `chimera.evoformer.EvoFormer`, because that local class is explicitly an approximation.

## CodonOptimizer

The CodonOptimizer can load the downloaded ESM-2 checkpoint with:

```bash
python scripts/train_codon_optimizer.py \
  --dataset data/fath2011_codon_training.jsonl \
  --esm-model-path ./weights/upstream/esm2_t30_150M_UR50D.pt
```

The ESM project documents `esm2_t30_150M_UR50D` as the 30-layer, 150M-parameter ESM-2 model with 640-dimensional residue representations. The checkpoint is publicly downloadable from Meta's ESM model store.

## Native structural backends

Native ProteinMPNN, RFdiffusion, and AlphaFold2/OpenFold checkpoints must be consumed through adapters built against their matching upstream implementations. A filename match is never treated as architectural compatibility.

## Cache helper

Python users can also invoke:

```bash
python scripts/pretrained_manager.py --asset esm2_t30_150m
python scripts/pretrained_manager.py --asset esmfold_v1
```

The cache location can be overridden with `PSC_CHIMERA_MODEL_CACHE`.
