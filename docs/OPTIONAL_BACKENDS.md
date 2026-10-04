# Optional upstream backends

This document covers optional upstream model boundaries, not CHIMERA
installation. For the supported package installation path, see
[`INSTALL.md`](../INSTALL.md). For pinned assets, licenses, cache identities,
and observed verification results, see
[`pretrained_models.md`](./pretrained_models.md) and
[`chimera/model_dependencies.json`](../chimera/model_dependencies.json).

The local CHIMERA representation, SE(3) flow, and ProteinMPNN-inspired
component are not native OpenFold/AlphaFold, RFdiffusion, or ProteinMPNN
implementations. Optional upstream tools must be invoked through their native
implementation and must not be described as interchangeable checkpoints for
the local models.

## Asset acquisition

Install CHIMERA normally, then use its explicit model-store commands to list,
fetch, inspect, and verify declared assets:

```console
pip install .
chimera models list
chimera models fetch proteinmpnn_v48_020
chimera models inspect proteinmpnn_v48_020
chimera models verify --offline
```

Acquisition is opt-in; installation and `import chimera` do not download
weights. The cache records asset identity and locally calculated integrity
metadata. Where the upstream does not publish a digest, local verification
does not authenticate upstream origin. See `pretrained_models.md` for that
distinction and the exact recorded evidence.

## ProteinMPNN

The local `chimera.proteinmpnn` implementation is a ProteinMPNN-inspired
component, not the upstream model. The optional `ProteinMPNNAdapter` calls the
canonical upstream implementation. Its checkpoint and required upstream
revision are separately declared in the dependency manifest and its adapter
smoke has been verified on CPU; see `pretrained_models.md`.

Provide the upstream checkout through the adapter's `proteinmpnn_repo`
configuration rather than adding a machine-specific checkout path to
`PYTHONPATH`. Keep that checkout at the exact manifest revision. Adapter
availability does not make upstream ProteinMPNN a required CHIMERA runtime
dependency.

## RFdiffusion

`chimera.se3_flow` and `chimera.schrodinger_bridge` implement CHIMERA's own
canonical SE(3) flow/bridge; they are not RFdiffusion. The old
`chimera.flow_matching` path is compatibility-only. The optional
`RFdiffusionCLIAdapter` invokes the upstream inference script in its native
environment. Its checkpoint is declared and safely inspected, but native
inference is unavailable in the verified CPU-only Python 3.12 environment:
the pinned upstream stack requires an older CUDA/DGL runtime. Run it only in a
separately provisioned, compatible upstream environment and do not load its
checkpoint into CHIMERA's local flow model.

## OpenFold and AlphaFold

The canonical MSA representation is a native PyTorch, AlphaFold-2-style
EvoFormer stack. It is not the full AlphaFold/OpenFold system, has no
pretrained AlphaFold/OpenFold weights, and is not checkpoint-compatible with
those models. AlphaFold's parameter archive is not a standalone EvoFormer
checkpoint. An `OpenFoldCLIAdapter` boundary can call a separately supplied
runner script and checkpoint, but this repository does not provide or validate
that runner, checkpoint, database setup, or a native OpenFold inference
environment. The direct `OpenFoldAdapter` is explicitly unwired.

## ESM-2 and ESMFold

ESM-2 is an optional dependency for the downstream codon-optimization
workflow, not the canonical structural design path. Install its declared
optional package extra only when using that workflow; acquire its asset
explicitly through the model store. The recorded CPU smoke and asset details
are in `pretrained_models.md`.

ESMFold is listed as an optional upstream asset, but no CHIMERA ESMFold
adapter is implemented. Do not treat it as part of canonical inference or as
a validated CHIMERA structure backend.

## Reproducibility boundary

Record the upstream repository revision, artifact identity, local digest,
runtime versions, and adapter smoke result for any optional backend that is
used. A pinned source revision and locally calculated digest are provenance
metadata; they are not evidence of biological validity or a published
upstream checksum.
