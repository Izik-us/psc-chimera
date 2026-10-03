# External pretrained models and assets

Model acquisition is an explicit operation. Installation and ordinary
`import chimera` do not download weights. The machine-readable dependency lock
is [`chimera/model_dependencies.json`](../chimera/model_dependencies.json); it
records immutable repository revisions, named variants, official sources,
licenses, intended components, and the absence of upstream-published SHA-256
values where applicable.

## Inventory and current integration boundary

| Dependency | Exact declared upstream artifact | Source / revision | License | CHIMERA status |
| --- | --- | --- | --- | --- |
| ESM-2 | `esm2_t30_150M_UR50D.pt`; 30 layers, about 150M parameters, 640-dimensional representations | [ESM](https://github.com/facebookresearch/esm), `2b369911bb5b4b0dda914521b9475cad1656b2ac`; [checkpoint](https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t30_150M_UR50D.pt) | MIT; [license](https://opensource.org/license/mit) | `NATIVE_VERIFIED` on CPU: real checkpoint passed through `CodonOptimizer.encode_protein`. |
| ESM-2 contact-regression auxiliary | `esm2_t30_150M_UR50D-contact-regression.pt` | Same pinned [ESM source](https://github.com/facebookresearch/esm/tree/2b369911bb5b4b0dda914521b9475cad1656b2ac); [official regression asset](https://dl.fbaipublicfiles.com/fair-esm/regression/esm2_t30_150M_UR50D-contact-regression.pt) | MIT; [license](https://opensource.org/license/mit) | Acquired and loaded with ESM-2 by the canonical `fair-esm` core loader; separate model-store identity. |
| ESMFold | `esmfold_3B_v1.pt` | [ESM](https://github.com/facebookresearch/esm), `2b369911bb5b4b0dda914521b9475cad1656b2ac`; [checkpoint](https://dl.fbaipublicfiles.com/fair-esm/models/esmfold_3B_v1.pt) | MIT; [license](https://opensource.org/license/mit) | Optional upstream asset; no CHIMERA ESMFold adapter identified. |
| ProteinMPNN | Vanilla full-backbone `v_48_020.pt`, 48-neighbor variant | [ProteinMPNN](https://github.com/dauparas/ProteinMPNN), `8907e6671bfbfc92303b5f79c4b5e6ce47cdef57`; [pinned checkpoint](https://github.com/dauparas/ProteinMPNN/raw/8907e6671bfbfc92303b5f79c4b5e6ce47/vanilla_model_weights/v_48_020.pt) | MIT; [license](https://github.com/dauparas/ProteinMPNN/blob/8907e6671bfbfc92303b5f79c4b5e6ce47/LICENSE) | `NATIVE_VERIFIED` on CPU: exact pinned runtime and checkpoint passed through `ProteinMPNNAdapter`. The local ProteinMPNN-inspired model remains a separate approximation. |
| RFdiffusion | `Base_ckpt.pt` | [RFdiffusion](https://github.com/RosettaCommons/RFdiffusion), `86507b6538f51fce57b5a72477165f03999ed7ae`; [upstream checkpoint](http://files.ipd.uw.edu/pub/RFdiffusion/6f5902ac237024bdd0c176cb93063dc4/Base_ckpt.pt) | BSD-3-Clause; [license](https://github.com/RosettaCommons/RFdiffusion/blob/86507b6538f51fce57b5a72477165f03999ed7ae/LICENSE) | Checkpoint acquired and safely inspected; `UNAVAILABLE` for native inference because the pinned runtime requires an unavailable CUDA/DGL stack. The CHIMERA SE(3) bridge is not RFdiffusion. |
| AlphaFold2 parameters | `alphafold_params_2022-12-06.tar`, a multi-model parameter archive | [AlphaFold](https://github.com/google-deepmind/alphafold), `c77e5d2a8961d1a353632c462914ff0a32a950f6`; [official archive](https://storage.googleapis.com/alphafold/alphafold_params_2022-12-06.tar) | CC BY 4.0; [terms](https://github.com/google-deepmind/alphafold#license-and-disclaimer) | Optional archive; not a standalone EvoFormer checkpoint. No wired OpenFold inference adapter. |

These repository revisions identify the upstream code snapshots reviewed for
the declared artifacts. They do not prove that an unmanifested local file came
from those sources. No upstream-published SHA-256 was located for these
checkpoint downloads. A SHA-256 computed during acquisition is consequently
recorded as **locally recorded**, never as an upstream checksum.

## Acquired artifacts and native smoke results

The following results were observed on 2026-10-02 using the default cache
(`~/.cache/psc-chimera/models`). The cache is outside the Git worktree. Each
`INTEGRITY_VERIFIED` result means the local bytes match the cache's self-hashed
manifest; every upstream-published SHA-256 remains absent.

| Artifact | Size | Local SHA-256 | Acquired/runtime evidence |
| --- | ---: | --- | --- |
| ESM-2 `esm2_t30_150M_UR50D.pt` | 592,774,773 bytes | `881c7176cf198ef8dec26a3c375d40eb58d0c33df95c22562ca6cc6d3f812c62` | `INTEGRITY_VERIFIED`; `fair-esm 2.0.0` loaded the 30-layer, 640-dimensional model (~150M parameters). CPU inference through `CodonOptimizer.encode_protein` returned finite `(1, 8, 32)` memory; repeated output agreed within `rtol=1e-5, atol=1e-6`. |
| ESM-2 contact-regression auxiliary | 3,431 bytes | `6a604b96722ed052eef8a094ad90b275ba2e987d406315dbed0bdc6b3c4238a7` | `INTEGRITY_VERIFIED`; loaded with ESM-2 through the canonical upstream `load_model_and_alphabet_core`. |
| ProteinMPNN `v_48_020.pt` | 6,681,301 bytes | `c9cb4a671d79604111231f8dbfc7c590e06f1197453b7a6854ac6661a642f5bd` | `INTEGRITY_VERIFIED`; exact upstream checkout at `8907e6671bfbfc92303b5f79c4b5e6ce47cdef57` loaded strictly. CPU adapter smoke produced `(1, 8)` sequence tokens and finite `(1, 8, 21)` log-probabilities while preserving a fixed residue. |
| RFdiffusion `Base_ckpt.pt` | 483,616,107 bytes | `0fcf7d7c32b4848030aca3a051e6768de194616f96ba6c38186351a33bfc6eca` | `INTEGRITY_VERIFIED`; safe tensor-only inspection found the expected checkpoint payload keys, 5,998 tensors, and 59,808,046 tensor elements. Native inference is `UNAVAILABLE`: pinned `SE3nv` specifies Python 3.9, PyTorch 1.9, CUDA 11.1, and DGL CUDA 11.1; this host has Python 3.12 and CPU-only PyTorch 2.13. |
| AlphaFold2 parameter archive | 5,587,968,000 bytes | `36d4b0220f3c735f3296d301152b738c9776d16981d054845a68a1370b26cfe3` | `INTEGRITY_VERIFIED`; tar listing contains 16 parameter files. This is not an EvoFormer checkpoint or a checkpoint for the local CHIMERA approximation. No native AlphaFold/OpenFold inference was run. |
| ESMFold `esmfold_3B_v1.pt` | 2,771,653,574 bytes (upstream `Content-Length`) | — | `MISSING`: acquisition timed out, then the model store stopped the retry when the cache volume ran out of space. The atomic staging directory was removed; no partial model is registered. Native inference was not attempted. |

Both cached native smoke tests passed offline (`2 passed`); they explicitly
fail if model-store acquisition attempts network access. CUDA is unavailable
on this host, so no CUDA validation is claimed. `NATIVE_VERIFIED` here means
the named native implementation, checkpoint, and CHIMERA adapter boundary ran
in this environment; it is not a claim of biological validation.

The RFdiffusion dependency URL was corrected from a pinned-README mismatch:
the official README at revision
`86507b6538f51fce57b5a72477165f03999ed7ae` specifies the URL ending
`93063dc4`, while the previous lock ended `93063dc6` (404). The manifest now
matches the official pinned README. The declared revision and checkpoint
identity were not changed.

## Cache and commands

Install the project, then inspect the lock and configured cache:

```console
pip install .
chimera --version
chimera models list
```

Set `CHIMERA_MODEL_CACHE` to a private/shared cache directory if needed. The
older `PSC_CHIMERA_MODEL_CACHE` variable remains a fallback. Acquisition is
explicit and stores each checkpoint under an identity directory, for example:

```console
chimera models fetch esm2_t30_150m
chimera models inspect proteinmpnn_v48_020
chimera models verify --offline
```

`models verify` never accesses the network. Fetch stages a file in a temporary
directory, checks any published digest, calculates SHA-256, writes a
versioned/self-hashed manifest, and atomically publishes the directory. An
existing identity is never overwritten. A failed/incomplete download is not
made visible as a usable cache entry.

When upstream does not publish a digest, successful local verification proves
only that the cached bytes still match the locally recorded manifest. It does
**not** authenticate upstream origin. The manifest reports
`CHECKSUM_UNCONFIRMED`. Newly acquired assets start as `NATIVE_UNVERIFIED`;
successful cached native smoke tests persist their status, test identity,
runtime, adapter contract, and output-shape evidence in the local cache
manifest. Matching filenames, equal tensor dimensions, successful
deserialization, and a local hash alone are not compatibility proofs. These
self-hashed local records detect accidental changes but are not signed
attestations.

The production gate exposes separate `declared`, `acquired`,
`integrity_verified`, `architecture_verified`, `runtime_verified`, and
`native_smoke_verified` fields for every declared dependency. Required
dependencies block release unless their cache and native-smoke evidence both
verify. No production CHIMERA artifact/configuration is selected yet, so these
dependency rows do not constitute a production inference pass.

The ESMFold and AlphaFold2 archives are large. Fetch them only when explicitly
needed. `pip install .` and package imports do not fetch them. Upstream terms
may restrict use or redistribution; consult the linked license before use.

## Workspace file reconciliation

The ignored workspace contained duplicate model files and two files without a
trustworthy identity. The exact paths below were removed after confirming that
they were Git-ignored and recording their hashes in
[`chimera/architecture_dependencies.json`](../chimera/architecture_dependencies.json).

| Removed file | SHA-256 | Disposition |
| --- | --- | --- |
| `Models/esm2_t30_150M_UR50D.pt` | `881c7176cf198ef8dec26a3c375d40eb58d0c33df95c22562ca6cc6d3f812c62` | Exact duplicate of the ESM-2 cache asset; use only the identity-addressed model-store path. |
| `Models/esm2_t30_150M_UR50D-contact-regression.pt` | `6a604b96722ed052eef8a094ad90b275ba2e987d406315dbed0bdc6b3c4238a7` | Exact duplicate of the ESM-2 auxiliary cache asset. |
| `v_48_020.pt` | `c9cb4a671d79604111231f8dbfc7c590e06f1197453b7a6854ac6661a642f5bd` | Exact duplicate of pinned ProteinMPNN cache asset. |
| `weights/proteinmpnn_v48_020.pt` | `c9cb4a671d79604111231f8dbfc7c590e06f1197453b7a6854ac6661a642f5bd` | Exact duplicate of pinned ProteinMPNN cache asset. |
| `weights/rfdiffusion_base.pt` | `ba77dd3c33876e4208bb1eeecefc7c60a35c32a48a16f86cf2e834f8b7f3127c` | Untrusted 22,328-byte file; it did not match the registered 483,616,107-byte RFdiffusion checkpoint and was removed. |
| `weights/poet_weights.pt` | `7af41d5b0ac87545a63947ac3705f9667fd2c498876644b88dbf71ec15ce63c1` | Unidentified 28,006,316-byte file with no source/revision or code consumer; removed. |

These values are locally observed digests, not upstream-published hashes. The
production gate now fails if any of the reconciled workspace filenames
reappears outside the identity-addressed model store.

An additional project inventory identified the Pouyet human codon-usage archive
and its extracted data, plus the Fath2011 research data. Their scientific
provenance and label boundaries are documented in
[`data/external/README.md`](../data/external/README.md). No persisted structural
retrieval index or external MSA/structure database was identified in the
tracked runtime source. User-provided MSA/PDB inputs are inputs, not silently
resolved pretrained dependencies.

## Adapter truth table

| Adapter / model family | Real upstream checkpoint tested here | Current truthful status |
| --- | --- | --- |
| Local EvoFormer-like representation | No | Local MSA row/column approximation; not pretrained EvoFormer/OpenFold. |
| OpenFold | No | Native implementation not wired to the CHIMERA representation contract. |
| Local sequence-recovery model | No | ProteinMPNN-inspired local model; not native pretrained ProteinMPNN. |
| Native ProteinMPNN | Yes | `NATIVE_VERIFIED` on CPU through the adapter and exact pinned upstream source. |
| CHIMERA custom SE(3) bridge | Not applicable | Custom CHIMERA component; not RFdiffusion. |
| Native RFdiffusion | Checkpoint only | `UNAVAILABLE` for native inference with this host's runtime; never substituted with CHIMERA SE(3) flow. |
| ESM-2 | Yes | `NATIVE_VERIFIED` on CPU through the ESM-backed CodonOptimizer encoder. |
| ESMFold | No | `MISSING`; no cache artifact or CHIMERA ESMFold adapter. |
| AlphaFold2 / OpenFold | Parameter archive only | Not a standalone EvoFormer checkpoint; no native adapter/inference integration established. |

Downloading or hashing weights establishes neither model compatibility nor
scientific or biological validation.
