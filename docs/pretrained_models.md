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
| ESM-2 | `esm2_t30_150M_UR50D.pt`; 30 layers, about 150M parameters, 640-dimensional representations | [ESM](https://github.com/facebookresearch/esm), `2b369911bb5b4b0dda914521b9475cad1656b2ac`; [checkpoint](https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t30_150M_UR50D.pt) | MIT; [license](https://opensource.org/license/mit) | Optional CodonOptimizer encoder; canonical `fair-esm` loader only. |
| ESMFold | `esmfold_3B_v1.pt` | [ESM](https://github.com/facebookresearch/esm), `2b369911bb5b4b0dda914521b9475cad1656b2ac`; [checkpoint](https://dl.fbaipublicfiles.com/fair-esm/models/esmfold_3B_v1.pt) | MIT; [license](https://opensource.org/license/mit) | Optional upstream asset; no CHIMERA ESMFold adapter identified. |
| ProteinMPNN | Vanilla full-backbone `v_48_020.pt`, 48-neighbor variant | [ProteinMPNN](https://github.com/dauparas/ProteinMPNN), `8907e6671bfbfc92303b5f79c4b5e6ce47cdef57`; [pinned checkpoint](https://github.com/dauparas/ProteinMPNN/raw/8907e6671bfbfc92303b5f79c4b5e6ce47/vanilla_model_weights/v_48_020.pt) | MIT; [license](https://github.com/dauparas/ProteinMPNN/blob/8907e6671bfbfc92303b5f79c4b5e6ce47/LICENSE) | Native adapter boundary exists but requires upstream source. The local ProteinMPNN-inspired model is not native ProteinMPNN. |
| RFdiffusion | `Base_ckpt.pt` | [RFdiffusion](https://github.com/RosettaCommons/RFdiffusion), `86507b6538f51fce57b5a72477165f03999ed7ae`; [upstream checkpoint](https://files.ipd.uw.edu/pub/RFdiffusion/6f5902ac237024bdd0c176cb93063dc6/Base_ckpt.pt) | BSD-3-Clause; [license](https://github.com/RosettaCommons/RFdiffusion/blob/86507b6538f51fce57b5a72477165f03999ed7ae/LICENSE) | CLI boundary exists but requires the upstream inference stack and auxiliary configuration/dependencies. The CHIMERA SE(3) bridge is not RFdiffusion. |
| AlphaFold2 parameters | `alphafold_params_2022-12-06.tar`, a multi-model parameter archive | [AlphaFold](https://github.com/google-deepmind/alphafold), `c77e5d2a8961d1a353632c462914ff0a32a950f6`; [official archive](https://storage.googleapis.com/alphafold/alphafold_params_2022-12-06.tar) | CC BY 4.0; [terms](https://github.com/google-deepmind/alphafold#license-and-disclaimer) | Optional archive; not a standalone EvoFormer checkpoint. No wired OpenFold inference adapter. |

These repository revisions identify the upstream code snapshots reviewed for
the declared artifacts. They do not prove that an unmanifested local file came
from those sources. No upstream-published SHA-256 was located for these
checkpoint downloads. A SHA-256 computed during acquisition is consequently
recorded as **locally recorded**, never as an upstream checksum.

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
`CHECKSUM_UNCONFIRMED` and keeps model compatibility `UNKNOWN` until the
canonical upstream loader and the relevant CHIMERA adapter have been exercised.
Matching filenames, equal tensor dimensions, successful deserialization, and a
local hash alone are not compatibility proofs.

The ESMFold and AlphaFold2 archives are large. Fetch them only when explicitly
needed. `pip install .` and package imports do not fetch them. Upstream terms
may restrict use or redistribution; consult the linked license before use.

## Workspace files observed during the production baseline

The following ignored, outside-Git files existed in the inspected workspace.
Their SHA-256 values are local observations, not upstream-published checksums.
None had an identity-addressed CHIMERA manifest, so none counted as verified
external dependencies or adapter-tested checkpoints:

| Local file | Size | Local SHA-256 | Provenance / compatibility |
| --- | ---: | --- | --- |
| `Models/esm2_t30_150M_UR50D.pt` | 592,774,773 bytes | `881c7176cf198ef8dec26a3c375d40eb58d0c33df95c22562ca6cc6d3f812c62` | Present, origin and canonical-loader compatibility unverified. |
| `Models/esm2_t30_150M_UR50D-contact-regression.pt` | 3,431 bytes | `6a604b96722ed052eef8a094ad90b275ba2e987d406315dbed0bdc6b3c4238a7` | Present; no active CHIMERA consumer found. |
| `v_48_020.pt` and `weights/proteinmpnn_v48_020.pt` | 6,681,301 bytes each | `c9cb4a671d79604111231f8dbfc7c590e06f1197453b7a6854ac6661a642f5bd` | Same local bytes; origin and native inference compatibility unverified. |
| `weights/rfdiffusion_base.pt` | 22,328 bytes | `ba77dd3c33876e4208bb1eeecefc7c60a35c32a48a16f86cf2e834f8b7f3127c` | Present; origin and checkpoint validity unverified. |
| `weights/poet_weights.pt` | 28,006,316 bytes | `7af41d5b0ac87545a63947ac3705f9667fd2c498876644b88dbf71ec15ce63c1` | Present but unidentified; no current source-code consumer or upstream identity found. Not declared as an operational dependency. |

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
| Native ProteinMPNN | No | Adapter needs upstream source and a manifest-verified checkpoint; no end-to-end upstream smoke is established. |
| CHIMERA custom SE(3) bridge | Not applicable | Custom CHIMERA component; not RFdiffusion. |
| Native RFdiffusion | No | Adapter needs the upstream runtime/assets; no end-to-end upstream smoke is established. |
| ESM-2 | No verified cache artifact | Optional canonical `fair-esm` loading path; no present local file is accepted without a cache manifest. |
| ESMFold / AlphaFold2 | Not applicable | No active CHIMERA adapter/inference integration established. |

Downloading or hashing weights establishes neither model compatibility nor
scientific or biological validation.
