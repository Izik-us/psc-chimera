# PSC-CHIMERA architecture dependency audit

## Scope and source identity

This inventory was grounded in the clean `chimera-repair` baseline
`faf721c5b9eda0eddf98a4c9b3c421621769349e` (`production implementations`,
software version `2.0.0`). The machine-readable inventory is
[`chimera/architecture_dependencies.json`](../chimera/architecture_dependencies.json);
the production gate validates its schema, IDs, component/runtime/asset edges,
test references, external cache identity records, dataset file hashes, and
selected-composition state. It rejects any previously identified local model
candidate if it reappears outside the identity-addressed cache. The
`chimera production-dependencies` command emits the closure and blockers as
JSON without implying that a candidate is release-ready.

The repository has a canonical *research composition*, not a production
composition. The public `chimera.CHIMERAv2` is
`chimera.architecture.CanonicalCHIMERAv2`. Its constructor creates trainable
modules with random initialization. `require_inference_ready()` refuses normal
design unless required modules, objective predictors, and calibrations carry
validated training/validation evidence. The separate `chimera-design` command
is a research/demo utility; `--demo`/`--experimental` explicitly permits
untrained execution and does not create release evidence.

Accordingly the inventory records `production_composition: null` and
`production_artifact: null`. It is not a claim that the architecture is
dependency-complete or biologically valid.

## Execution graph that exists today

```text
chimera-design / CanonicalCHIMERAv2.design (research API)
  -> caller-provided MSA tokens, pair features, source frames, constraints
  -> MSARepresentationBackbone (local EvoFormer-like approximation)
  -> pair connector + optional substrate and constraint conditioning
  -> local SE3SchrodingerBridge / velocity network (stochastic sampling)
  -> frames-to-backbone conversion + geometric sanity checks
  -> local ProteinMPNN-inspired geometric message passing
  -> MultiScaleNRPSDesigner + autoregressive sequence policy
  -> four randomly initialized learned objective heads
  -> transparent deterministic proxy evaluator, including deterministic
     icosahedral assembly geometry
  -> research result mapping and FASTA/JSON written by scripts/run_design.py
```

The CLI does not resolve a CHIMERA model artifact or consume the versioned
`InferenceConfig`. MSA preprocessing, pair-feature production, source-frame
preprocessing and their identities remain caller responsibilities. Its default
design path rejects the untrained composition; explicit experimental mode is
not production inference.

### Internal components

The manifest contains the component-by-component source symbol, graph edges,
scope, status, input/config identity, test references, verification facts and
blockers. Important classifications:

| Area | Implementation and present evidence | Release status |
| --- | --- | --- |
| MSA/EvoFormer boundary | `MSARepresentationBackbone` is a local row/column-attention approximation, not native EvoFormer/OpenFold. Synthetic construction/forward tests exist. | `VERIFIED_APPROXIMATION`; random/untrained weights |
| Pair representation, triangular updates, substrate and NRPS conditioning | Local modules wired in canonical composition and exercised in small synthetic tests. | `LOCAL_VERIFIED`; no trained release checkpoint |
| SO(3)/SE(3), frames, bridge, flow and sampling | Canonical geometry helpers are in `lie.py`; `flow_matching.py` imports those helpers. `schrodinger_bridge.py` supplies entropic coupling, Brownian bridge targets, and stochastic sampling. Equivariance/invariance and constraint tests exist. | `VERIFIED_APPROXIMATION`; local model, no trained release checkpoint; not RFdiffusion |
| Structure/geometry | Local frame-to-coordinate conversion and `validate_backbone` check geometric sanity. | `LOCAL_VERIFIED`; not force-field or biological validation |
| Sequence recovery | `ProteinMPNNBackbone`/`ProteinMPNN` are local ProteinMPNN-inspired approximations. The canonical composition also uses `MultiScaleNRPSDesigner` and an autoregressive policy. | Approximation is integrated; learned parameters are not production-trained |
| Native ProteinMPNN | `ProteinMPNNAdapter` loads the exact pinned external source and cached weight; CPU native inference through the adapter passed and preserves a fixed residue. | `NATIVE_VERIFIED` adapter; **not invoked by canonical CHIMERAv2** |
| Objectives | Four neural surrogate channels are untrained/uncalibrated. The assembly channel is a deterministic geometry proxy. Other transparent scores are proxies, not measurements. | Neural objectives `INTEGRATED_NOT_VERIFIED`; deterministic checks `LOCAL_VERIFIED` |
| Retrieval | Index creation API exists, but no index/compatible encoder identity exists; forward reports `RAG_UNAVAILABLE` when requested. | `UNAVAILABLE`, optional and off by default |
| ESM/codon | Real ESM-2 runs through the optional standalone `CodonOptimizer` workflow. The canonical MSA backbone does not consume ESM-2; canonical objective expression output is a rule-based proxy. | ESM native smoke `NATIVE_VERIFIED`, optional and not a production composition dependency |
| Bayesian/EI/Pareto/PROTEUS/DPO | Research/optimization and training utilities exist, but require trained/calibrated objective or sequence models and explicit experiment data. | Research/training only; not a selected release feature |
| Input, artifact and result boundaries | Caller supplies feature tensors. `InferenceConfig` is a valid JSON value object but is not consumed by `design`. External model store is not a CHIMERA artifact registry. `design`/script emit ad-hoc results. | Missing production preprocessing identity, artifact resolver, and stable public inference/result contract |
| Checkpoint provenance | Versioned checkpoint manifest and training resume machinery exist and are tested. No selected trained checkpoint links code, data, config, environment and objective schema. | Present, not integrated into a selected production artifact |
| Legacy code | `chimera_v1.py`, `chimera_v2.py` and legacy multi-objective classes remain for compatibility/research. Public `CHIMERAv2` points at the canonical composition, not legacy `CHIMERAv2`. | `DEPRECATED`; excluded from candidate graph |

## External models, source and licensing boundary

The dependency and cache state is independently represented in the inventory
and external model lock:

| Asset | Current status | Role in canonical candidate |
| --- | --- | --- |
| ESM-2 `esm2_t30_150M_UR50D.pt` and contact regression auxiliary | `NATIVE_VERIFIED` by cached CPU smoke; local SHA-256 only, no published upstream SHA-256 located | Optional standalone codon workflow; not the canonical MSA model |
| ProteinMPNN `v_48_020.pt`, pinned source `8907e6671bfbfc92303b5f79c4b5e6ce47cdef57` | `NATIVE_VERIFIED` by cached CPU adapter smoke; local SHA-256 only | Optional native adapter; not invoked by canonical sequence path |
| RFdiffusion `Base_ckpt.pt`, pinned source `86507b6538f51fce57b5a72477165f03999ed7ae` | `UNAVAILABLE` for native inference on this host; upstream runtime requires Python 3.9, PyTorch 1.9, CUDA 11.1 and DGL CUDA 11.1 | Not a substitute for the local SE(3) bridge |
| ESMFold v1 | `MISSING`; no adapter was found in the active composition | Optional; not required by the traced candidate |
| AlphaFold2 parameter archive | `INCOMPATIBLE` as a standalone EvoFormer checkpoint; no native AF2/OpenFold inference was attempted | Not used by the local EvoFormer-like backbone |
| Retrieval index | `MISSING` | Optional; no production index |

License and upstream revisions are in
[`chimera/model_dependencies.json`](../chimera/model_dependencies.json).
Local cache hashes establish only local byte integrity when the upstream does
not publish a checksum. The ProteinMPNN and RFdiffusion pinned source checkouts
are outside the repository. RFdiffusion's unavailable runtime remains distinct
from a missing checkpoint.

Six ignored workspace model-named files were reconciled by hash. Four were
duplicates of identity-addressed ESM-2/ProteinMPNN cache assets; the RFdiffusion-
named 22 KB file did not match the registered 483 MB checkpoint; and the PoET-
named file had no identified source or consumer. All six were removed from the
workspace. Their observed hashes, source match (where known), and disposition
are recorded in the manifest; this does not infer upstream origin from a name.

## Data, schemas, packages and runtime

- The five objective semantics are in `objective-schema-v2`; the assembly
  objective is deterministic and not a learned fifth neural head.
- `data/` includes dataset manifests and training/splitting code. The inventory
  records SHA-256 and size for 17 tracked training/reference files and a
  deterministic aggregate hash for the 504-file Ensembl CDS cache. Text assets
  are hashed and sized after CRLF/CR-to-LF normalization so Windows and Linux
  checkouts share the same identity; the source archive is hashed as raw bytes.
  The ignored, local-only `data/1AMU.pdb` is not represented as a checked-in
  asset. These assets remain training-only: no production checkpoint
  references a dataset manifest/hash, training run, held-out validation, or
  calibration set. Upstream licensing and provenance are explicitly partial
  where the repository does not record them.
- Token vocabularies and geometry conventions are implemented in source. A
  selected production preprocessing/tokenizer identity and serialized request
  schema do not exist.
- `setup.py` supports Python `>=3.10`; base installation declares only
  `torch>=2.1.0`, `einops>=0.7.0`, and `numpy>=1.24.0`. Data, ESM, retrieval,
  research, visualization, and development dependencies are separate extras.
  Previously declared but unused packages (including Transformers, pandas,
  h5py, CLI helpers, OpenMM and MDTraj) are recorded as removed in the machine
  inventory. Version ranges are not a production lockfile, and CUDA wheels
  remain platform-specific. `scripts/check_install.py` only hard-requires the
  core runtime and reports workflow extras as optional.
- CPU execution is the only native-model backend exercised here. CUDA cannot
  be reported as validated from the CPU-only host.

## Gate result and remaining blockers

The gate now consumes the machine-readable architecture inventory and verifies
that its graph references resolve and all 18 hashed training/reference assets
(17 files plus the Ensembl CDS directory) match their recorded identities.
The CLI exposes the declared candidate and its closure using `chimera
production-dependencies`. The gate blocks unless a production composition and
artifact are selected and closure is marked ready. The current inventory
intentionally fails that release check. Its exact blockers are:

1. No selected trained and validated CHIMERA checkpoint.
2. No production composition/artifact or independently resolvable artifact registry.
3. `InferenceConfig` is not consumed by the canonical design path.
4. No authoritative production preprocessing/input identity.
5. No complete checkpoint-to-data/config/source/dependency provenance chain.
6. No stable public structured inference result and serialization contract.
7. No end-to-end inference or deterministic production smoke.

This is not blocked by the missing ESMFold checkpoint or unavailable
RFdiffusion runtime: neither is in the candidate's runtime closure. The
scientific/training blocker is substantive. The available 588-row codon data
mixes measured and proxy labels and contains no structure/MSA examples or
independent calibration set adequate to validate the canonical representation,
flow, sequence generation, and learned objective channels. A production
checkpoint cannot be honestly produced from the current data. The gate reports
this as blocked; it does not substitute a random checkpoint or label proxy
training as scientific validation.

This audit does not fabricate an artifact, mark an approximation native, call a
unit test an end-to-end test, or turn optional external models into production
requirements. Production remains blocked internally until the training,
artifact, preprocessing, inference and provenance gaps are closed.
