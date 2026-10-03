# CHIMERA Architecture Cleanup Audit

## Scope and result

This is the Stage A repository audit for `chimera-repair`. It records the
checked-out source rather than treating earlier diagrams, module names, or
historical descriptions as proof of runtime use.

At inspection the branch was `chimera-repair`, HEAD was
`d82e46114674c77407f20379a1b9a76cb5d6204c`, and the worktree was clean. The
user-specified frozen baseline is `5c9ea0ac6b910f35f687def079c1d79235b57c9c`;
the current HEAD is a later production-engineering commit. No architecture
modules were moved or deleted in this audit.

The canonical composition root is `chimera/architecture.py`: package exports
alias `CHIMERAv2` to `CanonicalCHIMERAv2`, and the latter owns `forward`,
`design`, inference-readiness checks, and checkpoint-component transfer.
This is a model composition root, not an end-to-end input preprocessor. Callers
provide the tokenized MSA, pair features, source frames, and any required
conditioning data. `chimera/cli.py` and `scripts/run_design.py` are entry
surfaces, not alternate model compositions.

The classifications below are primary roles; a package export or test
reference is not, by itself, evidence that a module participates in a
canonical forward pass. “Imported by” lists source-level importers and
important call-path owners found in the package, scripts, and tests.

## `chimera/*.py` inventory

| File | Classification | Direct imports / important importers | Runtime role and disposition |
|---|---|---|---|
| `__init__.py` | `PRODUCTION_SUPPORT` | Imports version, configuration, errors, production gate, architecture, core APIs, codon APIs, adapters, training and schemas. | Public import surface. Eagerly exposes several distinct subsystems; not itself a computational path. Preserve. |
| `_version.py` | `PRODUCTION_SUPPORT` | Imported by `__init__.py`, `checkpoint.py`, `cli.py`, and `production_gate.py`. | Single software-version identity used by package, CLI and provenance. Preserve. |
| `adapters.py` | `OPTIONAL_RESEARCH` | Imports `backends.py` and `model_store.py`; used by package exports and adapter tests. | Isolates native/external model integration. Does not substitute external weights for canonical local models. Preserve as optional. |
| `architecture.py` | `CORE` | Imports conditioning, components, DPO, schemas, evaluators, flow, geometry, Lie, mixed objectives, Pareto, PCGrad API, ProteinMPNN, reproducibility, bridge and sequence policy. Imported by `__init__.py` and canonical architecture tests. | Canonical composition, forward/design and component-transfer path. Preference-trainer construction lazily calls `training.py`. Preserve. |
| `autoregressive_policy.py` | `CORE` | Imported by `architecture.py`, package exports and policy/architecture tests. | Canonical sequence policy used by design and sequence/preference training. Preserve. |
| `backends.py` | `OPTIONAL_RESEARCH` | Imported by `adapters.py`; exercised by backend tests. | Backend availability/process helpers, not part of the local canonical model. Preserve as optional infrastructure. |
| `bayesian.py` | `CORE_SUPPORT` | Imported by `architecture.py`, package exports and probabilistic-optimization tests. | Uncertainty/acquisition support invoked only when readiness and explicit utility conditions allow it; not an unconditional inference stage. Preserve. |
| `checkpoint.py` | `PRODUCTION_SUPPORT` | Imports `_version.py`; used by `training.py`, `objective_schema.py`, package exports and checkpoint/provenance tests. | Checkpoint identity, schemas, hashes and compatibility contracts. Preserve. |
| `chimera_v1.py` | `LEGACY_RESEARCH` | Imports `evoformer.py`, `se3_diffusion.py` and local `proteinmpnn.py`; no active canonical importer found. | Historical generation with a distinct implementation. No evidence found that canonical training, inference or current checkpoint migration instantiates it. Retain pending an explicit retirement/public-API decision. |
| `chimera_v2.py` | `LEGACY_COMPATIBILITY` | Imports legacy flow APIs, components, schemas, evaluators and mixed-objective symbols; explicitly exercised by legacy/checkpoint tests. | Historical implementation remains reachable through its import path and is involved in state-compatibility comparisons. Keep path and symbols; do not replace test references with assumptions based on the package alias. |
| `cli.py` | `PRODUCTION_SUPPORT` | Imports `_version.py`, errors and `model_store.py`; exposed by package entry point and CLI tests. | Installed command surface for identity/model/gate operations. It does not construct a complete production serving API. Preserve. |
| `codon_optimizer.py` | `DOWNSTREAM_CODON` | Imported by `evaluators.py`, `protein_fitness.py`, package exports and codon scripts/tests. | Protein-to-codon optimization and translation utilities. Downstream to protein design, despite limited proxy use in objective evaluation. Preserve separately. |
| `components.py` | `CORE` | Imports `domain_schema.py`; instantiated by `architecture.py` and legacy v2; used by canonical training and architecture tests. | Local MSA/representation, feature connectors, and residue trunk. Pair-feature mixing is not a geometric guarantee; constraint encoding is conditioning, not hard enforcement; the ProteinMPNN-like trunk uses node features while the multiscale designer consumes graph geometry. Historical parameter names are retained for checkpoint compatibility. Preserve APIs and computation. |
| `conditioning.py` | `CORE` | Imported by `architecture.py` and legacy v2; exercised by canonical/constraint tests. | Canonical substrate/pocket conditioning. Preserve. |
| `configuration.py` | `PRODUCTION_SUPPORT` | Imports errors; exported through `__init__.py`; exercised by inference-configuration tests. | Versioned serializable inference configuration. The current legacy `design()` flow does not consume it end-to-end; preserve the contract without claiming integration. |
| `domain_schema.py` | `CORE_SUPPORT` | Imported by components, architecture and legacy v2; package exports and schema tests. | Typed NRPS/domain/assembly constraints shared across model components. Preserve. |
| `dpo.py` | `CORE_SUPPORT` | Imported by architecture, training and package exports; tested with preference training. | DPO batch/loss/trainer utilities; canonical preference regime uses this path under readiness preconditions. Preserve. |
| `errors.py` | `PRODUCTION_SUPPORT` | Imported by configuration, CLI, model store and package exports. | Shared explicit error types. Preserve. |
| `evaluators.py` | `CORE_SUPPORT` | Imports codon utilities, geometry and icosahedral proxy; used by architecture and legacy v2; objective tests. | Candidate/objective evaluation. Includes proxies; results are not experimental measurements. Preserve. |
| `evoformer.py` | `LEGACY_COMPATIBILITY` | Imported by `chimera_v1.py`; directly referenced by historical/scientific tests and docs. | Older `EvoFormer` implementation; the active canonical model instead constructs its local representation through `components.py`. Retain due historical/test/API surface; label as legacy. |
| `flow_matching.py` | `MIXED_REQUIRES_REFACTOR` | Imports Lie and Schrödinger-bridge modules; imported by architecture, legacy v2, structure utilities, package exports and tests. | Canonical bridge velocity/backbone path and legacy deterministic OT/RK4-style flow APIs coexist. Both are referenced; do not move/delete in this audit. Separate only with a future compatibility plan and tests. |
| `geometry.py` | `CORE_SUPPORT` | Used by architecture, legacy v2, package exports and geometry/invariance tests. | Validates generated/provided backbone geometry on canonical and compatibility paths. Preserve. |
| `icosahedral.py` | `CORE_SUPPORT` | Imported by evaluators and scientific tests. | Deterministic assembly/interface proxy used during candidate evaluation, not a learned objective head. Preserve. |
| `lie.py` | `CORE` | Imported by flow, Schrödinger bridge and geometry/scientific tests. | SO(3)/Lie operations used by canonical SE(3) transport. Preserve; no mathematical changes in this phase. |
| `model_store.py` | `PRODUCTION_SUPPORT` | Imports errors; used by adapters, CLI, production gate, pretrained-manager script and tests. | Identity-addressed external dependency cache and integrity operations. Not a CHIMERA artifact registry or required model dependency. Preserve. |
| `multi_objective.py` | `MIXED_REQUIRES_REFACTOR` | Imported by architecture, `pareto_pcgrad.py`, legacy v2 and package exports; broad objective/legacy tests. | Contains live `MultiScaleNRPSDesigner`, `StructuralRetriever`, `ParetoObjectives` alongside legacy DPO, sequence-policy, Pareto-head and Bayesian symbols/aliases. Preserve all until symbol-level migration is separately approved and compatibility tests are updated. |
| `objective_schema.py` | `CORE_SUPPORT` | Imports checkpoint hashing; used by architecture, Pareto and training modules, exports and schema tests. | Typed labels/source kinds, validation and schema identity for objective training/evaluation. Preserve. |
| `pareto_pcgrad.py` | `CORE` | Imports active `ParetoObjectives` from `multi_objective.py` and objective schemas; used by architecture, training, exports and objective tests. | Canonical objective feature/head and gradient-conflict training path. Do not confuse with generic `pcgrad.py`. Preserve. |
| `pcgrad.py` | `LEGACY_COMPATIBILITY` | Imported/re-exported by architecture and package API; directly covered by tests and referenced by legacy objective code. | Generic PCGrad helper remains public and tested. Canonical objective regime uses `pareto_pcgrad.py`'s head/loss path; this file is not dead and is not evidence that both algorithms run in every training step. Preserve. |
| `production_gate.py` | `PRODUCTION_SUPPORT` | Imports version/model store; used by CLI, package API and gate/dependency tests. | Release/readiness validation only. It is not the neural model and does not establish scientific/biological validation. Preserve. |
| `protein_fitness.py` | `DOWNSTREAM_CODON` | Imports codon vocabulary; exported and exercised by codon/fitness paths. | Optional ESM-backed protein fitness utility for the codon subsystem; not required by canonical structural generation. Preserve separately. |
| `proteinmpnn.py` | `CORE` | Used by architecture, training, v1/v2 and model/constraint tests. | Local ProteinMPNN-inspired structural sequence-recovery model and graph utilities. Distinct from the optional native ProteinMPNN adapter. Preserve. |
| `readiness.py` | `LEGACY_COMPATIBILITY` | Exported by `__init__.py`; used by readiness tests. | Legacy pre-production evidence summary, explicitly distinct from `production_gate.py` and `inference_readiness()`. Retain for exported/tested API; do not present it as the production release gate. |
| `reproducibility.py` | `CORE_SUPPORT` | Used by architecture, training, exports and design scripts/tests. | Seeds and deterministic runtime controls; preserve, while avoiding a claim of whole-path bitwise reproducibility. |
| `schrodinger_bridge.py` | `CORE` | Imports Lie operations; used by architecture, flow and bridge tests. | Canonical stochastic bridge construction/sampling used by flow training/inference. Preserve. |
| `se3_diffusion.py` | `LEGACY_RESEARCH` | Imported by `chimera_v1.py`; historical tests/docs refer to it; no canonical architecture importer found. | Separate older denoising implementation, not the active canonical structural generator. Keep for the v1 historical closure; mark as non-canonical. |
| `structure_utils.py` | `CORE_SUPPORT` | Imports flow rotation helper; used by design/adapter scripts and structure tests. | User-input PDB/MSA loading and utility path, not a complete canonical preprocessing API. Preserve. |
| `training.py` | `CORE_SUPPORT` | Imports checkpoint, DPO, ProteinMPNN, Pareto feature and objective-schema APIs; exported and used by architecture's preference path and training tests. | Canonical staged `CanonicalTrainer`, regime/loss selection, validation and resumable checkpoints. Preserve. |

### Symbol-level high-risk findings

- `MultiScaleNRPSDesigner`, `StructuralRetriever` and `ParetoObjectives` are
  live canonical dependencies. `pareto_pcgrad.py` imports
  `ParetoObjectives`; architecture composes the designer/retriever and legacy
  v2 also imports these names.
- `LegacyDPOTrainer`, `LegacyAutoregressiveSequencePolicy`,
  `LegacyParetoMultiObjectiveHead` and
  `LegacyBayesianUncertaintyEstimator` (including compatibility aliases) are
  retained in `multi_objective.py`. They are not the canonical trainer/head/
  policy/acquisition implementations, but legacy v2 and package/test surfaces
  make broad deletion unsafe.
- Canonical training uses `pareto_pcgrad.py` objective components. Generic
  `pcgrad.py` is independently public/tested and is used by compatibility
  code; its existence does not establish it as the current trainer's loss.
- `flow_matching.py` is not cleanly classifiable as one generation: the
  canonical `FlowMatchingBackbone` uses the Schrödinger bridge, while older
  flow APIs remain in the same module and have legacy callers.

## Canonical dependency and runtime graphs

The graphs show code paths established from constructors/call sites, not a
claim that all paths run for every invocation.

### Canonical model/inference

```text
chimera.CHIMERAv2
  -> chimera.architecture.CanonicalCHIMERAv2
     -> components.MSARepresentationBackbone / pair and node connectors
     -> conditioning.SubstratePocketConditioner / constraint features
     -> flow_matching.FlowMatchingBackbone
        -> schrodinger_bridge.SE3SchrodingerBridge
        -> lie.py
     -> geometry.validate_backbone
     -> proteinmpnn.get_protein_graph / local sequence-recovery trunk
     -> multi_objective.MultiScaleNRPSDesigner
     -> autoregressive_policy.AutoregressiveSequencePolicy
     -> evaluators.BiologicalObjectiveEvaluator
        -> deterministic geometry/icosahedral proxies
        -> codon_optimizer proxy utility where relevant
     -> bayesian acquisition only when objective readiness and explicit
        utility conditions permit it
```

The caller supplies already prepared MSA tokens, pair features, source frames,
and conditioning inputs. `design()` repeats/batches candidate generation,
filters/ranks outputs, and can report unavailable retrieval when its embedding
space is not aligned. Native ProteinMPNN/RFdiffusion/OpenFold is not silently
substituted for local components.

### Canonical training

`architecture.py` exposes model components; `training.py` owns
`CanonicalTrainer`, `TrainingRegime`, and the actual regime-to-trainable-module
map. The concrete losses and validation are selected in `CanonicalTrainer`;
the presence of a class or README table alone is not evidence of a run.

| Regime | Entry and trainable components | Loss / validation path |
|---|---|---|
| `representation` | `CanonicalTrainer(..., REPRESENTATION)`; `evoformer` component (the model's local representation in `components.py`) | Masked-token representation objective; trainer's held-out `validate()` path. |
| `flow` | `FLOW`; representation, pair connector, flow model, evolutionary cross-attention, constraint encoder and substrate conditioner | Supervised Schrödinger-bridge drift regression; held-out trainer validation. Inference stochastic bridge sampling is separate from differentiable training. |
| `constraint` | `CONSTRAINT`; same flow set with constraint conditioning | Constrained bridge drift regression; held-out trainer validation. |
| `sequence` | `SEQUENCE`; representation, node connector, local ProteinMPNN trunk, multiscale designer/projection and autoregressive policy | Teacher-forced causal amino-acid cross-entropy; held-out trainer validation. |
| `objective` | `OBJECTIVE`; objective feature encoder and Pareto head | Per-objective supervised losses with canonical conflict handling; typed objective sources/labels and held-out validation. |
| `preference` | `PREFERENCE`; autoregressive sequence policy only | DPO against a frozen reference; rejected until supervised sequence training and separate held-out validation preconditions are met. |

`CanonicalTrainer.validate()` is the engineering validation path for these
regimes. It does not establish independent biological validation.

## Production infrastructure dependency graph

```text
chimera --version / installed entry point
  -> __init__.py -> _version.py

canonical checkpoint save/resume
  -> training.py
     -> checkpoint.py -> _version.py
     -> objective_schema.py -> checkpoint config hashing
     -> reproducibility.py

chimera production-gate / model commands
  -> cli.py -> production_gate.py -> model_store.py -> errors.py
  -> configuration.py / errors.py (public configuration contract)
  -> architecture_dependencies.json / model_dependencies.json
```

`production_gate.py` evaluates packaging/dependency/artifact/readiness
contracts. It is not called as the model's forward pass. `model_store.py`
stores external model dependencies, not trained CHIMERA artifacts.
`configuration.py` is versioned and serializable, but the current
`CanonicalCHIMERAv2.design()` path does not yet accept and fully honor an
`InferenceConfig`; the production documentation already marks that integration
unavailable. Packaging and version identity are defined by project metadata
and `chimera/_version.py`.

## Optional backend graph

```text
cli.py / scripts/pretrained_manager.py
  -> model_store.py -> identity-addressed local external-artifact cache
adapters.py
  -> backends.py + model_store.py
  -> optional native ProteinMPNN / RFdiffusion / OpenFold-facing integration
```

The local canonical MSA representation, local ProteinMPNN-inspired sequence
model, and custom CHIMERA SE(3) bridge remain distinct from upstream models.
Native ProteinMPNN has an adapter smoke path; RFdiffusion native execution is
unavailable in the recorded CPU environment; OpenFold is not integrated into
the canonical representation contract. ESM-2 is used through the optional
codon fitness encoder, not as the canonical CHIMERA representation.
`ESMFold` has no established canonical adapter. The adapters and cache remain
valid optional/research/dependency-management code, not required runtime
dependencies of the local forward path.

## Downstream codon subsystem

```text
protein design output / supplied amino-acid sequence
  -> codon_optimizer.py
     -> codon tables, translation, synonymous optimization
     -> protein_fitness.py (optional ESM-backed score)
     -> data/codon_dataset.py / data/splitting.py
     -> scripts for acquisition, preparation, merge, observation, training
```

The structural design boundary ends at protein candidates and their
structure/objective evaluations. Codon optimization is downstream. The
`evaluators.py` codon-derived proxy call does not make its training corpora
structural CHIMERA data or move the optimizer into the structural forward
path.

## Legacy and compatibility audit

| Candidate | Imports / instantiation and active-path result | Tests, API, docs and unique value | Disposition |
|---|---|---|---|
| `chimera_v1.py` | No canonical training/inference or current checkpoint migrator imports/instantiates it. It imports `evoformer.py`, `se3_diffusion.py` and `proteinmpnn.py`. | Its module/classes remain directly importable; historical references exist. Its older composite implementation is unique historical material. | `LEGACY_RESEARCH`, historical-only based on inspected call graph; preserve for now. No deletion without an explicit public/history retirement decision. |
| `chimera_v2.py` | Not the package's `CHIMERAv2` target; package alias points to canonical architecture. It remains explicitly imported in compatibility tests and used in migration/state comparisons. | Import path and old component/state behavior are exercised; its symbols bridge legacy objective/flow APIs. | `LEGACY_COMPATIBILITY`; retain exact module path and behavior. Quarantine only with compatibility shim and equivalent tests. |
| `se3_diffusion.py` | Only the v1 implementation imports it; no canonical architecture/trainer/design caller found. | v1 and historical tests/docs preserve the denoiser's historical role. | `LEGACY_RESEARCH`, not a canonical structural generator. Retain and label historical. |
| `pcgrad.py` | Public architecture/package exports and legacy objective code use it; the canonical objective regime is defined by `pareto_pcgrad.py`. | Direct tests and public names constitute compatibility surface; helper functionality differs from the canonical Pareto head integration. | `LEGACY_COMPATIBILITY`; preserve. |
| `readiness.py` | Exported and tested; not the model's `inference_readiness()` nor `production_gate.py`. | Public legacy pre-production evidence summary; its documentation explicitly limits its meaning. | `LEGACY_COMPATIBILITY`; preserve with clear naming/documentation. |
| `data/training.py` | Imports `data/dataset.py`; offers a dataset loader/DataLoader helper. Not `chimera.training.CanonicalTrainer` and not imported into the canonical trainer path. | May be used as structural dataset tooling; do not conflate names or remove without checking data workflows. | `CORE_SUPPORT` for data preparation utility, separate from the canonical trainer. |
| `data/training_data.py` | No canonical package import/call found in the inspected import graph. | Contains historical architecture/training-data narrative; scripts/docs may refer to it. No unique active API was established. | `LEGACY_RESEARCH`; preserve until content and provenance are fully reconciled. |

`tests/conftest.py` performs collection-time compatibility substitutions for
some old imports. Thus a test's spelling is not sufficient evidence that it
executes the corresponding historical class. Tests remain part of the safety
system and were not removed.

## Data inventory and consumers

There are 564 tracked paths under `data/` in this checkout, including the
Ensembl cache. The tracked families resolve to distinct roles:

| Family / files | Consumer / producer | Classification |
|---|---|---|
| `data/dataset.py`, `data/training.py`, `data/merged/{train,validation}.jsonl`, manifest | Structural JSONL dataset/DataLoader utilities and structural data records; not automatically used by `CanonicalTrainer`, whose input is a `CanonicalTrainingBatch`. | Structural data/support; exact dataset provenance and whether a file is used in a current job remain caller-selected. |
| `data/training_data.py` | Historical data/training preparation descriptions. | Legacy research; not proof of an active training path. |
| `data/codon_training.jsonl`; `fath2011_codon_training*.jsonl`; `human_expression_cds_proxy.jsonl` | `CodonJSONLDataset`, codon scripts and codon optimizer training/evaluation. | Downstream codon/expression data, with measured/proxy/provenance variants kept distinct. |
| `data/Fath2011_*`, `data/Fath2011_S1_download` | Fath2011 preparation script and resulting codon training records. | Codon/expression source data, not structural training data. |
| `data/external/pouyet_human_codon_usage/` and `data/external/ensembl_cds_cache/` | Human germline-expression codon preparation; Ensembl release-83 transcript CDS cache. | External cached codon/expression source; Pouyet label is a germline-expression proxy, not general mammalian expression. |
| `data/external/kudla2006/dataset_s1.txt` | External weak/species-specific codon-context source; external README records retrieval/provenance constraints. | Codon research data, not structural CHIMERA data. |
| `data/merged_codon/`, `data/merged_codon_v2/` | Codon merge/training workflow and manifests. | Downstream codon datasets; provenance/label-kind boundaries must stay visible. |
| `data/external/README.md` | Documents external codon sources, labels, and retrieval status. | Provenance documentation; preserve. |

The user-mentioned `merged_codon`, `merged_codon_v2`, Pouyet, Kudla, Fath2011,
human-expression and Ensembl families are tracked. No `1AMU.pdb` is tracked in
the audited tree; a previous local ignored copy was not a repository asset.
No data family was deleted or reclassified as canonical structural training
without a demonstrated consumer.

## Script inventory

| File | Classification | Observed role |
|---|---|---|
| `scripts/run_design.py` | `canonical operational tooling` | Loads user PDB/MSA inputs, seeds, and invokes design; not a separate model. |
| `scripts/check_install.py` | `production tooling` | Installation/import smoke check. |
| `scripts/pretrained_manager.py` | `model acquisition tooling` | Wraps model-store fetch/inspect/verify. |
| `scripts/validate_repo_contract.py` | `production tooling` | Validates repository/architecture contract used by CI. |
| `scripts/test_proteinmpnn_adapter.py` | `model acquisition tooling` | Explicit optional native ProteinMPNN adapter smoke/check script. |
| `scripts/download_weights.sh`, `scripts/download_weights.ps1` | `model acquisition tooling` | Historical/general weight download entrypoints; verify target manifests before relying on them. |
| `scripts/codon_observatory.py` | `codon tooling` | Codon usage statistics/inspection. |
| `scripts/fetch_human_expression_cds.py` | `codon tooling` | Fetches/caches human CDS used by codon proxy workflow. |
| `scripts/merge_codon_datasets.py` | `codon tooling` | Validates and merges codon records. |
| `scripts/prepare_fath2011_dataset.py` | `codon tooling` | Prepares Fath2011-derived codon labels. |
| `scripts/test_codon_optimizer.py` | `codon tooling` | Standalone codon optimizer checks. |
| `scripts/test_codon_optimizer_integrity.py` | `codon tooling` | Dataset/optimizer integrity checks. |
| `scripts/train_codon_optimizer.py` | `codon tooling` | Downstream codon optimizer training; uses clustered splitting. |
| `scripts/__init__.py` | `development tooling` | Package marker only. |

## Test inventory

Every `tests/test_*.py` file is retained. These labels identify the primary
contract protected; many files also cover regressions or multiple layers.

| File | Primary coverage |
|---|---|
| `test_architecture_contract.py` | Canonical architecture/public composition contract |
| `test_architecture_dependencies.py` | Dependency inventory and provenance |
| `test_backends.py` | Optional backend behavior |
| `test_canonical_architecture.py` | Canonical model, inference and staged training |
| `test_canonical_autoregressive_policy.py` | Canonical sequence policy |
| `test_checkpoint_contract.py` | Checkpoint schema/migration compatibility |
| `test_chimera_v2.py` | Historical v2 compatibility and cross-version comparisons |
| `test_cli_contract.py` | CLI behavior |
| `test_codon_observatory.py` | Codon tooling |
| `test_codon_optimizer_training.py` | Codon training/splitting |
| `test_dataset_contracts.py` | Dataset schemas and data utilities |
| `test_evoformer_masking.py` | Historical representation/scientific masking invariants |
| `test_inference_configuration.py` | Configuration serialization/version/validation |
| `test_native_model_smoke.py` | Optional native model integrations |
| `test_objective_contracts.py` | Learned/deterministic objective contracts |
| `test_packaging_contract.py` | Installation/import/entry-point packaging |
| `test_pretrained_manager.py` | External artifact store/acquisition |
| `test_probabilistic_optimization.py` | Bayesian/probabilistic optimization |
| `test_production_gate.py` | Release gate and readiness |
| `test_proteinmpnn_constraints.py` | Local sequence recovery and constraints |
| `test_provenance_contracts.py` | Checkpoint/data/runtime provenance |
| `test_public_contracts.py` | Public import/API contracts |
| `test_readiness_gate.py` | Legacy readiness summary and distinction from release gate |
| `test_regressions.py` | Historical bug regressions |
| `test_repair_contracts.py` | Repair/recovery contracts |
| `test_schrodinger_bridge_constraints.py` | Canonical bridge constraints |
| `test_schrodinger_bridge_regressions.py` | Bridge regression/scientific behavior |
| `test_scientific_contracts.py` | Scientific invariants |
| `test_scientific_repairs.py` | Scientific regression fixes |
| `test_se3_invariance.py` | SE(3)/invariance behavior |
| `test_split_and_resume_contracts.py` | Data split, training resume, checkpoint provenance |
| `test_structure_utils.py` | Structure/MSA input utilities |

## Documentation consistency and unresolved items

- The README currently leads with the local-approximation and untrained
  prototype limitation, describes the local structural and sequence models,
  separates external native adapters, lists current training regimes, and
  distinguishes production engineering from biological validation. This
  report does not find evidence to replace that architecture with external
  models.
- Four shell/PowerShell installation wrappers duplicated virtual-environment
  creation and `pip install`; the older pair also advertised an undeclared
  `.[md]` extra. They were removed in the later canonicalization pass.
  `INSTALL.md` now documents the package install commands directly; the
  independent `scripts/check_install.py` diagnostic remains.
- `chimera/evoformer.py`, `chimera/se3_diffusion.py`, and
  `data/training_data.py` contained historical descriptions in active-looking
  paths. Their module headers now label that material historical without
  changing model/data behavior.
- The retired root-level `push_to_github.sh` helper had no callers and was
  replaced with a no-op stub in the low-risk cleanup; this final pass removed
  that inert stub. Its predecessor had unsafe force-push/remote-replacement
  behavior and was never executed during the cleanup.
- A complete production input-preprocessing/request/result/telemetry path is
  not established. `InferenceConfig` is not yet wired end-to-end into
  `CanonicalCHIMERAv2.design()`.
- Full biological validation, objective calibration from independent
  measurements, and a validated trained CHIMERA checkpoint remain absent.
- Some legacy APIs are publicly importable but have no demonstrated current
  consumers. Exact downstream external user usage cannot be inferred from a
  repository-local search. This is a compatibility uncertainty, not grounds
  for deletion.
- No `DEAD_RESIDUE` classification is assigned: the audit did not satisfy the
  user's deletion safety conditions for any file.

## Cleanup decision

Stage A is complete. The follow-up low-risk cleanup labels misleading
historical module/data descriptions, clarifies the README's canonical
representation and PCGrad boundaries, labels the prior dependency inventory
as historical, and retired the unsafe push helper. The final pass removed its
inert stub and an unrelated zero-byte root artifact after verifying neither
had a consumer.
Architecture moves/splits require their own import-compatibility plan and
focused tests. No scientific architecture, objective semantics, or model
mathematics were changed.

## Second-order separation update (2026-10-03)

The later second-order pass implemented the previously deferred safe module
separation. `MultiScaleNRPSDesigner` now lives in `chimera/sequence_design.py`,
`StructuralRetriever` in `chimera/retrieval.py`, and `ParetoObjectives` in
`chimera/objective_schema.py`. The old `chimera/multi_objective.py` import path
is a compatibility shim; the historical DPO, autoregressive, Pareto, and
Bayesian implementations live in `chimera/legacy_optimization.py`.

The canonical `architecture.py` and `pareto_pcgrad.py` now import from those
owning modules directly and do not import the compatibility shim or
`legacy_optimization.py`. Tests verify both the owned class modules and
resolution of serialized globals through the old multi-objective path. This
supersedes this audit's earlier recommendation to retain the entire mixed
module in place. `flow_matching.py` and `components.py` remain unchanged:
source/test consumers demonstrate shared canonical building blocks in both,
and no state/checkpoint-safe extraction was justified in this pass.
