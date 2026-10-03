# PSC-CHIMERA repository cleanup: final audit

## Scope and outcome

This report records the source-based audit and conservative cleanup on
`chimera-repair`. The frozen architecture was not redesigned, scientific
semantics were not changed, and no uncertain model, data, test, or compatibility
file was deleted.

The branch was already beyond the frozen architectural baseline when this
work began. The inspected starting state was:

| Item | Starting value |
| --- | --- |
| Branch | `chimera-repair` |
| HEAD | `089e101757485216f5133d07315fff0323cf1d44` |
| Worktree | Clean |
| Python | 3.12.10 |
| PyTorch | 2.13.0+cpu |
| Tracked files | 680 |
| Compile check | Passed |
| Repository contract | Passed |
| Pytest | 212 passed, 41 warnings |

The frozen baseline commit remains `5c9ea0ac6b910f35f687def079c1d79235b57c9c`;
it is not the starting HEAD. This audit extends the prior source inventory in
[`ARCHITECTURE_CLEANUP_AUDIT.md`](./ARCHITECTURE_CLEANUP_AUDIT.md), which
contains package-module, data-family, script, and test classifications. The
root README's Architecture Contract remains the single authoritative
high-level description of current executable architecture.

## Canonical closure

The public package name `chimera.CHIMERAv2` resolves to
`chimera.architecture.CanonicalCHIMERAv2`. It is a research composition root
that expects caller-prepared tensors; it is not a complete preprocessing or
production serving interface.

### Inference and design

```text
chimera.CHIMERAv2
  -> chimera.architecture.CanonicalCHIMERAv2
  -> forward(msa_tokens, initial_pair_features, source_R, source_t, ...)
     -> components.MSARepresentationBackbone
     -> components.TriangularPairUpdateConnector
     -> conditioning.SubstratePocketConditioner (when requested)
     -> components.NRPSConstraintEncoder (when constraints are supplied)
     -> components.EvolCrossAttentionConnector
     -> flow_matching.FlowMatchingBackbone.sample
        -> schrodinger_bridge bridge/sampling implementation
        -> lie.py SO(3)/SE(3) helpers
     -> frame-to-coordinate conversion and geometry.validate_backbone
     -> components.NodeProjectionConnector and ProteinMPNNBackbone
     -> proteinmpnn.get_protein_graph
     -> multi_objective.MultiScaleNRPSDesigner
     -> AutoregressiveSequencePolicy.generate
     -> pareto_pcgrad.ObjectiveFeatureEncoder and
        MergeReadyParetoMultiObjectiveHead
     -> evaluators.BiologicalObjectiveEvaluator
        -> deterministic icosahedral assembly/interface proxy
     -> output mapping
```

`design()` batches calls through `forward()`, collects candidates, chooses
trained surrogates only when readiness evidence allows it, otherwise ranks
using explicitly named deterministic proxy scores, and optionally computes
Bayesian Expected Improvement only with validated objective state and
explicit utility weights. Retrieval is constructed and has an index-building
API, but the current inference path reports it unavailable rather than using
unaligned query/index embeddings. These conditions do not change the
architecture or assert biological validity.

The public composition is not the standalone native ESM-2, ProteinMPNN,
RFdiffusion, ESMFold, or OpenFold implementations. The local representation,
SE(3) bridge, and ProteinMPNN-inspired modules remain distinct local models or
approximations. Optional adapters and cached assets are not mandatory forward
dependencies.

### Training closure

`chimera.training.CanonicalTrainer` implements the active regime-to-component
map and separately enforces gradient, validation, and checkpoint contracts:

| Regime | Actual loss path and required learned components |
| --- | --- |
| Representation | Masked reconstruction through `MSARepresentationBackbone`; representation parameters are trained with explicitly frozen initialization/expansion pieces. |
| Flow | Shared conditioning path, pair connector, flow model, evolutionary cross-attention; calls the flow model's bridge loss with source/target frames. |
| Constraint | The flow path plus required NRPS constraints and constraint encoder. |
| Sequence | Target/source frames to coordinates, node projection, local ProteinMPNN backbone, protein graph, multiscale designer, logits-to-context projection, and autoregressive sequence-policy loss. |
| Objective | Typed objective feature construction, `ObjectiveFeatureEncoder`, canonical Pareto head task losses, and true PCGrad projection in `pareto_pcgrad.py`. Assembly compatibility is rejected as a supervised learned label because it is deterministic at inference. |
| Preference | `dpo.DPOTrainer` over the sequence policy and frozen reference policy, using an explicit `DPOBatch`. |

Held-out model validation is `CanonicalTrainer.validate()` and records dataset
identity, minimum/observed examples, structure source, losses, and validation
status. The training datasets and codon datasets do not automatically become
inputs to this trainer: callers supply `CanonicalTrainingBatch` values and
dataset manifests.

### Validation and production boundaries

These are different closures and remain separate:

* **Model validation:** `training.CanonicalTrainer.validate()` evaluates
  held-out training batches and updates training/validation provenance.
* **Candidate validation:** `geometry.validate_backbone()` performs geometric
  sanity checks; `evaluators.BiologicalObjectiveEvaluator` and
  `icosahedral.py` provide candidate scores, including explicit proxies.
  These checks do not establish biological validity.
* **Production validation:** `production_gate.run_production_gate()` checks
  repository/package contracts, tests, manifests, runtime, and evidence.
  `checkpoint.py`, `model_store.py`, `configuration.py`, and
  `reproducibility.py` serve provenance, external-asset, configuration, and
  determinism support. `model_store.py` is not a CHIMERA artifact registry.
  `readiness.py` remains a legacy evidence-summary API, not the release gate.

The production gate remains honest: the repository does not identify a
validated CHIMERA checkpoint and complete production inference/preprocessing/
provenance contract. The prior production engineering audit records this as a
release blocker; this cleanup did not weaken or re-run the gate.

## `multi_objective.py` symbol audit

The module is mixed-purpose, not dead. Its top-level symbols, current roles,
and disposition are:

| Symbol | Active callers and evidence | Classification and disposition |
| --- | --- | --- |
| `StructuralRetriever` | Constructed by canonical `architecture.py`; index setup is exposed there; also used by legacy v2 and retrieval tests. Canonical forward currently reports retrieval unavailable. | Canonical optional/support API. Retain. |
| `ProteusPreferencePair` | Used by `LegacyDPOTrainer`, imported by legacy v2, and instantiated in a compatibility test. | Legacy preference data contract. Retain its import path. |
| `LegacyDPOTrainer` / alias `DPOTrainer` | Imported by legacy v2/tests; intentionally distinct from canonical `chimera.dpo.DPOTrainer`. | Legacy compatibility API. Retain; do not conflate the two implementations. |
| `LegacyAutoregressiveSequencePolicy` / alias `AutoregressiveSequencePolicy` | Imported by legacy v2 and historical module-level tests. Test collection redirects selected model/policy names, so test spelling alone is not proof of runtime dispatch. | Legacy compatibility API. Retain pending explicit public API retirement. |
| `ParetoObjectives` | Constructed by canonical `architecture.py`; consumed by `pareto_pcgrad.py`; also imported by legacy v2 and scientific-contract tests. | Shared canonical value object. Retain. |
| `LegacyParetoMultiObjectiveHead` / alias `ParetoMultiObjectiveHead` | Used by legacy v2 and directly exercised through the old import in scientific-contract tests. Canonical training uses `MergeReadyParetoMultiObjectiveHead` instead. | Legacy compatibility implementation, not canonical PCGrad. Retain. |
| `LegacyBayesianUncertaintyEstimator` / alias `BayesianUncertaintyEstimator` | Legacy v2 compatibility imports; distinct from `chimera.bayesian.BayesianUncertaintyEstimator`, which the canonical architecture uses. Explicit identity tests distinguish them. | Legacy compatibility API. Retain. |
| `MultiScaleNRPSDesigner` | Instantiated and called by canonical inference and sequence training; also imported by legacy v2, package exports, and canonical/legacy tests. | Canonical sequence-generation component. Retain. |

No top-level symbol in this file was shown to be safe to purge as a whole.
Splitting the active classes into another module could improve file boundaries,
but the present legacy imports and alias identity tests create a real
compatibility surface. Moving class definitions would also change their
`__module__` identity, which can affect external imports and serialized Python
objects. No compatibility/deprecation policy currently authorizes that
change. Therefore the smallest safe decision is to keep the mixed module and
document its boundaries, not to move symbols speculatively.

One unreachable block was removed from
`LegacyDPOTrainer.compute_sequence_logprob()`: after returning from the
`sequence_logprob` branch, the function unconditionally raises `TypeError`.
The following generic model-forward scoring implementation could never run.
Removing it preserves the method's observable behavior.

## Repository inventory and classification

The tracked-path inventory was taken before changes (680 paths), and every
area was reviewed against source references, package/CLI configuration, tests,
CI, manifests, and documentation. Counts below refer to that baseline
inventory; the two removed root paths are accounted for separately.

| Area | Baseline paths | Classification and audit result |
| --- | ---: | --- |
| `chimera/` | 40 | Individually classified in the prior package inventory as canonical computation, core support, production support, optional backend, downstream codon, or legacy/compatibility. No uncertain package file was purged. |
| `data/` | 564 | Data/support, downstream codon, external source/cache, historical, or provenance material; no contents/manifests changed. Includes 541 external paths: 504 Ensembl cache paths, 35 Pouyet source paths, one Kudla source file, and one external README. `data/merged/` has 3 paths; `merged_codon/` and `merged_codon_v2/` have 3 paths each. |
| `tests/` | 34 | Canonical behavior, scientific invariants, data/provenance, checkpoint/API, production, optional-backend, and legacy compatibility tests. No test was removed. |
| `scripts/` | 19 | Current design/contract/install tools, optional model acquisition, and downstream codon preparation/inspection/training. Each script's role and consumer is listed in the prior script inventory. |
| `docs/` | 7 | README is the architecture overview; cleanup/dependency audits are dated snapshots; production gate/engineering docs describe the release boundary; setup and pretrained-model docs cover installation/assets. |
| `.github/` | 3 | CI, dependency update configuration, and repository Copilot instructions. CI consumes compile, repository-contract, and pytest checks. |
| `production-evidence/` | 3 | Historical gate evidence/provenance records, including records that preserve the exact earlier source/worktree state. Retained without rewriting. |
| Root | 10 | `.gitignore`, `INSTALL.md`, `LICENSE`, `README.md`, `pytest.ini`, `requirements-dev.txt`, `requirements.txt`, `setup.py`, plus the two removed artifacts below. |

For path-level review of the non-code directories: `.github/` contains
`copilot-instructions.md`, `dependbot.yml`, and `workflows/tests.yml`;
`production-evidence/` contains `phase1-final.json` and the two timestamped
`production-gate-*.json` records. The seven baseline documentation files are
`PIP_SETUP_WINDOWS.md`, `ARCHITECTURE_DEPENDENCY_AUDIT.md`,
`ARCHITECTURE_CLEANUP_AUDIT.md`, `pretrained_models.md`,
`PRE_PRODUCTION_ENGINEERING_GATE.md`, `PRODUCTION_ENGINEERING_BASELINE.md`,
and `PRODUCTION_ENGINEERING.md`. This final audit becomes the eighth tracked
documentation file. The baseline root paths are listed in the table; the
empty `python` artifact and retired `push_to_github.sh` stub are the only
removals.

Data is not one undifferentiated training corpus. The structural JSONL and
dataset helpers, Fath2011 codon source and derivatives, human-expression/CDS
proxies, Ensembl cache, Kudla source, and merged codon datasets remain
distinct. Codon material is downstream support, not evidence of the MSA,
structure, or held-out objective examples required for a trained production
CHIMERA artifact. No dataset was silently re-labeled, regenerated, or deleted.

Documentation roles are made explicit here: README is the one high-level
architecture authority; `ARCHITECTURE_DEPENDENCY_AUDIT.md` is an explicitly
historical inventory snapshot; `ARCHITECTURE_CLEANUP_AUDIT.md` is the prior
stage's detailed inventory; and this document is a dated cleanup disposition,
not a competing architecture specification. Existing production and
pretrained-model documents retain their stated scope and limitations.
The prior script inventory names every maintained script and its operational
role, and its test inventory names every retained `test_*.py` file and its
primary contract. Neither scripts nor tests were removed.

## Duplicate and legacy implementation audit

Overlapping implementations were compared by active call path, public import,
tests, and checkpoint compatibility. No equivalent-name match was treated as
proof of redundancy:

| Area | Canonical/current path | Other path and disposition |
| --- | --- | --- |
| Model composition | `architecture.CanonicalCHIMERAv2` | `chimera_v1.py` and `chimera_v2.py` are distinct historical compositions. V1 remains historical research; v2 remains import- and test-visible compatibility code. Retain both. |
| Representation | `components.MSARepresentationBackbone` | `evoformer.EvoFormer` belongs to the historical v1/compatibility closure. Retain; do not substitute one for the other. |
| Structural generation | `flow_matching.FlowMatchingBackbone` with `schrodinger_bridge.py` and `lie.py` | `se3_diffusion.py` and older flow APIs are separate historical implementations; flow APIs coexist in a mixed module. Retain pending a behavior/checkpoint-aware compatibility plan. |
| Objective head and PCGrad | `pareto_pcgrad.MergeReadyParetoMultiObjectiveHead` | `multi_objective.LegacyParetoMultiObjectiveHead` is explicitly heuristic; generic `pcgrad.py` also has public tested helpers. Preserve distinctions and compatibility. |
| DPO and sequence policy | `dpo.py`, `autoregressive_policy.py` | `multi_objective.py` exposes historical counterparts; old imports and identity/behavior tests exist. Retain. |
| Bayesian optimization | `bayesian.py` | Legacy estimator alias in `multi_objective.py` is a separate compatibility type. Retain both. |
| Readiness and release | `architecture.inference_readiness()` and `production_gate.py` | `readiness.py` is the exported evidence-summary compatibility surface. Retain and do not treat it as the release gate. |
| ProteinMPNN | Local ProteinMPNN-inspired canonical component | Native upstream integration is isolated behind optional adapter code. They are not interchangeable implementations. Retain both boundaries. |
| SO(3)/SE(3), geometry, data, checkpoints | Shared Lie/geometry helpers and explicit checkpoint/model-store contracts | Similar-purpose helpers have distinct callers, invariants, and artifact semantics. No behavior-equivalence evidence justified consolidation. Retain. |

The legacy files do not masquerade as canonical in the root README and the
prior cleanup inventory. No `chimera/legacy/` move was made because a
compatibility shim would not by itself prove checkpoint, pickle, or downstream
consumer compatibility.

## Evidence-backed changes and purge decisions

| Path/change | Evidence | Result |
| --- | --- | --- |
| Root `python` | Tracked blob was zero bytes; `setup.py` packages discovered Python packages, not this root-level empty file; no package, script, CI, documentation, or dynamic-import consumer was found. | Deleted. |
| Root `push_to_github.sh` | The tracked contents were an inert retirement stub that exited with failure and had no consumers. The unsafe predecessor's force-push/remote-replacement behavior was not run. | Deleted; prior audit updated. |
| `chimera/__init__.py` version literal | A local hard-coded `"2.0.0"` assignment was immediately overwritten by importing `_version.__version__`; package metadata, CLI, and checkpoint already use `_version.py`. | Removed redundant assignment; authoritative version unchanged. |
| `multi_objective.py` unreachable DPO block | Unconditional `TypeError` precedes the block for every input lacking `sequence_logprob`; the supported branch returns before it. | Removed unreachable duplicate scorer; observable behavior unchanged. |

No other file met the full removal criteria. In particular:

* `chimera_v1.py`, `chimera_v2.py`, `evoformer.py`, `se3_diffusion.py`,
  `flow_matching.py`, and mixed objective APIs remain due historical,
  scientific, import, test, or checkpoint-compatibility evidence.
* Data and production-evidence records remain because source provenance and
  cache/history are not disposable merely because they do not train the
  currently selected composition.
* Weight acquisition wrappers and install helpers remain because they are
  operational entry points; their optional role is already documented.
* Ignored bytecode/cache output from validation is not tracked repository
  content and was not used as purge evidence.

There are no quarantined files. Quarantine would not improve safety for the
retained code because its compatibility/import surface is still live. The
mixed module and other historical classes remain explicitly classified rather
than being presented as canonical.

## Validation after changes

Two incremental checkpoints were run before this final report:

1. After deleting the empty root artifact and inert push stub:
   compile passed, `scripts/validate_repo_contract.py` passed, and
   `pytest -q` reported **212 passed, 41 warnings**.
2. After removing the duplicate version assignment and unreachable scorer:
   compile passed, `scripts/validate_repo_contract.py` passed,
   `pytest -q` reported **212 passed, 41 warnings**, and `git diff --check`
   passed.

The warnings are existing PyTorch transformer/nested-tensor, migration,
deprecation, and test-conversion warnings; no test was skipped or failed.
No inference, training, objective, checkpoint, dataset, or release-gate
semantics were changed.

## Remaining uncertainty

The following are intentionally retained and not claimed to be unnecessary:

* External consumers of legacy module/class paths are not enumerable from this
  repository. A supported deprecation/versioning decision and compatibility
  tests are needed before moving or removing those APIs.
* Historic Python-object serialization using old class module paths was not
  exhaustively enumerated. This is another reason not to relocate legacy
  class definitions in this cleanup.
* Individual cached external records may be relevant to downstream
  provenance. Their family/source role is known, but no evidence justifies
  deleting or rewriting any member.
* The working source has no selected, trained, validated CHIMERA production
  checkpoint and does not yet supply a complete production preprocessing,
  request/result, telemetry, and determinism contract. This remains an
  explicit production blocker, not a reason to change model architecture or
  weaken the gate.

## Follow-up canonicalization pass (2026-10-03)

### Frozen starting state

The follow-up pass began from clean HEAD `6929c719d2b6191394597d6a3472ef9daa93e6a7`
on `chimera-repair`; `origin/chimera-repair` was
`d82e46114674c77407f20379a1b9a76cb5d6204c`. No user changes were present.
The runtime was Python 3.12.10, PyTorch 2.13.0+cpu, Windows 11 x64, with CUDA
unavailable. The baseline passed `compileall`, `scripts/validate_repo_contract.py`,
`git diff --check`, and pytest (**212 passed, 41 warnings**).

The earlier report above reflects the previous cleanup pass's inventory and
must not be read as the after-count for this pass. This pass started with 679
tracked files: 92 Python files, 34 test files, 19 scripts, 564 data files,
8 documentation files, and 8 root files. The 4 installer scripts and their
references are the only repository files removed in this pass:

| Inventory measure | Before this pass | After this pass |
| --- | ---: | ---: |
| Tracked files | 679 | 675 |
| Python files | 92 | 92 |
| Test files | 34 | 34 |
| Script files | 19 | 15 |
| Data files | 564 | 564 |
| Documentation files | 8 | 8 |
| Root-level files | 8 | 8 |

The Python total is unchanged because the removed wrappers were shell and
PowerShell files. The installation contract is now the package's direct
`pip install .` / `pip install -e .` path, which is tested from an isolated
environment; `scripts/check_install.py` remains a non-mutating diagnostic.
The four wrappers had no CI or code callers; `INSTALL.md` was their only user
workflow entry. The older `install.sh` and `install.ps1` also advertised a
nonexistent `.[md]` extra. `INSTALL.md` was rewritten to show explicit
Windows/Linux environment creation and package installation commands.

### Tagged `components.py` symbol disposition

The tagged file contains a cohesive collection of neural building blocks, not
an alternative composition root. Exact source references and callers were
checked; none of the model layers was safe to delete.

| Symbol | Role | Disposition |
| --- | --- | --- |
| `TriangularPairUpdateConnector` | Canonical and legacy pair representation update; owns triangular attention/multiplicative modules and optional retrieved-context path. | Keep; instantiated in both architecture compositions and used in inference/training. |
| `TriangularAttention` | Outgoing/incoming pair attention used by the connector. | Keep; instantiated by canonical and legacy connector constructors. |
| `TriangularMultiplicativeUpdate` | Outgoing/incoming triangle-indexed pair update used by the connector. | Keep; instantiated by canonical and legacy connector constructors. |
| `EvolCrossAttentionConnector` | Flow-time-conditioned evolutionary cross-attention in the structural flow. | Keep; canonical/legacy composition and flow training use it. |
| `NodeProjectionConnector` | Projects MSA single representation for local sequence recovery. | Keep; canonical inference and sequence training use it; legacy composition also constructs it. |
| `NRPSConstraintEncoder` | Encodes domain, PPant and Stachelhaus annotations into conditioning features. | Keep; canonical constraint/flow conditioning and legacy composition use it. |
| `MSARepresentationBackbone` | Local row/column attention representation and masked reconstruction. | Keep; canonical inference and representation training use it. |
| `EvoFormerBackbone` | Historical alias for `MSARepresentationBackbone`. | Keep as compatibility: `chimera_v2.py` and masking tests import it, with an explicit identity contract. |
| `ProteinMPNNBackbone` | Local residue-feature trunk, distinct from native ProteinMPNN. | Keep; canonical and legacy compositions use it. Its ignored legacy `backbone_coords`/constructor `edge_features` parameters are documented compatibility surface, not claims of geometric input use. |

Two constructor/configuration details are intentionally retained, not
misrepresented as active settings: `MSARepresentationBackbone.n_blocks` is
accepted but the current implementation fixes the row encoder at eight
layers; `ProteinMPNNBackbone.edge_features` is accepted but the local trunk
does not consume edge features. Both are candidate API-cleanup items requiring
an explicit deprecation/compatibility decision before changing signatures.
The canonical geometric graph is consumed later by `MultiScaleNRPSDesigner`.

### Installation, exact duplicates, and remaining candidates

The exact-content scan found these duplicate byte pairs:

* `data/fath2011_codon_training.jsonl` and
  `data/fath2011_codon_training_measured.jsonl`
* `data/external/ensembl_cds_cache/ENST00000394805.txt` and
  `data/external/ensembl_cds_cache/ENST00000618441.txt`

No duplicate was deleted. The first pair has distinct manifest identities and
dataset/provenance labels even though its bytes match; the cache pair may
represent two accession records that happen to have identical sequence text.
Removing either would require a deliberate data-manifest/provenance migration,
which is outside a source-only cleanup and has no demonstrated benefit.

`docs/PIP_SETUP_WINDOWS.md` remains an optional upstream-backend research note,
not an installation authority. It documents optional OpenFold/RFdiffusion
setup and makes explicit that those checkpoints do not load into CHIMERA's
local approximations. Its hard-coded local paths are historical and are not
used as package configuration. A general rewrite or deletion was deferred
because the upstream-specific setup steps are unique research material.

No component layer, scientific primitive, dataset, legacy class, adapter,
production mechanism, test, or model manifest was removed in this pass. The
previously documented release blocker remains unchanged.
