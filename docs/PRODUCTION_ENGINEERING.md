# Production engineering status

## Stage and source identity

This work preserves the frozen architecture at
`5c9ea0ac6b910f35f687def079c1d79235b57c9c`. It does not constitute biological
validation. The inspected starting state is recorded in
[`PRODUCTION_ENGINEERING_BASELINE.md`](./PRODUCTION_ENGINEERING_BASELINE.md).

The intended boundary remains:

```text
Research Prototype
    -> Pre-Production Engineering
    -> Production Engineering
    -> Biological Validation
```

Software correctness, checkpoint integrity, architecture compatibility,
scientific validation, and biological validation are separate claims.

## Packaging, version, and imports

Install with `pip install .`; use `pip install -e '.[dev]'` for editable
development. `chimera._version.__version__` is the single source used by the
package, setup metadata, checkpoints, and CLI. `python -m chimera.cli --help`
and the installed `chimera --help` expose the CLI.

Package smoke tests use an isolated interpreter mode from a temporary working
directory so the repository root is not implicitly added to imports. The design
command remains available as `chimera-design`; its source file no longer
manipulates `sys.path`.

The declared Python floor is 3.10. CI currently tests 3.10 and 3.11; the
baseline engineering environment ran Python 3.12. PyTorch/CUDA wheels remain
platform-specific. The base install includes PyTorch, NumPy, and einops.
Optional workflow dependencies are isolated in extras; the architecture
inventory records runtime scope and verification status.

## Configuration and inference

`InferenceConfig` is a versioned JSON-only value object with a canonical JSON
representation and SHA-256 `config_id`. It records model/artifact/checkpoint
identities, preprocessing, device/dtype, sampler/steps, batch/microbatch and
length policy, objective set, retrieval identity, seed/determinism, CPU threads,
workers, memory limit, and timeout. Unsupported Python objects and non-finite
numbers are rejected.

**Current limitation:** the legacy `CHIMERAv2.design(...)` entry point does not
consume `InferenceConfig` and does not yet expose a stable structured inference
result. Existing CLI inputs are validated in its design flow, but no complete
production request/result contract, inference telemetry, or independently
executable model-serving path is established. Consequently the production gate
reports configuration, inference, and determinism checks as `UNAVAILABLE`.

The existing deterministic helper seeds Python, NumPy, PyTorch, and CUDA and
sets PyTorch deterministic flags when requested. Same-seed reproducibility for
the complete production inference path has not been demonstrated; no
cross-environment or bitwise reproducibility claim is made.

## External model store and integrity

`chimera/model_dependencies.json` is the versioned external dependency lock.
`chimera models list|fetch|inspect|verify` uses identity-addressed cache
directories under `CHIMERA_MODEL_CACHE` (legacy
`PSC_CHIMERA_MODEL_CACHE` is also read). Fetches are explicit; imports and
installation do not fetch weights.

Cache manifests record upstream project/revision/release, source URL, variant,
architecture, license, file size, local SHA-256, checksum source, timestamps,
compatibility/native status, recorded smoke evidence, and a hash over manifest
contents. Fetches stage and verify before atomic directory publication; an
existing artifact identity is never overwritten. `models verify --offline`
only reads local files.

No published upstream SHA-256 was found for the listed checkpoint URLs. In
that case, local SHA-256 verifies byte integrity against the local manifest,
not upstream origin. Compatibility remains `UNKNOWN` until a canonical
upstream loader and adapter smoke establish it; successful cached native smoke
tests persist local evidence for the gate to inspect. Local files in the repository's
legacy `Models/` and `weights/` folders are unmanifested; their measured hashes
and status are in [`pretrained_models.md`](./pretrained_models.md).

The actual upstream acquisition pass confirmed native CPU integration for
ESM-2 through the CodonOptimizer encoder and ProteinMPNN through the pinned
native adapter. Both use locally recorded hashes because upstream SHA-256
values are unpublished. RFdiffusion's checkpoint is acquired and structurally
inspected, but native execution is `UNAVAILABLE` with this host's CPU-only
PyTorch and incompatible pinned CUDA/DGL environment. ESMFold remains `MISSING`
because its 2.77 GB checkpoint could not be staged on the available disk. The
AlphaFold parameter archive is not an EvoFormer checkpoint. Exact hashes,
runtime evidence, and these distinctions are maintained in
[`pretrained_models.md`](./pretrained_models.md).

The model store is a local filesystem registry for upstream dependencies, not
a registry for CHIMERA model artifacts. The machine-readable architecture
inventory reports that no CHIMERA production composition or trained artifact
is selected. The identified duplicate/untrusted files in ignored `Models/` and
`weights/` paths were removed; their hashes and dispositions are retained in
[`pretrained_models.md`](./pretrained_models.md), and the gate rejects their
reappearance.

`chimera production-dependencies` prints the production closure and current
release blockers as JSON. At present it reports a blocked research candidate,
not an executable production release. The available 588-row codon dataset
contains measured and proxy labels but does not supply the MSA/structure
training examples and independent held-out calibration required by the
canonical representation, flow, sequence, and objective components. It is
therefore not a valid source from which to manufacture a production checkpoint.

## CLI and production gate

```console
chimera --version
chimera models list
chimera models fetch esm2_t30_150m
chimera models fetch esm2_t30_150m_contact_regression
chimera models verify --offline
chimera production-dependencies
chimera production-gate --root C:\path\to\psc-chimera --output C:\path\to\evidence.json
```

The gate does not accept a caller-constructed evidence object. It executes:

- Git source, branch, and worktree inspection;
- an offline, no-dependency installation into a temporary target followed by an
  isolated import/version, console-script, and packaged model-inventory checks
  from outside the source tree;
- `python -m compileall -q chimera data scripts`;
- `python -m pytest -q`, parsing pass/fail/error/skip counts and treating skips
  as non-passing;
- external lock schema validation and local cache hash/manifest verification;
- architecture dependency graph, runtime records, data hashes, and reconciled
  workspace model-file policy;
- local CPU tensor smoke and CUDA tensor smoke when CUDA is available;
- checkpoint-format/migration-policy validation.

`--root` is required: the gate must know which source checkout to execute and
must not guess from the current working directory or write into the installed
package directory. The output is a create-only JSON evidence record with timestamp, source,
runtime/dependency versions, test command and counts, dependency manifest hash,
per-check statuses, explicit unavailable fields, and a SHA-256 evidence
identity. Existing evidence paths are never overwritten. The hash detects
accidental/tamper changes but is not a digital signature against an attacker
able to rewrite both evidence and hash.

There is no selected release checkpoint, CHIMERA artifact, or production
inference configuration in the repository. Accordingly checkpoint
compatibility, artifact integrity for a selected model, complete provenance,
inference smoke, and deterministic smoke are emitted as `UNAVAILABLE` and
block a release `PASS`. This is currently a training/data-adequacy blocker,
not a missing ESM-2, ProteinMPNN, RFdiffusion, ESMFold, or OpenFold production
dependency: none of those external foundation models is required by the
selected local approximation candidate. Optional missing external model
assets are listed individually; they are not reported as successfully
installed. This gate currently establishes engineering evidence collection,
not a releasable model.

`preproduction_readiness_report(PreProductionGateEvidence(...))` remains for
backwards compatibility as an evidence-summary API. It is not the production
gate and its caller-supplied booleans are not execution evidence.

## Trust boundary and errors

Production-facing tensor-only checkpoint paths use PyTorch's
`weights_only=True` loader; loading arbitrary Python objects from a checkpoint
is not required for inference. Full resumable training checkpoints currently
use `weights_only=False` to restore optimizer and RNG state. Only load those
from a trusted source. Upstream subprocess adapters use argument arrays rather
than a shell and validate output tensor mappings; upstream runner code and
checkpoints remain trusted external software.

The public error module defines configuration, artifact/integrity,
checkpoint-compatibility, input-validation, unsupported-runtime, resource,
inference, and provenance categories. The model acquisition CLI emits
machine-readable errors. Existing scientific APIs are not yet consistently
migrated to this taxonomy.

Structured run logging, latency/resource telemetry, comprehensive input
validation, OOM-specific recovery guidance, and a stable inference failure
schema remain outstanding. No silent model fallback was added. Local
approximations are explicitly distinguished from native ESM/OpenFold,
ProteinMPNN, and RFdiffusion.

## Test and performance policy

The project currently has one main pytest suite; CI tests CPU behavior on
Python 3.10 and 3.11 and excludes GPU/weight-dependent tests. The new focused
contracts cover canonical configuration, offline model inventory, local
manifest/hash detection, CLI semantics, evidence immutability, and isolated
package imports. GPU and real-upstream-checkpoint adapter tests remain
environment-dependent and unavailable without the corresponding hardware,
upstream source trees, and verified artifacts.

The measured pre-change baseline was `177 passed`, `0 failed`, `0 skipped` in
102.60 seconds on Python 3.12 in this workspace. That is a test-suite timing,
not a model performance benchmark. Startup, construction/load, preprocessing,
inference, postprocessing, memory, batch scaling, and sequence-length scaling
have no production artifact/configuration with which to measure them and
remain unavailable. No performance optimization or biological performance
claim is made.

## Release policy and limitations

An engineering release candidate must identify software version, clean source
commit, configuration identity, CHIMERA artifact/checkpoint identity,
external dependency manifest, environment, and self-verifying gate evidence.
The current gate cannot issue a release PASS while required evidence is
unavailable. Model acquisition alone does not establish adapter compatibility,
scientific validity, or biological validation.
