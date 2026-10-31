# CHIMERA Launch & Inference Harness: Implementation Plan

**Target branch:** `chimera-main`  
**Planning branch:** `chimera-launch-inference-plan`  
**Status:** The core checkpoint-driven training-to-inference and bounded synchronous design-run path are implemented on this branch. The remaining sections describe future work unless explicitly marked implemented in the design-run specification.

## 1. Objective

Build a first-class launch and inference harness for PSC-CHIMERA that sits around the canonical inference engine. It must make model launch, readiness checks, inference execution, validation, resource handling, and artifact provenance repeatable and auditable.

The harness is an operational system, not a claim of scientific or biological validation. A successful launch proves that the selected runtime path executed; it does not prove that generated proteins are functional.

## 2. Current foundations to preserve

Reuse rather than duplicate these existing components:

- `chimera/inference.py`: `InferenceRequest`, `InferenceResult`, `ValidationOptions`, and `run_inference()`.
- `chimera/configuration.py`: versioned `InferenceConfig`, canonical serialization, and configuration identity.
- `chimera/checkpoint.py`: checkpoint manifests, runtime/source provenance, state-schema identity, and compatibility validation.
- `chimera/errors.py`: typed domain errors and corrective-action metadata.
- `chimera/cli.py`: existing CLI parser and JSON/error conventions.
- `chimera/model_store.py`: declared model-asset inspection, fetching, and integrity verification.
- `chimera/candidate_validation.py`, `chimera/input_contract.py`, `chimera/readiness_policy.py`, and `chimera/composition_manifest.py`: existing validation, input, readiness, and component-composition contracts.
- `scripts/run_design.py`: legacy/design-oriented path to audit and clearly label; do not silently treat it as equivalent to canonical `run_inference()`.

Before implementation, inspect the current exports in `chimera/__init__.py`, package entry points, production-gate checks, and existing inference tests. Avoid duplicating checks already implemented.

## 3. Proposed module/file plan

### Phase 0: contract audit and design freeze

| File | Action | Deliverable |
|---|---|---|
| `chimera/inference.py` | Audit, minimally refactor only if required | Document exact runtime contract and known limitations |
| `chimera/__init__.py` | Audit exports | Stable public API for the harness |
| `chimera/cli.py` | Audit and extend later | Preserve existing command behavior and JSON output |
| `setup.py` / packaging configuration | Locate actual entry-point declaration before editing | Add commands only through the repository's existing packaging mechanism |
| `chimera/production_gate.py` | Audit | Distinguish engineering checks from inference smoke tests and scientific evaluation |
| `tests/test_inference_contract.py` | Audit current coverage | Reuse existing fixtures and avoid redundant tests |
| `docs/PRODUCTION_ENGINEERING.md` | Update after behavior is implemented | Accurate user-facing capability and limitation documentation |

**Exit criteria:** a reviewed map of the actual call graph, checkpoint formats, input preparation requirements, model-loading routes, and current tests. No guessed CLI syntax is shipped.

### Phase 1: preflight and readiness

| Proposed file | Responsibility |
|---|---|
| `chimera/launch/preflight.py` | Check Python/PyTorch versions, required dependencies, selected device, CUDA visibility and basic runtime capability |
| `chimera/launch/hardware.py` | Report CPU, RAM, GPU names, VRAM and relevant runtime versions; provide estimates, not false guarantees |
| `chimera/launch/checkpoint_inspection.py` | Inspect checkpoint path, safe deserialization format, manifest, tensor schema and compatibility without implying that weights are trained |
| `chimera/launch/readiness.py` | Combine checks into explicit readiness findings with severity, evidence and corrective action |
| `tests/launch/test_preflight.py` | Unit tests for missing dependencies, CPU-only systems, unavailable CUDA and malformed environment data |
| `tests/launch/test_checkpoint_inspection.py` | Missing, corrupt, incompatible, unverified-provenance and valid fixture cases |

Reuse existing checkpoint validation. Do not create a second, weaker checkpoint-compatibility implementation. Preflight should return structured JSON and meaningful exit codes.

### Phase 2: launch profiles and runtime lifecycle

| Proposed file | Responsibility |
|---|---|
| `chimera/launch/profile.py` | Validated launch profile: model/checkpoint references, device, precision, resource budget and policy |
| `chimera/launch/runtime.py` | One controlled path to resolve configuration, instantiate/load the intended model and prepare it for inference |
| `chimera/launch/session.py` | Lifecycle states: created, preflighted, loaded, ready, running, completed, failed, cancelled |
| `chimera/launch/errors.py` | Only add launch-specific errors if the existing shared error hierarchy cannot represent the failure cleanly |
| `tests/launch/test_runtime.py` | Verify correct loader selection, strict loading, device placement, evaluation mode and failure cleanup |
| `tests/launch/test_session.py` | Validate legal state transitions and terminal states |

Do not guess model architecture from a checkpoint filename. Require explicit model identity/configuration and route each supported checkpoint type through its established loader. Keep unverified component-transfer checkpoints distinguishable from canonical trained checkpoints.

### Phase 3: unified inference job orchestration

| Proposed file | Responsibility |
|---|---|
| `chimera/launch/jobs.py` | Define inference-job request, identifiers, status and result references |
| `chimera/launch/runner.py` | Orchestrate preflight → load → validate inputs → call canonical inference → validate outputs → persist artifacts |
| `chimera/launch/progress.py` | Structured progress events and stage timings |
| `chimera/launch/cancellation.py` | Cooperative cancellation at safe boundaries |
| `tests/launch/test_runner.py` | End-to-end orchestration with a tiny test model/checkpoint and prepared inputs |
| `tests/launch/test_progress_cancel.py` | Progress ordering, cancellation boundaries and cleanup |

The runner must call the canonical inference implementation instead of reimplementing generation. It must explicitly identify the chosen inference backend. Do not silently switch between `run_inference()` and `scripts/run_design.py`.

**Initial execution model:** synchronous and single-process, with microbatching delegated to the existing inference API. Add process isolation only if hard cancellation or memory containment is later required and can be tested reliably.

### Phase 4: artifacts, observability and recovery

| Proposed file | Responsibility |
|---|---|
| `chimera/launch/artifacts.py` | Create-only run directory, atomic writes, checksums, manifest and output inventory |
| `chimera/launch/events.py` | JSON Lines events for lifecycle transitions, progress, warnings and failures |
| `chimera/launch/inspection.py` | Inspect a prior run from its manifest and artifacts without rerunning it |
| `tests/launch/test_artifacts.py` | Atomicity, existing-file protection, checksum verification and partial-write handling |
| `tests/launch/test_recovery.py` | Interrupted run detection and safe restart behavior |

Persist at minimum: run ID and status; software version and Git/source identity; config ID and canonical config; checkpoint identity and compatibility status; prepared-input identity/manifest; seed and determinism settings; runtime/device details; start/end times and stage timings; candidate count; validation results; warnings/errors; output paths and hashes.

Do not store raw sensitive inputs by default. Record input hashes and declared provenance, and make any input retention explicit. A restart must not silently claim to resume a sampler trajectory unless sampler state is actually persisted. The first version may safely restart a job rather than offer unsupported mid-kernel resume.

### Phase 5: CLI surface

Extend `chimera/cli.py` after the APIs above are stable. Proposed commands, not yet implemented:

- `chimera launch preflight --profile <profile>`
- `chimera launch inspect --checkpoint <path>`
- `chimera launch smoke-test --checkpoint <path> --fixture <fixture>`
- `chimera infer validate-inputs --input-bundle <path>`
- `chimera infer run --config <json-path>`
- `chimera infer inspect-run --run-id <id> --runs-dir <path>`
- `chimera infer validate-output --run-id <id> --runs-dir <path>`

Before exposing these commands, confirm how the project currently packages CLI entry points. Keep existing `chimera version`, `models`, and production-gate commands backward-compatible. Output machine-readable JSON by default, with optional human-readable summaries. Errors must use existing structured error/exit-code conventions.

### Phase 6: benchmark and scientific evaluation integration

| Proposed file | Responsibility |
|---|---|
| `chimera/evaluation/inference_benchmark.py` | Execute a versioned held-out inference suite through the same runner |
| `chimera/evaluation/report.py` | Report runtime, validity, failure rates, diversity and declared metrics |
| `tests/evaluation/test_inference_benchmark.py` | Test split identity, leakage guards, report schema and deterministic fixtures |
| `docs/CHIMERA_LAUNCH_INFERENCE.md` | User guide, examples, limitations and troubleshooting |

Do not collapse engineering readiness and scientific readiness into one green check. Benchmark reports must label computational proxies as proxies, describe the dataset split and baseline, and state that in-silico performance does not establish experimental function.

## 4. Runtime contract and limitations to make explicit

The current canonical inference contract expects caller-prepared tensors, including MSA tokens, pair features and source backbone frames. The harness should validate and identify this prepared input bundle; it should not pretend that preprocessing is already automatic.

The current engineering guide describes these limitations that must remain visible unless implementation and tests change them:

- Canonical inference is currently restricted to float32.
- The supported sampler is the declared stochastic Euler–Maruyama path.
- Timeout checks occur between microbatches and cannot interrupt a long-running kernel.
- Nonzero worker counts and requested memory limits currently fail explicitly rather than being enforced.
- Retrieval identity does not by itself activate RAG; the current canonical inference path must not claim retrieval was used when it was not.
- Native RFdiffusion, OpenFold, ESMFold and ProteinMPNN are not automatically invoked by the canonical inference API.
- Component-transfer checkpoints may load strictly while retaining unverified training provenance.

These are not necessarily permanent restrictions. They are contract boundaries until corresponding implementation and tests prove otherwise.

## 5. Readiness levels

Use separate machine-readable statuses:

1. **Runtime ready:** environment and requested device are usable.
2. **Checkpoint compatible:** architecture and state schema match; provenance/training status remains separately reported.
3. **Inference smoke test passed:** a small fixture completed and produced validated outputs.
4. **Benchmark evaluated:** versioned held-out benchmark and baseline results exist.
5. **Experimental validation:** external wet-lab evidence is attached and independently described.

No earlier status may imply a later one.

## 6. Required test matrix

- CPU-only environment and CUDA-enabled environment when available.
- Missing, corrupt, incompatible and provenance-incomplete checkpoints.
- Input shape/dtype/range failures.
- Finite output and geometry-validation failures.
- Same-seed reproducibility and preservation of caller RNG state.
- Small batch and microbatch equivalence within declared tolerance.
- Device unavailable, OOM, timeout warning and cooperative cancellation.
- Atomic artifact creation, checksum mismatch and pre-existing output protection.
- RAG requested but unavailable, with explicit status rather than a false success.
- CLI JSON schema, exit codes and backward compatibility.
- End-to-end fixture from request through candidate validation and manifest creation.
- Full existing inference, checkpoint, architecture and production-gate regression suites.

GPU-specific tests should be conditional on hardware availability. The default CI suite must use small fixtures and must not instantiate the full-size model unnecessarily.

## 7. Suggested implementation order and completion gates

1. **Audit:** document actual APIs, loaders, packaging entry points and existing tests.
2. **Preflight:** environment/device/checkpoint diagnostics with structured findings.
3. **Runtime:** one explicit loader and a verified smoke-test fixture.
4. **Runner:** single unified job path that calls `run_inference()`.
5. **Artifacts:** durable manifests, logs, hashes and safe restart semantics.
6. **CLI:** expose stable commands after APIs are tested.
7. **Evaluation:** benchmark integration and separate scientific-readiness reporting.

Each phase should be a small reviewable change with tests. Do not label the harness production-ready until all required tests have run in the target environment and the known limitations are either enforced or explicitly reported.

## 8. Definition of done

- A user can run preflight and receive actionable, machine-readable results.
- A checkpoint can be inspected without conflating compatibility with training quality.
- One documented command can run a small controlled inference using a declared configuration and prepared inputs.
- The run produces validated candidates or a structured failure, never a silent partial success.
- Every run has a manifest linking source, checkpoint, config, inputs, runtime and outputs.
- Existing CLI and inference contracts remain compatible.
- CPU CI passes; GPU checks run where hardware is available.
- Documentation distinguishes software execution, benchmark performance and experimental evidence.

## 9. First implementation PR recommendation

Start with **Phase 0 plus Phase 1 only**: audit the call graph and implement preflight/checkpoint inspection with tests. This establishes trustworthy launch diagnostics before adding a larger orchestration layer. Keep all work on a feature branch and merge through review; do not modify the protected/mainline branch directly.
