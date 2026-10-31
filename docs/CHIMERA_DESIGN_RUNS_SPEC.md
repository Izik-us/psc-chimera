# CHIMERA Design Runs: Functional Specification

**Branch:** `chimera-launch-inference-plan`  
**Status:** The bounded synchronous canonical-inference path is implemented on this branch. Candidate clustering, diverse shortlist export, domain-specific NRPS evaluators, benchmark orchestration, and cancellation remain outside this milestone.

## 1. Purpose

A design run is a bounded, auditable computational job that turns one explicit design task and a declared model/checkpoint into candidate outputs plus computational validation, proxy scoring, provenance, and a final report.

The Design Run Manager orchestrates the existing canonical inference API; it does not implement a second model-generation path or silently substitute the legacy `scripts/run_design.py` path.

For CHIMERA's NRPS module/domain-linker compatibility objective, a run should record the exact design context and constraints, the candidates produced, the validators and scoring functions applied, and the evidence supporting each ranking. Computational scores are hypotheses/proxies, not proof of biochemical compatibility.

## 2. Proposed lifecycle

```text
Draft task
  -> Task/schema validation
  -> Preflight + checkpoint compatibility
  -> Input bundle validation
  -> Run plan and resource estimate
  -> Candidate generation
  -> Candidate normalization and deduplication
  -> Structural/sequence/constraint validation
  -> Scoring and uncertainty reporting
  -> Candidate selection/export
  -> Immutable run manifest + final report
```

Every transition should emit a structured event. A failed stage should yield a typed failure and preserve diagnostic artifacts where safe. The run must never label a partial candidate batch as complete.

## 3. Proposed module plan

| File | Responsibility |
|---|---|
| `chimera/design_runs/schema.py` | Versioned `DesignTask`, `DesignRunConfig`, `CandidateRecord`, and `DesignRunReport` schemas |
| `chimera/design_runs/planner.py` | Validate task, resolve configuration/checkpoint/input identities, estimate candidate count and resources, produce a dry-run plan |
| `chimera/design_runs/manager.py` | Orchestrate run lifecycle and state transitions |
| `chimera/design_runs/generation.py` | Adapter to the canonical inference API; isolate backend differences without duplicating generation |
| `chimera/design_runs/constraints.py` | Declare and evaluate hard constraints separately from soft objectives |
| `chimera/design_runs/validation.py` | Compose sequence, structure, geometry and input-specific validators |
| `chimera/design_runs/scoring.py` | Apply versioned scoring functions and preserve per-metric values and uncertainty |
| `chimera/design_runs/deduplication.py` | Detect exact duplicates and configurable sequence/structure similarity clusters |
| `chimera/design_runs/selection.py` | Select a diverse candidate shortlist under explicit rules |
| `chimera/design_runs/artifacts.py` | Persist candidate records, run manifest, logs, reports and hashes atomically |
| `chimera/design_runs/inspection.py` | Inspect, summarize, validate or export an existing run without rerunning it |
| `chimera/design_runs/cli.py` | Optional command handlers if they fit better than extending the current CLI directly |
| `tests/design_runs/` | Schema, planner, lifecycle, adapter, validation, scoring, deduplication and end-to-end tests |

Before creating these files, Phase 0 must inspect existing modules such as `chimera/candidate_validation.py`, `chimera/input_contract.py`, `chimera/composition_manifest.py`, `chimera/inference.py`, and the current CLI. Reuse existing contracts and avoid duplicate validators or manifests.

## 4. Design task contract

A versioned design task should explicitly identify:

- Task ID and task-schema version.
- Design objective and target context.
- Input bundle path or stable reference, hash, provenance, and preprocessing identity.
- Checkpoint/model identity and expected architecture configuration.
- Hard constraints, soft objectives, thresholds and metric versions.
- Requested candidate count, sampling configuration, random seed policy and batching limits.
- Retrieval configuration, if requested, and a record of whether retrieval was actually used.
- Allowed tools/backends and explicit policy for unavailable optional components.
- Output retention and export settings.

No field should imply a biological fact that the supplied inputs do not establish. For NRPS work, task metadata may include module/domain identity, substrate context, linker boundaries, required sequence/structure constraints and known evidence sources where available. The schema should not require speculative fields unsupported by the current model contract.

## 5. Candidate record

Each candidate record should contain:

- Stable candidate ID and generation ordinal.
- Sequence and/or structural artifact references as actually produced.
- Source run ID, seed, model/checkpoint identity and generation configuration.
- Input/task identity and constraints evaluated.
- Validation statuses and detailed failure reasons.
- Per-objective scores with metric version, units, calibration status and uncertainty if available.
- Deduplication/cluster assignment.
- Selection status and reason.
- Artifact hashes and provenance links.

Keep raw outputs immutable. Normalization, scoring and ranking should be separate derived artifacts so a new evaluator can rescore a candidate without regenerating it.

## 6. Hard constraints versus ranking

The system must distinguish three questions:

1. **Validity:** Is the output well-formed and compatible with the declared input/output contract?
2. **Feasibility:** Does the candidate satisfy the hard constraints for this task?
3. **Preference:** Among feasible candidates, which are most promising under the declared computational objectives?

A soft score must not override a failed hard constraint. If a candidate fails a required check, mark it infeasible and preserve the reason rather than hiding it through an aggregate score. If metrics conflict, expose the Pareto trade-offs or per-metric results rather than inventing a single authoritative ranking.

## 7. CLI

Implemented commands on this branch:

```bash
chimera design plan --task configs/design_task.json
chimera design prepare --task configs/design_task.json --runs-root runs/design
chimera design run --task configs/design_task.json --runs-root runs/design
chimera design inspect --run-dir runs/design/<run-id>
```

`run` creates an immutable planned run, verifies that the input bundle and checkpoint still match their planned hashes, validates the prepared tensor bundle, reconstructs the canonical model from the checkpoint, invokes `run_inference_from_checkpoint()`, and persists the complete inference result plus a compact report. It is synchronous and single-process. The task input bundle is a schema-versioned JSON object containing tensor arrays and a `PreparedInputManifest`; no raw biological preprocessing is synthesized.

Only `sequence_length` maximum/minimum hard constraints currently have an implemented evaluator. Any other declared hard constraint fails closed before generation. Soft objectives must be included in `inference_config.objective_set`. Retrieval remains explicitly unavailable in the current canonical inference backend. The command does not provide asynchronous progress polling, forceful cancellation, clustering, diversity-aware export, or experimental validation.

The `plan` command remains side-effect-light and does not load a model. `run` requires a bounded candidate count in the versioned task schema.

```bash
# Validate a task and show the planned work without generating candidates
chimera design plan --task configs/design_task.json

# Run a small smoke design against a test fixture
chimera design run --task configs/smoke_design.json --profile local

# Run a declared design task
chimera design run --task configs/nrps_task.json --profile gpu

# Inspect progress or the final report
chimera design inspect --run-id <run-id> --runs-dir runs/design
chimera design report --run-id <run-id> --runs-dir runs/design

# Validate outputs or export a shortlist
chimera design validate --run-id <run-id> --runs-dir runs/design
chimera design export --run-id <run-id> --top-k 20 --diversity-aware
```

A `plan` command should be side-effect-light and should not load a full-size model unless explicitly requested. `run` must require a valid task and must not default to an unbounded candidate count.

## 8. Run directory and reproducibility

Suggested layout:

```text
runs/design/<run-id>/
  task.json
  resolved_config.json
  plan.json
  manifest.json
  events.jsonl
  candidates/
    candidate-000001.json
    candidate-000001.sequence.fasta
    candidate-000001.structure.cif
  validation.jsonl
  scores.jsonl
  clusters.json
  report.json
  logs/
```

Only write artifacts that the selected backend actually produces. Do not create placeholder structure files or imply coordinates exist when they do not.

Manifest fields should include run ID/status, source revision, environment, checkpoint and config hashes, input bundle hash, seed and deterministic settings, backend, requested and completed candidate counts, stage timings, warnings, errors, evaluator identities, output hashes, and explicit retrieval-used status. Avoid storing sensitive raw input data by default.

Use atomic writes and create-only semantics for immutable artifacts. A rerun should create a new run ID unless the system can prove it is resuming the same compatible state. Do not promise bitwise reproducibility across different hardware or nondeterministic kernels.

## 9. Failure and partial-run policy

- Preflight or task-validation failure: no candidate generation.
- Model load failure: mark run failed and retain diagnostics.
- Candidate-level validation failure: preserve the candidate with a failed status and reason; do not fail the entire run unless the failure indicates a systemic defect.
- Batch/runtime failure: mark run incomplete; record generated count and stage reached.
- Cancellation: mark cancelled only after control returns to a safe boundary.
- Timeout: report whether it was a cooperative stage-boundary timeout; do not claim a running GPU kernel was forcibly interrupted.
- Artifact write failure: never report successful completion if required manifest/report persistence failed.
- Missing RAG, structure evaluator or optional dependency: fail or degrade only according to explicit task policy, and record the actual behavior.

## 10. Evaluation and selection

Version each evaluator and its configuration. Separate generic validity metrics from domain-specific metrics. For NRPS compatibility, report the exact evidence available for interface geometry, sequence constraints, substrate context and other supported metrics. If a metric is an uncalibrated learned score or a heuristic, label it as such.

Shortlisting should consider feasibility, score uncertainty and diversity. Avoid ranking thousands of near-identical candidates as independent discoveries. Maintain a complete candidate ledger so top-k export does not erase failures or rejected candidates.

A design-run report should include:
- Requested, generated, valid, feasible, unique and selected counts.
- Failure rates grouped by stage and reason.
- Per-metric score distributions and uncertainty summaries where supported.
- Diversity/cluster summary.
- Runtime and resource usage observations.
- Baseline comparison and held-out benchmark identity, when available.
- Clear limitations and a statement that computational evaluation does not establish experimental function.

## 11. Test plan

1. Schema versioning and invalid-task rejection.
2. Dry-run planner with bounded candidate counts.
3. Canonical inference adapter contract tests.
4. Hard-constraint rejection and soft-objective ranking tests.
5. Candidate-level failure does not corrupt other records.
6. Exact and near-duplicate detection on small fixtures.
7. Ranking stability and explicit tie handling.
8. Missing evaluator/RAG policy behavior.
9. Interrupted run and atomic artifact writes.
10. Manifest integrity and output hash verification.
11. End-to-end tiny fixture from task JSON to report and candidate ledger.
12. Regression tests for existing inference API and legacy design command.

Use mocked or tiny model fixtures for default CI. Hardware tests should be conditional and separately reported.

## 12. Implementation sequence

1. Audit current `scripts/run_design.py`, `model.design()`, `run_inference()`, validation, checkpoint and CLI contracts.
2. Define schemas and planner/dry-run command.
3. Implement run manager and canonical inference adapter.
4. Add candidate ledger, validators, explicit hard constraints and per-metric scoring.
5. Add durable artifacts, inspection and export.
6. Add progress/cancellation semantics consistent with what the runtime can enforce.
7. Add benchmark integration and domain-specific NRPS evaluation only where the underlying metrics are implemented and testable.

## 13. Definition of done

A user can submit a versioned design task, dry-run the plan, execute a bounded canonical inference job, inspect its durable status and report, and reproduce the run's provenance from its manifest. Candidate-level validation and per-objective proxy scores are retained in the immutable inference-result artifact. Unsupported hard constraints fail closed. Failures are explicit, partial runs are never mislabeled as complete, and scientific claims are limited to the evidence actually produced. Clustering, diverse shortlist export, benchmark integration, and experimental validation are not claimed by this milestone.
