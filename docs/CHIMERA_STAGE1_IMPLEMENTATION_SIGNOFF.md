# CHIMERA Stage 1 implementation sign-off record

## Scope

This record tracks implementation correctness for the canonical research composition on
`chimera-main`. It is not a biological-validation claim and does not declare CHIMERA
production-ready.

Audited source revision: `2f6f42dd91433f6185c0e61225078bd49295e829`
Audited: 2026-10-27

## Findings

| Area | Current source evidence | Status |
|---|---|---|
| Canonical composition | `chimera/architecture.py` instantiates the public canonical composition and owns the explicit component inventory. | Source inspected |
| EvoFormer / representation | `chimera/components.py` defines the canonical MSA/pair stack. `tests/test_evoformer_masking.py` covers padding invariance, MSA information exchange, and reconstruction gradients. | Focused tests exist; execution must be confirmed on the audited revision |
| Training-stage ownership | `chimera/training.py` explicitly maps representation, flow, sequence, constraint, objective, and preference regimes to trainable modules. `CanonicalTrainer.train_step` checks required gradient paths before optimizer updates. | Source inspected; focused tests exist |
| Frozen EvoFormer behavior | The sequence regime freezes the shared EvoFormer by default and fingerprints its state. Flow/constraint training obtains representations through `CanonicalTrainer._conditioning` without `no_grad`, allowing gradients where the regime requires them. The `no_grad` block in the inference-oriented canonical forward path is separate from those trainer loss paths. | Earlier concern narrowed; not a demonstrated current defect |
| Checkpoint compatibility | `from_pretrained` constructs the requested architecture and loads component state dictionaries strictly. Therefore the configured EvoFormer depth must match the checkpoint depth; strict loading rejects mismatches. A compatible load is recorded as `loaded_unverified`, not as trained or validated. | Fail-closed behavior present; regression tests added |
| Forward / generation | `tests/test_canonical_architecture.py` exercises a synthetic forward/backward path and experimental design output. `scripts/audit_full_size_model.py` is a full-default-size construction, one-step synthetic training-contract, and tiny generation smoke audit. | Existing test/audit coverage; current execution not verified here |
| Checkpoint provenance | Training/checkpoint code records configuration, dataset, preprocessing, optimizer and runtime provenance. A source-compatible checkpoint alone does not establish that it was trained well or validated. | Mechanisms present; a selected scientifically validated artifact is still absent |
| Reproducible baseline | Existing scripts and tests provide building blocks, but this audit did not execute the complete suite, record the environment, or capture fresh test output on this revision. | **Open** |

## Regression added

`tests/test_stage1_signoff.py` adds two focused cases:

1. A checkpoint from a two-block EvoFormer must fail closed when loaded into a three-block configuration.
2. A checkpoint with matching depth may load, but its status must remain `loaded_unverified`, with training status `not_started` and model readiness `UNINITIALIZED`.

These tests intentionally separate shape compatibility from evidence of training quality.

## Verification limitation

The repository connector permits source inspection and commits, but no test runner was available through this audit. The newly added tests and the existing suite have **not** been executed in this session. No passing result is claimed.

Run on the target checkout:

```bash
python -m pytest -q tests/test_stage1_signoff.py tests/test_evoformer_masking.py tests/test_canonical_architecture.py
python scripts/audit_full_size_model.py
```

Capture the exact commit SHA, Python/PyTorch versions, device, command lines, exit codes, and complete test summaries. The full-size audit can be memory-intensive; run it only on a machine with sufficient RAM and expect the CPU version to take time.

## Sign-off decision

**Stage 1 is substantially implemented, but not signed off by this record.** Sign-off requires fresh passing output for the focused tests and relevant full suite on the exact commit, review of any failures, and a recorded environment. Biological validity, trained-checkpoint quality, leakage-resistant benchmarking, and experimental NRPS function remain separate downstream claims.
