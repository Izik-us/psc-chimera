# Pre-Production Engineering Gate

## Stage Boundary

```text
Research Prototype
        -> Pre-Production Engineering Gate
        -> Production Engineering
        -> Biological Validation
```

The engineering gate evaluates software contracts only. A passing gate does not establish biological validity, experimental predictive performance, clinical utility, or readiness for biological deployment. The current local MSA row/column model, custom SE(3) Schrödinger bridge, and ProteinMPNN-inspired sequence-recovery module retain their documented approximation boundaries.

## Gate Evidence

The independent API is `chimera.preproduction_readiness_report`. Callers must supply evidence for:

- Full test suite and compile/diff checks.
- Architecture, checkpoint-compatibility, objective-contract, gradient-flow, and real checkpoint/resume smoke tests.
- Complete checkpoint provenance and environment reproducibility.
- Documentation review.

Missing or failed evidence returns `BLOCKED`. Complete engineering evidence returns `PASS`, while the scientific status remains `NOT SCIENTIFICALLY VALIDATED`. This report does not bypass or weaken model inference-readiness checks.

Validation uses an iterable of held-out `CanonicalTrainingBatch` values. The default floor is 30 examples, matching the repository's existing minimum calibration evidence count. This is a conservative engineering threshold for avoiding one-batch validation, not a statistically or biologically validated sample-size claim. The configured minimum and observed counts are included in the validation record.

## Objective Limits

Four learned objective surrogates remain: evolutionary plausibility, structural stability, expression efficiency, and substrate selectivity. Their typed schema fixes feature sources, shape, dtype, activation, target scale/range, loss, provenance kinds, and semantic version. Structural features are local-frame invariant and residue streams use positional attention pooling; neither property validates the learned predictors scientifically.

Assembly compatibility is a deterministic inference-time interface-geometry proxy, not a learned head or an assembly free-energy predictor. It is not trained against its own evaluator output and is not an experimental measurement.

Structural-stability batches must mark structures as `reference`, `generated`, or `synthetic`. A predictor trained on reference structures and used on generated structures reports a distribution-shift risk. Backbone sanity checks provide a quality signal but do not demonstrate mitigation of that shift.

All learned objectives remain proxies/surrogates until their labels, held-out evaluation, and calibration are independently supported. No biological measurements are introduced by the engineering gate.

## Production Engineering Boundary

The next stage may address packaging, stable CLI/API configuration, artifact integrity and migration, deployment layout, inference serving, CPU/GPU execution, batching and resource management, performance, structured logs and telemetry, failure handling, deterministic inference, registries, security, containers, operations testing, and release/versioning policy. These deployment concerns are intentionally outside this gate.