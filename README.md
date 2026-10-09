# PSC-CHIMERA

**Compositional Hierarchical Inference Model for Evolutionary Representation and Architecture**

[![Tests](https://github.com/Izik-us/psc-chimera/actions/workflows/tests.yml/badge.svg)](https://github.com/Izik-us/psc-chimera/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c.svg)](https://pytorch.org/)

Stage 1 computational design prototype for the theoretical Pharmacosynthetic Constructor (PSC) engineering pipeline.

> **Research-status notice:** CHIMERA is an untrained research prototype by default. Its canonical evolutionary representation is a CHIMERA PyTorch implementation of the AlphaFold-2 EvoFormer core; it is not the full AlphaFold system, is not pretrained, and is not checkpoint-compatible with AlphaFold/OpenFold. The structural generator is a custom local SE(3) Schrödinger-bridge model, not RFdiffusion; and the sequence-recovery module is ProteinMPNN-inspired, not native ProteinMPNN. Random modules remain trainable and production inference fails closed until required components are trained, validated, and objective predictors calibrated. Proxy values are not biological measurements.

---

## Architecture contract

The public `chimera.CHIMERAv2` entry point is assembled explicitly by the canonical architecture composition root. Importing the package performs no runtime monkey-patching. The diagram below is the authoritative current architecture overview. The model consumes caller-prepared MSA tokens, pair features, source frames, and required conditioning inputs; it does not provide an end-to-end preprocessing pipeline.

```text
Caller-prepared MSA tokens and pair features
        │
        ▼
Evolutionary representation
(138 coupled EvoFormer blocks by default; CHIMERA-owned parameters, not pretrained AlphaFold weights)
        ├─ MSA stream: pair-biased row attention → column attention → transition
        ├─ MSA-to-pair: masked Outer Product Mean
        └─ pair stream: triangle multiplication/attention → pair transition
        │
        ├──────────────► pair representation
        │
        ▼
Triangular pair connector + substrate conditioning
(retrieval is opt-in and currently reports RAG_UNAVAILABLE without aligned embeddings)
        │
        ▼
SE(3) Schrödinger-bridge backbone transport
        │
        │  entropic Sinkhorn endpoint coupling
        │  Brownian-bridge conditional training targets
        │  Euler-Maruyama stochastic sampling
        ▼
Hierarchical geometric sequence designer
        │
        ├─ residue scale: geometric message passing
        ├─ domain scale: domain attention
        ├─ module scale: interface attention
        └─ assembly scale: symmetry/interface representation
        │
        ▼
Five explicitly sourced objective channels
        │
        ├─ evolutionary plausibility
        ├─ structural validity/stability proxy
        ├─ expression proxy
        ├─ substrate selectivity proxy
        └─ assembly compatibility proxy
        │
        ▼
Calibrated surrogate Pareto filtering, or explicitly named deterministic proxies
        │
        ▼
Optional Bayesian uncertainty + Gaussian Expected Improvement
(only when objective readiness and explicit utility conditions permit)
        │
        ▼
Candidate batch for external experimental evaluation
        │
        ▼
PROTEUS preference data
        │
        ▼
DPO update of the actual autoregressive sequence policy
```

Implementation ownership follows that path: `chimera/architecture.py`
composes the model; `evoformer_stack.py` owns the canonical evolutionary
representation and `components.py` owns its CHIMERA integration/connectors;
`se3_flow.py` owns the canonical velocity network and bridge-backed structural
sampler, while `schrodinger_bridge.py` owns bridge coupling, training targets,
and stochastic integration. `flow_matching.py` preserves historical imports
and delegates its old deterministic OT utilities to `legacy_flow.py`.
`sequence_design.py` owns hierarchical sequence design; `objective_schema.py`
and `pareto_pcgrad.py` own objective contracts and prediction. Optional
retrieval is isolated in `retrieval.py`. `multi_objective.py` is only a
compatibility import shim; its historical implementations are isolated in
`legacy_optimization.py` and are not imported by canonical architecture code.

The default canonical constructor uses 138 EvoFormer blocks. The EvoFormer computation, masks, configuration, and training objective are
documented in [docs/EVOFORMER_REPRESENTATION.md](./docs/EVOFORMER_REPRESENTATION.md).

### Scientific status of the transport model

The canonical bridge implementation is in `chimera/schrodinger_bridge.py`.

It is an **entropic Brownian Schrödinger-bridge approximation in a local SE(3) coordinate chart**. Translation is Euclidean and rotation is represented locally with

\[
\omega = \log(R_0^T R), \qquad R = R_0\exp(\omega).
\]

The endpoint coupling is computed with log-domain Sinkhorn scaling. For the reference SDE

\[
dX_t = \sqrt{2D}\,dW_t,
\]

the conditional Brownian bridge is

\[
X_t\mid X_0,X_1 \sim
\mathcal N((1-t)X_0+tX_1,\;2Dt(1-t)I),
\]

with conditional drift

\[
b^*(x,t\mid x_1)=\frac{x_1-x}{1-t}.
\]

Training samples target endpoints from the **entropic endpoint coupling**, rather than simply pairing source `i` with target `i`. Sampling uses Euler-Maruyama because the Schrödinger bridge is stochastic.

This is intentionally documented as a local Lie-algebra approximation. It is not presented as an exact closed-form heat-kernel Schrödinger bridge on SO(3).

---

## Optimization contracts

### PCGrad

The canonical objective regime calls `MergeReadyParetoMultiObjectiveHead.pcgrad_loss`
in `chimera.pareto_pcgrad`; it computes per-objective gradients and projects
conflicts. The separately public `chimera.pcgrad` module provides reusable
PCGrad helpers and remains covered by its own API/tests. The two paths should
not be conflated:

1. compute one gradient per objective;
2. randomly permute the other objectives for each task;
3. when `g_i · g_j < 0`, project the conflicting component out;
4. sum the projected gradients and write them to `.grad`.

The canonical trainer does not use loss-magnitude weighting as a substitute
for gradient conflict detection.

### DPO

`chimera.dpo.DPOTrainer` implements the standard reference-policy DPO objective

\[
-\log\sigma\left(\beta[(\log\pi_\theta(y_w|x)-\log\pi_{ref}(y_w|x))
-(\log\pi_\theta(y_l|x)-\log\pi_{ref}(y_l|x))]\right).
\]

The reference policy is frozen. If sequence masks are supplied, padding positions are excluded from the sequence log probability. A policy can provide `logprob(context, tokens)` or callable token logits for masked likelihoods.

The legacy preference-training implementation remains isolated from the canonical DPO API and is not used by the public CHIMERAv2 path.

### Bayesian uncertainty and Expected Improvement

`chimera.bayesian.BayesianUncertaintyEstimator` uses MC dropout as an **approximate Bayesian posterior method**. It reports:

- predictive mean;
- epistemic variance;
- epistemic standard deviation;
- optional aleatoric variance when a model supplies predictive variance.

Expected Improvement uses the standard deviation, not the variance:

\[
EI(x)=(\mu-f^*-\xi)\Phi(Z)+\sigma\phi(Z),
\qquad
Z=\frac{\mu-f^*-\xi}{\sigma}.
\]

Heterogeneous objectives are normalized before scalarization. The estimator also preserves each module's original training/evaluation mode while activating only dropout layers.

**Important:** MC dropout is not an exact Bayesian posterior and should not be described as one. Deep ensembles or a calibrated probabilistic surrogate can be added for stronger uncertainty estimates.

---

## Geometry contract

Backbone geometry uses `(B, L, 4, 3)` coordinates in `N/CA/C/O` order and optional `(B, L, 3, 3)` residue frames.

The validator checks:

- finite coordinates;
- frame orthogonality and determinant;
- Cα spacing;
- peptide and local bond lengths;
- backbone angle sanity;
- steric clashes.

The clash detector builds the backbone covalent graph and excludes atom pairs at graph distance one or two from the simple steric-distance test. This prevents expected bonded/near-bonded backbone distances from being mislabeled as steric clashes while still detecting genuinely close non-local atoms.

The validator is a deterministic sanity check, **not** a molecular mechanics force field.

---

## ProteinMPNN contract

`chimera.proteinmpnn.ProteinMPNN` is a ProteinMPNN-inspired local model, not the original pretrained ProteinMPNN.

Its geometric edge features are:

```text
16 radial basis distance features
+ 3 query-frame local displacement features
+ 9 relative-frame rotation features
= 28 edge features
```

The representation is invariant to a shared global rigid transformation. Fixed residues are hard constrained at the sequence-logit level.

Native ProteinMPNN integration belongs behind `ProteinMPNNAdapter` and must be performed with the upstream implementation and its compatible checkpoint.

---

## Quick start

```python
import torch
from chimera import CHIMERAv2, NRPSConstraints

model = CHIMERAv2()

# The local classes are research approximations. Native upstream weights
# require the corresponding adapters and environments.

constraints = NRPSConstraints(
    fixed_mask=None,
    stachelhaus_positions=torch.tensor([235, 236, 239, 278, 299, 301, 322, 330, 517, 518]),
    domain_boundaries=torch.tensor([[[0, 300], [300, 400], [400, 500], [500, 580], [580, 600]]]),
    module_boundaries=torch.tensor([[[0, 300], [300, 400], [400, 500], [500, 580], [580, 600]]]),
    icosahedral_face=torch.tensor([7]),
    ppt_serine_position=519,
    hotspot_coords=None,
    hotspot_indices=None,
    target_substrate="PHE",
)
```

A real design run requires compatible MSA, pair features, source backbone frames, and any substrate geometry required by the selected conditioning path. Canonical modules begin uninitialized; the CLI refuses production inference unless readiness is verified. `--experimental` (or synthetic `--demo`) permits explicit proxy-ranked exploratory output and records readiness, objective provenance, and retrieval status.

For callers that already provide model-ready tensors, `chimera.run_inference`
accepts an `InferenceRequest` with a versioned `InferenceConfig` and returns
candidate coordinates/sequences, geometry sanity checks, and a provenance-rich
`InferenceResult`. The result can be written once as JSON with `write_json()`.
This API uses an isolated seeded sampler, does not activate unavailable
retrieval or external backends, and does not imply production or biological
validation. See [Production Engineering](docs/PRODUCTION_ENGINEERING.md) for
the configuration-to-runtime mapping, checkpoint rules, and remaining gate
blockers.

---

## Training

`CanonicalTrainer` owns one explicit regime at a time. It chooses the active loss and trainable module set; it does not sum unrelated flow, sequence, objective, and preference losses.

| Regime | Supervision | Trainable components | Active loss |
|---|---|---|---|
| `representation` | MSA tokens and pair features | Coupled EvoFormer stack | Masked-MSA recovery + pairwise MSA mutual-information supervision |
| `flow` | Source/target SE(3) frames | MSA encoder, pair connector, flow backbone, evolutionary cross-attention; supplied substrate/constraint encoders | Brownian Schrödinger-bridge drift regression |
| `sequence` | Ground-truth structure and amino-acid sequence | MSA encoder, node connector, local sequence-recovery trunk, multiscale designer, projection, autoregressive policy | Teacher-forced causal token cross entropy |
| `constraint` | Source/target frames plus explicit NRPS constraints | Flow-regime modules and constraint encoder | Constrained bridge drift regression |
| `objective` | Named labels with per-objective source metadata | Typed feature encoder and four learned objective heads | Per-task supervised losses with PCGrad |
| `preference` | Chosen/rejected sequences and conditioning context | Autoregressive sequence policy only | DPO against a frozen reference |

Objective inputs are declared per learned head in `OBJECTIVE_FEATURE_PLAN`: evolutionary plausibility consumes sequence and MSA features; structural stability consumes sequence and local-frame-invariant structure features; expression efficiency consumes sequence features; substrate selectivity consumes sequence and substrate features. Residue streams use positional encodings and learned attention pooling rather than unconditional mean pooling. Structural descriptors use local relative translations/rotations, neighbor distances, and explicit face context; global translation/rotation changes them by less than `1e-5` in the synthetic invariance regression. Each learned objective label must supply both a source identity and an `ObjectiveLabelKind` (`proxy`, `surrogate`, `validated_surrogate`, `deterministic_evaluator`, or `experimental_measurement`) and satisfy its versioned shape, dtype, finite, range, scale, and loss contract. MSA-derived inputs are frozen and detached in the objective regime.

Assembly compatibility is not a learned objective head. Inference reports the deterministic `icosahedral_interface_geometry_proxy` directly from candidate geometry and face assignment. It is not used as a training label for a model consuming the same geometry, and it is not an experimental measurement or physical assembly-energy prediction. This geometric score is not Bayesian-calibrated and contributes no learned epistemic uncertainty.

Structural-stability training must identify whether its structures are `reference`, `generated`, or `synthetic`. Training on reference structures and applying the predictor to generated structures is recorded as a distribution-shift risk. Geometry validation is a sanity gate, not a demonstrated correction for that shift.

Flow training uses the supervised bridge loss; the inference sampler is stochastic Euler-Maruyama under `no_grad`. The sampler is not used as a differentiable training shortcut. Preference optimization is rejected until the sequence policy has supervised training and a separate held-out validation manifest. `CanonicalTrainer.validate()` consumes an iterable of held-out batches, aggregates losses, and requires at least 30 examples by default before recording `validated`; the minimum is configurable and recorded. This is an engineering evidence floor, not a scientifically established sample-size guarantee. Objective heads remain uncalibrated unless explicit evidence is added; MC dropout does not make random predictions meaningful. Calibration requires at least 30 independent examples, a distinct calibration manifest, and empirical one-sigma coverage between 0.58 and 0.78; the recorded RMSE, MAE, and coverage remain visible in provenance.

```python
from chimera import CanonicalTrainer, CanonicalTrainingBatch, TrainingRegime, seed_everything

seed_everything(seed)
trainer = CanonicalTrainer(
        model,
        TrainingRegime.FLOW,
        dataset_manifest={"dataset_version": dataset_version, "split": "train"},
        dataset_path=training_jsonl_path,
        preprocessing_config=preprocessing_config,
        random_seed=seed,
)
step = trainer.train_step(batch)
gradient_matrix = step["gradient_flow"]
trainer.save_checkpoint("flow-stage.pt")
```

`dataset_path` fingerprints the exact file bytes; the dataset manifest/version and preprocessing configuration must describe the inputs actually used. When provenance cannot be obtained, the manifest retains `None`, and canonical checkpoint save/resume fails closed rather than treating it as verified. The manifest also records the Git commit/source tree, model, state-schema and objective-schema fingerprints, runtime, regime, validation floor, optimizer/scheduler implementation and configuration, and an explicitly configured seed. Canonical resume restores saved Python, NumPy, PyTorch, CUDA, and explicit-generator RNG states.

`CanonicalTrainer.save_checkpoint()` writes a tagged resumable training checkpoint. `model.save()` and `load_connectors()` are a separate, versioned component-transfer path: they transfer module weights only, contain no optimizer/trainer state, and cannot be passed to canonical resume. The loader still accepts legacy raw component maps for migration.

`gradient_flow_report(model)` reports every parameter's `requires_grad`, gradient presence, norm, and finite status. `trainer.resume(path)` restores the full model, optimizer, optional scheduler, regime, component status, and manifest after compatibility checks. Held-out validation marks components validated; a mere optimizer step does not. `model.inference_readiness()` explains what is still missing, and `model.design(..., experimental=True)` uses clearly named deterministic proxies until validated, calibrated surrogate objectives are available.

## Pre-production boundary

```text
Research Prototype
        -> Pre-Production Engineering Gate
        -> Production Engineering
        -> Biological Validation
```

`preproduction_readiness_report(PreProductionGateEvidence(...))` is a legacy evidence summary; it does not alter `inference_readiness()`. It may summarize caller-provided booleans and is not the production release gate. The report distinguishes engineering status from `NOT SCIENTIFICALLY VALIDATED`. Biological activity, stability, selectivity, expression, and assembly remain unvalidated absent independent experimental evidence.

## Production engineering entry points

Install with `pip install .` (or `pip install -e .` for editable development).
The core install includes only PyTorch, NumPy, and einops. Optional feature
groups are `esm`, `retrieval`, `data`, `research`, and `viz`; development tools
are in `dev`. Use only the extras required by the selected workflow.
The installed CLI exposes `chimera --version`, identity-addressed model
commands (`chimera models list|fetch|inspect|verify`), and the self-executing
`chimera production-gate`. `chimera production-dependencies` emits the
machine-readable production-candidate closure and its current blockers. It
reports `BLOCKED_INTERNAL`: no validated CHIMERA training artifact exists.
See
[`docs/PRODUCTION_ENGINEERING.md`](docs/PRODUCTION_ENGINEERING.md) for current
contracts and explicit gaps, and
[`docs/ARCHITECTURE_DEPENDENCY_AUDIT.md`](docs/ARCHITECTURE_DEPENDENCY_AUDIT.md)
for the component-level dependency inventory and release blockers.

The AlphaFold-2-style CHIMERA EvoFormer (without pretrained weights), ProteinMPNN-inspired model, and custom
SE(3) bridge remain distinct from upstream ESM/OpenFold, ProteinMPNN, and
RFdiffusion. Model acquisition alone does not establish adapter compatibility,
scientific validation, or biological validation.
