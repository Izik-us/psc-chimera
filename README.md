# PSC-CHIMERA

**Compositional Hierarchical Inference Model for Evolutionary Representation and Architecture**

Stage 1 computational design prototype for the theoretical Pharmacosynthetic Constructor (PSC) engineering pipeline.

> **Research-status notice:** CHIMERA is an untrained research prototype by default. Its local MSA row/column attention model is an approximation, not native EvoFormer/OpenFold; its structural generator is a custom local SE(3) Schrödinger-bridge model, not RFdiffusion; and its sequence-recovery module is ProteinMPNN-inspired, not native ProteinMPNN. Random modules remain trainable and production inference fails closed until required components are trained, validated, and objective predictors calibrated. Proxy values are not biological measurements.

---

## Architecture contract

The public `chimera.CHIMERAv2` entry point is assembled explicitly by the canonical architecture composition root. Importing the package performs no runtime monkey-patching. The intended pipeline is:

```text
Animal / target-family MSA
        │
        ▼
Evolutionary representation
(local MSA row/column-attention approximation; not native EvoFormer/OpenFold)
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
Bayesian uncertainty + Gaussian Expected Improvement
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

`chimera.pcgrad` implements canonical Projected Conflicting Gradients:

1. compute one gradient per objective;
2. randomly permute the other objectives for each task;
3. when `g_i · g_j < 0`, project the conflicting component out;
4. sum the projected gradients and write them to `.grad`.

The repository no longer treats loss-magnitude differences as a substitute for gradient conflict detection.

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

---

## Training

`CanonicalTrainer` owns one explicit regime at a time. It chooses the active loss and trainable module set; it does not sum unrelated flow, sequence, objective, and preference losses.

| Regime | Supervision | Trainable components | Active loss |
|---|---|---|---|
| `representation` | MSA tokens | MSA row/column encoder and masked-token head | Masked-token cross entropy |
| `flow` | Source/target SE(3) frames | MSA encoder, pair connector, flow backbone, evolutionary cross-attention; supplied substrate/constraint encoders | Brownian Schrödinger-bridge drift regression |
| `sequence` | Ground-truth structure and amino-acid sequence | MSA encoder, node connector, local sequence-recovery trunk, multiscale designer, projection, autoregressive policy | Teacher-forced causal token cross entropy |
| `constraint` | Source/target frames plus explicit NRPS constraints | Flow-regime modules and constraint encoder | Constrained bridge drift regression |
| `objective` | Named labels with per-objective source metadata | Sequence projection and five-objective head | Per-task supervised losses with PCGrad |
| `preference` | Chosen/rejected sequences and conditioning context | Autoregressive sequence policy only | DPO against a frozen reference |

Flow training uses the supervised bridge loss; the inference sampler is stochastic Euler-Maruyama under `no_grad`. The sampler is not used as a differentiable training shortcut. Preference optimization is rejected until the sequence policy has supervised training and a separate held-out validation manifest. Objective heads remain uncalibrated unless explicit evidence is added; MC dropout does not make random predictions meaningful. Calibration requires at least 30 independent examples, a distinct calibration manifest, and empirical one-sigma coverage between 0.58 and 0.78; the recorded RMSE, MAE, and coverage remain visible in provenance.

```python
from chimera import CanonicalTrainer, CanonicalTrainingBatch, TrainingRegime

trainer = CanonicalTrainer(model, TrainingRegime.FLOW, dataset_manifest="train-v1")
step = trainer.train_step(batch)
gradient_matrix = step["gradient_flow"]
trainer.save_checkpoint("flow-stage.pt")
```

`gradient_flow_report(model)` reports every parameter's `requires_grad`, gradient presence, norm, and finite status. `trainer.resume(path)` restores the full model, optimizer, optional scheduler, regime, component status, and manifest under strict schema/config/dataset checks. Held-out validation marks components validated; a mere optimizer step does not. `model.inference_readiness()` explains what is still missing, and `model.design(..., experimental=True)` uses clearly named deterministic proxies until validated, calibrated surrogate objectives are available.
