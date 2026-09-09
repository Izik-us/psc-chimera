# PSC-CHIMERA

**Compositional Hierarchical Inference Model for Evolutionary Representation and Architecture**

Stage 1 computational design prototype for the theoretical Pharmacosynthetic Constructor (PSC) engineering pipeline.

> **Research-status notice:** CHIMERA is a research prototype. The local EvoFormer, RFdiffusion/SE(3), and ProteinMPNN components are explicitly **approximations**, not drop-in replacements for OpenFold, RFdiffusion, or dauparas/ProteinMPNN. Native upstream checkpoints are not loaded into structurally incompatible local classes. Proxy objectives are not experimentally calibrated and must not be interpreted as biological validation.

---

## Architecture contract

The public `chimera.CHIMERAv2` entry point is assembled explicitly by the canonical architecture composition root. Importing the package performs no runtime monkey-patching. The intended pipeline is:

```text
Animal / target-family MSA
        │
        ▼
Evolutionary representation
(local EvoFormer approximation or future native OpenFold adapter)
        │
        ├──────────────► pair representation
        │
        ▼
Triangular pair connector + substrate conditioning + structural retrieval
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
Five-objective prediction / evaluation
        │
        ├─ evolutionary plausibility
        ├─ structural validity/stability proxy
        ├─ expression proxy
        ├─ substrate selectivity proxy
        └─ assembly compatibility proxy
        │
        ▼
Pareto non-dominated filtering
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
    module_boundaries=torch.tensor([[[0, 600], [0, 0], [0, 0], [0, 0], [0, 0]]]),
    icosahedral_face=torch.tensor([7]),
    ppt_serine_position=519,
    hotspot_coords=None,
    hotspot_indices=None,
    target_substrate="PHE",
)
```

A real design run requires compatible MSA, pair features, source backbone frames, and any substrate geometry required by the selected conditioning path. The CLI refuses missing biological inputs unless demo mode is explicitly requested.

---

## Training

The intended optimization stages are:

```text
1. Supervised training of trainable connectors/heads
2. Multi-objective gradient surgery with canonical PCGrad
3. External experimental evaluation
4. Frozen-reference DPO update from preference pairs
5. Bayesian uncertainty estimation
6. EI acquisition of the next experimental batch
7. Repeat
```

The canonical low-level APIs are intentionally separate so each stage can be tested independently:

```python
from chimera import (
    BayesianUncertaintyEstimator,
    DPOBatch,
    DPOTrainer,
    SchrodingerBridge,
    pcgrad_step,
)
```
