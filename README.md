# PSC-CHIMERA

**Compositional Hierarchical Inference Model for Evolutionary Representation and Architecture**

Stage 1 computational design prototype for the theoretical Pharmacosynthetic Constructor (PSC) engineering pipeline.

> **Research-status notice:** CHIMERA is a research prototype. The local EvoFormer, RFdiffusion/SE(3), and ProteinMPNN components are explicitly **approximations**, not drop-in replacements for OpenFold, RFdiffusion, or dauparas/ProteinMPNN. Native upstream checkpoints are not loaded into structurally incompatible local classes. Proxy objectives are not experimentally calibrated and must not be interpreted as biological validation.

---

## Architecture contract

The public `chimera.CHIMERAv2` entry point installs the current merge-safe component contracts before model construction. The intended pipeline is:

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

The older DPO implementation in `multi_objective.py` remains available as `LegacyDPOTrainer` for compatibility, but it is not the canonical API.

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

For multi-objective supervised training, obtain independent task losses and pass them to `pcgrad_step` instead of summing them and calling ordinary `backward()`.

For DPO, initialize a frozen reference policy **before** the preference update. Do not update the reference during the same DPO round.

For active learning, `best_observed` must represent the best **experimentally observed** scalarized objective in the same normalized utility space as the candidate predictions. It should not silently be replaced by the maximum predicted candidate.

---

## Native upstream integrations

The repository exposes explicit adapter boundaries for:

| Component | Local implementation | Production boundary |
|---|---|---|
| Evolutionary trunk | small Transformer approximation | `OpenFoldAdapter` / `OpenFoldCLIAdapter` |
| Backbone generation | local SE(3) SB/velocity architecture | `RFdiffusionAdapter` / `RFdiffusionCLIAdapter` |
| Sequence design | ProteinMPNN-inspired model | `ProteinMPNNAdapter` |

The adapters fail closed rather than loading an incompatible checkpoint into a different architecture.

---

## Installation

```bash
git clone https://github.com/Izik-us/psc-chimera.git
cd psc-chimera
pip install -e .
```

Optional native upstream projects must be installed in their own supported environments. Their checkpoints must be consumed by compatible upstream code or explicit adapters. A file merely being named `rfdiffusion_weights.pt` or `proteinmpnn_weights.pt` does not make it compatible with a local approximation.

Run the CPU-safe test suite with:

```bash
pytest tests/ -v
```

---

## Testing philosophy

The test suite is designed to reject silent scientific regressions, not merely import errors. It covers:

- SO(3) exponential/logarithmic-map consistency;
- SE(3) interpolation;
- geometric frame covariance/invariance;
- fixed-residue constraints;
- covalent-bond-aware clash detection;
- PDB completeness validation;
- Sinkhorn marginal constraints;
- Brownian bridge statistics;
- canonical PCGrad conflict handling;
- DPO preference behavior and masking;
- MC-dropout mode restoration;
- EI numerical behavior;
- ProteinMPNN geometric invariance and shape contracts;
- CHIMERAv2 integration shapes.

CI runs the CPU test suite on supported Python versions without requiring pretrained biological weights.

---

## What is not yet scientifically validated

The following are deliberately **not** claimed as experimentally validated:

- mammalian NRPS functional prediction;
- pLDDT or structural stability from the local proxy heads;
- PoET/evolutionary likelihood calibration;
- substrate selectivity prediction without validated A-domain labels;
- icosahedral assembly compatibility;
- de novo catalytic chemistry;
- wet-lab expression, PPant loading, product yield, or intracellular assembly.

These require domain-specific datasets, calibrated predictors, native structural backbones, controlled experimental evaluation, and external validation.

---

## Repository status

### Implemented

- [x] Strict foundation-model adapter boundaries
- [x] Geometric clash validation repair
- [x] ProteinMPNN fixed-residue constraints
- [x] Rigid-transform-invariant ProteinMPNN edge geometry
- [x] Canonical PCGrad
- [x] Entropic Sinkhorn endpoint coupling
- [x] Brownian Schrödinger-bridge training/sampling approximation
- [x] Canonical DPO objective
- [x] Mask-aware DPO likelihoods
- [x] MC-dropout epistemic uncertainty
- [x] Analytic Gaussian Expected Improvement
- [x] Regression and scientific-contract tests
- [x] Incremental CI linting and CPU test execution

### Still required before a production biological claim

- [ ] Native OpenFold integration and checkpoint compatibility validation
- [ ] Native RFdiffusion integration or validated bridge training data
- [ ] Native ProteinMPNN integration
- [ ] Domain-specific animal NRPS datasets and labels
- [ ] Calibrated biological objective evaluators
- [ ] Experimental PROTEUS data pipeline
- [ ] External validation of generated structures and sequences
- [ ] Wet-lab validation

---

## References

- Lipman et al., *Flow Matching for Generative Modeling*, 2022/2023.
- Yim et al., *SE(3) Diffusion Model with Application to Protein Backbone Generation*, 2023.
- Liu et al., *I²SB: Image-to-Image Schrödinger Bridge*, 2023.
- Bose et al., *SE(3)-Stochastic Flow Matching for Protein Backbone Generation*, 2023.
- Yu et al., *Gradient Surgery for Multi-Task Learning*, NeurIPS 2020.
- Rafailov et al., *Direct Preference Optimization: Your Language Model is Secretly a Reward Model*, NeurIPS 2023.
- Gal & Ghahramani, *Dropout as a Bayesian Approximation*, 2016.
- Jones et al., *Efficient Global Optimization of Expensive Black-Box Functions*, 1998.

---

## License

MIT
