# CHIMERA SEQUENCE-regime training protocol

## Purpose

The SEQUENCE regime must do two things at once:

1. learn new structure/domain/module/substrate relationships;
2. avoid erasing the evolutionary representation learned during REPRESENTATION.

The existing trainer already provides an explicit isolation boundary: by default the EvoFormer is frozen during SEQUENCE, while the relational conditioning and sequence-design components remain trainable. `allow_sequence_backbone_finetuning=True` is an explicit opt-in when the experiment requires backbone plasticity.

This document defines the experimental protocol before biological training begins.

## Why this design

Continual-learning literature describes the same stability/plasticity problem in different architectures. Learning without Forgetting uses outputs from an earlier model as a distillation target while learning a new task. EWC instead slows updates to parameters estimated to be important to previous tasks. Adapter methods preserve a fixed backbone and put new task capacity in small trainable modules. Feature- and relationship-level distillation extends the idea beyond final logits.

For CHIMERA, the most natural hierarchy is therefore:

```text
REPRESENTATION checkpoint
        |
        |  stable evolutionary knowledge
        v
+-----------------------+
| EvoFormer             |
| MSA + pair reasoning  |
+-----------+-----------+
            |
            | pair-conditioned context
            v
+-----------------------+
| sequence conditioning |
| + constraints         |
| + substrate           |
+-----------+-----------+
            |
            v
   hierarchical designer
            |
            v
     sequence policy
```

The default SEQUENCE experiment keeps the EvoFormer fixed and learns through the downstream relational path. This is the lowest-risk route to learning new relationships without changing the foundation representation.

A second experiment permits EvoFormer fine-tuning. That experiment is **not** treated as equivalent to the default regime. It must report representation drift and old-task retention.

## Two SEQUENCE tracks

### Track A: relational adaptation, frozen EvoFormer

```text
EvoFormer:                 FROZEN
Pair connector:            TRAIN
Node connector:            TRAIN
Constraint encoder:        TRAIN when constraints exist
Substrate conditioner:     TRAIN when substrate exists
Base MPNN:                 TRAIN
Multi-scale designer:      TRAIN
Sequence projection:       TRAIN
Sequence policy:           TRAIN
```

This is the first biological training target. It allows CHIMERA to learn new relationships using the representation already established by the evolutionary stage.

### Track B: representation plasticity

```text
EvoFormer:                 TRAIN, explicit opt-in
All sequence components:   TRAIN
```

Track B is used only after Track A establishes a credible baseline. It tests whether the EvoFormer itself benefits from adaptation to the NRPS/substrate distribution.

## Stability/plasticity measurements

Every Track B experiment must report at least:

- sequence negative log-likelihood;
- token recovery accuracy;
- substrate-conditioned recovery accuracy;
- domain/module-conditioned recovery accuracy;
- old-distribution representation drift;
- new-distribution representation improvement;
- pair-relation drift;
- parameter drift in the EvoFormer;
- held-out old-task performance before and after adaptation.

A lower training loss by itself is not evidence of successful continual learning.

## Representation probes

For an old/reference batch, capture EvoFormer outputs before adaptation:

```text
single_old = EvoFormer_old(MSA, pair)
pair_old   = EvoFormer_old(MSA, pair)
```

After SEQUENCE adaptation:

```text
single_new = EvoFormer_new(MSA, pair)
pair_new   = EvoFormer_new(MSA, pair)
```

Report normalized feature drift:

```text
D_single = ||single_new - single_old||_2 / (||single_old||_2 + eps)
D_pair   = ||pair_new   - pair_old||_2   / (||pair_old||_2   + eps)
```

These are diagnostics, not training losses. The first implementation should not force representations to remain identical because that would defeat plasticity.

## Controlled plasticity experiment

The sequence experiment should be run as an ablation matrix:

| Condition | EvoFormer | Relational path | Purpose |
|---|---|---|---|
| A0 | frozen | train | baseline |
| A1 | frozen | train + substrate/constraint conditioning | relationship-learning baseline |
| B0 | train | train | maximum plasticity reference |
| B1 | train | train + representation preservation | stability/plasticity candidate |

B0 is intentionally a stress test. It tells us how much forgetting unrestricted fine-tuning causes. B1 becomes the candidate production strategy only if it improves new-task performance without unacceptable old-task degradation.

## Representation preservation strategy

The first preservation mechanism to evaluate is **representation-level distillation** from the previous validated representation checkpoint. The teacher supplies stable single and pair representations for the same examples while the student learns the new sequence objective.

Conceptually:

```text
L_sequence = L_recovery + lambda_rep * L_representation

L_representation =
    alpha * D(single_student, single_teacher)
  + beta  * D(pair_student, pair_teacher)
```

The teacher is frozen. The student remains trainable. `lambda_rep`, `alpha`, and `beta` are experimental hyperparameters and must be recorded in the checkpoint provenance.

The important CHIMERA-specific point is that pair representations should be included. The architecture's biological value is not only residue identity prediction; it is learning relationships among residues, domains, modules, and substrate-conditioned contexts.

The teacher targets should be produced from the **validated previous-stage checkpoint**, never from the student's current weights.

## Why not simply freeze everything?

Freezing EvoFormer is a strong baseline, but it limits where new evolutionary relationships can be represented. The downstream pair connector and conditioning stack can learn a new task-specific transformation of the old representation, but they cannot create arbitrary new information that the EvoFormer never encoded.

Therefore the research question is not:

> frozen or unfrozen?

It is:

> how much representation plasticity is useful before the stability cost becomes unacceptable?

That is what Tracks A and B measure.

## Stage transition rule

Do not promote Track B merely because its training loss is lower.

A candidate SEQUENCE checkpoint should require:

1. finite training and validation losses;
2. improvement over the frozen baseline on the new-task validation set;
3. no unacceptable degradation on the held-out old-distribution set;
4. measured EvoFormer single/pair representation drift;
5. reproducible checkpoint reload;
6. unchanged dataset/preprocessing provenance;
7. independent structural evaluation before biological claims are made.

## Relationship to CHIMERA's SHA-256 isolation machinery

The existing state fingerprints remain a hard ownership check for components that are intentionally frozen. They answer:

> did this supposedly frozen representation actually change?

The stability/plasticity experiment answers a different question:

> when we intentionally allow change, did that change improve the new relationship while preserving the old behavior?

Both are required. A hash can prove that a frozen module did not change; it cannot prove that a changed module learned something useful.

## Literature basis

- Li & Hoiem, *Learning without Forgetting*, ECCV 2016 / TPAMI 2017.
- Kirkpatrick et al., *Overcoming catastrophic forgetting in neural networks*, PNAS 2017.
- Houlsby et al., *Parameter-Efficient Transfer Learning for NLP*, ICML 2019.
- Kang et al., *Class-Incremental Learning by Knowledge Distillation With Adaptive Feature Consolidation*, CVPR 2022.
- Zhang et al., *Feature Forgetting in Continual Representation Learning*, 2022.
- Sim et al., *Learning without Forgetting Attention Scores*, ICCSIT 2025 / publication 2026.

The adaptation to CHIMERA is deliberately conservative: preserve the established MSA/pair representation as a teacher target while allowing a controlled amount of new representation learning. This is an experimental hypothesis for CHIMERA, not a claim that any of these papers already validated the exact EvoFormer/NRPS setting.
