# ProteinMPNN reference integration

## Scope and source provenance

CHIMERA now vendors the upstream Dauparas ProteinMPNN utility implementation
from https://github.com/dauparas/ProteinMPNN, currently pinned to upstream
blob `a8fd00821a538d2d82ac4ba0a86d29252bed0293`. The source and its MIT
license are retained under `chimera/vendor/`.

The wrapper in `chimera/proteinmpnn_reference.py` instantiates the upstream
`ProteinMPNN` class directly. It does not rename native parameter keys,
project hidden states, replace graph layers, or fuse CHIMERA features into the
reference computation. Standard model defaults are hidden width 128, three
encoder layers, three decoder layers, and 48 neighbors. Official checkpoints
are loaded into the native module with strict state-dictionary loading.

## Input contract

- Backbone coordinates: `(B, L, 4, 3)`, atom order N, CA, C, O. A fifth atom
  may be supplied and is ignored by the full-backbone model.
- Sequence: integer tokens in the upstream alphabet
  `ACDEFGHIKLMNPQRSTVWYX`.
- Residue mask: boolean `(B, L)`, true for valid residues.
- Design mask: boolean `(B, L)`, true for residues to redesign and false for
  fixed residues.
- Residue indices and chain encodings: integer `(B, L)` tensors.
- Optional `randn` and decoding order allow deterministic reference comparisons.

The native model uses the original backbone-distance featurization, virtual Cβ,
relative sequence/chain features, message-passing encoder and autoregressive
graph decoder. Sampling delegates to the upstream sampler, including its
randomized decoding order and fixed-position behavior.

## Boundaries

This is the first-stage reference baseline, not yet a replacement for
`ProteinMPNNBackbone`, `MultiScaleNRPSDesigner`, or
`AutoregressiveSequencePolicy`. The wrapper is intentionally not inserted into
the default canonical composition, so existing CHIMERA parameter counts,
checkpoint schemas, and execution paths remain unchanged.

The opt-in `ProteinMPNNConditioningAdapter` in
`chimera/proteinmpnn_conditioning.py` now accepts native log probabilities,
EvoFormer single-residue features, CHIMERA geometric residue states, and
`MultiScaleNRPSDesigner` logits. It adds a zero-initialized residual correction
and emits normalized probabilities over the 20 standard amino acids. The
zero-initialized head begins as the native distribution renormalized without
the upstream X token, so the adapter initially contributes no learned preference.

The canonical forward output now exposes `geometric_residue_features` as an
explicit integration point. The adapter is not inserted into the default model
and does not change the canonical parameter count or checkpoint schema.

Important boundary: this adapter adjusts residue distributions after a native
forward pass. It does not modify the internals of the upstream autoregressive
sampler. Keep native sampling as its own baseline; evaluate the adapter as an
explicit distribution-level conditioning/reranking interface before considering
any integration into CHIMERA's separate autoregressive sequence policy. Compare
native ProteinMPNN alone against each conditioned variant under identical
structures, masks, decoding contexts, and held-out evaluation sets.

## Validation gates

1. Strict load of an upstream official checkpoint without missing/unexpected keys.
2. Reference log-probability parity against the upstream source for identical
   inputs and decoding order.
3. Rigid-motion invariance, valid masks, chain separation, and fixed residues.
4. Held-out sequence recovery and diversity.
5. Independent structure prediction/fold agreement before biological claims.

A locally saved state-dictionary round trip verifies wrapper loading mechanics,
not parity with a published pretrained checkpoint. No pretrained weights are
bundled in the repository by this change.


## Research notes and model variants

- Dauparas et al., *Robust deep learning-based protein sequence design using
  ProteinMPNN*, Science (2022), DOI
  [10.1126/science.add2187](https://doi.org/10.1126/science.add2187). The model
  featurizes inter-residue distances across N, CA, C, O, and virtual Cβ atoms,
  uses graph message passing, and trains with randomized autoregressive decoding
  order. Fixed positions, chain-aware design, tied positions, amino-acid biases,
  and probability-scoring modes are part of the reference implementation.
- Official code and checkpoints:
  [dauparas/ProteinMPNN](https://github.com/dauparas/ProteinMPNN). The repository
  supplies multiple backbone-noise variants, CA-only models, and soluble-model
  weights. A checkpoint's neighborhood size and model family must match the
  selected constructor; checkpoint loading must remain strict.
- Dauparas et al., *Atomic context-conditioned protein sequence design with
  LigandMPNN*, Nature Methods (2025), DOI
  [10.1038/s41592-025-02626-1](https://doi.org/10.1038/s41592-025-02626-1).
  LigandMPNN explicitly models nonprotein atoms and can predict side-chain
  conformations. It is a strong later candidate for substrate-pocket and
  ligand-contact design, but should remain a separate model family rather than
  being mislabeled as the vanilla ProteinMPNN reference.
- Goverde et al., *Computational design of soluble and functional membrane
  protein analogues*, Nature (2024), DOI
  [10.1038/s41586-024-07601-y](https://doi.org/10.1038/s41586-024-07601-y).
  SolubleMPNN uses a training set excluding annotated transmembrane proteins.
  That is a distribution-specific model choice, not a universally superior
  checkpoint for every enzyme or membrane-associated target.
- *ProteinMPNN Recovers Complex Sequence Properties of Transmembrane
  β-barrels* (2024), [PMC article](https://pmc.ncbi.nlm.nih.gov/articles/PMC10862708/).
  This benchmark emphasizes the importance of accurate, refined input
  backbones and cautions against treating sequence recovery as the only design
  quality measure.

For CHIMERA, the immediate baseline is vanilla full-backbone ProteinMPNN. Future
NRPS substrate-pocket experiments may compare it with LigandMPNN, while any
SolubleMPNN use should be justified by the intended expression/solubility target.
These alternatives should be compared as controlled ablations, not silently
mixed into one purportedly faithful model.
