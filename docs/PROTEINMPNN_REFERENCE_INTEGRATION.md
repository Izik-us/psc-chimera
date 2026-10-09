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

The next integration stage should add explicit adapters for EvoFormer residue
representations and NRPS domain/module annotations, then compare native
ProteinMPNN alone against each conditioned variant under identical structures,
masks, decoding contexts, and held-out evaluation sets. Any added conditioning
must be opt-in and must not alter native checkpoint parity.

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
