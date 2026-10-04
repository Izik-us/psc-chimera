# Canonical EvoFormer evolutionary representation

## Three levels of the claim

1. **AlphaFold-2 EvoFormer core.** The reproduced core is the coupled MSA/pair
   stack: pair-biased MSA row attention, MSA column attention, MSA and pair
   transitions, Outer Product Mean, outgoing/incoming triangle multiplication,
   and starting/ending triangle attention. These operations are based on
   Jumper et al.'s Evoformer algorithms and the DeepMind/OpenFold reference
   implementations.
2. **CHIMERA EvoFormer implementation.** CHIMERA implements those information
   flows in its own PyTorch modules, masks, configuration, and caller-provided
   input/output interfaces. It is an **AlphaFold-2 architectural
   replication**, not a port of their implementation or parameterization.
3. **Full AlphaFold system.** The AlphaFold input pipeline, template and Extra
   MSA stacks, recycling schedule, structure module, auxiliary structure
   heads, training infrastructure, and trained weights are out of scope.

The EvoFormer exists to repeatedly exchange information between aligned
evolutionary observations and residue-pair features before CHIMERA's SE(3)
structural generator consumes those representations. It is not pretrained and
is not checkpoint-compatible with AlphaFold/OpenFold.

## Canonical path

```text
Caller-prepared MSA tokens (B,N_seq,L)
             │
             ▼
       amino-acid embedding ───────────────┐
             │                            │
             │            caller pair features + signed relative-position embedding
             │                            │
             ▼                            ▼
       M ∈ R[B,N_seq,L,C_m]         Z ∈ R[B,L,L,C_z]
             │                            │
       ┌─────┴────────────────────────────┴─────────────────────┐
       │                  48 EvoFormer blocks                    │
       │                                                        │
       │  MSA row attention + learned pair bias ────────────┐   │
       │  MSA column attention                              │   │
       │  MSA transition                                    │   │
       │  masked Outer Product Mean ────────────────────────┤   │
       │  triangle multiplication outgoing/incoming        │   │
       │  triangle attention starting/ending node           │   │
       │  pair transition                                   └───┘
       └────────────────────────────────────────────────────────┘
             │                            │
             └── query-row projection     └── refined pair representation
                      │                                 │
                      └──────────────┬──────────────────┘
                                     ▼
                CHIMERA pair connector and SE(3) structural conditioning
```

`CanonicalCHIMERAv2` constructs `MSARepresentationBackbone`, implemented by
`EvoFormerStack`. The `evoformer_n_blocks` model option defaults to **48** and
is passed directly to the `ModuleList`; a smaller count is an explicit
development/test configuration, not a silent production substitution. The
block count, widths, dropout rates, chunk sizes, relative-position range, and
gradient-checkpointing selection are part of `model_configuration()` and thus
the checkpoint architecture identity.

The block operations are separate residual updates in this exact order:

1. Per-MSA-row gated residue attention with learned pair-derived per-head bias.
2. Per-residue MSA-column attention over valid aligned sequences.
3. LayerNorm → `C_m → 4*C_m` → ReLU → `4*C_m → C_m` transition.
4. Sequence-mask-normalized Outer Product Mean from two projected MSA branches.
5. Outgoing and incoming triangle multiplicative updates.
6. Starting-node and ending-node triangle attention.
7. LayerNorm → `C_z → 4*C_z` → ReLU → `4*C_z → C_z` pair transition.

The row attention bias is computed from normalized pair features and a
no-bias projection to heads; its two residue axes are respectively the query
residue and key residue. Triangle-attention bias comes from each key-side pair
(`Z[i,k]` in starting-node orientation) and is broadcast over query pairs, not
the reverse. Both MSA attention operations use explicit valid-token masks;
all-masked rows produce zero attention rather than NaNs. Outer Product Mean
divides by the number of aligned sequences valid at both residues plus the
OpenFold `1e-3` numerical epsilon **after the output projection**, and emits
zero when that count is zero. The outer product is contracted and flattened,
projected to `C_z`, then divided by the jointly valid row count plus epsilon.
Its two linear branches and output projection include biases. Pair masks are the
outer product of residue validity and are applied throughout the triangle
stack. Invalid pair states are also explicitly zeroed after residual updates;
this is deliberate padding isolation beyond DeepMind's historically unmasked
pair-transition implementation. Fully padded examples remain finite and
yield zero representations.

The pair state begins as the sum of the caller-provided pair features and a
learned signed relative-position embedding. `residue_index` can be supplied to
represent gaps or nonconsecutive residue numbering; otherwise positions are
consecutive indices. Existing caller pair features are not replaced by
single-feature concatenation. NRPS constraints and substrate signals continue
to enter through their existing downstream conditioning paths.

After all blocks, `single_repr` is a learned projection of the **first/query
MSA row**, not an average over MSA sequences. The compatibility `forward()`
returns `(single_repr, pair_repr)`. `forward_with_representations()` returns an
`EvoformerOutput` exposing `msa_repr`, `pair_repr`, `single_repr`, and optional
diagnostics.

## Engineering choices and boundaries

- CHIMERA's default channels remain `C_m = 256`, `C_z = 128`, and
  `C_s = 256`; test models may set smaller compatible widths.
- Row attention, column attention, and triangle attention can be chunked along
  independent MSA rows/residue nodes. Outer Product Mean can chunk target
  residues, avoiding materialization of the full unchunked outer-product
  intermediate. `None` selects full/un-chunked computation; chunked and
  unchunked execution implement the same operations. Optional gradient
  checkpointing trades compute for activation memory.
- Outer Product Mean projection, contraction, and sequence normalization run
  in float32 under autocast and return the update in the MSA input dtype to
  avoid low-precision accumulation of evolutionary pair statistics.
- Default residual-update dropout rates are 0.15 for MSA row attention, 0 for
  MSA column attention, and 0.25 for pair triangle operations. Following
  OpenFold's `DropoutRowwise` broadcast dimension `-3`, MSA residual updates
  share dropout across MSA sequence rows (`[B,1,L,C]`); pair residual updates
  share across the first residue axis (`[B,1,L,C]`). Triangle-attention
  dropout is applied after chunks are reassembled, so chunk size does not
  change its sharing domain.
- The transition expansion defaults to 4 and is part of the architecture
  configuration. Configuration records widths, head counts, dropout rates,
  chunk settings, gradient-checkpointing, relative-position range, transition
  factor, and OPM epsilon; checkpoint validation rejects mismatches.
- Initialization is intentionally CHIMERA-specific: Xavier initialization
  with residual output scaling by `1/sqrt(n_blocks)`, and sigmoid gates with
  zero weights/unit biases. This differs from OpenFold's mix of LeCun,
  ReLU-He, gating, and zero-final initializers. It is not parameter-level
  parity; the operation equations, orientations, masking, and dropout
  semantics are the fidelity target.
- This is architectural replication, not pretrained-weight compatibility.
  CHIMERA's operation names, state-dict keys, initializations, and downstream
  dimensions are independently owned.
- There is no dedicated Extra-MSA stack, template stack, or representation
  recycling loop. Callers provide the MSA rows they intend the main stack to
  process; previous pair/query states are not implicitly recycled.
- The existing `TriangularPairUpdateConnector` is retained as a distinct
  downstream adapter: it projects the 128-channel EvoFormer pair state into
  the structural conditioner's configured width and supports explicit
  retrieved-context fusion. Its post-projection triangle updates operate in
  that conditioning space; they do not substitute for or feed back into the
  EvoFormer stack.

## Representation training

The `representation` regime trains the whole EvoFormer, including relative
position, single/pair projections, and both communication streams. Its loss is
the sum of:

1. Masked-MSA token reconstruction cross entropy.
2. Smooth-L1 supervision of a pair-state head against normalized pairwise
   categorical mutual information calculated from the uncorrupted aligned
   MSA. Invalid/gap residues are excluded and pairs with fewer than two
   jointly observed sequences are omitted.

This is a CHIMERA-specific objective, not AlphaFold's structural training
objective or full training recipe. AlphaFold jointly trained its structure
prediction system using structural supervision (including FAPE and auxiliary
distogram and masked-MSA losses), large-scale structure/sequence data,
recycling, and distributed accelerator training. The CHIMERA objective does
not establish structural accuracy. Structural utility is trained separately
through the existing flow regime, where gradients propagate from the
supervised SE(3) bridge loss through the pair connector and EvoFormer. The
existing training gradient audit continues to verify trainable-component
gradients. No AlphaFold-scale structural objective, dataset, recycling
schedule, or pretrained representation is implied.

## References

- Jumper et al., *Highly accurate protein structure prediction with
  AlphaFold*, Nature 2021, including Evoformer Algorithms 6–15 in the
  Supplementary Information.
- [DeepMind AlphaFold source](https://github.com/google-deepmind/alphafold),
  particularly `alphafold/model/modules.py`.
- [OpenFold source](https://github.com/aqlaboratory/openfold), particularly
  [`evoformer.py`](https://github.com/aqlaboratory/openfold/blob/main/openfold/model/evoformer.py),
  MSA, Outer Product Mean, triangle attention/multiplication, transition,
  primitive, and dropout modules.

The implementation was written as CHIMERA-owned PyTorch code using these
architectural references; it does not import or redistribute their model
implementations or weights. **AlphaFold/OpenFold checkpoint compatibility:
NO.**
