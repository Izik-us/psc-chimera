# P0-P2 Merge Audit

This file is a temporary audit ledger for the final consolidation branch.

## P0
- Canonical composition must not subclass the legacy CHIMERAv2 implementation.
- SO(3) operations must have one authoritative implementation.
- IPA must depend on canonical Lie operations rather than a legacy flow module.
- Candidate generation and DPO must share the same autoregressive policy.

## P1
- Bayesian uncertainty must evaluate fixed candidates rather than regenerate stochastic candidates.
- Multi-objective acquisition semantics must be explicit.
- Structural retrieval must be rigid-transform invariant.
- FAISS index must be tied to a frozen/indexed encoder version.
- Native ProteinMPNN fixed residues must preserve their actual identities.
- PDB parsing must deterministically handle MODEL/altLoc/occupancy/insertion codes.

## P2
- PCGrad RNG must be explicitly controllable.
- PCGrad conflict telemetry must count unique pairs.
- Checkpoints need architectural and tensor-schema fingerprints.
- Retrieval K must be bounded by index size.
- End-to-end deterministic-mode coverage is required.

This ledger is retained only until the implementation and CI validation are complete.