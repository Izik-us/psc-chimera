# Folding generator and validator

## Why
`BiologicalObjectiveEvaluator` scores `structural_validity` from `validate_backbone` on the single
generated backbone, which `architecture.py` copies across all `n_draws` sampled sequences. The score is
therefore identical for every sequence of a backbone and says nothing about whether a sequence folds
into it. This package closes that gap: sequence -> independent folder -> compare with the intended backbone.

## Components
| Module | Role |
|---|---|
| `folding/types.py` | `FoldPrediction`; `FoldingBackend.predict(tokens, mask, seed, msa_tokens)` - the **independence firewall** (no structural parameter exists). |
| `folding/model.py` | `CHIMERAFold`: Pairformer trunk (reuses CHIMERA triangle blocks) + recycling + shared-weight `IPABlock` structure module with group-valued frame updates + confidence heads. `CHIMERAFoldBackend` fails closed without a checkpoint. |
| `folding/heads.py`, `losses.py` | pLDDT/PAE/distogram heads trained against *realised* error; FAPE, SO(3) geodesic, violation, carbonyl-angle losses. |
| `folding/adapters.py` | `ExternalFoldingBackend`: wraps any CLI folder (ESMFold/OpenFold) via a command template; never fabricates output. |
| `folding/ensemble.py` | Multi-seed/backend consensus: pairwise TM, medoid, Karcher mean + geodesic dispersion on SO(3). Deterministic reruns count as one effective member. |
| `validation/metrics.py` | TM-score (iterative superposition search), GDT-TS, lDDT, contact F1, frame/torsion agreement, domain arrangement. |
| `validation/calibration.py` | Isotonic (PAV) + one-sided split-conformal lower bound with length-Mondrian groups; fails closed (`-inf`) when unsupported. |
| `validation/fold_validator.py` | `FoldValidator` -> `FoldValidationReport` (ACCEPT / PENALIZE / REJECT, 0-100 score), optional shuffled-sequence null, three-way design/fold/experimental triangle, `structural_objective_labels`, `pareto_funnel`. |
| `data/folding_dataset.py` | Structure-QC gates, cluster-balanced weights, calibration partition carved by cluster, design-overlap exclusion. |

## Wiring (applied: `CHIMERAv2.design`, opt-in)
```python
validator = FoldValidator([ExternalFoldingBackend([...], name="esmfold", checkpoint_path=...)],
                          calibrator=ConfidenceCalibrator.from_json(open("fold_cal.json").read()))
result = model.design(..., fold_validator=validator, fold_budget=200, fold_n_null=8, fold_require_accept=False)
result["fold_validation"]   # per-candidate reports, decision counts, label kind, status
```
Flow: cheap objectives -> `pareto_funnel` chooses `fold_budget` candidates -> sequence-only folding ->
the sequence-independent `backbone_sanity_validity` column (index 1) is replaced by the fold score ->
REJECTed candidates (and, with `fold_require_accept=True`, anything not ACCEPTed) are dropped ->
Pareto front over survivors. Never-folded candidates are excluded. If nobody survives,
`pareto_sequences` is empty and `fold_validation["status"]` says why (fail-closed).
Without `fold_validator`, behaviour is unchanged.

Note: in `design()` each design has its own backbone (`n_mpnn_seqs=1`); the old `structural_validity`
is constant across sequences only when several sequences are drawn from one backbone, but in every case it
depends on backbone geometry alone, never on the sequence.

## Honest limits
* Default thresholds are unvalidated; every report records `thresholds_provenance`.
* Until a calibrator fitted on cluster-held-out data is supplied, nothing is ACCEPTed and labels are `proxy`.
* Conformal validity needs exchangeability between calibration data and deployment designs; designed NRPS
  sequences may be out of distribution - check `CalibrationState.domain`.
* Agreement with a folder shows foldability, not stability, expression or catalytic activity; multi-domain NRPS
  modules can legitimately adopt other conformations, hence per-domain/arrangement metrics.
* `ESMFold` weights are `MISSING` in this repo and `CHIMERAFold` is untrained: no real validator is available until one is supplied.
