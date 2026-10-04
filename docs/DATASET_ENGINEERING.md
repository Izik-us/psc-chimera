# CHIMERA Dataset Engineering

## 1. Repository audit summary

The repository already contains several useful building blocks, but it does not yet have a canonical dataset-engine layer that is pinned to actual structural provenance and leakage-safe splits.

Relevant implementation points:

- [data/dataset.py](../data/dataset.py) defines a JSONL training dataset that validates tensorized MSA and geometry records.
- [data/splitting.py](../data/splitting.py) contains deterministic cluster-based split utilities using exact-identity and sequence-identity fallback logic.
- [data/training.py](../data/training.py) contains helpers for deduplication, stats, and download-with-checksum workflows.
- [chimera/structure_utils.py](../chimera/structure_utils.py) contains a deterministic PDB backbone loader that honors model selection, residue ordering, and altLoc selection.
- [chimera/proteinmpnn.py](../chimera/proteinmpnn.py) contains the canonical geometric graph and invariant edge-feature construction used for local geometric sequence design.
- [chimera/domain_schema.py](../chimera/domain_schema.py) defines explicit NRPS schema primitives for domain spans and modules.
- [chimera/architecture.py](../chimera/architecture.py) and [chimera/evoformer_stack.py](../chimera/evoformer_stack.py) define the model contract that expects evolutionary context, pair features, source frames, and downstream structural-conditioning inputs.

Important architectural boundary:

- The dataset layer must not be used to replace the model architecture or the current ProteinMPNNBackbone compatibility path.
- The data engine is instead the provenance and supervision layer for a future CHIMERA geometric sequence designer that learns conditional sequence design from structure, evolution, and NRPS context.

## 2. Canonical dataset architecture

The dataset foundation is organized around explicit entities:

```text
Structure
  ├── Chain
  │    ├── Sequence
  │    ├── MSAExample
  │    └── NRPSDomain
  │
  └── ExperimentalMetadata

NRPSDomain
  ├── DomainType
  ├── Module
  ├── SelectivityPositions
  ├── SubstrateAssociation
  └── StructuralRegion
```

The implemented schema foundation lives in [data/dataset_engine.py](../data/dataset_engine.py) and includes:

- `ProvenanceRecord`
- `DataQualityResult`
- `AcquisitionManifest`
- `DatasetSplit`
- `DatasetVersion`
- `StructuralManifest`
- `ExternalSourceInventory`
- `stable_hash()`
- `summarize_records()`

These primitives preserve provenance, support QC verdicts, and keep dataset-version metadata deterministic.

## 3. External-source inventory

The repository includes a machine-readable inventory at [data/external_sources.json](../data/external_sources.json). Selected high-value sources are:

| Source | Purpose | Status |
| --- | --- | --- |
| RCSB PDB / wwPDB | Experimental structures in PDBx/mmCIF | verified |
| AlphaFold Protein Structure Database | Predicted structure coverage | verified |
| UniProt | Sequence and taxonomy metadata | verified |
| NCBI Entrez / RefSeq / GenBank | Sequence accessioning and cross-references | verified |
| ENA | Sequence archive and assembly metadata | verified |
| Pfam | Domain family annotation | verified |
| InterPro | Integrated domain annotation | verified |
| MIBiG | NRPS / BGC metadata and product annotation | verified |
| antiSMASH | NRPS/PKS cluster detection and annotation | verified |
| PubChem | Chemical identifiers and ligand metadata | verified |
| ChEBI | Chemical ontology and substrate normalization | verified |

This inventory records what is currently verified and marks fields that cannot be confirmed as `UNKNOWN` rather than guessing.

## 4. Source rationale

The following authoritative sources are the primary ones for the current dataset-engine phase:

- RCSB PDB / wwPDB for experimental structures and mmCIF download provenance.
- AlphaFold DB for predicted-structure augmentation and coverage expansion with explicit provenance separation.
- UniProt and NCBI/ENA for sequence identity, taxonomic mapping, and MSA linkage.
- Pfam and InterPro for domain and family boundaries.
- MIBiG and antiSMASH for NRPS/BGC annotation; these remain distinct from experimentally confirmed annotations.
- PubChem and ChEBI for substrate and chemical mapping.

The source inventory intentionally preserves the distinction between `experimental` and `predicted` structure provenance and avoids silently mixing the two.

## 5. Structural ingestion and QC design

The canonical pipeline is:

1. manifest-driven record selection
2. source download with checksum verification
3. raw artifact retention
4. parsing into canonical mmCIF or source-specific normalized records
5. quality control with accepted / rejected / quarantined statuses
6. backbone extraction and residue normalization
7. derived geometric tensors, frames, contact maps, and graph features
8. MSA linkage and split assignment
9. dataset version pinning and immutability

The quality-control contract lives in `DataQualityResult`, and the provenance contract lives in `ProvenanceRecord`.

## 6. Dataset splits and leakage control

The repository already has leak-resistant split logic in [data/splitting.py](../data/splitting.py). The new dataset-engine layer formalizes the split contract through `DatasetSplit` and the versioned dataset metadata in `DatasetVersion`.

The intended future split strategy is configured as:

- train
- validation
- test
- OOD

with family-aware, sequence-aware, and structural-identity leakage controls rather than naive random splitting.

## 7. Reproducibility and dataset versioning

Every dataset release should be pinned by:

- source database versions
- release dates
- processing pipeline version
- annotation versions
- QC configuration
- split algorithm version
- feature-generation version
- manifest checksums

This is represented by `DatasetVersion` and `stable_hash()` in [data/dataset_engine.py](../data/dataset_engine.py).

## 8. Operational command patterns

The dataset-engine foundation is designed to support commands such as:

```bash
python -c "import json; from pathlib import Path; import data.dataset_engine as d; print(d.stable_hash({'source':'RCSB PDB'}))"
python -m pytest tests/test_dataset_engineering.py
```

This is intentionally a foundation only; it does not claim to have downloaded or processed a full scientific corpus.

## 9. Implementation status

| Subsystem | Status |
| --- | --- |
| Repository audit | implemented |
| Source inventory | implemented |
| Canonical dataset schema | implemented |
| Provenance layer | implemented |
| Structural QC result contract | implemented |
| Dataset versioning | implemented |
| Split manifest contract | implemented |
| Full external corpus acquisition | not yet complete |
| Full MSA/NRPS construction | partial / foundation only |
| End-to-end structural training corpus | not yet complete |

This task intentionally focuses on the data-engine foundation required before architectural redesign of the sequence designer.
