# CHIMERA Dataset Engineering

**Status checked 2026-10-04 on `chimera-repair`.** The data path is now executable from raw RCSB mmCIF through canonical chain/sequence records, MSA and annotation context, geometry, split assignment and immutable storage. It is an interface-ready controlled seed, not a production-scale training corpus.

Relevant implementation points:

- [data/rcsb.py](../data/rcsb.py) provides RCSB query discovery, ID manifests, cache-aware/retryable downloads, assembly files, per-record stage history and checksum semantics.
- [data/structures.py](../data/structures.py) parses PDBx/mmCIF into every entity/label-chain record, retaining source numbering, atom masks, QC, ligand coordinates/contacts and provenance.
- [data/uniprot.py](../data/uniprot.py), [data/msa.py](../data/msa.py) and [data/sequence_linkage.py](../data/sequence_linkage.py) resolve inactive accessions, retrieve UniRef members, form target-projected alignments and report sequence identity/coverage/mismatches/gaps.
- [data/interpro.py](../data/interpro.py), [data/annotations.py](../data/annotations.py) and the source adapters preserve evidence-separated NRPS domains, modules, PPant sites, substrate reactions and ligand context.
- [data/geometry.py](../data/geometry.py) derives masked frames, pair geometry, torsions, contacts and k-NN graphs.
- [data/leakage_splits.py](../data/leakage_splits.py) writes deterministic group-safe TRAIN/VALIDATION/TEST/OOD manifests.
- [data/dataset_api.py](../data/dataset_api.py) constructs one structured sample per chain and writes immutable SQLite/NPZ storage with random access and batching.
- [data/external_sources.json](../data/external_sources.json) is the date-pinned source record; it distinguishes verified access methods from sources actually used in the seed.

`data/real_acquisition.py` remains only as a compatibility facade. It no longer contains a second parser or longest-chain implementation. The model architecture has not been changed during data implementation.

## 2. Canonical dataset architecture

The stable sample is organized around these entities:

```text
Structure
  ├── Experimental metadata and assembly
  ├── Entity
  │    └── Chain
  │         ├── polymer sequence and source/canonical/sequence/MSA maps
  │         ├── N/CA/C/O/CB coordinates and atom/residue masks
  │         ├── QC and provenance
  │         ├── raw/normalized/model-ready MSA
  │         ├── geometric derivatives
  │         └── evidence-typed NRPS domains/modules
  └── CCD ligand instances and chain-aware binding-site contacts

NRPSDomain
  ├── DomainType
  ├── Module
  ├── SelectivityPositions
  ├── SubstrateAssociation
  └── StructuralRegion
```

The schema/provenance contracts live in [data/dataset_engine.py](../data/dataset_engine.py); the executable sample and storage API lives in [data/dataset_api.py](../data/dataset_api.py). `sample = dataset[i]` contains tensors, metadata, linkage, annotations, MSA, masks, geometry and provenance.

The version contract includes:

- `ProvenanceRecord`
- `DataQualityResult`
- `AcquisitionManifest`
- `DatasetSplit`
- `DatasetVersion`
- `StructuralManifest`
- `ExternalSourceInventory`
- `stable_hash()`
- `summarize_records()`

These primitives are paired with explicit schema, QC, geometry, MSA-generation, split and manifest versions. See [data/dataset_spec_v0_1.json](../data/dataset_spec_v0_1.json).

## 3. External-source inventory

The machine-readable inventory at [data/external_sources.json](../data/external_sources.json) was checked against official sources on 2026-10-04. Sources actually used for the controlled seed:

| Source | Purpose | Status |
| --- | --- | --- |
| RCSB PDB / wwPDB | Entry and assembly mmCIF, entity/chain metadata | Used; 1AMU entry and assembly 1 acquired |
| wwPDB CCD | Component name/formula/descriptors embedded in mmCIF | Used for observed ligand metadata; no PubChem/ChEBI crosswalk in the seed |
| UniProtKB / UniRef90 | Historical accession resolution, reviewed sequence/features and family members | Used; P14687 resolved to P0C062/P0C061 family |
| InterPro / Pfam | Protein domain match coordinates and family/signature evidence | Used; InterPro 110.0, Pfam member 38.2 |

Other verified but not attached to the 1AMU seed: MIBiG 4.0 JSON/GenBank, antiSMASH 8.0 result formats, PubChem PUG REST, ChEBI downloads, and NCBI E-utilities. The MIBiG/antiSMASH parsers are implemented and unit-tested; their seed record counts remain zero. AlphaFold DB and ENA are intentionally omitted from this selected inventory because predicted structures and a second nucleotide archive are not required to prove this experimental NRPS path.

This inventory records what is currently verified and marks fields that cannot be confirmed as `UNKNOWN` rather than guessing.

## 4. Source rationale

Source version, official URL, retrieval mechanism, format, verification date, usage terms, limits, identifiers and intended CHIMERA use are recorded per source in the inventory. Selected source rationale:

- RCSB/wwPDB supplies experimental coordinates, entity sequences, author/label numbering and assembly files. Its APIs are rate-limited; static files are not. RCSB recommends starting at a handful of API requests per second; this client defaults to 2 requests/second.
- UniProtKB/UniRef90 supplies sequence records and homolog clusters. The tested accession P14687 is inactive and demereged to P0C061/P0C062; the resolver follows that source record and sequence alignment chooses the structure-linked accession.
- InterPro 110.0 supplies computational match spans, including GrsA A and carrier domains. UniProt feature evidence remains distinct from InterPro signatures.
- MIBiG and antiSMASH remain separate curated/computed evidence sources; no BGC was guessed from a protein name.
- PubChem is optional and subject to its published 5-requests/second PUG REST policy. ChEBI is optional and has monthly releases plus nightly ontology updates. Neither is used to infer a ligand's biochemical role from its name.

The source inventory intentionally preserves the distinction between `experimental` and `predicted` structure provenance and avoids silently mixing the two.

## 5. Structural ingestion and QC

The canonical pipeline is:

Implemented state history: REQUESTED, DISCOVERED, DOWNLOADING, DOWNLOADED, CHECKSUM_COMPUTED, optionally CHECKSUM_VERIFIED, PARSED, QC_PASSED/QC_FAILED and ACCEPTED/REJECTED/QUARANTINED. SHA-256 computation is never treated as external verification. The 1AMU source endpoint supplied no expected digest, so the seed records `sha256_computed=true`, `checksum_verified=false`, `checksum_status=UNVERIFIED`.

Canonical chain records retain entity ID, label/auth chain IDs, author residue numbers, label sequence numbers, insertion codes, residue names, alternate locations, occupancy/B-factor, N/CA/C/O/CB coordinates and masks, unresolved polymer positions and a source-to-sequence map. QC checks chain length, unresolved/backbone/side-chain missingness, duplicate author numbering, nonstandard residues, peptide continuity and backbone bond lengths. Ligand contacts are computed across protein chains and carry chain/entity/residue references. Assembly files are separate explicit acquisition records; no longest-chain selection is used.

## 6. Dataset splits and leakage control

[data/leakage_splits.py](../data/leakage_splits.py) merges records sharing computed sequence-identity groups or supplied structural, protein-family, domain-family, NRPS-family, taxonomy, substrate and module-composition labels before assigning TRAIN/VALIDATION/TEST/OOD. OOD holdout is a named novelty dimension. Missing grouping dimensions are explicitly reported, not claimed as guarded. Small controlled corpora use a deterministic pairwise aligner; larger corpora must provide externally computed sequence cluster IDs. The 1AMU integration fixture has one linked sample and therefore demonstrates TRAIN assignment only; it does not provide independent validation, test or OOD examples.

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

The frozen specification [data/dataset_spec_v0_1.json](../data/dataset_spec_v0_1.json) records source versions, processing/schema/QC/geometry/MSA/split versions and raw-source/manifest hashes. `DatasetVersion.content_hash` and split manifests are deterministic. `write_immutable_dataset` refuses to overwrite a versioned SQLite dataset.

## 8. Operational command patterns

Useful validation commands:

```bash
python -m pytest tests/unit tests/integration/offline -q
set CHIMERA_RUN_LIVE_EXTERNAL=1
python -m pytest tests/integration/external -q
```

Live tests are opt-in and ordinary CI does not require internet.

## 9. Implementation status

| Subsystem | Evidence/status |
| --- | --- |
| RCSB entry + biological assembly acquisition | Live tested on 1AMU; 1 accepted entry, 0 rejected/quarantined, two canonical chains; assembly 1 acquired |
| Integrity | SHA-256 computed for both files; source checksum unverified because no authoritative expected checksum was supplied |
| Structural QC and canonical representation | Offline unit tests plus live 1AMU parse; both chains accepted; masks and author/label mappings retained |
| Sequence linkage | One chain linked to P0C062 from historical P14687 by global alignment; live UniRef pathway also returned a ≥99% match |
| MSA | One UniRef90 family; 4 raw aligned rows (query plus 3 source sequence records), one model-ready row after 90% deduplication, Neff 1.0 |
| NRPS annotation | One real UniProt/InterPro GrsA annotation with computed A/T spans, source-coded PPant site, A/T module relationship and curated reaction context |
| MIBiG / antiSMASH | Source methods verified and parsers tested; zero linked records in the seed |
| Ligand/substrate | Eight ligand instances in the two-chain asymmetric unit; chain-aware experimental contacts retained. The four chain-A contexts include PHE/AMP/Mg/SO4; role remains UNASSIGNED without a chemical crosswalk/experimental role assignment |
| Geometry | Derived geometry and masks run in offline full path; rigid transform, masking and index-permutation properties unit-tested |
| Splits | Hashed leakage-safe manifest generated; controlled one-sample seed assigned TRAIN. No independent VAL/TEST/OOD examples are present |
| Storage/version | SQLite/NPZ random-access round-trip passed; seed spec is CHIMERA-DATASET-v0.1 |
| MIBiG/antiSMASH and chemistry limitations | No MIBiG BGC association or antiSMASH result imported; PubChem/ChEBI canonical ligand mapping not performed |

This is a frozen, tested data interface and a controlled real-data proof, not a claim that the single-family seed is sufficient for model training.
