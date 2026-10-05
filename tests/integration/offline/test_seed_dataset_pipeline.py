import json
from pathlib import Path

from data.annotations import combine_uniprot_interpro_nrps, parse_interpro_domain_matches
from data.dataset_api import SQLiteChimeraDataset, build_dataset_sample, write_immutable_dataset
from data.leakage_splits import SplitConfig, generate_leakage_safe_splits
from data.msa import MSAConfig, build_msa_record
from data.sequence_linkage import SequenceCandidate, link_structure_sequence
from data.structures import StructuralQCConfig, parse_mmcif_structure


FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "data_engineering"


def test_1amu_derived_fixture_flows_through_canonical_dataset_pipeline_offline(tmp_path):
    # This fixture is a deterministic normalized reconstruction from the pinned real 1AMU acquisition record.
    # The live external test remains the authoritative raw-RCSB acquisition path.
    entry_directory = FIXTURES / "rcsb" / "entry"
    acquisition = json.loads((entry_directory / "acquisition_manifest.jsonl").read_text(encoding="utf-8").splitlines()[0])
    structure = parse_mmcif_structure(
        entry_directory / "1AMU.cif",
        structure_id="1AMU",
        # The offline fixture is the canonical N/CA/C/O/CB representation, not a
        # full raw atom dump. Do not interpret omitted non-canonical sidechain
        # atoms as experimental missingness in this derived fixture.
        config=StructuralQCConfig(maximum_missing_sidechain_fraction=1.0),
    )
    chain = next(item for item in structure.chains if item.label_asym_id == "A")

    msa_snapshot = json.loads(
        (FIXTURES / "uniprot" / "uniref90_P0C061_target_alignment.json").read_text(encoding="utf-8")
    )
    msa = build_msa_record(
        msa_snapshot["raw_alignment"],
        msa_family_id=msa_snapshot["cluster_id"],
        target_sequence_id="1AMU:A",
        source=msa_snapshot["source"],
        source_version=msa_snapshot["source_release"],
        target_sequence=chain.sequence,
        raw_sequence_records=msa_snapshot["raw_sequence_records"],
        generation_method=msa_snapshot["generation_method"],
        generation_version=msa_snapshot["generation_version"],
        config=MSAConfig(minimum_coverage=0.4, maximum_gap_fraction=0.7, maximum_depth=16),
    )

    uniprot = json.loads((FIXTURES / "uniprot" / "uniprotkb_P0C062.json").read_text(encoding="utf-8"))
    interpro_snapshot = json.loads(
        (FIXTURES / "interpro" / "P0C062.json").read_text(encoding="utf-8")
    )
    interpro = parse_interpro_domain_matches(
        interpro_snapshot["payload"],
        accession="P0C062",
        release=interpro_snapshot["release"],
        verification_date="2026-10-04",
    )
    annotation = combine_uniprot_interpro_nrps(
        interpro,
        uniprot,
        uniprot_release="2026_03",
        interpro_release=interpro_snapshot["release"],
        verification_date="2026-10-04",
    )
    linkage = link_structure_sequence(
        structure_id=structure.structure_id,
        chain_id=chain.label_asym_id,
        entity_id=chain.entity_id,
        sequence=chain.sequence,
        candidates=[
            SequenceCandidate(
                accession="P0C062",
                sequence=uniprot["sequence"]["value"],
                source="UniProtKB",
                source_version="2026_03",
            )
        ],
        reported_accession="P14687",
    )

    split_manifest = generate_leakage_safe_splits(
        [
            {
                "record_id": f"{structure.structure_id}:{chain.label_asym_id}:{chain.entity_id}",
                "sequence": chain.sequence,
                "sequence_cluster_id": msa.msa_family_id,
                "protein_family_id": "IPR010071",
                "nrps_family_id": "UniProt:P0C062:GrsA",
                "taxonomy_id": "1393",
                "substrate_ids": ["PHE"],
                "module_composition": "A-T/PCP",
            }
        ],
        dataset_version="CHIMERA-DATASET-v0.1",
        config=SplitConfig(
            train_fraction=1.0,
            validation_fraction=0.0,
            test_fraction=0.0,
            ood_fraction=0.0,
            ood_dimension=None,
        ),
    )
    sample_id = f"{structure.structure_id}:{chain.label_asym_id}:{chain.entity_id}"
    sample = build_dataset_sample(
        structure,
        chain,
        msa=msa,
        sequence_linkage=linkage,
        nrps_annotation=annotation,
        acquisition_record=acquisition,
        dataset_version="CHIMERA-DATASET-v0.1",
        split=split_manifest.assignments[sample_id],
    )
    store_path = tmp_path / "chimera-seed-v0.1.sqlite"
    write_immutable_dataset(
        store_path,
        [sample],
        dataset_version="CHIMERA-DATASET-v0.1",
        split_manifest_sha256=split_manifest.manifest_sha256,
    )
    stored = SQLiteChimeraDataset(store_path)[0]

    assert acquisition["sha256_computed"] is True
    assert acquisition["checksum_status"] == "UNVERIFIED"
    assert acquisition["status"] == "accepted"
    assert len(structure.chains) == 2
    assert chain.quality.status == "accepted"
    assert linkage.reported_accession == "P14687"
    assert linkage.external_accession == "P0C062"
    assert linkage.sequence_identity > 0.9
    assert msa.num_rows == 4
    assert msa.model_ready_row_count == 1
    assert msa.neff == 1.0
    assert any(domain.domain_type == "A" for domain in annotation.domains)
    assert any(domain.domain_type == "T/PCP" for domain in annotation.domains)
    assert annotation.modules[0].adenylation_domain_id
    assert annotation.modules[0].carrier_domain_id
    assert annotation.modules[0].evidence.evidence_class.value == "SEQUENCE_INFERRED"
    assert annotation.products[0].substrate_name == "L-phenylalanine"
    assert annotation.products[0].evidence.evidence_class.value == "DATABASE_CURATED"
    assert sample["ligand_contexts"]
    assert all(context["role"] == "UNASSIGNED" for context in sample["ligand_contexts"])
    assert sample["geometry"]["edge_index"].shape[0] == len(chain.sequence)
    assert stored["sequence"] == chain.sequence
    assert stored["split"] == "TRAIN"
    assert stored["acquisition"]["checksum_verified"] is False