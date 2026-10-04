import json

from data.dataset_engine import (
    AcquisitionManifest,
    DataQualityResult,
    DatasetSplit,
    DatasetVersion,
    ExternalSourceInventory,
    ProvenanceRecord,
    StructuralManifest,
)


def test_acquisition_manifest_tracks_requested_download_and_processing_states():
    manifest = AcquisitionManifest(
        source="RCSB PDB",
        query_filter="resolution <= 2.5 A and experimental_method == X-ray",
        source_release="2026-09-30",
        record_identifiers=["1AMU", "2GRY"],
        requested_timestamp="2026-10-04T00:00:00Z",
        download_status="downloaded",
        checksum="abc123",
        file_path="/tmp/1AMU.cif",
        processing_status="validated",
    )

    payload = manifest.to_dict()
    assert payload["source"] == "RCSB PDB"
    assert payload["download_status"] == "downloaded"
    assert payload["processing_status"] == "validated"
    assert payload["record_identifiers"] == ["1AMU", "2GRY"]


def test_data_quality_result_preserves_rejection_reason_and_status():
    result = DataQualityResult(
        status="rejected",
        reasons=["missing_backbone_atoms", "broken_peptide_bond"],
        qc_version="qc-v1",
    )

    assert result.status == "rejected"
    assert result.reasons == ["missing_backbone_atoms", "broken_peptide_bond"]
    assert result.qc_version == "qc-v1"


def test_dataset_version_and_split_are_deterministic_and_serializable():
    version = DatasetVersion(
        name="CHIMERA-DATASET-v0.1",
        source_versions={"RCSB PDB": "2026-09-30"},
        processing_version="dataset-engine-v0.1",
        annotation_versions={"MIBiG": "2026-04"},
        qc_config={"resolution_max_a": 2.5},
        split_algorithm="family-aware-cluster-split-v1",
        feature_generation_version="geometry-v1",
        manifest_checksums={"experimentals": "sha256:abc"},
    )

    split = DatasetSplit(
        name="train",
        record_ids=["s1", "s2"],
        split_policy="sequence_and_family_leakage_guard",
        leakage_control={"max_sequence_identity": 0.3},
    )

    assert version.name == "CHIMERA-DATASET-v0.1"
    assert split.to_dict()["name"] == "train"
    assert json.dumps(version.to_dict(), sort_keys=True)


def test_external_source_inventory_records_only_verified_or_unknown_values():
    inventory = ExternalSourceInventory(
        sources=[
            {
                "source_name": "RCSB PDB",
                "source_type": "experimental_structure",
                "official_url": "https://www.rcsb.org/",
                "purpose": "Experimental protein structures",
                "authoritative_status": "verified",
                "data_type": "PDBx/mmCIF",
                "identifier_scheme": "PDB accession",
                "current_release_or_version": "2026-09-30",
                "release_date": "2026-09-30",
                "access_method": "FTP/REST",
                "API_or_download_endpoint": "https://files.rcsb.org/download/",
                "file_format": "mmCIF",
                "license_or_usage_terms": "Open access",
                "checksum_support": "yes",
                "rate_limit_information": "UNKNOWN",
                "update_strategy": "weekly release",
                "required_for_training": True,
                "optional_for_training": False,
                "provenance_requirements": "download timestamp + checksum + release pin",
            }
        ]
    )

    payload = inventory.to_dict()
    assert payload["sources"][0]["source_name"] == "RCSB PDB"
    assert payload["sources"][0]["rate_limit_information"] == "UNKNOWN"


def test_structural_manifest_and_provenance_round_trip():
    provenance = ProvenanceRecord(
        source_database="RCSB PDB",
        source_accession="1AMU",
        source_version="2026-09-30",
        download_timestamp="2026-10-04T00:00:00Z",
        source_file_checksum="sha256:source",
        processed_file_checksum="sha256:processed",
        processing_pipeline_version="dataset-engine-v0.1",
        parser_version="pdbx-reader-v1",
        normalization_version="canonical-structure-v1",
        qc_version="qc-v1",
    )

    manifest = StructuralManifest(
        structure_id="1AMU:A",
        source="RCSB PDB",
        source_version="2026-09-30",
        release_date="2026-09-30",
        experimental_method="X-RAY DIFFRACTION",
        resolution=2.0,
        entity_id="1",
        chain_id="A",
        assembly_id="1",
        sequence_sha256="sha256:seq",
        coordinate_sha256="sha256:coords",
        processing_version="dataset-engine-v0.1",
        quality_flags=[],
        num_residues=100,
        num_resolved_residues=98,
        num_missing_residues=2,
        num_missing_backbone_atoms=0,
        ligand_count=1,
        polymer_type="protein",
        organism="Escherichia coli",
        taxonomy_id="562",
        split="train",
        provenance=provenance,
    )

    assert manifest.structure_id == "1AMU:A"
    assert manifest.provenance.source_accession == "1AMU"
    assert manifest.to_dict()["split"] == "train"
