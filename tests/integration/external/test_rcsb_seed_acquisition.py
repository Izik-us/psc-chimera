import os
from datetime import datetime, timezone

import pytest

from data.annotations import combine_uniprot_interpro_nrps, parse_interpro_domain_matches
from data.interpro import InterProClient
from data.msa import MSAConfig
from data.rcsb import RCSBAcquirer
from data.structures import parse_mmcif_structure
from data.uniprot import UniProtClient


pytestmark = pytest.mark.external


@pytest.mark.skipif(
    not os.getenv("CHIMERA_RUN_LIVE_EXTERNAL"),
    reason="live external integration tests require CHIMERA_RUN_LIVE_EXTERNAL=1",
)
def test_live_multisource_seed_acquisition_and_annotation_pipeline(tmp_path):
    acquirer = RCSBAcquirer(requests_per_second=1.0, maximum_workers=1)
    result = acquirer.acquire_many(["1AMU"], tmp_path / "rcsb-entry")
    assembly = acquirer.acquire_many(["1AMU"], tmp_path / "rcsb-assembly", assembly_id="1")

    assert result["requested"] == 1
    assert result["discovered"] == 1
    assert result["downloaded"] == 1
    assert result["sha256_computed"] == 1
    assert result["checksum_verified"] == 0
    assert result["parsed"] == 1
    assert result["accepted"] == 1
    assert result["rejected"] == 0
    assert result["canonical_chain_count"] == 2
    assert assembly["downloaded"] == 1
    assert assembly["parsed"] == 1

    record = result["records"][0]
    assert record["structure_id"] == "1AMU"
    assert record["sha256_computed"] is True
    assert record["checksum_verified"] is False
    assert record["checksum_status"] == "UNVERIFIED"
    assert record["parsed"] is True
    states = [item["state"] for item in record["state_history"]]
    assert "CHECKSUM_COMPUTED" in states
    assert "PARSED" in states
    assert "QC_PASSED" in states

    structure = parse_mmcif_structure(record["raw_path"], structure_id="1AMU")
    chain = next(item for item in structure.chains if item.label_asym_id == "A")
    source_accessions = [item["accession"] for item in structure.source_accessions]
    uniprot = UniProtClient(requests_per_second=2.0)
    msa, sequence_links = uniprot.retrieve_msa(
        target_sequence=chain.sequence,
        source_accessions=source_accessions,
        target_sequence_id="1AMU:A",
        structure_id="1AMU",
        chain_id="A",
        entity_id=chain.entity_id,
        max_members=16,
        config=MSAConfig(maximum_depth=16, minimum_coverage=0.4, maximum_gap_fraction=0.7),
    )
    assert msa.source == "UniRef90 / UniProtKB"
    assert msa.model_ready_row_count >= 1
    assert sequence_links and sequence_links[0].reported_accession == "P14687"
    assert sequence_links[0].sequence_identity > 0.9

    uniprot_payload, uniprot_response = uniprot.fetch_uniprot_entry("P0C062")
    uniprot_release = uniprot_response.headers.get("x-uniprot-release", "2026_03")
    interpro_payload, interpro_release = InterProClient(requests_per_second=1.0).fetch_protein_matches("P0C062")
    verification_date = datetime.now(timezone.utc).date().isoformat()
    interpro_annotation = parse_interpro_domain_matches(
        interpro_payload,
        accession="P0C062",
        release=interpro_release,
        verification_date=verification_date,
    )
    annotation = combine_uniprot_interpro_nrps(
        interpro_annotation,
        uniprot_payload,
        uniprot_release=uniprot_release,
        interpro_release=interpro_release,
        verification_date=verification_date,
    )
    assert any(domain.domain_type == "A" for domain in annotation.domains)
    assert any(domain.domain_type == "T/PCP" for domain in annotation.domains)
    assert annotation.modules