from types import SimpleNamespace

import numpy as np
import pytest

from data.rcsb import AcquisitionState, RCSBAcquirer, RCSBQuery, build_rcsb_query_payload


def _fake_structure():
    chain = SimpleNamespace(
        label_asym_id="A",
        auth_asym_id="X",
        entity_id="1",
        sequence="ACDE",
        sequence_sha256="sequence-hash",
        residue_mask=np.ones(4, dtype=bool),
        quality=SimpleNamespace(status="accepted", to_dict=lambda: {"status": "accepted", "reasons": []}),
        to_dict=lambda include_arrays=False: {"sequence": "ACDE"},
    )
    return SimpleNamespace(
        chains=[chain],
        experimental_method="X-RAY DIFFRACTION",
        resolution_angstrom=2.0,
        release_date="2024-01-01",
        assembly_ids=["1"],
        ligands=[],
        source_accessions=[],
    )


def _mock_download(_self, _structure_id, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(b"fixed-mmCIF-fixture")
    return True, None


def test_rcsb_query_supports_source_filters_and_nrps_search():
    payload = build_rcsb_query_payload(
        RCSBQuery(
            release_date_from="2020-01-01",
            release_date_to="2025-01-01",
            experimental_methods=("X-RAY DIFFRACTION", "ELECTRON MICROSCOPY"),
            maximum_resolution_angstrom=2.5,
            minimum_sequence_length=100,
            maximum_sequence_length=1000,
            taxonomy_ids=(562, 287),
            nrps_targeted=True,
        )
    )

    serialized = str(payload)
    assert "initial_release_date" in serialized
    assert "resolution_combined" in serialized
    assert "rcsb_sample_sequence_length" in serialized
    assert "taxonomy_lineage.id" in serialized
    assert "nonribosomal peptide synthetase" in serialized
    assert payload["request_options"]["paginate"]["rows"] == 100


def test_acquisition_records_unverified_checksum_and_all_stages(tmp_path, monkeypatch):
    acquirer = RCSBAcquirer(requests_per_second=1000000)
    monkeypatch.setattr(RCSBAcquirer, "_download", _mock_download)
    monkeypatch.setattr("data.rcsb.parse_mmcif_structure", lambda *_args, **_kwargs: _fake_structure())

    result = acquirer.acquire_id("1abc", tmp_path)

    states = [entry["state"] for entry in result["state_history"]]
    assert result["status"] == "accepted"
    assert result["sha256_computed"] is True
    assert result["checksum_verified"] is False
    assert result["checksum_status"] == "UNVERIFIED"
    assert result["canonical_chain_count"] == 1
    assert states == [
        "REQUESTED", "DISCOVERED", "DOWNLOADING", "DOWNLOADED",
        "CHECKSUM_COMPUTED", "PARSED", "QC_PASSED", "ACCEPTED",
    ]

    cached = acquirer.acquire_id("1ABC", tmp_path)
    assert cached["cached"] is True
    assert "DOWNLOADING" not in [event["state"] for event in cached["state_history"]]


def test_authoritative_expected_checksum_controls_verification_and_quarantine(tmp_path, monkeypatch):
    acquirer = RCSBAcquirer(requests_per_second=1000000)
    monkeypatch.setattr(RCSBAcquirer, "_download", _mock_download)
    monkeypatch.setattr("data.rcsb.parse_mmcif_structure", lambda *_args, **_kwargs: _fake_structure())

    verified = acquirer.acquire_id(
        "1ABC", tmp_path / "match", expected_sha256=__import__("hashlib").sha256(b"fixed-mmCIF-fixture").hexdigest()
    )
    assert verified["checksum_verified"] is True
    assert verified["checksum_verification_source"] == "caller_supplied_manifest"

    rejected = acquirer.acquire_id("1ABC", tmp_path / "mismatch", expected_sha256="0" * 64)
    assert rejected["status"] == "quarantined"
    assert rejected["failure_reason"] == "CHECKSUM_FAILURE"
    assert rejected["parsed"] is False
    assert rejected["state_history"][-1]["state"] == AcquisitionState.QUARANTINED.value


def test_invalid_rcsb_identifier_is_rejected_before_network(tmp_path):
    with pytest.raises(ValueError, match="invalid RCSB structure identifier"):
        RCSBAcquirer().acquire_id("../1ABC", tmp_path)


def test_biological_assembly_uses_official_assembly_endpoint_and_path(tmp_path, monkeypatch):
    acquirer = RCSBAcquirer(requests_per_second=1000000)
    monkeypatch.setattr(RCSBAcquirer, "_download", _mock_download)
    parse_arguments = {}

    def parse(_path, **kwargs):
        parse_arguments.update(kwargs)
        return _fake_structure()

    monkeypatch.setattr("data.rcsb.parse_mmcif_structure", parse)
    result = acquirer.acquire_id("1ABC", tmp_path, assembly_id="1")

    assert result["raw_path"].endswith("1ABC-assembly1.cif")
    assert result["download_url"].endswith("1abc-assembly1.cif")
    assert result["assembly_id"] == "1"
    assert parse_arguments["assembly_id"] == "1"