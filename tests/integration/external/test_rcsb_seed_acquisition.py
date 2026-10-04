import os

import pytest

from data.real_acquisition import run_external_seed_pipeline


pytestmark = pytest.mark.external


@pytest.mark.skipif(
    not os.getenv("CHIMERA_RUN_LIVE_EXTERNAL"),
    reason="live external integration tests require CHIMERA_RUN_LIVE_EXTERNAL=1",
)
def test_live_rcsb_seed_acquisition_verifies_real_mmcif_and_manifest(tmp_path):
    result = run_external_seed_pipeline(["1AMU"], tmp_path / "rcsb-seed")

    assert result["requested"] == 1
    assert result["discovered"] == 1
    assert result["downloaded"] == 1
    assert result["checksum_verified"] == 1
    assert result["parsed"] == 1
    assert result["accepted"] == 1
    assert result["rejected"] == 0

    record = result["records"][0]
    assert record["structure_id"] == "1AMU"
    assert record["checksum_verified"] is True
    assert record["parsed"] is True
    assert "manifest" in record