from pathlib import Path

from scripts.pretrained_manager import PUBLIC_ASSETS, cache_root, ensure_asset


def test_public_manifest_contains_core_assets():
    assert "esm2_t30_150m" in PUBLIC_ASSETS
    assert "esmfold_v1" in PUBLIC_ASSETS
    assert "proteinmpnn_v48_020" in PUBLIC_ASSETS
    assert "rfdiffusion_base" in PUBLIC_ASSETS


def test_cache_root_is_overrideable(tmp_path: Path):
    assert cache_root(tmp_path) == tmp_path


def test_existing_asset_is_reused(tmp_path: Path):
    asset = PUBLIC_ASSETS["esm2_t30_150m"]
    destination = tmp_path / asset.filename
    destination.write_bytes(b"already-present")
    assert ensure_asset("esm2_t30_150m", tmp_path) == destination
    assert destination.read_bytes() == b"already-present"
