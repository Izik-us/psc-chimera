from pathlib import Path

from chimera.model_store import (
    ModelIntegrityError,
    PUBLIC_ASSETS,
    asset_path,
    cache_root,
    ensure_asset,
    verify_asset,
)


def test_public_manifest_contains_core_assets():
    assert "esm2_t30_150m" in PUBLIC_ASSETS
    assert "esmfold_v1" in PUBLIC_ASSETS
    assert "proteinmpnn_v48_020" in PUBLIC_ASSETS
    assert "rfdiffusion_base" in PUBLIC_ASSETS


def test_cache_root_is_overrideable(tmp_path: Path):
    assert cache_root(tmp_path) == tmp_path.resolve()


def test_unknown_checkpoint_without_manifest_is_not_reused(tmp_path: Path):
    asset = PUBLIC_ASSETS["esm2_t30_150m"]
    destination = asset_path("esm2_t30_150m", tmp_path)
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b"already-present")
    assert verify_asset("esm2_t30_150m", tmp_path)["state"] == "UNKNOWN"
    try:
        ensure_asset("esm2_t30_150m", tmp_path)
    except ModelIntegrityError as exc:
        assert "without a CHIMERA integrity manifest" in str(exc)
    else:
        raise AssertionError("unmanifested local weights must not be trusted")


def test_download_is_atomic_and_manifest_records_local_hash(tmp_path: Path, monkeypatch):
    import io

    from chimera import model_store

    class Response(io.BytesIO):
        headers = {"Content-Length": "16"}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.close()

    monkeypatch.setattr(model_store.urllib.request, "urlopen", lambda *args, **kwargs: Response(b"test-model-bytes"))
    path = ensure_asset("esm2_t30_150m", tmp_path)

    result = verify_asset("esm2_t30_150m", tmp_path)
    assert path.name == "esm2_t30_150M_UR50D.pt"
    assert result["state"] == "INTEGRITY_VERIFIED"
    assert result["integrity_basis"] == "LOCALLY_RECORDED_SHA256"
    assert result["upstream_origin"] == "CHECKSUM_UNCONFIRMED"
    assert result["compatibility"] == "UNKNOWN"


def test_cached_artifact_tampering_is_detected(tmp_path: Path, monkeypatch):
    import io

    from chimera import model_store

    class Response(io.BytesIO):
        headers = {"Content-Length": "16"}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.close()

    monkeypatch.setattr(model_store.urllib.request, "urlopen", lambda *args, **kwargs: Response(b"test-model-bytes"))
    path = ensure_asset("esm2_t30_150m", tmp_path)
    path.write_bytes(b"tampered-model!")

    assert verify_asset("esm2_t30_150m", tmp_path)["state"] == "CORRUPT"


def test_manifest_metadata_tampering_is_detected_even_if_manifest_hash_is_recomputed(
    tmp_path: Path, monkeypatch
):
    import io
    import json

    from chimera import model_store

    class Response(io.BytesIO):
        headers = {"Content-Length": "16"}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.close()

    monkeypatch.setattr(
        model_store.urllib.request,
        "urlopen",
        lambda *args, **kwargs: Response(b"test-model-bytes"),
    )
    ensure_asset("esm2_t30_150m", tmp_path)
    manifest_path = asset_path("esm2_t30_150m", tmp_path).parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["license"] = "MIT (modified)"
    manifest["manifest_sha256"] = model_store._manifest_hash(manifest)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    assert verify_asset("esm2_t30_150m", tmp_path)["state"] == "CORRUPT"
