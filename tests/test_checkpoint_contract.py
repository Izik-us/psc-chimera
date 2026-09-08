import torch
import torch.nn as nn

from chimera.checkpoint import CheckpointManifest, load_manifest, save_manifest


def test_checkpoint_manifest_roundtrip(tmp_path):
    path = tmp_path / "manifest.json"
    manifest = CheckpointManifest()
    save_manifest(path, manifest)
    loaded = load_manifest(path)
    assert loaded == manifest


def test_checkpoint_manifest_rejects_non_autoregressive_policy(tmp_path):
    path = tmp_path / "manifest.json"
    save_manifest(path)
    data = path.read_text(encoding="utf-8").replace('"autoregressive"', '"parallel"')
    path.write_text(data, encoding="utf-8")
    try:
        load_manifest(path)
    except ValueError as exc:
        assert "autoregressive" in str(exc)
    else:
        raise AssertionError("invalid checkpoint policy was accepted")
