from pathlib import Path


def test_no_runtime_compatibility_shim():
    root = Path(__file__).resolve().parents[1]
    assert not (root / "chimera" / "runtime_compat.py").exists()


def test_package_init_has_no_compat_patch_import():
    root = Path(__file__).resolve().parents[1]
    text = (root / "chimera" / "__init__.py").read_text(encoding="utf-8")
    assert "runtime_compat" not in text


def test_flow_matching_uses_canonical_lie_module():
    root = Path(__file__).resolve().parents[1]
    text = (root / "chimera" / "flow_matching.py").read_text(encoding="utf-8")
    assert "from .lie import" in text or "from chimera.lie import" in text
