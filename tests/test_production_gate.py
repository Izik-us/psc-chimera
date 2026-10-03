import inspect
import json
import subprocess

import pytest

from chimera.production_gate import (
    Check,
    _canonical_hash,
    _evidence_identity,
    _manifest_check,
    _test_check,
    run_production_gate,
)


def test_gate_executes_required_test_command_and_does_not_count_skips(monkeypatch, tmp_path):
    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(
            args[0], 0, stdout="2 passed, 1 skipped in 0.2s\n", stderr=""
        )

    monkeypatch.setattr("chimera.production_gate.subprocess.run", fake_run)
    check = _test_check(tmp_path)

    assert check.status == "FAIL"
    assert check.evidence["passed"] == 2
    assert check.evidence["skipped"] == 1
    assert check.evidence["command"][-2:] == ["pytest", "-q"]


def test_gate_executes_required_test_command_and_records_failures(monkeypatch, tmp_path):
    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(
            args[0], 1, stdout="1 passed, 1 failed in 0.2s\n", stderr=""
        )

    monkeypatch.setattr("chimera.production_gate.subprocess.run", fake_run)
    check = _test_check(tmp_path)

    assert check.status == "FAIL"
    assert check.evidence["failures"] == 1


def test_required_dependency_needs_verified_native_smoke(monkeypatch):
    from chimera.model_store import DEPENDENCIES

    artifact_id = "esm2-t30-150m-ur50d"
    metadata = dict(DEPENDENCIES[artifact_id], required=True)
    monkeypatch.setattr(
        "chimera.production_gate.list_assets",
        lambda model_cache: [
            {
                "artifact_id": artifact_id,
                "metadata": metadata,
                "present": True,
                "manifest_present": True,
                "verification": {},
            }
        ],
    )
    monkeypatch.setattr(
        "chimera.production_gate.verify_asset",
        lambda key, model_cache: {
            "state": "INTEGRITY_VERIFIED",
            "integrity": "VALID",
            "compatibility": "UNKNOWN",
            "native_status": "NATIVE_UNVERIFIED",
            "native_validation": None,
            "sha256": "local-checksum",
            "manifest_sha256": "manifest-checksum",
        },
    )

    check, rows, _ = _manifest_check(None)

    assert check.status == "FAIL"
    assert rows[0]["declared"] is True
    assert rows[0]["acquired"] is True
    assert rows[0]["integrity_verified"] is True
    assert rows[0]["architecture_verified"] is False
    assert rows[0]["runtime_verified"] is False
    assert rows[0]["native_smoke_verified"] is False
    assert f"{artifact_id} native smoke" in check.evidence["blockers"]


def test_unregistered_workspace_model_file_blocks_dependencies(monkeypatch, tmp_path):
    monkeypatch.setattr("chimera.production_gate.list_assets", lambda model_cache: [])
    candidate = tmp_path / "Models" / "poet_weights.pt"
    candidate.parent.mkdir()
    candidate.write_bytes(b"unregistered")

    check, _, _ = _manifest_check(None, workspace_root=tmp_path)

    assert check.status == "FAIL"
    assert any(
        blocker.startswith("unregistered workspace model file:")
        for blocker in check.evidence["blockers"]
    )


def test_gate_cannot_be_given_caller_authored_pass_evidence(tmp_path, monkeypatch):
    assert "evidence" not in inspect.signature(run_production_gate).parameters
    passed = Check("tests", "PASS", "test command executed", evidence={"passed": 1})
    monkeypatch.setattr("chimera.production_gate._test_check", lambda root: passed)
    monkeypatch.setattr(
        "chimera.production_gate._git_check",
        lambda root: (
            Check("source_identity", "PASS", "clean", evidence={"commit": "abc", "worktree_status": [], "worktree_clean": True}),
            {"commit": "abc", "worktree_status": [], "worktree_clean": True},
        ),
    )
    monkeypatch.setattr(
        "chimera.production_gate._package_check",
        lambda root: Check("package_import", "PASS", "installed import"),
    )
    monkeypatch.setattr(
        "chimera.production_gate._compile_check",
        lambda root: Check("compile", "PASS", "compiled"),
    )
    monkeypatch.setattr(
        "chimera.production_gate._manifest_check",
        lambda model_cache, workspace_root=None: (
            Check(
                "external_dependency_manifest",
                "PASS",
                "manifest checked",
                evidence={"manifest_sha256": "abc"},
            ),
            [],
            "abc",
        ),
    )
    monkeypatch.setattr(
        "chimera.production_gate._runtime_checks",
        lambda: [Check("cpu_runtime", "PASS", "CPU smoke")],
    )
    evidence_file = tmp_path / "production-gate.json"

    evidence = run_production_gate(root=tmp_path, output_path=evidence_file)
    saved = json.loads(evidence_file.read_text(encoding="utf-8"))

    assert evidence["engineering_status"] == "BLOCKED"
    assert evidence["inference_smoke_result"] == "UNAVAILABLE"
    assert evidence["determinism_result"] == "UNAVAILABLE"
    assert evidence["evidence_sha256"] == _evidence_identity(saved)
    assert saved["evidence_sha256"] == _canonical_hash(
        {key: value for key, value in saved.items() if key != "evidence_sha256"}
    )
    with pytest.raises(FileExistsError, match="Refusing to replace immutable"):
        run_production_gate(root=tmp_path, output_path=evidence_file)
