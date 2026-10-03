import json
from pathlib import Path

from chimera.production_gate import (
    _architecture_manifest_check,
    production_dependency_inventory,
)
from chimera.cli import main

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPOSITORY_ROOT / "chimera" / "architecture_dependencies.json"


def _matching_dependency_rows(document):
    return [
        {
            "artifact_id": asset["artifact_id"],
            "native_status": asset["status"],
            "sha256": asset["sha256"],
        }
        for asset in document["external_assets"]
        if asset["artifact_id"] is not None
    ]


def test_architecture_inventory_is_structurally_closed_but_release_blocked(tmp_path):
    document = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    test_manifest = tmp_path / "architecture_dependencies.json"
    test_manifest.write_text(json.dumps(document), encoding="utf-8")

    check, manifest_hash = _architecture_manifest_check(
        _matching_dependency_rows(document),
        manifest_path=test_manifest,
        repository_root=REPOSITORY_ROOT,
    )

    assert check.status == "FAIL"
    assert check.evidence["component_count"] == len(document["components"])
    assert check.evidence["external_asset_count"] == len(document["external_assets"])
    assert check.evidence["workspace_model_file_disposition_count"] == len(
        document["workspace_model_file_dispositions"]
    )
    assert check.evidence["data_asset_count"] == len(
        next(
            asset["files"]
            for asset in document["external_assets"]
            if asset["id"] == "data_manifests"
        )
    )
    assert check.evidence["data_asset_discrepancies"] == []
    assert check.evidence["external_cache_discrepancies"] == []
    assert manifest_hash == check.evidence["manifest_sha256"]
    assert "no selected production composition" in check.evidence["blockers"]
    assert "no selected production artifact" in check.evidence["blockers"]
    assert not any(
        "unresolved dependency" in blocker or "missing test reference" in blocker
        for blocker in check.evidence["blockers"]
    )


def test_production_dependency_command_reports_blocked_candidate_without_claiming_release():
    result = production_dependency_inventory(MANIFEST_PATH)

    assert result["production_status"] == "BLOCKED_INTERNAL"
    assert result["production_composition"] is None
    assert result["production_artifact"] is None
    assert result["candidate_composition"]["composition_id"] == (
        "canonical-chimerav2-research-candidate"
    )
    assert len(result["candidate_composition"]["required_components"]) == 20
    assert len(
        result["candidate_composition"]["required_runtime_dependencies"]
    ) == 4
    assert "No selected trained and validated CHIMERA checkpoint." in result["blockers"]


def test_production_dependencies_cli_emits_json(capsys):
    assert main(["production-dependencies"]) == 0
    output = json.loads(capsys.readouterr().out)

    assert output["production_status"] == "BLOCKED_INTERNAL"
    assert output["production_artifact"] is None


def test_architecture_inventory_detects_dataset_hash_mismatch(tmp_path):
    document = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    data_manifest = next(
        asset
        for asset in document["external_assets"]
        if asset["id"] == "data_manifests"
    )
    data_manifest["files"][0]["sha256"] = "0" * 64
    test_manifest = tmp_path / "architecture_dependencies.json"
    test_manifest.write_text(json.dumps(document), encoding="utf-8")

    check, _ = _architecture_manifest_check(
        _matching_dependency_rows(document),
        manifest_path=test_manifest,
        repository_root=REPOSITORY_ROOT,
    )

    assert check.status == "FAIL"
    assert (
        "dataset_fath2011_expression_csv: SHA-256 differs from inventory"
        in check.evidence["data_asset_discrepancies"]
    )


def test_architecture_inventory_rejects_reappearing_model_file(tmp_path):
    document = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    test_manifest = tmp_path / "architecture_dependencies.json"
    test_manifest.write_text(json.dumps(document), encoding="utf-8")
    candidate = tmp_path / "weights" / "poet_weights.pt"
    candidate.parent.mkdir()
    candidate.write_bytes(b"unregistered test candidate")

    check, _ = _architecture_manifest_check(
        _matching_dependency_rows(document),
        manifest_path=test_manifest,
        repository_root=tmp_path,
    )

    assert check.status == "FAIL"
    assert any(
        "weights/poet_weights.pt: removed or unregistered model file is still present"
        in blocker
        for blocker in check.evidence["blockers"]
    )


def test_architecture_inventory_rejects_dangling_dependency(tmp_path):
    document = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    document["components"][0]["dependencies"].append("undeclared_dependency")
    test_manifest = tmp_path / "architecture_dependencies.json"
    test_manifest.write_text(json.dumps(document), encoding="utf-8")

    check, _ = _architecture_manifest_check(
        _matching_dependency_rows(document),
        manifest_path=test_manifest,
        repository_root=REPOSITORY_ROOT,
    )

    assert check.status == "FAIL"
    assert any(
        "input_contract: unresolved dependency undeclared_dependency" in blocker
        for blocker in check.evidence["blockers"]
    )
