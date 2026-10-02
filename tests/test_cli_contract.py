import json
from pathlib import Path

from chimera.cli import main


def test_version_command_reports_canonical_version(capsys):
    assert main(["version"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["software"] == "psc-chimera"
    assert result["version"]


def test_models_list_is_offline_and_reports_missing_artifacts(tmp_path: Path, capsys):
    assert main(["models", "list", "--cache-dir", str(tmp_path)]) == 0
    entries = json.loads(capsys.readouterr().out)
    assert entries
    assert all(entry["verification"]["state"] == "MISSING" for entry in entries)


def test_models_verify_offline_does_not_claim_missing_assets(capsys, tmp_path: Path):
    assert main(["models", "verify", "--offline", "--cache-dir", str(tmp_path)]) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["offline"] is True
    assert all(item["state"] == "MISSING" for item in report["results"])
