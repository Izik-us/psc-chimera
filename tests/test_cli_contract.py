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



def test_train_command_routes_to_canonical_runner(monkeypatch, tmp_path: Path):
    from scripts import train_canonical

    captured = {}

    def fake_main(argv):
        captured["argv"] = argv
        return 23

    monkeypatch.setattr(train_canonical, "main", fake_main)
    dataset = tmp_path / "train.jsonl"
    checkpoint = tmp_path / "CHIMERAv2.pt"
    assert main([
        "train", "sequence", "resume",
        "--dataset", str(dataset),
        "--checkpoint", str(checkpoint),
        "--epochs", "2000",
        "--lr", "4e-4",
    ]) == 23
    assert captured["argv"][captured["argv"].index("--regime") + 1] == "sequence"
    assert captured["argv"][captured["argv"].index("--resume") + 1] == str(checkpoint)
    assert captured["argv"][captured["argv"].index("--learning-rate") + 1] == "4e-4"


def test_checkpoint_inspect_reports_single_checkpoint_stage_history(tmp_path: Path, capsys):
    import torch

    checkpoint = tmp_path / "CHIMERAv2.pt"
    torch.save({
        "artifact_type": "canonical_training_checkpoint",
        "checkpoint_schema_version": 2,
        "manifest": {"dataset_version": "v1", "dataset_hash": "abc"},
        "training": {
            "regime": "sequence",
            "epoch": 4,
            "global_step": 19,
            "trainable_components": ["sequence_policy"],
            "stage_history": [{"regime": "representation", "status": "stage_transitioned"}],
        },
    }, checkpoint)
    assert main(["checkpoint", "inspect", "--checkpoint", str(checkpoint)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["active_regime"] == "sequence"
    assert report["global_step"] == 19
    assert report["stage_history"][0]["regime"] == "representation"
