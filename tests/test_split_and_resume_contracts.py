import hashlib
import json

import torch

from chimera.checkpoint import CheckpointManifest, validate_checkpoint_compatibility
from data.splitting import (
    ExactIdentityClusterer,
    SequenceIdentityClusterer,
    build_cluster_split_manifest,
    make_clustered_split,
)


def test_exact_identity_clusterer_keeps_cluster_members_together():
    records = [
        {"aa_sequence": "MKT", "source": "a", "label_type": "proxy"},
        {"aa_sequence": "MKT", "source": "b", "label_type": "proxy"},
        {"aa_sequence": "WQF", "source": "c", "label_type": "measured"},
    ]

    result = make_clustered_split(
        records,
        train_fraction=0.5,
        validation_fraction=0.25,
        seed=7,
        clusterer=ExactIdentityClusterer(),
    )

    assert result["train_record_count"] + result["validation_record_count"] + result["test_record_count"] == 3
    assert result["train_cluster_ids"].isdisjoint(result["validation_cluster_ids"])
    assert result["validation_cluster_ids"].isdisjoint(result["test_cluster_ids"])
    assert result["train_cluster_ids"].isdisjoint(result["test_cluster_ids"])
    assert result["cluster_count"] == 2
    assert sum(
        {0, 1}.issubset(set(result[f"{partition}_indices"]))
        for partition in ("train", "validation", "test")
    ) == 1


def test_cluster_split_manifest_is_deterministic_and_hash_stable():
    records = [
        {"aa_sequence": "MKT", "source": "a", "label_type": "proxy"},
        {"aa_sequence": "MKT", "source": "b", "label_type": "proxy"},
        {"aa_sequence": "WQF", "source": "c", "label_type": "measured"},
        {"aa_sequence": "GPA", "source": "d", "label_type": "measured"},
    ]

    one = build_cluster_split_manifest(records, seed=11, dataset_version="v1")
    two = build_cluster_split_manifest(records, seed=11, dataset_version="v1")
    assert one == two
    payload = json.dumps(
        {key: value for key, value in one.items() if key != "manifest_hash"},
        sort_keys=True,
        separators=(",", ":"),
    )
    assert hashlib.sha256(payload.encode("utf-8")).hexdigest() == one["manifest_hash"]


def test_sequence_identity_heuristic_is_not_labeled_homology_aware():
    clusterer = SequenceIdentityClusterer()
    records = [{"aa_sequence": "MKT"}, {"aa_sequence": "MKT"}]

    result = make_clustered_split(records, clusterer=clusterer)

    assert clusterer.threshold == 0.3
    assert clusterer.homology_aware is False
    assert result["homology_aware"] is False
    assert result["manifest"]["homology_aware"] is False


def test_checkpoint_resume_roundtrip_restores_optimizer_and_metadata(tmp_path):
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    x = torch.randn(4, 2)
    y = model(x)
    loss = y.pow(2).mean()
    loss.backward()
    optimizer.step()

    manifest = CheckpointManifest(
        dataset_manifest="dataset-v1",
        dataset_hash="abc",
        preprocessing_hash="def",
        environment_hash="ghi",
        git_commit="deadbeef",
        config_hash="cfg",
        state_schema_hash="state",
    )
    checkpoint_path = tmp_path / "resume.pt"
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "metadata": {"manifest": manifest.__dict__},
    }, checkpoint_path)

    loaded = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert loaded["metadata"]["manifest"]["dataset_manifest"] == "dataset-v1"

    valid = validate_checkpoint_compatibility(manifest, manifest)
    assert valid == "exact-compatible"

    model2 = torch.nn.Linear(2, 2)
    optimizer2 = torch.optim.SGD(model2.parameters(), lr=0.1)
    model2.load_state_dict(model.state_dict())
    optimizer2.load_state_dict(optimizer.state_dict())
    x2 = torch.randn(4, 2)
    loss2 = model2(x2).pow(2).mean()
    loss2.backward()
    optimizer2.step()
    assert torch.isfinite(loss2)
