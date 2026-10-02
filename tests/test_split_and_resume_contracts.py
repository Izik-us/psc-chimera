import hashlib
import json
from dataclasses import replace

import torch

from chimera.checkpoint import CheckpointManifest, validate_checkpoint_compatibility
from data.splitting import (
    ExactIdentityClusterer,
    SequenceIdentityClusterer,
    build_cluster_split_manifest,
    make_clustered_split,
)


def _complete_manifest(**overrides):
    values = {
        "format_version": 4,
        "config_hash": "model-config",
        "state_schema_hash": "state-schema",
        "objective_schema_hash": "objective-schema",
        "git_commit": "commit-a",
        "git_worktree_clean": True,
        "source_tree_hash": "source-a",
        "dataset_manifest": "dataset-manifest-v1",
        "dataset_version": "v1",
        "dataset_hash": "dataset-hash",
        "preprocessing_hash": "preprocessing-hash",
        "environment_hash": "environment-hash",
        "python_version": "3.12.0",
        "torch_version": "2.8.0",
        "platform": "Windows-test",
        "numpy_version": "2.0.0",
        "cuda_available": False,
        "cuda_version": None,
        "cudnn_version": None,
        "cuda_device_names": None,
        "deterministic_algorithms_enabled": False,
        "cudnn_deterministic": False,
        "cudnn_benchmark": False,
        "matmul_allow_tf32": False,
        "cudnn_allow_tf32": False,
        "training_regime": "sequence",
        "random_seed": 11,
        "min_validation_examples": 30,
        "optimizer_class": "torch.optim.AdamW",
        "optimizer_hash": "optimizer-config",
        "scheduler_class": "none",
        "scheduler_hash": "no-scheduler",
    }
    values.update(overrides)
    return CheckpointManifest(**values)


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

    manifest = _complete_manifest(dataset_manifest="dataset-v1", dataset_hash="abc")
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


def test_checkpoint_compatibility_verdicts_cover_provenance_and_contract_fields():
    manifest = _complete_manifest()
    assert validate_checkpoint_compatibility(manifest, manifest) == "exact-compatible"

    source_update = replace(
        manifest,
        git_commit="commit-b",
        git_worktree_clean=False,
        source_tree_hash="source-b",
    )
    assert validate_checkpoint_compatibility(source_update, manifest) == "incompatible"
    source_tree_only_update = replace(manifest, source_tree_hash="source-b")
    assert validate_checkpoint_compatibility(source_tree_only_update, manifest) == "incompatible"
    serialization_update = replace(manifest, serialization_revision=0)
    assert validate_checkpoint_compatibility(
        serialization_update,
        manifest,
        migration_id="v4-serialization-metadata-v0-to-v1",
    ) == "expected-compatible"

    incompatible_fields = (
        "state_schema_hash",
        "objective_schema_hash",
        "config_hash",
        "dataset_manifest",
        "dataset_version",
        "dataset_hash",
        "preprocessing_hash",
        "environment_hash",
        "training_regime",
        "random_seed",
        "transport",
        "sequence_policy",
        "objective_count",
        "python_version",
        "torch_version",
        "platform",
        "numpy_version",
        "cuda_version",
        "cudnn_version",
        "cuda_device_names",
        "cudnn_deterministic",
        "cudnn_benchmark",
        "matmul_allow_tf32",
        "cudnn_allow_tf32",
        "optimizer_class",
        "optimizer_hash",
        "scheduler_class",
        "scheduler_hash",
        "cuda_available",
        "deterministic_algorithms_enabled",
        "min_validation_examples",
    )
    for field in incompatible_fields:
        changed_value = (
            12 if field in ("random_seed", "min_validation_examples", "cudnn_version")
            else True if field in (
                "cuda_available", "deterministic_algorithms_enabled", "cudnn_deterministic",
                "cudnn_benchmark", "matmul_allow_tf32", "cudnn_allow_tf32",
            )
            else "changed"
        )
        changed = replace(manifest, **{field: changed_value})
        assert validate_checkpoint_compatibility(changed, manifest) == "incompatible", field

    legacy_v2 = replace(
        manifest,
        format_version=2,
        objective_schema_hash=None,
        dataset_version=None,
        training_regime=None,
        random_seed=None,
        git_worktree_clean=None,
        source_tree_hash=None,
    )
    assert validate_checkpoint_compatibility(legacy_v2, manifest) == "incompatible"

    legacy_v1 = replace(
        legacy_v2,
        format_version=1,
        config_hash=None,
        state_schema_hash=None,
        git_commit=None,
        dataset_manifest=None,
        dataset_hash=None,
        preprocessing_hash=None,
        environment_hash=None,
        python_version=None,
        torch_version=None,
        platform=None,
    )
    assert validate_checkpoint_compatibility(legacy_v1, manifest) == "incompatible"

    incomplete = CheckpointManifest()
    assert validate_checkpoint_compatibility(incomplete, incomplete) == "unverified-provenance"
