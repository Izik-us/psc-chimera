import hashlib
import json

from chimera.checkpoint import (
    CheckpointManifest,
    git_provenance,
    runtime_provenance,
    validate_checkpoint_compatibility,
)
from data.training import build_dataset_manifest


def test_checkpoint_manifest_keeps_unavailable_provenance_explicit():
    manifest = CheckpointManifest(
        dataset_manifest="dataset-v1",
        dataset_hash="abc123",
        preprocessing_hash="def456",
        environment_hash="ghi789",
        git_commit="deadbeef",
    )
    manifest.validate()

    assert manifest.dataset_manifest == "dataset-v1"
    assert manifest.dataset_hash == "abc123"
    assert manifest.preprocessing_hash == "def456"
    assert manifest.environment_hash == "ghi789"
    assert manifest.python_version is None
    assert manifest.torch_version is None
    assert manifest.platform is None
    assert manifest.has_complete_provenance() is False


def test_checkpoint_compatibility_does_not_verify_missing_provenance():
    manifest = CheckpointManifest()

    assert manifest.has_complete_provenance() is False
    assert validate_checkpoint_compatibility(manifest, manifest) == "unverified-provenance"


def test_runtime_fingerprint_records_backend_flags_and_explicit_cuda_availability():
    runtime = runtime_provenance()

    assert runtime["python_version"]
    assert runtime["torch_version"]
    assert runtime["numpy_version"]
    assert runtime["platform"]
    assert runtime["environment_hash"]
    for name in (
        "cuda_available",
        "deterministic_algorithms_enabled",
        "cudnn_deterministic",
        "cudnn_benchmark",
        "matmul_allow_tf32",
        "cudnn_allow_tf32",
    ):
        assert isinstance(runtime[name], bool)
    if not runtime["cuda_available"]:
        assert runtime["cuda_version"] is None
        assert runtime["cuda_device_names"] is None


def test_git_provenance_is_explicitly_unavailable_outside_repository(tmp_path):
    provenance = git_provenance(tmp_path)

    assert provenance == {
        "git_commit": None,
        "git_worktree_clean": None,
        "source_tree_hash": None,
    }


def test_dataset_manifest_is_stable_and_hashable():
    records = [
        {"aa_sequence": "MKT", "codon_sequence": "ATGAAAACC", "expression": 0.2, "source": "src-a"},
        {"aa_sequence": "MKT", "codon_sequence": "ATGAAAACC", "expression": 0.2, "source": "src-a"},
        {"aa_sequence": "WQF", "codon_sequence": "TGGCAATTT", "expression": 0.4, "source": "src-b"},
    ]

    manifest = build_dataset_manifest(records, split_name="demo")
    payload = json.dumps(
        {key: value for key, value in manifest.items() if key != "digest"},
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()

    assert manifest["record_count"] == 3
    assert manifest["unique_record_count"] == 2
    assert manifest["split_name"] == "demo"
    assert manifest["digest"] == digest
