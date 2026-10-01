import hashlib
import json

from chimera.checkpoint import CheckpointManifest
from data.training import build_dataset_manifest


def test_checkpoint_manifest_records_runtime_and_dataset_provenance():
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
    assert manifest.python_version
    assert manifest.torch_version
    assert manifest.platform


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
