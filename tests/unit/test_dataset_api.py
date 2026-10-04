import pytest
import torch

from data.dataset_api import (
    SQLiteChimeraDataset,
    build_dataset_sample,
    collate_chimera_samples,
    write_immutable_dataset,
)
from data.msa import build_msa_record
from data.sequence_linkage import SequenceCandidate
from data.structures import StructuralQCConfig, parse_mmcif_structure
from tests.unit.test_structural_pipeline import _write_mmcif


def test_canonical_sample_contains_geometry_msa_linkage_and_qc(tmp_path):
    structure = parse_mmcif_structure(
        _write_mmcif(tmp_path),
        config=StructuralQCConfig(minimum_chain_length=1),
    )
    chain = structure.chains[0]
    msa = build_msa_record(
        f">query\n{chain.sequence}\n>homolog\n{chain.sequence}\n",
        msa_family_id="fixture-family",
        target_sequence_id="target",
        source="UniRef90",
        source_version="2026-06-10",
        target_sequence=chain.sequence,
    )
    sample = build_dataset_sample(
        structure,
        chain,
        msa=msa,
        sequence_candidates=[
            SequenceCandidate("P0TEST", chain.sequence, "UniProtKB", "2026_03")
        ],
        split="TRAIN",
    )

    assert sample["sample_id"] == "TEST:A:1"
    assert sample["coordinates"].shape == (3, 5, 3)
    assert sample["geometry"]["edge_index"].shape[0] == 3
    assert sample["msa_tokens"].shape[1] == len(chain.sequence)
    assert sample["sequence_linkage"]["external_accession"] == "P0TEST"
    assert sample["sequence_to_msa_columns"].tolist() == [0, 1, 2]
    assert sample["quality"]["status"] == "accepted"


def test_sqlite_dataset_is_immutable_random_access_and_streamable(tmp_path):
    path = tmp_path / "dataset.sqlite"
    records = [
        {
            "sample_id": "sample-2",
            "dataset_version": "v1",
            "split": "TEST",
            "sequence": "ACDE",
            "coordinates": torch.ones(4, 5, 3),
            "msa_tokens": torch.tensor([[0, 1, 2, 3]]),
        },
        {
            "sample_id": "sample-1",
            "dataset_version": "v1",
            "split": "TRAIN",
            "sequence": "ACD",
            "coordinates": torch.zeros(3, 5, 3),
            "msa_tokens": torch.tensor([[0, 1, 2]]),
        },
    ]
    manifest = write_immutable_dataset(
        path,
        records,
        dataset_version="v1",
        split_manifest_sha256="split-hash",
    )
    dataset = SQLiteChimeraDataset(path)
    train = SQLiteChimeraDataset(path, split="TRAIN")

    assert dataset.sample_ids == ["sample-1", "sample-2"]
    assert len(train) == 1
    assert dataset[1]["coordinates"].shape == (4, 5, 3)
    assert manifest["manifest_sha256"]
    assert [len(batch) for batch in dataset.iter_batches(batch_size=1)] == [1, 1]
    with pytest.raises(FileExistsError, match="immutable dataset already exists"):
        write_immutable_dataset(path, records, dataset_version="v1", split_manifest_sha256="split-hash")


def test_collator_pads_variable_length_tensors_and_msa_tokens():
    batch = collate_chimera_samples(
        [
            {"coordinates": torch.ones(2, 4, 3), "msa_tokens": torch.tensor([[0, 1]])},
            {"coordinates": torch.ones(3, 4, 3), "msa_tokens": torch.tensor([[0, 1, 2], [3, 4, 5]])},
        ]
    )

    assert batch["coordinates"].shape == (2, 3, 4, 3)
    assert batch["msa_tokens"].shape == (2, 2, 3)
    assert batch["msa_tokens"][0, 0, 2].item() == 22