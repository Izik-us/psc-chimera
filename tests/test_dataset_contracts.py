import json

import pytest
import torch

from data.dataset import ChimeraJSONLDataset


def _record(length=4):
    eye = torch.eye(3).repeat(length, 1, 1).tolist()
    coords = torch.arange(length * 3, dtype=torch.float32).reshape(length, 3).tolist()
    return {
        "msa_tokens": [[0, 1, 2, 3][:length]],
        "source_R": eye,
        "source_t": coords,
        "target_R": eye,
        "target_t": coords,
    }


def _write(tmp_path, record):
    path = tmp_path / "train.jsonl"
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    return path


def test_dataset_accepts_consistent_frames(tmp_path):
    dataset = ChimeraJSONLDataset(_write(tmp_path, _record()))
    item = dataset[0]
    assert item["msa_tokens"].shape == (1, 4)
    assert item["source_R"].shape == (4, 3, 3)


def test_dataset_rejects_length_mismatch(tmp_path):
    record = _record()
    record["target_t"] = record["target_t"][:-1]
    dataset = ChimeraJSONLDataset(_write(tmp_path, record))
    with pytest.raises(ValueError, match="same residue length"):
        dataset[0]


def test_dataset_rejects_reflection(tmp_path):
    record = _record()
    record["source_R"][0][0][0] = -1.0
    dataset = ChimeraJSONLDataset(_write(tmp_path, record))
    with pytest.raises(ValueError, match="reflections"):
        dataset[0]
