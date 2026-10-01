import json

import pytest
import torch

from data.dataset import ChimeraJSONLDataset
from data.codon_dataset import CodonJSONLDataset
from scripts.merge_codon_datasets import split_by_accession


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


def test_codon_split_keeps_identical_proteins_together():
    records = [
        {
            "aa_sequence": "MKT",
            "codon_sequence": "ATGAAAACC",
            "expression": 0.2,
            "accession": "transcript-a",
        },
        {
            "aa_sequence": "MKT",
            "codon_sequence": "ATGAAGACT",
            "expression": 0.8,
            "accession": "transcript-b",
        },
        {
            "aa_sequence": "WQF",
            "codon_sequence": "TGGCAATTT",
            "expression": 0.4,
            "accession": "transcript-c",
        },
    ]

    train, validation = split_by_accession(records, validation_fraction=0.5)

    train_proteins = {record["aa_sequence"] for record in train}
    validation_proteins = {record["aa_sequence"] for record in validation}
    assert train_proteins.isdisjoint(validation_proteins)
    assert len(train) + len(validation) == len(records)


def test_codon_dataset_preserves_label_provenance(tmp_path):
    record = {
        "aa_sequence": "M",
        "codon_sequence": "ATG",
        "expression": 0.7,
        "source": "Pouyet_HumanCodonUsage",
        "label_type": "proxy",
        "expression_metric": "Exp_Meiosis",
        "expression_phenotype": "meiotic_germline_FPKM",
        "expression_components": ["F_PGC_17W", "M_PS", "M_RS"],
        "normalization": "max_positive_source_FPKM",
    }
    path = tmp_path / "proxy.jsonl"
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    item = CodonJSONLDataset(path)[0]

    assert item["label_type"] == "proxy"
    assert item["source"] == "Pouyet_HumanCodonUsage"
    assert item["expression_metric"] == "Exp_Meiosis"
    assert item["expression_phenotype"] == "meiotic_germline_FPKM"
    assert item["expression_components"] == ["F_PGC_17W", "M_PS", "M_RS"]
    assert item["normalization"] == "max_positive_source_FPKM"
