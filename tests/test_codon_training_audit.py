import torch

from scripts.train_codon_optimizer import split_dataset
from chimera.codon_optimizer import CodonOptimizer, AA_TO_IDX, tokenize_protein, tokenize_dna, translate_dna, CODON_TO_IDX, PAD_TOKEN


class TinyDataset:
    def __init__(self, records):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        return {
            "aa_sequence": [record["aa_sequence"]],
            "protein_tokens": tokenize_protein(record["aa_sequence"]),
            "codon_tokens": tokenize_dna(record["codon_sequence"]),
            "expression": torch.tensor(record.get("expression", 0.5), dtype=torch.float32),
        }


def test_split_keeps_same_protein_in_one_partition():
    dataset = TinyDataset([
        {"aa_sequence": "MTE", "codon_sequence": "ATGACCGAA"},
        {"aa_sequence": "MTE", "codon_sequence": "ATGACCGAG"},
        {"aa_sequence": "GKT", "codon_sequence": "GGTAAGACT"},
        {"aa_sequence": "FLV", "codon_sequence": "TTTCTGGTT"},
    ])
    train, val = split_dataset(dataset, 0.5, seed=7)
    train_keys = {dataset[i]["aa_sequence"][0] for i in train.indices}
    val_keys = {dataset[i]["aa_sequence"][0] for i in val.indices}
    assert train_keys.isdisjoint(val_keys)


def test_translation_round_trip_for_synonymous_targets():
    protein = "MTEYK"
    dna = "ATGACCGAGTATAAG"
    assert translate_dna(dna) == protein


def test_causal_mask_blocks_future_positions():
    mask = CodonOptimizer._causal_mask(5, torch.device("cpu"))
    assert mask.dtype == torch.bool
    assert torch.equal(mask, torch.triu(torch.ones((5, 5), dtype=torch.bool), diagonal=1))
    assert not mask[0, 0]
    assert mask[0, 1]
    assert not mask[4, 0]


def test_pad_token_is_not_a_valid_codon():
    assert PAD_TOKEN not in CODON_TO_IDX.values()
