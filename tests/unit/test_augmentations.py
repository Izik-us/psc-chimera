import torch

from data.augmentations import AugmentationConfig, augment_sample


def _sample(length=6):
    coords = torch.zeros(length, 5, 3)
    for index in range(length):
        x = index * 3.8
        coords[index, 0] = torch.tensor([x, 0.0, 0.0])
        coords[index, 1] = torch.tensor([x + 1.4, 0.2, 0.1])
        coords[index, 2] = torch.tensor([x + 2.7, 0.0, 0.2])
        coords[index, 3] = torch.tensor([x + 3.0, 1.0, 0.3])
        coords[index, 4] = torch.tensor([x + 1.4, 1.4, 0.1])
    return {
        "sample_id": "seed:A:1",
        "sequence": "ACDEFG"[:length],
        "label_asym_id": "A",
        "coordinates": coords,
        "atom_mask": torch.ones(length, 5, dtype=torch.bool),
        "residue_mask": torch.ones(length, dtype=torch.bool),
        "sequence_mask": torch.ones(length, dtype=torch.bool),
        "fixed_positions": torch.tensor([True] + [False] * (length - 1)),
        "msa_tokens": torch.tensor([[0, 1, 2, 3, 4, 5][:length], [0, 1, 2, 3, 4, 5][:length]]),
        "msa_mask": torch.ones(2, length, dtype=torch.bool),
        "sequence_to_msa_columns": torch.arange(length),
        "residue_mappings": [{"canonical_residue_index": index} for index in range(length)],
    }


def test_augmentation_is_reproducible_and_rigid_transform_preserves_distances():
    sample = _sample()
    config = AugmentationConfig(seed=17, translation_std_angstrom=4.0)
    first = augment_sample(sample, config, epoch=2, draw=1)
    second = augment_sample(sample, config, epoch=2, draw=1)

    assert torch.equal(first["coordinates"], second["coordinates"])
    assert torch.equal(first["geometry"]["distances"], second["geometry"]["distances"])
    assert torch.allclose(first["geometry"]["distances"], first["geometry"]["distances"].T)
    assert first["augmentation"]["seed"] == second["augmentation"]["seed"]


def test_crop_msa_subsample_and_residue_mask_preserve_query_and_fixed_positions():
    sample = _sample()
    config = AugmentationConfig(
        seed=3,
        crop_interval=(0, 4),
        crop_kind="domain",
        msa_row_limit=1,
        residue_mask_probability=1.0,
    )
    augmented = augment_sample(sample, config)

    assert augmented["sequence"] == "ACDE"
    assert augmented["coordinates"].shape[0] == 4
    assert augmented["msa_tokens"].shape == (1, 4)
    assert augmented["msa_tokens"][0].tolist() == [0, 1, 2, 3]
    assert not augmented["residue_corruption_mask"][0]
    assert augmented["residue_corruption_mask"][1:].all()
    assert augmented["augmentation"]["crop"] == {"kind": "domain", "start": 0, "end": 4}