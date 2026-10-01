import torch
import pytest
from itertools import product

from chimera.codon_optimizer import (
    AA_PAD_TOKEN,
    ALL_CODONS,
    CodonOptimizer,
    codon_optimizer_loss,
    count_bad_motifs,
    motif_penalty_from_logits,
    protein_fitness_loss_from_logits,
    tokenize_protein,
)
from scripts.train_codon_optimizer import (
    LengthBucketBatchSampler,
    _run_eval,
    collate_records,
)


def test_fitness_weighting_ignores_padded_protein_tokens():
    codon_logits = torch.randn(1, 3, 64)
    protein_tokens = torch.tensor([[0, 1, AA_PAD_TOKEN]])
    padding_mask = torch.tensor([[False, False, True]])
    fitness_logits = torch.randn(1, 3, 20)

    loss = protein_fitness_loss_from_logits(
        codon_logits,
        protein_tokens,
        padding_mask,
        fitness_logits=fitness_logits,
    )

    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_fitness_loss_is_padding_invariant():
    torch.manual_seed(3)
    valid_logits = torch.randn(1, 2, 64)
    valid_protein = torch.tensor([[0, 1]])
    valid_fitness = torch.randn(1, 2, 20)
    unpadded_loss = protein_fitness_loss_from_logits(
        valid_logits,
        valid_protein,
        torch.zeros(1, 2, dtype=torch.bool),
        fitness_logits=valid_fitness,
    )

    padded_loss = protein_fitness_loss_from_logits(
        torch.cat([valid_logits, torch.zeros(1, 1, 64)], dim=1),
        torch.tensor([[0, 1, AA_PAD_TOKEN]]),
        torch.tensor([[False, False, True]]),
        fitness_logits=torch.cat([valid_fitness, torch.zeros(1, 1, 20)], dim=1),
    )

    assert torch.allclose(unpadded_loss, padded_loss)


class _LengthDataset:
    def __init__(self, lengths):
        self.lengths = lengths

    def __len__(self):
        return len(self.lengths)

    def __getitem__(self, index):
        return {"protein_tokens": torch.zeros(self.lengths[index], dtype=torch.long)}


def test_length_bucket_sampler_separates_extreme_lengths():
    sampler = LengthBucketBatchSampler(
        _LengthDataset([80, 95, 2426]),
        batch_size=3,
        max_protein_length=3000,
        max_attention_elements=6_000_000,
    )
    batches = list(sampler)

    assert sorted(index for batch in batches for index in batch) == [0, 1, 2]
    assert all(not ({0, 1} <= set(batch) and 2 in batch) for batch in batches)


def test_length_bucket_sampler_requires_explicit_oversize_policy():
    with pytest.raises(ValueError, match="oversize-policy skip"):
        LengthBucketBatchSampler(
            _LengthDataset([80, 2426]),
            batch_size=2,
            max_protein_length=1022,
        )

    sampler = LengthBucketBatchSampler(
        _LengthDataset([80, 2426]),
        batch_size=2,
        max_protein_length=1022,
        oversize_policy="skip",
    )
    assert sampler.excluded_indices == [1]
    assert list(sampler) == [[0]]


def test_model_rejects_proteins_above_configured_context():
    model = CodonOptimizer(
        d_model=32,
        n_heads=4,
        n_dec_layers=1,
        dim_ff=64,
        max_codons=128,
        max_protein_length=128,
    )
    protein = "M" * 129
    with pytest.raises(ValueError, match="exceeds configured maximum"):
        model.encode_protein(
            tokenize_protein(protein).unsqueeze(0),
            [protein],
        )


def test_codon_decoder_capacity_tracks_protein_context():
    model = CodonOptimizer(
        d_model=32,
        n_heads=4,
        n_dec_layers=1,
        dim_ff=64,
        max_protein_length=40,
    )
    assert model.max_codons == 120
    assert model.pos_encoding.num_embeddings == model.max_codons

    with pytest.raises(ValueError, match="between max_protein_length"):
        CodonOptimizer(
            d_model=32,
            n_heads=4,
            n_dec_layers=1,
            dim_ff=64,
            max_protein_length=40,
            max_codons=121,
        )


def test_cpg_is_not_unconditionally_counted_as_a_bad_motif():
    assert count_bad_motifs("ACG") == 0


def test_motif_probability_matches_exact_two_codon_enumeration():
    logits = torch.zeros(2, len(ALL_CODONS))
    observed = float(motif_penalty_from_logits(logits))
    expected = 0.0
    for first_codon, second_codon in product(ALL_CODONS, repeat=2):
        dna = first_codon + second_codon
        upa = sum(dna[index : index + 2] == "TA" for index in range(len(dna) - 1)) / 5
        poly_a = float("AATAAA" in dna) + float("ATTAAA" in dna)
        au_rich = float("ATTTAT" in dna)
        expected += (upa + 2.0 * poly_a + au_rich) / (len(ALL_CODONS) ** 2)

    assert observed == pytest.approx(expected, abs=1e-6)


def test_unlabeled_expression_does_not_train_critic_to_inflate_score():
    logits = torch.randn(1, 1, 64, requires_grad=True)
    predicted_expression = torch.tensor([0.9], requires_grad=True)
    loss_weights = {
        "cai": 0.0,
        "gc": 0.0,
        "upa": 0.0,
        "motif": 0.0,
        "fitness": 0.0,
        "expr": 1.0,
    }

    loss, metrics = codon_optimizer_loss(
        logits,
        torch.tensor([[0]]),
        predicted_expression,
        lambdas=loss_weights,
    )
    loss.backward()

    assert metrics["expression"] == pytest.approx(0.9)
    assert predicted_expression.grad is None


def test_validation_metrics_remain_separate_by_label_type():
    from chimera.codon_optimizer import CODON_TO_IDX, AA_TO_IDX

    records = [
        {
            "protein_tokens": torch.tensor([AA_TO_IDX["M"]]),
            "codon_tokens": torch.tensor([CODON_TO_IDX["ATG"]]),
            "aa_sequence": ["M"],
            "expression": torch.tensor(0.8),
            "label_type": "measured",
        },
        {
            "protein_tokens": torch.tensor([AA_TO_IDX["M"]]),
            "codon_tokens": torch.tensor([CODON_TO_IDX["ATG"]]),
            "aa_sequence": ["M"],
            "expression": torch.tensor(0.2),
            "label_type": "proxy",
        },
    ]
    batch = collate_records(records)

    class FixedModel:
        def eval(self):
            return self

        def __call__(self, protein_tokens, codon_tokens, aa_sequence, **kwargs):
            logits = torch.zeros(codon_tokens.shape[0], codon_tokens.shape[1], 64)
            expression = torch.tensor([0.75, 0.25])
            return {"logits": logits, "expression": expression}

    metrics = _run_eval(FixedModel(), [batch], torch.device("cpu"))

    assert "measured_ce" in metrics
    assert "proxy_ce" in metrics
    assert metrics["measured_expression"] == pytest.approx(0.75)
    assert metrics["proxy_expression"] == pytest.approx(0.25)