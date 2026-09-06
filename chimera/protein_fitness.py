"""External protein-fitness adapters for mutation-mode training."""

from __future__ import annotations

import argparse
import pickle
from typing import Sequence

import torch

from .codon_optimizer import AA_VOCAB


class ESMProteinFitnessScorer:
    """Create masked-marginal amino-acid preferences with a local ESM-2 model.

    The returned logits are evolutionary plausibility preferences, not a
    validated assay for thermostability or catalytic activity. Use them as a
    conservative prior or to bootstrap preference-pair generation.
    """

    def __init__(
        self,
        model_path: str,
        device: str | torch.device = "cpu",
        repr_layer: int | None = None,
    ) -> None:
        try:
            import esm
        except ImportError as exc:
            raise RuntimeError(
                "fair-esm is required for ESMProteinFitnessScorer"
            ) from exc

        self.device = torch.device(device)
        try:
            self.model, self.alphabet = esm.pretrained.load_model_and_alphabet_local(
                model_path
            )
        except (pickle.UnpicklingError, RuntimeError) as exc:
            if "Weights only load failed" not in str(exc):
                raise
            torch.serialization.add_safe_globals([argparse.Namespace])
            self.model, self.alphabet = esm.pretrained.load_model_and_alphabet_local(
                model_path
            )
        self.model.to(self.device).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad = False
        self.batch_converter = self.alphabet.get_batch_converter()
        self.mask_idx = int(self.alphabet.mask_idx)
        self.repr_layer = repr_layer
        self.aa_indices = torch.tensor(
            [self.alphabet.get_idx(amino_acid) for amino_acid in AA_VOCAB],
            dtype=torch.long,
            device=self.device,
        )

    @torch.no_grad()
    def score_preferences(self, sequences: Sequence[str]) -> torch.Tensor:
        """Return ESM amino-acid preference logits with shape ``(B, L, 20)``.

        Each residue is masked in turn and the model's masked-token logits are
        collected for the 20 standard amino acids. Sequences in one call must
        have the same length so the result can feed ``codon_optimizer_loss``.
        """
        if not sequences:
            return torch.empty((0, 0, len(AA_VOCAB)), device=self.device)
        lengths = {len(sequence) for sequence in sequences}
        if len(lengths) != 1:
            raise ValueError("ESM fitness scoring requires equal-length sequences")
        if any(not sequence for sequence in sequences):
            raise ValueError("ESM fitness scoring requires non-empty sequences")

        batch = [(str(index), sequence) for index, sequence in enumerate(sequences)]
        _, _, tokens = self.batch_converter(batch)
        tokens = tokens.to(self.device)
        sequence_length = len(sequences[0])
        preferences = torch.empty(
            (len(sequences), sequence_length, len(AA_VOCAB)),
            dtype=torch.float32,
            device=self.device,
        )

        for position in range(sequence_length):
            masked_tokens = tokens.clone()
            masked_tokens[:, position + 1] = self.mask_idx
            output = self.model(masked_tokens, return_contacts=False)
            preferences[:, position] = output["logits"][:, position + 1].index_select(
                -1, self.aa_indices
            )
        return preferences

    @torch.no_grad()
    def score_sequence(self, sequence: str) -> float:
        """Return the mean masked-marginal logit of the observed sequence."""
        preferences = self.score_preferences([sequence])[0]
        observed = torch.tensor(
            [AA_VOCAB.index(amino_acid) for amino_acid in sequence],
            dtype=torch.long,
            device=self.device,
        )
        return float(preferences.log_softmax(dim=-1).gather(1, observed[:, None]).mean())

    @torch.no_grad()
    def rank_single_mutations(self, sequence: str) -> list[tuple[str, float]]:
        """Rank all single-residue substitutions by masked ESM preference."""
        preferences = self.score_preferences([sequence])[0].log_softmax(dim=-1)
        observed = [AA_VOCAB.index(amino_acid) for amino_acid in sequence]
        candidates: list[tuple[str, float]] = []
        for position, current_index in enumerate(observed):
            for amino_acid_index, amino_acid in enumerate(AA_VOCAB):
                if amino_acid_index == current_index:
                    continue
                candidate = list(sequence)
                candidate[position] = amino_acid
                score_delta = (
                    preferences[position, amino_acid_index]
                    - preferences[position, current_index]
                )
                candidates.append(("".join(candidate), float(score_delta)))
        return sorted(candidates, key=lambda item: item[1], reverse=True)
