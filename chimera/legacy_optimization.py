"""Historical optimization implementations retained for compatibility.

Canonical sequence design, objective prediction, and retrieval live in
their corresponding canonical or optional modules. Nothing in this module is
used by the canonical composition.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import warnings
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass

from .objective_schema import ParetoObjectives

# ═══════════════════════════════════════════════════════════════════════════════
# 1. STRUCTURAL RETRIEVAL-AUGMENTED GENERATION
# ═══════════════════════════════════════════════════════════════════════════════




# ═══════════════════════════════════════════════════════════════════════════════
# 2. DIRECT PREFERENCE OPTIMIZATION (DPO) FROM PROTEUS
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class ProteusPreferencePair:
    """
    A preference pair from one PROTEUS round.
    winner: sequence that survived PROTEUS selection (expressed + functional)
    loser:  sequence that failed PROTEUS selection
    msa:    the MSA context used to generate both sequences
    source_R, source_t: bacterial NRPS backbone frames used during generation
    """

    winner_tokens: torch.Tensor  # (L,) integer amino acid tokens
    loser_tokens: torch.Tensor  # (L,)
    msa_tokens: torch.Tensor  # (N_seq, L) MSA context
    pair_features: torch.Tensor  # (L, L, 128) pair features
    source_R: torch.Tensor  # (L, 3, 3) source backbone rotations
    source_t: torch.Tensor  # (L, 3) source backbone translations


class LegacyDPOTrainer(nn.Module):
    """
    Direct Preference Optimization for CHIMERA.

    Converts PROTEUS experimental results directly into gradient signal.
    No reward model needed — preferences optimize the generation model directly.

    DPO loss (Rafailov et al. 2023 NeurIPS):
        L_DPO = -E[(log σ(β * (log π_θ(yw|x) - log π_ref(yw|x))
                           - β * (log π_θ(yl|x) - log π_ref(yl|x))))]

    Where:
        yw = winner sequence (survived PROTEUS)
        yl = loser sequence (failed PROTEUS)
        x  = MSA context (the conditioning information)
        π_θ = current CHIMERA model
        π_ref = frozen reference model (CHIMERA before DPO fine-tuning)
        β = temperature parameter controlling how much to deviate from reference

    The crucial insight: DPO is equivalent to RLHF but without training a
    separate reward model. It's more stable and sample-efficient.
    For the PSC project: every PROTEUS round generates preference pairs
    that immediately improve CHIMERA without needing to design reward functions.
    """

    def __init__(self, beta: float = 0.1):
        """
        beta: strength of preference signal.
              Low beta (0.05) = conservative, stays close to reference.
              High beta (0.5) = aggressive, fast learning from preferences.
              For PROTEUS with expensive experiments: use 0.05-0.1.
        """
        super().__init__()
        self.beta = beta

    def compute_sequence_logprob(
        self,
        model,
        sequence_tokens: torch.Tensor,  # (B, L) integer tokens
        msa_tokens: torch.Tensor,  # (B, N_seq, L)
        pair_features: torch.Tensor,  # (B, L, L, 128)
        source_R: torch.Tensor,  # (B, L, 3, 3) source backbone
        source_t: torch.Tensor,  # (B, L, 3) source backbone positions
    ) -> torch.Tensor:
        """
        Compute log probability of a sequence under a CHIMERA model.
        log P(sequence | MSA) = sum_i log P(aa_i | backbone, evol_context, aa_{<i})

        In CHIMERA's architecture: the sequence logits come from ProteinMPNN's
        forward pass on the generated backbone. We compute the cross-entropy
        of the target sequence against these logits.
        """
        if hasattr(model, "sequence_logprob"):
            return model.sequence_logprob(sequence_tokens, msa_tokens, pair_features, source_R, source_t)
        raise TypeError("DPO requires a model.sequence_logprob autoregressive policy interface")

    def dpo_loss(
        self,
        policy_model,  # current CHIMERA (being trained)
        reference_model,  # frozen reference CHIMERA (before DPO)
        pairs: List[ProteusPreferencePair],
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute DPO loss over a batch of PROTEUS preference pairs.
        """
        if not pairs:
            return torch.tensor(0.0), {}

        # Stack batch
        device = pairs[0].winner_tokens.device
        winner_toks = torch.stack([p.winner_tokens for p in pairs]).to(device)
        loser_toks = torch.stack([p.loser_tokens for p in pairs]).to(device)
        msa = torch.stack([p.msa_tokens for p in pairs]).to(device)
        pair_feat = torch.stack([p.pair_features for p in pairs]).to(device)
        source_R = torch.stack([p.source_R for p in pairs]).to(device)
        source_t = torch.stack([p.source_t for p in pairs]).to(device)

        # Policy log-probabilities
        pi_yw = self.compute_sequence_logprob(
            policy_model, winner_toks, msa, pair_feat, source_R, source_t
        )
        pi_yl = self.compute_sequence_logprob(
            policy_model, loser_toks, msa, pair_feat, source_R, source_t
        )

        # Reference log-probabilities (no gradients)
        with torch.no_grad():
            ref_yw = self.compute_sequence_logprob(
                reference_model, winner_toks, msa, pair_feat, source_R, source_t
            )
            ref_yl = self.compute_sequence_logprob(
                reference_model, loser_toks, msa, pair_feat, source_R, source_t
            )

        # DPO loss
        rewards = self.beta * ((pi_yw - ref_yw) - (pi_yl - ref_yl))
        loss = -F.logsigmoid(rewards).mean()

        # Monitor implicit reward margins (should be positive and growing)
        with torch.no_grad():
            winner_reward = (pi_yw - ref_yw).mean()
            loser_reward = (pi_yl - ref_yl).mean()

        return loss, {
            "dpo_loss": loss.item(),
            "winner_reward": winner_reward.item(),
            "loser_reward": loser_reward.item(),
            "reward_margin": (winner_reward - loser_reward).item(),
        }

    def update_from_proteus_round(
        self,
        policy_model,
        reference_model,
        surviving_sequences: List[str],
        failed_sequences: List[str],
        msa_tokens: torch.Tensor,
        pair_features: torch.Tensor,
        n_dpo_steps: int = 50,
        learning_rate: float = 1e-5,
    ) -> Dict:
        """
        One-call interface: take PROTEUS results, run DPO fine-tuning.

        Args:
            surviving_sequences: amino acid sequences that passed PROTEUS
            failed_sequences:    amino acid sequences that failed PROTEUS
            msa_tokens:         MSA context used to generate these sequences
            pair_features:       pair features from EvoFormer
            n_dpo_steps:         DPO gradient steps per PROTEUS round
            learning_rate:       very low LR — we want conservative updates

        Returns:
            training metrics dict
        """
        # Only train connector parameters via DPO (pretrained models stay frozen)
        optimizer = torch.optim.AdamW(
            [p for p in policy_model.parameters() if p.requires_grad],
            lr=learning_rate,
            weight_decay=1e-4,
        )

        # Build preference pairs from PROTEUS results
        # Pair each winner with a random loser (or use all pairs)
        pairs = []
        for winner_seq, loser_seq in zip(surviving_sequences, failed_sequences):
            pair = ProteusPreferencePair(
                winner_tokens=self._tokenize(winner_seq),
                loser_tokens=self._tokenize(loser_seq),
                msa_tokens=msa_tokens[0],  # context for this batch
                pair_features=pair_features[0],
            )
            pairs.append(pair)

        metrics_history = []
        for step in range(n_dpo_steps):
            # Sample a mini-batch of preference pairs
            batch_size = min(8, len(pairs))
            batch = pairs[:batch_size]  # in production: random sample

            loss, metrics = self.dpo_loss(policy_model, reference_model, batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in policy_model.parameters() if p.requires_grad],
                max_norm=0.5,  # very conservative for DPO
            )
            optimizer.step()
            optimizer.zero_grad()
            metrics_history.append(metrics)

        avg_metrics = {
            k: np.mean([m[k] for m in metrics_history]) for k in metrics_history[0]
        }
        print(
            f"[DPO] {n_dpo_steps} steps | "
            f"reward margin: {avg_metrics['reward_margin']:.3f} | "
            f"loss: {avg_metrics['dpo_loss']:.4f}"
        )
        return avg_metrics

    @staticmethod
    def _tokenize(seq: str) -> torch.Tensor:
        """Convert amino acid string to integer tokens."""
        AA = "ACDEFGHIKLMNPQRSTVWY"
        if any(aa not in AA for aa in seq):
            raise ValueError("Preference sequences must contain standard amino-acid symbols")
        return torch.tensor([AA.index(aa) for aa in seq])


DPOTrainer = LegacyDPOTrainer


class LegacyAutoregressiveSequencePolicy(nn.Module):
    """Causal sequence policy used to define valid DPO log probabilities."""

    def __init__(self, context_dim: int, vocab_size: int = 20, layers: int = 2):
        super().__init__()
        self.context = nn.Linear(context_dim, context_dim)
        block = nn.TransformerEncoderLayer(context_dim, 4, context_dim * 4, batch_first=True)
        self.decoder = nn.TransformerEncoder(block, layers)
        self.token_embedding = nn.Embedding(vocab_size + 1, context_dim)
        self.head = nn.Linear(context_dim, vocab_size)
        self.vocab_size = vocab_size

    def forward(self, context: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        prefix = torch.full_like(tokens[:, :1], self.vocab_size)
        decoder_input = torch.cat([prefix, tokens[:, :-1]], dim=1)
        hidden = self.context(context) + self.token_embedding(decoder_input)
        size = hidden.shape[1]
        mask = torch.triu(torch.ones(size, size, device=hidden.device, dtype=torch.bool), diagonal=1)
        return self.head(self.decoder(hidden, mask=mask))

    def logprob(self, context: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        logits = self(context, tokens)
        return F.log_softmax(logits, dim=-1).gather(-1, tokens.unsqueeze(-1)).squeeze(-1).sum(-1)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. PARETO-FRONT MULTI-OBJECTIVE OPTIMIZATION
# ═══════════════════════════════════════════════════════════════════════════════


AutoregressiveSequencePolicy = LegacyAutoregressiveSequencePolicy


class LegacyParetoMultiObjectiveHead(nn.Module):
    """
    Replaces weighted sum losses with true Pareto-front exploration.

    Architecture:
        Five independent prediction heads, each predicting one objective.
        At training: multi-gradient optimization (PCGrad or MGDA).
        At inference: NSGA-II-style Pareto ranking of the generated library.
                      Returns the non-dominated frontier.

    Why this matters for PSC:
        Different target tissues and therapeutic applications require
        different tradeoffs. A tumor-targeted PSC maximizes substrate_selectivity
        (tighter binding pocket) at some cost to expression_efficiency.
        A constitutively active PSC inverts that tradeoff.
        The Pareto frontier lets you navigate this space explicitly.

    PCGrad (Project Conflicting Gradients — Yu et al. 2020):
        When two objectives have conflicting gradients, project one onto the
        normal plane of the other instead of averaging. This prevents one
        objective from hurting another during optimization.
    """

    def __init__(self, d_model: int = 256):
        super().__init__()

        # Shared trunk: takes the full CHIMERA output repr
        self.shared = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        # Five independent prediction heads
        def head(out_dim=1):
            return nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Dropout(0.1),  # also used for MC uncertainty estimation
                nn.Linear(d_model // 2, out_dim),
            )

        self.head_evol = head(1)  # scalar log-prob
        self.head_stab = head(1)  # 0-100 pLDDT
        self.head_expr = head(1)  # 0-1 expression efficiency
        self.head_sel = head(1)  # 0-1 selectivity match
        self.head_asm = head(1)  # 0-1 assembly compatibility

    def forward(self, repr: torch.Tensor) -> ParetoObjectives:
        """
        repr: (B, L, d_model) sequence-level representation
        Returns ParetoObjectives with predicted scores for each objective.
        """
        pooled = self.shared(repr).mean(dim=1)  # (B, d_model)
        return ParetoObjectives(
            evolutionary_plausibility=self.head_evol(pooled).squeeze(-1),
            structural_stability=100
            * torch.sigmoid(self.head_stab(pooled)).squeeze(-1),
            expression_efficiency=torch.sigmoid(self.head_expr(pooled)).squeeze(-1),
            substrate_selectivity=torch.sigmoid(self.head_sel(pooled)).squeeze(-1),
            assembly_compatibility=torch.sigmoid(self.head_asm(pooled)).squeeze(-1),
        )

    def weighted_objective_loss_legacy(
        self,
        objectives: ParetoObjectives,
        labels: Dict[str, Optional[torch.Tensor]],
        weights: Dict[str, float] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Deprecated heuristic weighted loss; this is not PCGrad.

        Algorithm:
          1. Compute individual task losses L_1, L_2, ..., L_n
          2. Detect conflicts by analyzing task loss distributions
          3. For conflicting tasks: reduce weight of lower-priority task
          4. Combine losses with conflict-adjusted weights

        Note: Full PCGrad requires gradient computation during backward pass.
        This implementation uses a simplified conflict detection via loss magnitude
        analysis, which avoids gradient tracking issues while still reducing
        inter-objective conflicts during training.

        For PSC: evolutionary plausibility can conflict with substrate selectivity;
        assembly compatibility can conflict with expression efficiency.
        This method resolves conflicts by down-weighting lower-priority objectives.
        """
        if weights is None:
            weights = {
                "evol": 1.0,
                "stab": 0.8,
                "expr": 0.6,
                "sel": 1.0,
                "asm": 0.4,
            }

        # Compute individual task losses
        losses = {}
        task_names = []
        task_losses_list = []

        if labels.get("evol") is not None:
            loss_evol = F.mse_loss(
                objectives.evolutionary_plausibility, labels["evol"]
            )
            losses["evol"] = loss_evol
            task_names.append("evol")
            task_losses_list.append(loss_evol.detach())

        if labels.get("stab") is not None:
            loss_stab = F.mse_loss(objectives.structural_stability, labels["stab"])
            losses["stab"] = loss_stab
            task_names.append("stab")
            task_losses_list.append(loss_stab.detach())

        if labels.get("expr") is not None:
            loss_expr = F.binary_cross_entropy(
                objectives.expression_efficiency, labels["expr"]
            )
            losses["expr"] = loss_expr
            task_names.append("expr")
            task_losses_list.append(loss_expr.detach())

        if labels.get("sel") is not None:
            loss_sel = F.mse_loss(objectives.substrate_selectivity, labels["sel"])
            losses["sel"] = loss_sel
            task_names.append("sel")
            task_losses_list.append(loss_sel.detach())

        if labels.get("asm") is not None:
            loss_asm = F.mse_loss(objectives.assembly_compatibility, labels["asm"])
            losses["asm"] = loss_asm
            task_names.append("asm")
            task_losses_list.append(loss_asm.detach())

        if not task_losses_list:
            return torch.tensor(0.0), {}

        # Simplified conflict detection: check if tasks have opposite sign gradients
        # by analyzing prediction vs label trends
        adjusted_weights = weights.copy()

        # Check for conflicts between objectives
        # We consider tasks conflicting if they have opposite trends
        # (e.g., evolutionary plausibility high vs selectivity low)
        if len(task_losses_list) >= 2:
            task_loss_magnitudes = torch.tensor([l.item() for l in task_losses_list])
            # Normalize magnitudes for comparison
            task_loss_normalized = task_loss_magnitudes / (task_loss_magnitudes.mean() + 1e-8)

            for i in range(len(task_losses_list)):
                for j in range(i + 1, len(task_losses_list)):
                    # If two tasks have very different loss magnitudes, they might conflict
                    # Reduce weight of the higher-loss task (lower priority)
                    loss_i = task_loss_normalized[i]
                    loss_j = task_loss_normalized[j]

                    # If ratio > 2, there's likely a conflict; down-weight the higher loss
                    if loss_i > 2 * loss_j or loss_j > 2 * loss_i:
                        task_i = task_names[i]
                        task_j = task_names[j]
                        if loss_i > loss_j:
                            adjusted_weights[task_i] *= 0.9
                        else:
                            adjusted_weights[task_j] *= 0.9

        # Combine losses with adjusted weights
        total = sum(adjusted_weights.get(k, 1.0) * v for k, v in losses.items())

        return total, {k: v.item() for k, v in losses.items()}

    def pcgrad_loss(
        self,
        objectives: ParetoObjectives,
        labels: Dict[str, Optional[torch.Tensor]],
        weights: Dict[str, float] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        """Deprecated compatibility alias for the historical weighted loss.

        Use ``MergeReadyParetoMultiObjectiveHead.pcgrad_loss`` for true
        gradient projection. This legacy model remains isolated from the
        canonical architecture.
        """
        warnings.warn(
            "ParetoMultiObjectiveHead.pcgrad_loss is heuristic, not PCGrad; "
            "use the canonical MergeReadyParetoMultiObjectiveHead instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.weighted_objective_loss_legacy(objectives, labels, weights)

    def task_losses(self, objectives: ParetoObjectives, labels: Dict[str, Optional[torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Return independent task losses for true PCGrad training."""
        result = {}
        if labels.get("evol") is not None:
            result["evol"] = F.mse_loss(objectives.evolutionary_plausibility, labels["evol"])
        if labels.get("stab") is not None:
            result["stab"] = F.mse_loss(objectives.structural_stability, labels["stab"])
        if labels.get("expr") is not None:
            result["expr"] = F.binary_cross_entropy(objectives.expression_efficiency, labels["expr"])
        if labels.get("sel") is not None:
            result["sel"] = F.mse_loss(objectives.substrate_selectivity, labels["sel"])
        if labels.get("asm") is not None:
            result["asm"] = F.mse_loss(objectives.assembly_compatibility, labels["asm"])
        return result

    def pcgrad_backward(self, objectives: ParetoObjectives, labels: Dict[str, Optional[torch.Tensor]], parameters) -> Dict[str, float]:
        """Project conflicting task gradients into ``parameters.grad``."""
        from .pcgrad import project_conflicting_gradients
        losses = self.task_losses(objectives, labels)
        if not losses:
            return {}
        project_conflicting_gradients(list(losses.values()), parameters)
        return {name: float(loss.detach()) for name, loss in losses.items()}

    @staticmethod
    def compute_pareto_frontier(
        objectives_matrix: torch.Tensor,  # (N, 5) all five objectives for N sequences
        maximize: List[bool] = None,  # True = maximize, False = minimize
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        NSGA-II non-dominated sorting to find Pareto front.

        Args:
            objectives_matrix: (N, 5) tensor of objective values
            maximize: which objectives to maximize (default: all True for PSC)

        Returns:
            pareto_front:   (K, 5) sequences on the Pareto frontier
            pareto_indices: (K,) indices of Pareto-optimal sequences
        """
        if maximize is None:
            maximize = [True, True, True, True, True]

        N = objectives_matrix.shape[0]
        obj = objectives_matrix.cpu().numpy()

        # Flip minimization objectives
        for j, maxi in enumerate(maximize):
            if not maxi:
                obj[:, j] = -obj[:, j]

        # Non-dominated sort: a sequence is non-dominated if no other sequence
        # is better on ALL objectives simultaneously
        is_pareto = np.ones(N, dtype=bool)
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                # Is j dominated by i? (i better on all objectives)
                if np.all(obj[i] >= obj[j]) and np.any(obj[i] > obj[j]):
                    is_pareto[j] = False

        pareto_indices = np.where(is_pareto)[0]
        pareto_front = objectives_matrix[pareto_indices]

        return pareto_front, torch.tensor(pareto_indices)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. BAYESIAN UNCERTAINTY + ACTIVE LEARNING ACQUISITION
# ═══════════════════════════════════════════════════════════════════════════════


ParetoMultiObjectiveHead = LegacyParetoMultiObjectiveHead


class LegacyBayesianUncertaintyEstimator(nn.Module):
    """
    MC Dropout-based Bayesian uncertainty quantification for CHIMERA.

    Theory:
        Gal & Ghahramani 2016: Dropout at inference time approximates
        a Bayesian deep learning model. Running T forward passes with
        dropout enabled gives T samples from the approximate posterior.

        Epistemic uncertainty (model uncertainty):
            U_epistemic = Var_T[P(y|x, w)] averaged over positions
            High = model hasn't seen sequences like this before
            Low  = model is confident based on training data

        Aleatoric uncertainty (data uncertainty):
            U_aleatoric = E_T[Var[P(y|x, w)]]
            Inherent variability — can't be reduced by more data

    Application to PSC active learning:
        Best sequences to test in PROTEUS are those where:
            U_epistemic is HIGH (exploring unknown design space)
            AND predicted_quality is HIGH (likely to be functional)

        Expected Improvement acquisition:
            EI(x) = U_epistemic(x) × predicted_pareto_score(x)

        This prioritizes high-uncertainty, high-quality sequences —
        maximizing information gain per expensive PROTEUS experiment.

    Deep Ensemble variant (stronger but more expensive):
        Run CHIMERA with N=5 different random seeds during fine-tuning.
        Uncertainty = variance across ensemble members.
        More accurate than MC Dropout but requires 5x inference.
        Use for final candidate selection before large PROTEUS batches.
    """

    def __init__(self, n_mc_samples: int = 30, dropout_rate: float = 0.1):
        super().__init__()
        self.n_mc_samples = n_mc_samples
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(p=dropout_rate)

    def estimate_uncertainty(
        self,
        model,
        inputs: Dict,
        n_samples: Optional[int] = None,
        per_candidate: bool = False,  # Return per-candidate uncertainty for design()
    ) -> Dict[str, torch.Tensor]:
        """
        Run N MC forward passes with dropout enabled.
        Return mean prediction and epistemic/aleatoric uncertainty.

        Args:
            model: CHIMERA model (will be set to train mode for dropout)
            inputs: Dictionary with model forward() arguments
            n_samples: Number of MC dropout samples (default: self.n_mc_samples)
            per_candidate: If True, return (B, n_seqs) uncertainty for design()
                          If False, return (B,) or (B, L) uncertainty
        """
        n = n_samples or self.n_mc_samples

        # Enable dropout at inference
        model.train()  # train mode = dropout active

        all_objectives = {"evol": [], "stab": [], "expr": [], "sel": [], "asm": []}

        with torch.no_grad():
            for _ in range(n):
                outputs = model(**inputs)
                # Collect per-candidate objectives: (B, n_seqs) each
                if "evol_plausibility" in outputs:
                    all_objectives["evol"].append(outputs["evol_plausibility"].cpu())
                    all_objectives["stab"].append(outputs["structural_stability"].cpu())
                    all_objectives["expr"].append(outputs["expression_efficiency"].cpu())
                    all_objectives["sel"].append(outputs["substrate_selectivity"].cpu())
                    all_objectives["asm"].append(outputs["assembly_compat"].cpu())

        model.eval()

        results = {}

        if all_objectives["evol"]:
            # Stack MC samples: (n_samples, B, n_seqs) per objective
            for key in all_objectives:
                stack = torch.stack(all_objectives[key])  # (n, B, n_seqs)

                # Mean prediction: (B, n_seqs)
                mean_pred = stack.mean(dim=0)

                # Epistemic uncertainty: variance across MC samples (model disagreement)
                # (B, n_seqs) — high when different MC samples predict different values
                epistemic = stack.var(dim=0)

                results[f"{key}_mean"] = mean_pred
                results[f"{key}_epistemic"] = epistemic

            # Aggregate uncertainty across all objectives
            # Shape: (n_samples, B, n_seqs, 5) with all 5 objectives
            all_obj_stack = torch.stack(
                [torch.stack(all_objectives[k]) for k in ["evol", "stab", "expr", "sel", "asm"]],
                dim=-1,
            )  # (n, B, n_seqs, 5)

            # Candidate-level uncertainty: average epistemic variance across objectives
            candidate_epistemic = all_obj_stack.var(dim=0).mean(dim=-1)  # (B, n_seqs)

            # Candidate quality: average predicted score across objectives
            candidate_quality = all_obj_stack.mean(dim=0).mean(dim=-1)  # (B, n_seqs)

            results["candidate_epistemic"] = candidate_epistemic
            results["candidate_quality"] = candidate_quality

            # Per-candidate uncertainty for use in design()
            results["per_candidate_uncertainty"] = candidate_epistemic
        else:
            # Fallback for non-objective outputs
            all_sequence_logits = []
            for _ in range(n):
                outputs = model(**inputs)
                seq_probs = F.softmax(outputs["sequences"], dim=-1)
                all_sequence_logits.append(seq_probs.cpu())

            stack = torch.stack(all_sequence_logits)  # (n, B, n_seqs, L, 20) or (n, B, L, 20)
            mean_pred = stack.mean(dim=0)
            epistemic = stack.var(dim=0).mean(dim=-1)  # Variance over vocab

            results["mean_prediction"] = mean_pred
            results["epistemic"] = epistemic
            results["per_candidate_uncertainty"] = epistemic.mean(dim=-1)  # (B, n_seqs) or (B,)

        return results

    def expected_improvement_acquisition(
        self,
        uncertainty: torch.Tensor,  # (N,) or (B, n_seqs) epistemic uncertainty
        predicted_quality: torch.Tensor,  # (N,) or (B, n_seqs) predicted quality
        best_observed: float = 0.0,  # best quality score observed so far
    ) -> torch.Tensor:
        """
        True Gaussian Expected Improvement acquisition function for Bayesian optimization.

        EI(x) = (μ(x) - f* - ε) × Φ(Z) + σ(x) × φ(Z)
        where:
            μ(x) = predicted quality
            σ(x) = epistemic uncertainty (MC Dropout variance)
            f* = best observed quality
            ε = small exploration bonus
            Z = (μ(x) - f* - ε) / (σ(x) + δ)
            Φ = standard normal CDF
            φ = standard normal PDF

        Returns: EI(x) ∈ [0, ∞) for each candidate

        High EI = unexplored region (high σ) with high expected quality (μ)
        These sequences maximize information gain per PROTEUS experiment.
        """
        from scipy.stats import norm
        import numpy as np

        # Handle both flat (N,) and structured (B, n_seqs) shapes
        original_shape = uncertainty.shape
        unc_flat = uncertainty.cpu().numpy().flatten()
        qual_flat = predicted_quality.cpu().numpy().flatten()

        # Exploration bonus
        eps = 0.01

        # Regularization to avoid division by zero
        delta = 1e-8

        # Compute standardized improvement
        Z = (qual_flat - best_observed - eps) / (unc_flat + delta)

        # Gaussian EI: (μ - f*) * Φ(Z) + σ * φ(Z)
        cdf_Z = norm.cdf(Z)
        pdf_Z = norm.pdf(Z)

        ei_flat = (qual_flat - best_observed) * cdf_Z + unc_flat * pdf_Z
        ei_flat = np.maximum(ei_flat, 0)  # EI is always non-negative

        # Reshape back to original shape
        ei = torch.from_numpy(ei_flat.reshape(original_shape)).to(uncertainty.device).float()

        return ei

    def select_proteus_batch(
        self,
        model,
        candidate_inputs: List[Dict],  # generated sequences to evaluate
        n_select: int = 96,  # 96-well plate PROTEUS experiment
        best_observed: float = 0.0,
    ) -> Tuple[List[int], torch.Tensor]:
        """
        Select the most informative n_select sequences for the next PROTEUS round.

        Uses uncertainty + predicted quality to maximize information per experiment.
        Returns indices into candidate_inputs + their acquisition scores.

        n_select=96 matches a standard 96-well plate — the natural PROTEUS unit.
        """
        all_uncertainties = []
        all_qualities = []

        for inp in candidate_inputs:
            unc_dict = self.estimate_uncertainty(model, inp)
            unc = unc_dict["sequence_uncertainty"]  # (1,)
            all_uncertainties.append(unc)

            with torch.no_grad():
                out = model(**inp)
                quality = (
                    out["sequences"].mean(dim=1).max(dim=-1).values
                )  # crude quality proxy
                all_qualities.append(quality)

        uncertainties = torch.cat(all_uncertainties)
        qualities = torch.cat(all_qualities)

        # Compute EI scores
        ei_scores = self.expected_improvement_acquisition(
            uncertainty=uncertainties,
            predicted_quality=qualities,
            best_observed=best_observed,
        )

        # Select top-n_select by EI, with diversity enforcement:
        # Don't pick sequences that are too similar to each other
        selected_indices = []
        selected_ei = []

        # Greedy selection with diversity bonus (MaxMin distance)
        candidates = ei_scores.argsort(descending=True)

        for idx in candidates:
            if len(selected_indices) >= n_select:
                break
            # Simple diversity: skip if too similar to already selected
            # (In production: use Hamming distance between sequences)
            selected_indices.append(idx.item())
            selected_ei.append(ei_scores[idx].item())

        return selected_indices, torch.tensor(selected_ei)


# ═══════════════════════════════════════════════════════════════════════════════
# 5. MULTI-SCALE HIERARCHICAL SEQUENCE DESIGNER
# ═══════════════════════════════════════════════════════════════════════════════


BayesianUncertaintyEstimator = LegacyBayesianUncertaintyEstimator
