"""Direct Preference Optimization for autoregressive sequence policies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F


@dataclass
class DPOBatch:
    context: torch.Tensor
    chosen: torch.Tensor
    rejected: torch.Tensor
    chosen_mask: Optional[torch.Tensor] = None
    rejected_mask: Optional[torch.Tensor] = None


class DPOTrainer:
    """Stable DPO objective with a frozen, deterministic reference policy."""

    def __init__(self, beta: float = 0.1, label_smoothing: float = 0.0, reference_free: bool = False):
        if beta <= 0:
            raise ValueError("beta must be positive")
        if not 0.0 <= label_smoothing < 0.5:
            raise ValueError("label_smoothing must be in [0, 0.5)")
        self.beta = float(beta)
        self.label_smoothing = float(label_smoothing)
        self.reference_free = bool(reference_free)

    @staticmethod
    def _sequence_logprob(policy, context, tokens, mask=None):
        """Return one masked sequence log-probability per sample.

        Context dimensionality is preserved for ``logprob`` implementations.
        Some policies consume a single conditioning vector ``(B,D)`` while the
        canonical token-level interface consumes ``(B,L,D)``.  Expanding the
        former before calling it changes its semantics and can create a rank-4
        tensor inside otherwise valid reference policies.
        """
        if tokens.ndim != 2 or tokens.shape[1] < 1:
            raise ValueError("tokens must have shape (B,L) with L >= 1")
        original_context = context
        if context.ndim == 2:
            sequence_context = context.unsqueeze(1).expand(-1, tokens.shape[1], -1)
        elif context.ndim == 3:
            sequence_context = context
        else:
            raise ValueError("context must have shape (B,D) or (B,L,D)")
        if sequence_context.shape[:2] != tokens.shape:
            raise ValueError("context sequence dimensions must match token dimensions")

        token_logp = None
        if hasattr(policy, "token_logprobs"):
            token_logp = policy.token_logprobs(sequence_context, tokens)
        elif hasattr(policy, "logprob"):
            # Preserve the caller's context rank.  A policy exposing logprob()
            # owns its conditioning contract and may intentionally expect (B,D).
            logp = policy.logprob(original_context, tokens)
            if logp.ndim == 3:
                expected = (*tokens.shape, logp.shape[-1])
                if logp.shape != expected:
                    raise ValueError("policy.logprob logits must have shape (B,L,V)")
                token_logp = F.log_softmax(logp, dim=-1).gather(
                    -1, tokens.unsqueeze(-1)
                ).squeeze(-1)
            elif logp.ndim == 2 and logp.shape == tokens.shape:
                token_logp = logp
            elif logp.ndim == 1 and logp.shape[0] == tokens.shape[0]:
                return logp
            else:
                raise ValueError("policy.logprob must return shape (B,), (B,L), or (B,L,V)")
        elif callable(policy):
            logits = policy(sequence_context, tokens)
            expected = (*tokens.shape, logits.shape[-1])
            if logits.shape != expected:
                raise ValueError("policy forward must return logits with shape (B,L,V)")
            token_logp = F.log_softmax(logits, dim=-1).gather(-1, tokens.unsqueeze(-1)).squeeze(-1)

        if token_logp is None:
            raise TypeError("policy must expose token_logprobs, logprob, or a forward returning logits")
        if token_logp.shape != tokens.shape:
            raise ValueError("token log-probabilities must have shape (B,L)")
        if mask is None:
            mask = torch.ones_like(tokens, dtype=torch.bool)
        elif mask.shape != tokens.shape:
            raise ValueError("sequence mask must have shape (B,L)")
        if not mask.any(dim=-1).all():
            raise ValueError("every DPO sequence must contain at least one unmasked token")
        return (token_logp * mask.to(token_logp.dtype)).sum(dim=-1)

    def loss(self, policy, reference, batch: DPOBatch):
        """Return DPO loss and detached diagnostics."""
        if batch.chosen.shape != batch.rejected.shape:
            raise ValueError("chosen and rejected sequences must have identical shapes")
        policy.train()
        pi_chosen = self._sequence_logprob(policy, batch.context, batch.chosen, batch.chosen_mask)
        pi_rejected = self._sequence_logprob(policy, batch.context, batch.rejected, batch.rejected_mask)
        if self.reference_free:
            ref_chosen = torch.zeros_like(pi_chosen)
            ref_rejected = torch.zeros_like(pi_rejected)
        else:
            if reference is None:
                raise ValueError("reference policy is required unless reference_free=True")
            reference_training = reference.training
            reference.eval()
            try:
                with torch.no_grad():
                    ref_chosen = self._sequence_logprob(reference, batch.context, batch.chosen, batch.chosen_mask)
                    ref_rejected = self._sequence_logprob(reference, batch.context, batch.rejected, batch.rejected_mask)
            finally:
                reference.train(reference_training)

        chosen_adv = pi_chosen - ref_chosen
        rejected_adv = pi_rejected - ref_rejected
        logits = self.beta * (chosen_adv - rejected_adv)
        positive = F.logsigmoid(logits)
        negative = F.logsigmoid(-logits)
        loss = -((1.0 - self.label_smoothing) * positive + self.label_smoothing * negative).mean()
        with torch.no_grad():
            accuracy = (logits > 0).float().mean()
            chosen_reward = chosen_adv.mean()
            rejected_reward = rejected_adv.mean()
        return loss, {
            "loss": float(loss.detach()),
            "preference_accuracy": float(accuracy),
            "chosen_reward": float(chosen_reward),
            "rejected_reward": float(rejected_reward),
            "reward_margin": float(chosen_reward - rejected_reward),
        }

    def step(self, optimizer, policy, reference, batch: DPOBatch, max_grad_norm: float = 1.0):
        """One optimizer step."""
        if max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be positive")
        optimizer.zero_grad(set_to_none=True)
        loss, metrics = self.loss(policy, reference, batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
        optimizer.step()
        return metrics
