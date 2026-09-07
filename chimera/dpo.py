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
    """Stable DPO objective with a frozen reference policy."""

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
        """Return one masked sequence log-probability per sample."""
        if context.ndim == 2:
            context = context.unsqueeze(1).expand(-1, tokens.shape[1], -1)
        elif context.ndim != 3:
            raise ValueError("context must have shape (B,D) or (B,L,D)")
        if context.shape[:2] != tokens.shape:
            raise ValueError("context sequence dimensions must match token dimensions")

        if mask is not None:
            if mask.shape != tokens.shape:
                raise ValueError("sequence mask must have shape (B,L)")
            if hasattr(policy, "token_logprobs"):
                token_logp = policy.token_logprobs(context, tokens)
            elif callable(policy):
                logits = policy(context, tokens)
                expected = (*tokens.shape, logits.shape[-1])
                if logits.shape != expected:
                    raise ValueError("policy forward must return logits with shape (B,L,V)")
                token_logp = F.log_softmax(logits, dim=-1).gather(-1, tokens.unsqueeze(-1)).squeeze(-1)
            else:
                raise TypeError("masked DPO requires token_logprobs or a callable policy")
            return (token_logp * mask.to(token_logp.dtype)).sum(dim=-1)

        if not hasattr(policy, "logprob"):
            raise TypeError("policy must expose logprob(context, tokens) for unmasked DPO")
        logp = policy.logprob(context, tokens)
        if logp.ndim != 1 or logp.shape[0] != tokens.shape[0]:
            raise ValueError("policy.logprob must return shape (B,)")
        return logp

    def loss(self, policy, reference, batch: DPOBatch):
        """Return DPO loss and detached diagnostics."""
        pi_chosen = self._sequence_logprob(policy, batch.context, batch.chosen, batch.chosen_mask)
        pi_rejected = self._sequence_logprob(policy, batch.context, batch.rejected, batch.rejected_mask)
        if self.reference_free:
            ref_chosen = torch.zeros_like(pi_chosen)
            ref_rejected = torch.zeros_like(pi_rejected)
        else:
            if reference is None:
                raise ValueError("reference policy is required unless reference_free=True")
            with torch.no_grad():
                ref_chosen = self._sequence_logprob(reference, batch.context, batch.chosen, batch.chosen_mask)
                ref_rejected = self._sequence_logprob(reference, batch.context, batch.rejected, batch.rejected_mask)

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
        optimizer.zero_grad(set_to_none=True)
        loss, metrics = self.loss(policy, reference, batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
        optimizer.step()
        return metrics
