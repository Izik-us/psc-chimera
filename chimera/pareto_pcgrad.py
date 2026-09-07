"""True gradient-surgery Pareto head for CHIMERAv2.

The historical Pareto head's ``pcgrad_loss`` adjusted scalar loss weights from
loss magnitudes. That is not PCGrad. This implementation computes each task
gradient, performs the canonical pairwise conflict projection, then expresses
the projected sum as coefficients of the original task losses. The detached
coefficients preserve the exact projected first-order gradient when the caller
later invokes ``total.backward()``.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn.functional as F

from .multi_objective import ParetoMultiObjectiveHead as _LegacyParetoHead


class MergeReadyParetoMultiObjectiveHead(_LegacyParetoHead):
    """Pareto head whose compatibility ``pcgrad_loss`` is real PCGrad."""

    def forward(self, repr: torch.Tensor):
        self._last_repr = repr
        return super().forward(repr)

    @staticmethod
    def _task_losses(objectives, labels: Dict[str, Optional[torch.Tensor]]):
        losses = {}
        if labels.get("evol") is not None:
            losses["evol"] = F.mse_loss(objectives.evolutionary_plausibility, labels["evol"])
        if labels.get("stab") is not None:
            losses["stab"] = F.mse_loss(objectives.structural_stability, labels["stab"])
        if labels.get("expr") is not None:
            losses["expr"] = F.binary_cross_entropy(objectives.expression_efficiency, labels["expr"])
        if labels.get("sel") is not None:
            losses["sel"] = F.mse_loss(objectives.substrate_selectivity, labels["sel"])
        if labels.get("asm") is not None:
            losses["asm"] = F.mse_loss(objectives.assembly_compatibility, labels["asm"])
        return losses

    def pcgrad_loss(self, objectives, labels, weights=None):
        """Return a scalar whose backward gradient is canonical PCGrad."""
        del weights
        losses = self._task_losses(objectives, labels)
        if not losses:
            return torch.zeros((), device=self._last_repr.device), {}
        names = list(losses)
        task_losses = [losses[name] for name in names]
        parameters = [p for p in self.parameters() if p.requires_grad]
        variables = [self._last_repr] + parameters

        # Each row is one complete task gradient over the input representation
        # plus Pareto-head parameters. ``allow_unused`` is required because a
        # head can be absent from a particular task configuration.
        flat_grads = []
        for loss in task_losses:
            grads = torch.autograd.grad(
                loss,
                variables,
                retain_graph=True,
                allow_unused=True,
            )
            pieces = []
            for variable, grad in zip(variables, grads):
                if grad is None:
                    pieces.append(torch.zeros_like(variable).reshape(-1))
                else:
                    pieces.append(grad.reshape(-1))
            flat_grads.append(torch.cat(pieces))
        G = torch.stack(flat_grads)

        # Track each projected gradient as a linear combination of original
        # task gradients. The coefficient matrix starts as identity and is
        # transformed alongside the gradient vectors.
        projected = G.clone()
        coefficients = torch.eye(len(names), device=G.device, dtype=G.dtype)
        order = torch.randperm(len(names), device=G.device)
        for i in order.tolist():
            others = torch.randperm(len(names), device=G.device).tolist()
            for j in others:
                if i == j:
                    continue
                dot = torch.dot(projected[i], G[j])
                denom = torch.dot(G[j], G[j]).clamp_min(1e-12)
                if dot < 0:
                    correction = dot / denom
                    projected[i] = projected[i] - correction * G[j]
                    coefficients[i] = coefficients[i] - correction * coefficients[j]

        combined_coefficients = coefficients.sum(dim=0).detach()
        total = sum(c * loss for c, loss in zip(combined_coefficients, task_losses))
        metrics = {name: float(loss.detach()) for name, loss in losses.items()}
        metrics["pcgrad_task_count"] = len(names)
        metrics["pcgrad_conflict_pairs"] = int(
            sum(1 for i in range(len(names)) for j in range(len(names)) if i != j and torch.dot(G[i], G[j]) < 0)
        )
        return total, metrics
