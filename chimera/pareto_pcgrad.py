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

    def pcgrad_loss(self, objectives, labels, weights=None, generator: Optional[torch.Generator] = None):
        """Return a scalar whose backward gradient is canonical PCGrad.

        ``generator`` makes task ordering reproducible without coupling PCGrad
        to unrelated global RNG consumption. Conflict telemetry counts each
        unordered task pair once.
        """
        del weights
        losses = self._task_losses(objectives, labels)
        if not losses:
            return torch.zeros((), device=self._last_repr.device), {}
        names = list(losses)
        task_losses = [losses[name] for name in names]
        parameters = [p for p in self.parameters() if p.requires_grad]
        variables = [self._last_repr] + parameters

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
                pieces.append(
                    torch.zeros_like(variable).reshape(-1)
                    if grad is None
                    else grad.reshape(-1)
                )
            flat_grads.append(torch.cat(pieces))
        G = torch.stack(flat_grads)

        projected = G.clone()
        coefficients = torch.eye(len(names), device=G.device, dtype=G.dtype)
        conflicts = set()

        def permutation(n: int):
            if generator is None:
                return torch.randperm(n, device=G.device).tolist()
            return torch.randperm(n, generator=generator, device=G.device).tolist()

        for i in permutation(len(names)):
            for j in permutation(len(names)):
                if i == j:
                    continue
                dot = torch.dot(projected[i], G[j])
                if dot < 0:
                    conflicts.add((min(i, j), max(i, j)))
                    denom = torch.dot(G[j], G[j]).clamp_min(torch.finfo(G.dtype).eps)
                    correction = dot / denom
                    projected[i] = projected[i] - correction * G[j]
                    coefficients[i] = coefficients[i] - correction * coefficients[j]

        combined_coefficients = coefficients.sum(dim=0).detach()
        total = sum(c * loss for c, loss in zip(combined_coefficients, task_losses))
        metrics = {name: float(loss.detach()) for name, loss in losses.items()}
        metrics["pcgrad_task_count"] = len(names)
        metrics["pcgrad_conflict_pairs"] = len(conflicts)
        return total, metrics
