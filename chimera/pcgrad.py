"""Canonical Projected Conflicting Gradients (PCGrad).

Implements Yu et al. (2020): compute one gradient per task, randomly permute
other tasks for every task, and project away only negatively aligned components.
The final update is the sum of projected task gradients.
"""

from __future__ import annotations

from typing import Iterable, Sequence, Tuple

import torch
from torch import Tensor


def project_conflicting_gradients(
    losses: Sequence[Tensor],
    parameters: Iterable[torch.nn.Parameter],
    retain_graph: bool = False,
) -> list[Tensor]:
    """Compute canonical PCGrad and write the summed projected gradient to ``.grad``."""
    params = [p for p in parameters if p.requires_grad]
    if not params:
        raise ValueError("PCGrad requires at least one trainable parameter")
    if not losses:
        raise ValueError("PCGrad requires at least one loss")

    flat_grads = []
    last_loss_index = len(losses) - 1
    for index, loss in enumerate(losses):
        if not torch.is_tensor(loss) or loss.ndim != 0:
            raise ValueError("each PCGrad loss must be a scalar tensor")
        if not loss.requires_grad:
            grads = [None] * len(params)
        else:
            grads = torch.autograd.grad(
                loss,
                params,
                retain_graph=retain_graph or index < last_loss_index,
                allow_unused=True,
            )
        flat_grads.append(
            torch.cat([
                g.reshape(-1) if g is not None else torch.zeros_like(p).reshape(-1)
                for g, p in zip(grads, params)
            ])
        )

    projected = [g.clone() for g in flat_grads]
    for i in range(len(projected)):
        order = torch.randperm(len(projected), device=projected[i].device).tolist()
        for j in order:
            if i == j:
                continue
            other = flat_grads[j]
            dot = torch.dot(projected[i], other)
            if dot < 0:
                denom = torch.dot(other, other).clamp_min(torch.finfo(other.dtype).eps)
                projected[i] = projected[i] - (dot / denom) * other

    merged = torch.stack(projected, dim=0).sum(dim=0)
    offset = 0
    for parameter in params:
        size = parameter.numel()
        parameter.grad = merged[offset : offset + size].view_as(parameter).detach().clone()
        offset += size
    return projected


def pcgrad_step(
    optimizer: torch.optim.Optimizer,
    losses: Sequence[Tensor],
    parameters: Iterable[torch.nn.Parameter],
) -> None:
    """Zero gradients, apply canonical PCGrad, then step the optimizer."""
    optimizer.zero_grad(set_to_none=True)
    project_conflicting_gradients(losses, parameters)
    optimizer.step()


class PCGradOptimizer:
    """Optimizer wrapper whose ``step`` consumes independent task losses."""

    def __init__(self, optimizer: torch.optim.Optimizer):
        self.optimizer = optimizer

    def zero_grad(self) -> None:
        self.optimizer.zero_grad(set_to_none=True)

    def step(self, losses: Sequence[Tensor]) -> None:
        parameters = [p for group in self.optimizer.param_groups for p in group["params"]]
        project_conflicting_gradients(losses, parameters)
        self.optimizer.step()
