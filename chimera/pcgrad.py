"""Actual Projected Conflicting Gradients (PCGrad)."""

from typing import Iterable, Sequence
import torch
from torch import Tensor


def project_conflicting_gradients(
    losses: Sequence[Tensor],
    parameters: Iterable[torch.nn.Parameter],
    retain_graph: bool = False,
) -> list[Tensor]:
    """Compute PCGrad: project each task gradient away from conflicts."""
    params = [p for p in parameters if p.requires_grad]
    if not params:
        raise ValueError("PCGrad requires at least one trainable parameter")
    if not losses:
        raise ValueError("PCGrad requires at least one loss")

    flat_grads = []
    last_loss_index = len(losses) - 1
    for index, loss in enumerate(losses):
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
            torch.cat(
                [
                    g.reshape(-1) if g is not None else torch.zeros_like(p).reshape(-1)
                    for g, p in zip(grads, params)
                ]
            )
        )

    projected = [gradient.clone() for gradient in flat_grads]
    for i, gradient in enumerate(projected):
        for j in torch.randperm(len(flat_grads)).tolist():
            if i == j:
                continue
            other = flat_grads[j]
            dot = torch.dot(gradient, other)
            if dot < 0:
                gradient -= dot / other.pow(2).sum().clamp_min(1e-12) * other

    merged = torch.stack(projected).mean(dim=0)
    offset = 0
    for parameter in params:
        size = parameter.numel()
        parameter.grad = merged[offset : offset + size].view_as(parameter).detach().clone()
        offset += size
    return projected


class PCGradOptimizer:
    """Optimizer wrapper whose ``step`` consumes independent task losses."""

    def __init__(self, optimizer: torch.optim.Optimizer):
        self.optimizer = optimizer

    def zero_grad(self) -> None:
        self.optimizer.zero_grad()

    def step(self, losses: Sequence[Tensor]) -> None:
        parameters = [p for group in self.optimizer.param_groups for p in group["params"]]
        project_conflicting_gradients(losses, parameters)
        self.optimizer.step()
