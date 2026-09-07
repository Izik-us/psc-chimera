"""Bayesian uncertainty and acquisition utilities.

MC dropout is used only as an approximate posterior over model parameters. Its
sample variance is epistemic uncertainty. Aleatoric uncertainty is reported
only when the predictor explicitly supplies a predictive variance; it is not
silently conflated with MC variance.

Expected Improvement uses the standard deviation sigma, not variance:
    EI = (mu - f_best - xi) Phi(Z) + sigma phi(Z)
    Z  = (mu - f_best - xi) / sigma
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Callable, Dict, Optional, Sequence

import torch
import torch.nn as nn


@contextmanager
def _dropout_only(model: nn.Module):
    """Enable Dropout layers while preserving every module's prior mode."""
    states = {module: module.training for module in model.modules()}
    try:
        model.eval()
        for module in model.modules():
            if isinstance(
                module,
                (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d, nn.AlphaDropout),
            ):
                module.train()
        yield
    finally:
        for module, state in states.items():
            module.train(state)


class BayesianUncertaintyEstimator:
    """MC-dropout posterior predictive uncertainty estimator.

    This is an approximate Bayesian method, not an exact posterior. The
    estimator intentionally exposes both variance and standard deviation so
    acquisition functions cannot accidentally treat variance as sigma.
    """

    def __init__(self, n_samples: int = 30):
        if n_samples < 2:
            raise ValueError("n_samples must be at least 2")
        self.n_samples = int(n_samples)

    def predict(
        self,
        model: nn.Module,
        inputs: Dict,
        output_getter: Optional[Callable] = None,
        n_samples: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """Return predictive mean, epistemic uncertainty and optional aleatoric variance."""
        n = int(n_samples or self.n_samples)
        if n < 2:
            raise ValueError("n_samples must be at least 2")
        samples = []
        predictive_variances = []
        with torch.no_grad(), _dropout_only(model):
            for _ in range(n):
                output = model(**inputs)
                if output_getter is not None:
                    value = output_getter(output)
                    variance = None
                elif isinstance(output, dict) and "mean" in output:
                    value = output["mean"]
                    variance = output.get("variance")
                elif torch.is_tensor(output):
                    value = output
                    variance = None
                else:
                    raise TypeError("provide output_getter for non-tensor model outputs")
                if not torch.is_tensor(value):
                    raise TypeError("output_getter must return a tensor")
                samples.append(value.detach())
                if variance is not None:
                    predictive_variances.append(variance.detach())

        stack = torch.stack(samples, dim=0)
        mean = stack.mean(dim=0)
        variance = stack.var(dim=0, unbiased=True)
        result = {
            "mean": mean,
            "epistemic_variance": variance,
            "epistemic_std": variance.clamp_min(0).sqrt(),
            "samples": stack,
        }
        if predictive_variances:
            aleatoric = torch.stack(predictive_variances, dim=0).mean(dim=0).clamp_min(0)
            result["aleatoric_variance"] = aleatoric
            result["total_variance"] = variance + aleatoric
        return result

    def estimate_uncertainty(
        self,
        model: nn.Module,
        inputs: Dict,
        n_samples: Optional[int] = None,
        per_candidate: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Compatibility API for CHIMERA's candidate-generation path.

        CHIMERA returns five objectives with different units. Uncertainty is
        therefore computed after objective-wise normalization rather than
        averaging pLDDT (0-100) directly with [0,1] scores. The returned
        ``per_candidate_uncertainty`` is a standard deviation, suitable for EI.
        """
        objective_keys = (
            "evol_plausibility",
            "structural_stability",
            "expression_efficiency",
            "substrate_selectivity",
            "assembly_compat",
        )

        def getter(output):
            if not isinstance(output, dict) or any(k not in output for k in objective_keys):
                raise TypeError("CHIMERA uncertainty requires all five objective outputs")
            values = []
            for key in objective_keys:
                value = output[key]
                if value.ndim == 1:
                    value = value.unsqueeze(-1)
                values.append(value)
            stacked = torch.stack(values, dim=-1)
            # Robust within-batch normalization removes the pLDDT 0-100 scale.
            lo = stacked.detach().amin(dim=1, keepdim=True)
            hi = stacked.detach().amax(dim=1, keepdim=True)
            return (stacked - lo) / (hi - lo).clamp_min(1e-6)

        result = self.predict(model, inputs, output_getter=getter, n_samples=n_samples)
        result["candidate_epistemic"] = result["epistemic_variance"].mean(dim=-1)
        result["per_candidate_uncertainty"] = result["epistemic_std"].mean(dim=-1)
        result["candidate_quality"] = result["mean"].mean(dim=-1)
        return result

    @staticmethod
    def expected_improvement(
        mean: torch.Tensor,
        std: torch.Tensor,
        best_observed: float,
        xi: float = 0.0,
    ) -> torch.Tensor:
        """Analytic Gaussian EI. ``std`` is a standard deviation, never variance."""
        if mean.shape != std.shape:
            raise ValueError("mean and std must have identical shapes")
        if xi < 0:
            raise ValueError("xi must be non-negative")
        sigma = std.clamp_min(1e-12)
        improvement = mean - float(best_observed) - float(xi)
        z = improvement / sigma
        normal = torch.distributions.Normal(torch.zeros_like(z), torch.ones_like(z))
        ei = improvement * normal.cdf(z) + sigma * torch.exp(normal.log_prob(z))
        deterministic = (mean - float(best_observed) - float(xi)).clamp_min(0)
        return torch.where(std > 1e-7, ei, deterministic)

    @staticmethod
    def scalarize_objectives(
        objectives: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
        lower: Optional[torch.Tensor] = None,
        upper: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Normalize heterogeneous objective scales before scalarization."""
        if objectives.ndim < 1:
            raise ValueError("objectives must have an objective dimension")
        if lower is None or upper is None:
            q_lo = torch.quantile(objectives.detach(), 0.05, dim=0)
            q_hi = torch.quantile(objectives.detach(), 0.95, dim=0)
            lower, upper = q_lo, q_hi
        scale = (upper - lower).clamp_min(1e-6)
        normalized = ((objectives - lower) / scale).clamp(0, 1)
        if weights is None:
            weights = torch.ones(objectives.shape[-1], device=objectives.device, dtype=objectives.dtype)
        if weights.shape != (objectives.shape[-1],):
            raise ValueError("weights must match the objective dimension")
        weights = weights / weights.sum().clamp_min(1e-8)
        return (normalized * weights).sum(dim=-1)

    @classmethod
    def rank_candidates(
        cls,
        predicted_mean: torch.Tensor,
        epistemic_std: torch.Tensor,
        best_observed: float,
        n_select: int,
        xi: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return top candidate indices and EI scores for an external experiment round."""
        if predicted_mean.ndim != 1 or epistemic_std.shape != predicted_mean.shape:
            raise ValueError("predicted_mean and epistemic_std must have shape (N,)")
        if not 1 <= n_select <= predicted_mean.numel():
            raise ValueError("n_select must be between 1 and the number of candidates")
        ei = cls.expected_improvement(predicted_mean, epistemic_std, best_observed, xi)
        scores, indices = torch.topk(ei, n_select)
        return indices, scores
