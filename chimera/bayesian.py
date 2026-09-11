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
from typing import Callable, Dict, Optional

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


class BayesianUncertaintyEstimator(nn.Module):
    """MC-dropout posterior predictive uncertainty estimator.

    This is an approximate Bayesian method, not an exact posterior. The
    estimator intentionally exposes both variance and standard deviation so
    acquisition functions cannot accidentally treat variance as sigma.
    """

    def __init__(self, n_samples: int = 30, *, n_mc_samples: Optional[int] = None):
        super().__init__()
        if n_mc_samples is not None:
            if n_samples != 30 and int(n_samples) != int(n_mc_samples):
                raise ValueError("n_samples and n_mc_samples disagree")
            n_samples = int(n_mc_samples)
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

    def predict_fixed_candidate(
        self,
        predictor: nn.Module,
        inputs: Dict,
        output_getter: Optional[Callable] = None,
        n_samples: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """Estimate uncertainty for an already-generated candidate.

        This API deliberately separates Bayesian evaluation from candidate
        generation. The supplied predictor is evaluated repeatedly with only
        dropout stochasticity enabled, so SB noise or autoregressive sampling
        cannot silently enter the epistemic estimate.
        """
        return self.predict(
            predictor,
            inputs,
            output_getter=output_getter,
            n_samples=n_samples,
        )

    @staticmethod
    def _normalize_chimera_objectives(output: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Map CHIMERA's five heterogeneous objective channels into [0,1]."""
        keys = (
            "evol_plausibility",
            "structural_stability",
            "expression_efficiency",
            "substrate_selectivity",
            "assembly_compat",
        )
        if any(k not in output for k in keys):
            raise TypeError("CHIMERA uncertainty requires all five objective outputs")
        evol = torch.sigmoid(output["evol_plausibility"])
        stab = output["structural_stability"].clamp(0, 100) / 100.0
        expr = output["expression_efficiency"].clamp(0, 1)
        sel = output["substrate_selectivity"].clamp(0, 1)
        asm = output["assembly_compat"].clamp(0, 1)
        return torch.stack([evol, stab, expr, sel, asm], dim=-1)

    def estimate_uncertainty(
        self,
        model: nn.Module,
        inputs: Dict,
        n_samples: Optional[int] = None,
        per_candidate: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Return normalized five-objective uncertainty for CHIMERA candidates."""
        del per_candidate
        result = self.predict(
            model,
            inputs,
            output_getter=self._normalize_chimera_objectives,
            n_samples=n_samples,
        )
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
        deterministic = improvement.clamp_min(0)
        return torch.where(std > 1e-7, ei, deterministic)

    @staticmethod
    def scalarize_objectives(
        objectives: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
        lower: Optional[torch.Tensor] = None,
        upper: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Scalarize objectives using explicit, fixed semantic bounds.

        If bounds are omitted, objectives are assumed already normalized to
        [0,1]. Candidate-batch quantiles are intentionally not used because
        moving normalization would change the acquisition target as the batch
        changes and can manufacture apparent improvement.
        """
        if objectives.ndim < 1:
            raise ValueError("objectives must have an objective dimension")
        n_obj = objectives.shape[-1]
        if lower is None:
            lower = torch.zeros(n_obj, device=objectives.device, dtype=objectives.dtype)
        if upper is None:
            upper = torch.ones(n_obj, device=objectives.device, dtype=objectives.dtype)
        lower = torch.as_tensor(lower, device=objectives.device, dtype=objectives.dtype)
        upper = torch.as_tensor(upper, device=objectives.device, dtype=objectives.dtype)
        if lower.shape != (n_obj,) or upper.shape != (n_obj,):
            raise ValueError("lower and upper must match the objective dimension")
        if torch.any(upper <= lower):
            raise ValueError("each upper bound must be greater than its lower bound")
        normalized = ((objectives - lower) / (upper - lower)).clamp(0, 1)
        if weights is None:
            weights = torch.ones(n_obj, device=objectives.device, dtype=objectives.dtype)
        if weights.shape != (n_obj,):
            raise ValueError("weights must match the objective dimension")
        if torch.any(weights < 0) or weights.sum() <= 0:
            raise ValueError("weights must be non-negative with positive total mass")
        weights = weights / weights.sum()
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
