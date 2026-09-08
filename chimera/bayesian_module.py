"""Neural-module boundary for the Bayesian acquisition estimator."""

from __future__ import annotations

import torch.nn as nn

from .bayesian import BayesianUncertaintyEstimator as _BayesianUncertaintyEstimator


class BayesianUncertaintyModule(_BayesianUncertaintyEstimator, nn.Module):
    """Module-compatible Bayesian estimator used by CHIMERAv2.

    The estimator itself has no trainable parameters, but CHIMERAv2 treats
    model components uniformly when freezing pretrained modules. Making the
    public integration type an ``nn.Module`` preserves that invariant without
    inventing trainable Bayesian parameters.
    """

    def __init__(self, *args, **kwargs):
        nn.Module.__init__(self)
        _BayesianUncertaintyEstimator.__init__(self, *args, **kwargs)
