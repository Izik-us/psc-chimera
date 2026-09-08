"""Reproducibility helpers for CHIMERA experiments."""

from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = False) -> None:
    """Seed Python, NumPy and PyTorch RNGs and optionally request deterministic kernels."""
    if not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False


def make_generator(seed: int, device: Optional[torch.device] = None) -> torch.Generator:
    """Create an explicitly seeded torch Generator for isolated stochastic paths."""
    if seed < 0:
        raise ValueError("seed must be non-negative")
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return generator
