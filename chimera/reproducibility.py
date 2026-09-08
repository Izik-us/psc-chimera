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
    if deterministic:
        # Must be set before a CUDA context is initialized for CUDA BLAS
        # operations to honor the requested deterministic workspace.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def seed_worker(worker_id: int) -> None:
    """Seed NumPy/Python RNGs from PyTorch's per-worker initial seed.

    Pass this as ``DataLoader(worker_init_fn=seed_worker)`` when stochastic
    dataset transforms are used. PyTorch creates a distinct initial seed for
    each worker, so this avoids every worker producing the same NumPy stream.
    """
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_generator(seed: int, device: Optional[torch.device] = None) -> torch.Generator:
    """Create an explicitly seeded torch Generator for isolated stochastic paths."""
    if not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return generator
