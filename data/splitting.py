"""Leakage-resistant dataset splitting utilities for protein families.

Sequence-level random splits can leak homologs across train/validation/test and
inflate generalization metrics. This module provides deterministic family/group
splits; callers should supply a precomputed family or cluster identifier (for
example from a sequence-identity clustering pipeline).
"""

from __future__ import annotations

from typing import Hashable, Sequence, Tuple

import numpy as np


def grouped_split(
    groups: Sequence[Hashable],
    train_fraction: float = 0.8,
    validation_fraction: float = 0.1,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split samples by group so no group occurs in more than one partition."""
    if not groups:
        raise ValueError("groups must not be empty")
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must lie in (0,1)")
    if not 0.0 <= validation_fraction < 1.0:
        raise ValueError("validation_fraction must lie in [0,1)")
    if train_fraction + validation_fraction >= 1.0:
        raise ValueError("train_fraction + validation_fraction must be < 1")
    if seed < 0:
        raise ValueError("seed must be non-negative")

    unique = np.asarray(list(dict.fromkeys(groups)), dtype=object)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    n_groups = len(unique)
    n_train = max(1, int(round(train_fraction * n_groups)))
    n_val = int(round(validation_fraction * n_groups))
    if n_train + n_val >= n_groups:
        n_val = max(0, n_groups - n_train - 1)

    train_groups = set(unique[:n_train].tolist())
    val_groups = set(unique[n_train : n_train + n_val].tolist())
    group_array = np.asarray(groups, dtype=object)
    train_idx = np.flatnonzero(np.isin(group_array, list(train_groups)))
    val_idx = np.flatnonzero(np.isin(group_array, list(val_groups)))
    test_idx = np.flatnonzero(~np.isin(group_array, list(train_groups | val_groups)))
    return train_idx, val_idx, test_idx
