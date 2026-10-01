"""Leakage-resistant dataset splitting utilities for protein families.

The live pipeline distinguishes between exact-sequence identity families and
heuristic positional sequence-identity clustering. Exact identity is a safe
fallback for duplicate isolation; the heuristic does not establish homology.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Hashable, Mapping, Sequence

import numpy as np


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _sequence_from_record(record: Mapping[str, Any]) -> str:
    for key in ("aa_sequence", "protein_sequence", "sequence", "protein"):
        value = record.get(key)
        if value is None:
            continue
        if isinstance(value, (list, tuple)):
            return "".join(str(part) for part in value).strip().upper()
        return str(value).strip().upper()
    raise ValueError(f"Record has no protein sequence field: {record!r}")


def _sequence_identity(left: Sequence[int] | str, right: Sequence[int] | str) -> float:
    if isinstance(left, str) and isinstance(right, str):
        left_values = list(left)
        right_values = list(right)
        if len(left_values) != len(right_values):
            return 0.0
        if not left_values:
            return 1.0
        matches = sum(a == b for a, b in zip(left_values, right_values))
        return matches / len(left_values)

    left_values = [int(token) for token in left]
    right_values = [int(token) for token in right]
    if len(left_values) != len(right_values):
        return 0.0
    if not left_values:
        return 1.0
    comparable = [a != 21 and b != 21 for a, b in zip(left_values, right_values)]
    if not any(comparable):
        return 1.0
    matches = sum(a == b for a, b, keep in zip(left_values, right_values, comparable) if keep)
    return matches / sum(comparable)


class SequenceClusterer:
    """Base class for leakage-free grouping strategies."""

    name = "base"
    homology_aware = False

    def cluster(self, records: Sequence[Mapping[str, Any]], *, seed: int = 0) -> list[list[int]]:
        raise NotImplementedError


@dataclass(frozen=True)
class ExactIdentityClusterer(SequenceClusterer):
    """Group records by identical protein sequence. This is the explicit fallback path."""

    name: str = "exact-identity"
    homology_aware: bool = False

    def cluster(self, records: Sequence[Mapping[str, Any]], *, seed: int = 0) -> list[list[int]]:
        groups: dict[str, list[int]] = {}
        for index, record in enumerate(records):
            sequence = _sequence_from_record(record)
            groups.setdefault(sequence, []).append(index)
        return [indices for _, indices in sorted(groups.items())]


@dataclass(frozen=True)
class SequenceIdentityClusterer(SequenceClusterer):
    """Greedily group equal-length sequences by positional identity; not a validated homology backend."""

    threshold: float = 0.3
    name: str = "sequence-identity"
    homology_aware: bool = False

    def __post_init__(self) -> None:
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("threshold must be between 0 and 1")

    def cluster(self, records: Sequence[Mapping[str, Any]], *, seed: int = 0) -> list[list[int]]:
        clusters: list[list[int]] = []
        representatives: list[str] = []
        for index, record in enumerate(records):
            sequence = _sequence_from_record(record)
            if sequence == "":
                raise ValueError(f"Record {index} has an empty protein sequence")
            for cluster_index, representative in enumerate(representatives):
                if _sequence_identity(sequence, representative) >= self.threshold:
                    clusters[cluster_index].append(index)
                    break
            else:
                representatives.append(sequence)
                clusters.append([index])
        return clusters


def _cluster_split_state(
    records: Sequence[Mapping[str, Any]],
    train_fraction: float = 0.8,
    validation_fraction: float = 0.1,
    seed: int = 0,
    clusterer: SequenceClusterer | None = None,
) -> tuple[list[list[int]], set[int], set[int], set[int], dict[str, str]]:
    if not records:
        raise ValueError("records must not be empty")
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must lie in (0, 1)")
    if not 0.0 <= validation_fraction < 1.0:
        raise ValueError("validation_fraction must lie in [0, 1)")
    if train_fraction + validation_fraction >= 1.0:
        raise ValueError("train_fraction + validation_fraction must be < 1")
    if seed < 0:
        raise ValueError("seed must be non-negative")

    clusterer = clusterer or SequenceIdentityClusterer(threshold=0.3)
    cluster_groups = clusterer.cluster(records, seed=seed)
    cluster_order = list(range(len(cluster_groups)))
    rng = np.random.default_rng(seed)
    rng.shuffle(cluster_order)
    n_clusters = len(cluster_order)
    n_train = max(1, int(round(train_fraction * n_clusters))) if n_clusters > 1 else 1
    n_val = int(round(validation_fraction * n_clusters))
    if n_train + n_val >= n_clusters:
        n_val = max(0, n_clusters - n_train - 1)

    train_cluster_indices = {cluster_order[i] for i in range(min(n_train, len(cluster_order)))}
    val_cluster_indices = {
        cluster_order[n_train + i]
        for i in range(min(n_val, max(0, len(cluster_order) - n_train)))
    }
    test_cluster_indices = set(range(n_clusters)) - train_cluster_indices - val_cluster_indices

    cluster_membership: dict[str, str] = {}
    for cluster_index, group in enumerate(cluster_groups):
        cluster_id = f"cluster-{cluster_index}"
        for index in group:
            seq = _sequence_from_record(records[index])
            cluster_membership.setdefault(seq, cluster_id)
    return cluster_groups, train_cluster_indices, val_cluster_indices, test_cluster_indices, cluster_membership


def make_clustered_split(
    records: Sequence[Mapping[str, Any]],
    train_fraction: float = 0.8,
    validation_fraction: float = 0.1,
    seed: int = 0,
    clusterer: SequenceClusterer | None = None,
    dataset_version: str = "unknown",
) -> dict[str, Any]:
    """Return a deterministic cluster split with metadata for the lineage manifest."""
    clusterer = clusterer or SequenceIdentityClusterer(threshold=0.3)
    cluster_groups, train_cluster_indices, val_cluster_indices, test_cluster_indices, cluster_membership = _cluster_split_state(
        records,
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
        seed=seed,
        clusterer=clusterer,
    )
    train_indices = sorted(index for cluster_index in train_cluster_indices for index in cluster_groups[cluster_index])
    val_indices = sorted(index for cluster_index in val_cluster_indices for index in cluster_groups[cluster_index])
    test_indices = sorted(index for cluster_index in test_cluster_indices for index in cluster_groups[cluster_index])
    result = {
        "cluster_count": len(cluster_groups),
        "clusterer": clusterer.name,
        "homology_aware": bool(clusterer.homology_aware),
        "dataset_version": dataset_version,
        "split_seed": seed,
        "train_cluster_ids": {f"cluster-{index}" for index in train_cluster_indices},
        "validation_cluster_ids": {f"cluster-{index}" for index in val_cluster_indices},
        "test_cluster_ids": {f"cluster-{index}" for index in test_cluster_indices},
        "train_indices": train_indices,
        "validation_indices": val_indices,
        "test_indices": test_indices,
        "train_record_count": len(train_indices),
        "validation_record_count": len(val_indices),
        "test_record_count": len(test_indices),
        "cluster_membership": cluster_membership,
    }
    result["manifest"] = build_cluster_split_manifest(
        records,
        seed=seed,
        dataset_version=dataset_version,
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
        clusterer=clusterer,
    )
    return result


def build_cluster_split_manifest(
    records: Sequence[Mapping[str, Any]],
    seed: int = 0,
    dataset_version: str = "unknown",
    train_fraction: float = 0.8,
    validation_fraction: float = 0.1,
    clusterer: SequenceClusterer | None = None,
) -> dict[str, Any]:
    """Build a deterministic split manifest that records cluster lineage and hash provenance."""
    clusterer = clusterer or SequenceIdentityClusterer(threshold=0.3)
    cluster_groups, train_cluster_indices, val_cluster_indices, test_cluster_indices, _ = _cluster_split_state(
        records,
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
        seed=seed,
        clusterer=clusterer,
    )
    manifest = {
        "dataset_version": dataset_version,
        "split_seed": seed,
        "train_fraction": train_fraction,
        "validation_fraction": validation_fraction,
        "clusterer": clusterer.name,
        "homology_aware": bool(clusterer.homology_aware),
        "record_count": len(records),
        "cluster_count": len(cluster_groups),
        "train_record_count": sum(len(cluster_groups[idx]) for idx in train_cluster_indices),
        "validation_record_count": sum(len(cluster_groups[idx]) for idx in val_cluster_indices),
        "test_record_count": sum(len(cluster_groups[idx]) for idx in test_cluster_indices),
        "train_cluster_ids": sorted(f"cluster-{index}" for index in train_cluster_indices),
        "validation_cluster_ids": sorted(f"cluster-{index}" for index in val_cluster_indices),
        "test_cluster_ids": sorted(f"cluster-{index}" for index in test_cluster_indices),
    }
    payload = _canonical_json({key: value for key, value in manifest.items() if key != "manifest_hash"})
    manifest["manifest_hash"] = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return manifest


def grouped_split(
    groups: Sequence[Hashable],
    train_fraction: float = 0.8,
    validation_fraction: float = 0.1,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
