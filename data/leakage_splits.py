"""Deterministic leakage-safe split generation with explicit novelty holdouts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

from .sequence_linkage import _alignment_metrics

SPLIT_DIMENSIONS = {
    "sequence_cluster": "sequence_cluster_id",
    "structural_similarity": "structural_cluster_id",
    "protein_family": "protein_family_id",
    "domain_family": "domain_family_ids",
    "nrps_family": "nrps_family_id",
    "taxonomy": "taxonomy_id",
    "substrate": "substrate_ids",
    "module_composition": "module_composition",
}


@dataclass(frozen=True)
class SplitConfig:
    train_fraction: float = 0.7
    validation_fraction: float = 0.1
    test_fraction: float = 0.1
    ood_fraction: float = 0.1
    sequence_identity_threshold: float = 0.3
    minimum_alignment_coverage: float = 0.8
    seed: int = 0
    ood_dimension: str | None = "nrps_family"
    maximum_pairwise_sequences: int = 5000

    def __post_init__(self) -> None:
        fractions = (
            self.train_fraction,
            self.validation_fraction,
            self.test_fraction,
            self.ood_fraction,
        )
        if any(value < 0.0 or value > 1.0 for value in fractions):
            raise ValueError("split fractions must be in [0, 1]")
        if abs(sum(fractions) - 1.0) > 1e-8:
            raise ValueError("train/validation/test/OOD fractions must sum to 1")
        if not 0.0 <= self.sequence_identity_threshold <= 1.0:
            raise ValueError("sequence_identity_threshold must be in [0, 1]")
        if not 0.0 <= self.minimum_alignment_coverage <= 1.0:
            raise ValueError("minimum_alignment_coverage must be in [0, 1]")
        if self.seed < 0 or self.maximum_pairwise_sequences < 1:
            raise ValueError("seed must be non-negative and maximum_pairwise_sequences positive")
        if self.ood_dimension is not None and self.ood_dimension not in SPLIT_DIMENSIONS:
            raise ValueError(f"unsupported OOD dimension: {self.ood_dimension}")


@dataclass(frozen=True)
class SplitManifest:
    dataset_version: str
    algorithm_version: str
    seed: int
    assignments: Mapping[str, str]
    record_groups: Mapping[str, str]
    leakage_dimensions: tuple[str, ...]
    unavailable_dimensions: tuple[str, ...]
    ood_dimension: str | None
    held_out_novelty_values: tuple[str, ...]
    dimension_group_counts: Mapping[str, int]
    manifest_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class _UnionFind:
    def __init__(self, size: int):
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        root_left, root_right = self.find(left), self.find(right)
        if root_left == root_right:
            return
        if self.rank[root_left] < self.rank[root_right]:
            root_left, root_right = root_right, root_left
        self.parent[root_right] = root_left
        if self.rank[root_left] == self.rank[root_right]:
            self.rank[root_left] += 1


def _values(value: Any) -> tuple[str, ...]:
    if value is None or value == "":
        return ()
    if isinstance(value, (list, tuple, set)):
        return tuple(sorted({str(item) for item in value if item is not None and str(item)}))
    return (str(value),)


def _stable_order(seed: int, key: str) -> str:
    return hashlib.sha256(f"{seed}:{key}".encode("utf-8")).hexdigest()


def _assign_grouped_components(
    components: list[list[int]],
    records: Sequence[Mapping[str, Any]],
    config: SplitConfig,
    ood_field: str | None,
) -> tuple[dict[int, str], tuple[str, ...]]:
    total = len(records)
    if ood_field is None or config.ood_fraction == 0.0:
        held_out: tuple[str, ...] = ()
        ood_components: set[int] = set()
    else:
        values_to_components: dict[str, set[int]] = {}
        for component_index, members in enumerate(components):
            for record_index in members:
                for value in _values(records[record_index].get(ood_field)):
                    values_to_components.setdefault(value, set()).add(component_index)
        if not values_to_components:
            raise ValueError(f"OOD field {ood_field!r} has no values in the dataset")
        ordered_values = sorted(values_to_components, key=lambda value: _stable_order(config.seed, f"ood:{value}"))
        target_count = max(1, round(total * config.ood_fraction))
        selected_values: list[str] = []
        ood_components = set()
        covered_records = 0
        for value in ordered_values:
            selected_values.append(value)
            new_components = values_to_components[value] - ood_components
            ood_components.update(new_components)
            covered_records = sum(len(components[index]) for index in ood_components)
            if covered_records >= target_count:
                break
        held_out = tuple(sorted(selected_values))

    assignments: dict[int, str] = {}
    remaining = []
    for component_index, members in enumerate(components):
        if component_index in ood_components:
            assignments[component_index] = "OOD"
        else:
            remaining.append((component_index, members))

    remaining.sort(
        key=lambda item: _stable_order(
            config.seed,
            "|".join(sorted(str(records[index].get("record_id", index)) for index in item[1])),
        )
    )
    targets = {
        "TRAIN": round(total * config.train_fraction),
        "VALIDATION": round(total * config.validation_fraction),
        "TEST": round(total * config.test_fraction),
    }
    counts = {name: 0 for name in targets}
    for component_index, members in remaining:
        candidates = [name for name in ("TRAIN", "VALIDATION", "TEST") if counts[name] < targets[name]]
        split_name = candidates[0] if candidates else "TEST"
        assignments[component_index] = split_name
        counts[split_name] += len(members)
    return assignments, held_out


def generate_leakage_safe_splits(
    records: Sequence[Mapping[str, Any]],
    *,
    dataset_version: str,
    config: SplitConfig | None = None,
    dimensions: Sequence[str] = tuple(SPLIT_DIMENSIONS),
) -> SplitManifest:
    """Group any records sharing a declared leakage dimension before assignment.

    For controlled corpora without a supplied sequence cluster label, pairs are
    aligned and merged when identity and query coverage exceed configured
    thresholds. Large builds must provide an external sequence cluster label.
    """
    config = config or SplitConfig()
    if not records:
        raise ValueError("records must not be empty")
    if any(dimension not in SPLIT_DIMENSIONS for dimension in dimensions):
        raise ValueError("unknown leakage dimension requested")
    identifiers = [str(record.get("record_id", "")) for record in records]
    if any(not identifier for identifier in identifiers) or len(set(identifiers)) != len(identifiers):
        raise ValueError("every record must have a unique non-empty record_id")

    union_find = _UnionFind(len(records))
    dimension_group_counts: dict[str, int] = {}
    unavailable_dimensions: list[str] = []
    for dimension in dimensions:
        field_name = SPLIT_DIMENSIONS[dimension]
        groups: dict[str, list[int]] = {}
        present = False
        for index, record in enumerate(records):
            values = _values(record.get(field_name))
            present |= bool(values)
            for value in values:
                groups.setdefault(value, []).append(index)
        dimension_group_counts[dimension] = len(groups)
        if not present:
            unavailable_dimensions.append(dimension)
        for members in groups.values():
            for member in members[1:]:
                union_find.union(members[0], member)

    has_external_sequence_clusters = all(record.get("sequence_cluster_id") for record in records)
    sequences = [str(record.get("sequence", record.get("protein_sequence", ""))).upper() for record in records]
    if not has_external_sequence_clusters:
        if len(records) > config.maximum_pairwise_sequences:
            raise ValueError(
                "large split input requires externally computed sequence_cluster_id values"
            )
        for left in range(len(records)):
            if not sequences[left]:
                raise ValueError(f"record {identifiers[left]} lacks a sequence for leakage control")
            for right in range(left + 1, len(records)):
                identity, coverage, _mismatches, _gaps, _mapping = _alignment_metrics(
                    sequences[left], sequences[right]
                )
                if identity >= config.sequence_identity_threshold and coverage >= config.minimum_alignment_coverage:
                    union_find.union(left, right)

    grouped: dict[int, list[int]] = {}
    for index in range(len(records)):
        grouped.setdefault(union_find.find(index), []).append(index)
    components = list(grouped.values())

    ood_field = SPLIT_DIMENSIONS[config.ood_dimension] if config.ood_dimension else None
    if ood_field and ood_field not in {SPLIT_DIMENSIONS[item] for item in dimensions}:
        if not any(record.get(ood_field) for record in records):
            raise ValueError(f"OOD dimension {config.ood_dimension!r} has no annotation values")
    component_assignments, held_out_values = _assign_grouped_components(
        components, records, config, ood_field
    )
    assignments: dict[str, str] = {}
    record_groups: dict[str, str] = {}
    for component_index, members in enumerate(components):
        group_id = "group-" + _stable_order(config.seed, "|".join(sorted(identifiers[index] for index in members)))[:16]
        for index in members:
            assignments[identifiers[index]] = component_assignments[component_index]
            record_groups[identifiers[index]] = group_id

    body = {
        "dataset_version": dataset_version,
        "algorithm_version": "multi-dimension-union-split-v0.1",
        "seed": config.seed,
        "assignments": dict(sorted(assignments.items())),
        "record_groups": dict(sorted(record_groups.items())),
        "leakage_dimensions": list(dimensions),
        "unavailable_dimensions": sorted(unavailable_dimensions),
        "ood_dimension": config.ood_dimension,
        "held_out_novelty_values": list(held_out_values),
        "dimension_group_counts": dimension_group_counts,
    }
    manifest_hash = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return SplitManifest(
        dataset_version=dataset_version,
        algorithm_version="multi-dimension-union-split-v0.1",
        seed=config.seed,
        assignments=assignments,
        record_groups=record_groups,
        leakage_dimensions=tuple(dimensions),
        unavailable_dimensions=tuple(sorted(unavailable_dimensions)),
        ood_dimension=config.ood_dimension,
        held_out_novelty_values=held_out_values,
        dimension_group_counts=dimension_group_counts,
        manifest_sha256=manifest_hash,
    )