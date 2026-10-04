"""Backward-compatible public entry points for the canonical RCSB pipeline."""

from __future__ import annotations

from typing import Any, Iterable

from .rcsb import AcquisitionSummary, RCSBAcquirer


def acquire_rcsb_structure(
    structure_id: str,
    output_dir: str,
    *,
    expected_checksum: str | None = None,
    assembly_id: str | None = None,
) -> dict[str, Any]:
    """Acquire one entry through staged cache-aware RCSB ingestion."""
    return RCSBAcquirer().acquire_id(
        structure_id,
        output_dir,
        expected_sha256=expected_checksum,
        assembly_id=assembly_id,
    )


def run_external_seed_pipeline(
    structure_ids: Iterable[str] | str,
    output_dir: str,
    *,
    assembly_id: str | None = None,
) -> dict[str, Any]:
    """Acquire a deterministic controlled set of RCSB structure IDs."""
    return RCSBAcquirer().acquire_many(
        [structure_ids] if isinstance(structure_ids, str) else structure_ids,
        output_dir,
        assembly_id=assembly_id,
    )


__all__ = [
    "AcquisitionSummary",
    "acquire_rcsb_structure",
    "run_external_seed_pipeline",
]