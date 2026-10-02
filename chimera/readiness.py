"""Explicit engineering gate; it is separate from model inference readiness."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class PreProductionGateEvidence:
    pytest_passed: bool | None
    compileall_passed: bool | None
    diff_check_passed: bool | None
    architecture_contracts_passed: bool | None
    checkpoint_compatibility_passed: bool | None
    objective_contracts_passed: bool | None
    gradient_smoke_passed: bool | None
    checkpoint_smoke_passed: bool | None
    provenance_complete: bool | None
    environment_reproducibility_passed: bool | None
    documentation_reviewed: bool | None
    test_count: int | None = None
    failures: int | None = None
    skipped: int | None = None
    known_scientific_limitations: tuple[str, ...] = (
        "No independent biological or experimental validation is established by engineering tests.",
        "Objective labels and deterministic evaluator scores remain proxies unless separately validated.",
        "Generated-structure objective use may differ in distribution from reference structures.",
    )


def preproduction_readiness_report(evidence: PreProductionGateEvidence) -> dict[str, Any]:
    """Summarize engineering evidence without implying biological validation."""
    values = asdict(evidence)
    required = {
        key: value
        for key, value in values.items()
        if key.endswith("_passed") or key in ("provenance_complete", "documentation_reviewed")
    }
    blockers = [
        f"{name} is {'missing' if value is None else 'not satisfied'}"
        for name, value in required.items()
        if value is not True
    ]
    if evidence.test_count is None or evidence.test_count < 1:
        blockers.append("test_count is missing or zero")
    if evidence.failures is None:
        blockers.append("failure count is missing")
    elif evidence.failures != 0:
        blockers.append(f"test failures reported: {evidence.failures}")
    if evidence.skipped is None:
        blockers.append("skipped-test count is missing")
    elif evidence.skipped != 0:
        blockers.append(f"tests were skipped: {evidence.skipped}")
    return {
        "engineering_status": "PASS" if not blockers else "BLOCKED",
        "scientific_status": "NOT SCIENTIFICALLY VALIDATED",
        "evidence": values,
        "blockers": blockers,
        "production_engineering_started": False,
    }