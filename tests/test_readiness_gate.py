from chimera.readiness import PreProductionGateEvidence, preproduction_readiness_report


def _passing_evidence():
    return PreProductionGateEvidence(
        pytest_passed=True,
        compileall_passed=True,
        diff_check_passed=True,
        architecture_contracts_passed=True,
        checkpoint_compatibility_passed=True,
        objective_contracts_passed=True,
        gradient_smoke_passed=True,
        checkpoint_smoke_passed=True,
        provenance_complete=True,
        environment_reproducibility_passed=True,
        documentation_reviewed=True,
        test_count=1,
        failures=0,
        skipped=0,
    )


def test_engineering_gate_pass_does_not_claim_scientific_validation():
    report = preproduction_readiness_report(_passing_evidence())

    assert report["engineering_status"] == "PASS"
    assert report["scientific_status"] == "NOT SCIENTIFICALLY VALIDATED"
    assert report["production_engineering_started"] is False


def test_engineering_gate_blocks_missing_or_failed_evidence():
    evidence = _passing_evidence()
    report = preproduction_readiness_report(
        PreProductionGateEvidence(**{**evidence.__dict__, "checkpoint_compatibility_passed": None})
    )

    assert report["engineering_status"] == "BLOCKED"
    assert "checkpoint_compatibility_passed is missing" in report["blockers"]