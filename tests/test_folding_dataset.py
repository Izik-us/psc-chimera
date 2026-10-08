import pytest

from data.folding_dataset import (
    FoldingQC, apply_qc, assert_no_cluster_overlap, carve_calibration_partition, cluster_balanced_weights,
    evaluate_record,
)


def rec(**kw):
    ca = [(3.8 * i, 0.0, 0.0) for i in range(100)]
    base = dict(id="a", sequence="A" * 100, method="X-RAY DIFFRACTION", resolution=1.8, rfree=0.22, ca=ca, mean_bfactor=30.0)
    base.update(kw)
    return base


def test_good_record_passes_and_each_defect_is_named():
    assert evaluate_record(rec()).passed
    assert "resolution" in evaluate_record(rec(resolution=3.4)).reasons[0]
    assert any("R-free" in r for r in evaluate_record(rec(rfree=0.4)).reasons)
    gap = [(3.8 * i + (20 if i > 50 else 0), 0.0, 0.0) for i in range(100)]
    gap[60] = (gap[60][0] + 25, 0, 0)
    assert any("chain breaks" in r for r in evaluate_record(rec(ca=[(10.0 * i, 0, 0) for i in range(100)])).reasons)
    assert any("unaligned" in r for r in evaluate_record(rec(ca=[(0, 0, 0)] * 90)).reasons)
    assert any("non-standard" in r for r in evaluate_record(rec(sequence="X" * 20 + "A" * 80)).reasons)


def test_bfactor_outliers_and_report():
    rows = [rec(id=str(i), mean_bfactor=30.0 + (i % 5)) for i in range(40)] + [rec(id="bad", mean_bfactor=400.0)]
    kept, report = apply_qc(rows)
    assert "bad" not in {r["id"] for r in kept} and report["n_rejected"] == 1


def test_weights_and_calibration_carve_is_cluster_disjoint():
    w = cluster_balanced_weights(["c1", "c1", "c1", "c2"])
    assert abs(sum(w) / 4 - 1) < 1e-9 and w[3] > w[0]
    ids = [f"r{i}" for i in range(200)]
    group = {r: f"g{i // 4}" for i, r in enumerate(ids)}
    split = {r: ("validation" if i % 2 else "train") for i, r in enumerate(ids)}
    # make clusters split-pure first, as the leakage-safe splitter guarantees
    split = {r: ("validation" if int(group[r][1:]) % 2 else "train") for r in ids}
    out = carve_calibration_partition(ids, split, group)
    assert {"train", "validation", "calibration"} == set(out.values())
    assert_no_cluster_overlap(out, group)
    out["r0"], out["r1"] = "train", "validation"
    with pytest.raises(ValueError):
        assert_no_cluster_overlap(out, group)
