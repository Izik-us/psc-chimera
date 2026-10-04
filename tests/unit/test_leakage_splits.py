from data.leakage_splits import SplitConfig, generate_leakage_safe_splits


def test_sequence_similarity_family_and_ood_groups_never_cross_splits():
    records = [
        {"record_id": "a", "sequence": "ACDEFG", "sequence_cluster_id": "s1", "nrps_family_id": "fam-a", "taxonomy_id": "10"},
        {"record_id": "b", "sequence": "ACDEFG", "sequence_cluster_id": "s1", "nrps_family_id": "fam-a", "taxonomy_id": "10"},
        {"record_id": "c", "sequence": "YYYYYY", "sequence_cluster_id": "s2", "nrps_family_id": "fam-b", "taxonomy_id": "20"},
        {"record_id": "d", "sequence": "WWWWWW", "sequence_cluster_id": "s3", "nrps_family_id": "fam-c", "taxonomy_id": "30"},
        {"record_id": "e", "sequence": "VVVVVV", "sequence_cluster_id": "s4", "nrps_family_id": "fam-d", "taxonomy_id": "40"},
    ]
    manifest = generate_leakage_safe_splits(
        records,
        dataset_version="CHIMERA-DATASET-v0.1",
        config=SplitConfig(seed=7, ood_dimension="nrps_family"),
    )

    assert manifest.assignments["a"] == manifest.assignments["b"]
    assert manifest.record_groups["a"] == manifest.record_groups["b"]
    assert set(manifest.assignments.values()) <= {"TRAIN", "VALIDATION", "TEST", "OOD"}
    assert any(value.startswith("fam-") for value in manifest.held_out_novelty_values)
    assert len(manifest.manifest_sha256) == 64


def test_split_manifest_is_deterministic_and_reports_unavailable_dimensions():
    records = [
        {"record_id": f"id-{index}", "sequence": sequence, "sequence_cluster_id": f"cluster-{index}", "nrps_family_id": f"family-{index}"}
        for index, sequence in enumerate(("ACDE", "YYYY", "WWWW", "VVVV", "GGGG"))
    ]
    first = generate_leakage_safe_splits(
        records,
        dataset_version="v1",
        config=SplitConfig(seed=2, ood_dimension="nrps_family"),
    )
    second = generate_leakage_safe_splits(
        records,
        dataset_version="v1",
        config=SplitConfig(seed=2, ood_dimension="nrps_family"),
    )

    assert first.manifest_sha256 == second.manifest_sha256
    assert "taxonomy" in first.unavailable_dimensions
    assert first.assignments == second.assignments


def test_partial_external_clusters_do_not_disable_pairwise_leakage_control():
    records = [
        {"record_id": "clustered", "sequence": "ACDEFG", "sequence_cluster_id": "family-1", "nrps_family_id": "fam-a"},
        {"record_id": "unlabeled-duplicate", "sequence": "ACDEFG", "nrps_family_id": "fam-b"},
        {"record_id": "different", "sequence": "YYYYYY", "sequence_cluster_id": "family-2", "nrps_family_id": "fam-c"},
    ]
    manifest = generate_leakage_safe_splits(
        records,
        dataset_version="v1",
        config=SplitConfig(seed=3, ood_dimension="nrps_family"),
    )

    assert manifest.assignments["clustered"] == manifest.assignments["unlabeled-duplicate"]