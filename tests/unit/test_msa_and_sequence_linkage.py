import pytest

from data.msa import MSAConfig, build_msa_record
from data.sequence_linkage import SequenceCandidate, link_structure_sequence
from data.uniprot import RetrievedProtein, align_sequences_to_query


def test_msa_preserves_three_views_query_and_padding_mask():
    raw = ">query\nACDE\n>homolog\nATx-E\n>low-coverage\nA---\n"
    record = build_msa_record(
        raw,
        msa_family_id="family-1",
        target_sequence_id="target-1",
        source="fixture",
        source_version="v1",
        target_sequence="ACDE",
        taxonomy_by_row={0: "9606", 1: "10090"},
        config=MSAConfig(minimum_coverage=0.5, maximum_gap_fraction=0.5, maximum_depth=4),
    )

    assert record.raw_alignment == raw
    assert record.raw_rows == ["ACDE", "ATx-E", "A---"]
    assert record.normalized_rows == ["ACDE", "AT-E", "A---"]
    assert record.model_ready_row_count == 2
    assert record.model_ready_tokens.shape == (4, 4)
    assert record.model_ready_mask.any(dim=1).sum().item() == 2
    assert record.model_ready_mask[:2].all()
    assert not record.model_ready_mask[2:].any()
    assert record.model_ready_tokens[0].tolist() == [0, 1, 2, 3]
    assert record.query_residue_to_msa_column == [0, 1, 2, 3]
    assert record.neff == 2.0


def test_msa_rejects_target_mismatch_and_unequal_alignment_lengths():
    with pytest.raises(ValueError, match="target sequence"):
        build_msa_record(
            ">query\nAAAA\n",
            msa_family_id="f",
            target_sequence_id="t",
            source="fixture",
            source_version="v1",
            target_sequence="AAAT",
        )
    with pytest.raises(ValueError, match="equal alignment length"):
        build_msa_record(
            ">q\nAAA\n>h\nAA\n",
            msa_family_id="f",
            target_sequence_id="t",
            source="fixture",
            source_version="v1",
        )


def test_sequence_linkage_uses_alignment_metrics_not_names():
    result = link_structure_sequence(
        structure_id="1ABC",
        chain_id="A",
        entity_id="1",
        sequence="ACDEFG",
        candidates=[
            SequenceCandidate("unrelated-looking-name", "ACDEFG", "UniProt", "release-1"),
            SequenceCandidate("better-name-but-wrong", "YYYYYY", "UniProt", "release-1"),
        ],
    )

    assert result.external_accession == "unrelated-looking-name"
    assert result.match_method == "global_pairwise_alignment"
    assert result.sequence_identity == 1.0
    assert result.coverage == 1.0
    assert result.confidence == 1.0


def test_uniprot_homologs_are_projected_to_target_columns_without_losing_raw_sequences():
    proteins = [
        RetrievedProtein("P0TEST", "ACFDE", "562", "E. coli", "2026_03", 1),
        RetrievedProtein("P0GAP", "ACE", "287", "B. subtilis", "2026_03", 3),
    ]
    alignment, raw_records = align_sequences_to_query("ACDE", proteins)

    assert alignment.splitlines() == [">target-query", "ACDE", ">P0TEST", "ACDE", ">P0GAP", "AC-E"]
    assert [record["sequence"] for record in raw_records] == ["ACFDE", "ACE"]
    assert raw_records[0]["taxonomy_id"] == "562"