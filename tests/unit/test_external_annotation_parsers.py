from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqFeature import FeatureLocation, SeqFeature
from Bio.SeqRecord import SeqRecord

from data.annotations import (
    EvidenceClass,
    parse_antismash_genbank,
    parse_interpro_domain_matches,
    parse_uniprot_nrps_features,
    combine_uniprot_interpro_nrps,
)
from data.interpro import InterProClient


def test_uniprot_carrier_and_ppant_features_preserve_evidence_and_numbering():
    annotation = parse_uniprot_nrps_features(
        {
            "primaryAccession": "P0C062",
            "sequence": {"value": "A" * 1098},
            "genes": [{"geneName": {"value": "grsA"}}],
            "features": [
                {
                    "type": "Domain",
                    "featureId": "carrier",
                    "description": "Carrier",
                    "location": {"start": {"value": 538}, "end": {"value": 612}},
                    "evidences": [{"evidenceCode": "ECO:0000255", "source": "PROSITE-ProRule"}],
                },
                {
                    "type": "Modified residue",
                    "description": "O-(pantetheine 4'-phosphoryl)serine",
                    "location": {"start": {"value": 573}, "end": {"value": 573}},
                    "evidences": [{"evidenceCode": "ECO:0000269", "source": "PubMed", "id": "7718566"}],
                },
            ],
        },
        accession="P0C062",
        release="2026_03",
        verification_date="2026-10-04",
    )

    carrier = annotation.domains[0]
    assert carrier.domain_type == "T/PCP"
    assert (carrier.start_aa, carrier.end_aa) == (537, 612)
    assert carrier.boundary_evidence.evidence_class == EvidenceClass.DATABASE_COMPUTED
    assert carrier.ppant_attachment_residue == 572
    assert carrier.ppant_evidence.evidence_class == EvidenceClass.EXPERIMENTAL


def test_antismash_genbank_parser_maps_domains_to_protein_and_module_relationships(tmp_path):
    record = SeqRecord(Seq("ATG" * 2000), id="cluster-record", name="cluster-record", description="test cluster")
    record.annotations["molecule_type"] = "DNA"
    record.features = [
        SeqFeature(
            FeatureLocation(0, 600, strand=1),
            type="CDS",
            qualifiers={"locus_tag": ["grsA"], "translation": ["A" * 200]},
        ),
        SeqFeature(
            FeatureLocation(30, 300, strand=1),
            type="aSDomain",
            qualifiers={"locus_tag": ["grsA"], "aSDomain": ["AMP-binding"], "domain_id": ["A1"], "specificity": ["phenylalanine"]},
        ),
        SeqFeature(
            FeatureLocation(300, 450, strand=1),
            type="aSDomain",
            qualifiers={"locus_tag": ["grsA"], "aSDomain": ["PCP"], "domain_id": ["T1"]},
        ),
        SeqFeature(
            FeatureLocation(0, 450, strand=1),
            type="aSModule",
            qualifiers={"module_number": ["1"], "domains": ["A1", "T1"]},
        ),
        SeqFeature(
            FeatureLocation(0, 900, strand=1),
            type="region",
            qualifiers={"product": ["gramicidin S"]},
        ),
    ]
    path = tmp_path / "antismash.gbk"
    SeqIO.write(record, path, "genbank")

    annotation = parse_antismash_genbank(
        path,
        cluster_id="record:region-1",
        tool_version="8.0",
        verification_date="2026-10-04",
    )

    assert [domain.domain_type for domain in annotation.domains] == ["A", "T/PCP"]
    assert [(domain.start_aa, domain.end_aa) for domain in annotation.domains] == [(10, 100), (100, 150)]
    assert annotation.modules[0].adenylation_domain_id == "A1"
    assert annotation.modules[0].carrier_domain_id == "T1"
    assert annotation.modules[0].substrate_association is None
    assert annotation.products[0].evidence.evidence_class == EvidenceClass.DATABASE_COMPUTED


def test_interpro_domain_locations_are_normalized_and_remain_computed():
    annotation = parse_interpro_domain_matches(
        {
            "metadata": {"accession": "P0C062", "gene": "grsA", "length": 1098},
            "results": [
                {
                    "metadata": {"accession": "IPR010071", "name": "Amino acid adenylation domain", "type": "domain"},
                    "proteins": [{
                        "accession": "p0c062",
                        "entry_protein_locations": [{"fragments": [{"start": 66, "end": 460, "dc-status": "CONTINUOUS"}]}],
                    }],
                },
                {
                    "metadata": {"accession": "IPR009081", "name": "Phosphopantetheine binding ACP domain", "type": "domain"},
                    "proteins": [{
                        "accession": "p0c062",
                        "entry_protein_locations": [{"fragments": [{"start": 538, "end": 612, "dc-status": "CONTINUOUS"}]}],
                    }],
                },
            ],
        },
        accession="P0C062",
        release="110.0",
        verification_date="2026-10-04",
    )

    assert [(domain.domain_type, domain.start_aa, domain.end_aa) for domain in annotation.domains] == [
        ("A", 65, 460),
        ("T/PCP", 537, 612),
    ]
    assert all(domain.boundary_evidence.evidence_class == EvidenceClass.DATABASE_COMPUTED for domain in annotation.domains)


def test_interpro_uniprot_merge_preserves_a_t_relationship_and_source_evidence():
    interpro = parse_interpro_domain_matches(
        {
            "metadata": {"accession": "P0C062", "gene": "grsA", "length": 1098},
            "results": [
                {
                    "metadata": {"accession": "IPR010071", "name": "Amino acid adenylation domain", "type": "domain"},
                    "proteins": [{"accession": "P0C062", "entry_protein_locations": [{"fragments": [{"start": 66, "end": 460}]}]}],
                },
                {
                    "metadata": {"accession": "IPR009081", "name": "Phosphopantetheine binding ACP domain", "type": "domain"},
                    "proteins": [{"accession": "P0C062", "entry_protein_locations": [{"fragments": [{"start": 538, "end": 612}]}]}],
                },
            ],
        },
        accession="P0C062",
        release="110.0",
        verification_date="2026-10-04",
    )
    uniprot = {
        "primaryAccession": "P0C062",
        "sequence": {"value": "A" * 1098},
        "genes": [{"geneName": {"value": "grsA"}}],
        "features": [
            {
                "type": "Domain", "description": "Carrier",
                "location": {"start": {"value": 538}, "end": {"value": 612}},
                "evidences": [{"evidenceCode": "ECO:0000255", "source": "PROSITE"}],
            },
            {
                "type": "Modified residue", "description": "phosphopantetheine serine",
                "location": {"start": {"value": 573}, "end": {"value": 573}},
                "evidences": [{"evidenceCode": "ECO:0000269", "source": "PubMed", "id": "7718566"}],
            },
        ],
        "comments": [
            {"commentType": "DOMAIN", "texts": [{"value": "One-module-bearing peptide synthase with adenylation and thiolation domains", "evidences": [{"evidenceCode": "ECO:0000250"}]}]},
            {"commentType": "CATALYTIC ACTIVITY", "reaction": {"name": "L-phenylalanine + ATP = D-phenylalanine + AMP"}, "evidences": [{"evidenceCode": "ECO:0000250"}]},
        ],
    }
    merged = combine_uniprot_interpro_nrps(
        interpro,
        uniprot,
        uniprot_release="2026_03",
        interpro_release="110.0",
        verification_date="2026-10-04",
    )

    assert merged.modules[0].adenylation_domain_id.startswith("InterPro:IPR010071")
    carrier = next(domain for domain in merged.domains if domain.domain_type == "T/PCP")
    assert carrier.ppant_attachment_residue == 572
    assert carrier.boundary_evidence.evidence_class == EvidenceClass.DATABASE_COMPUTED
    assert carrier.ppant_evidence.evidence_class == EvidenceClass.EXPERIMENTAL
    assert merged.products[0].substrate_name == "L-phenylalanine"
    assert merged.products[0].evidence.evidence_class == EvidenceClass.SEQUENCE_INFERRED


def test_interpro_client_follows_pagination_and_records_release(monkeypatch):
    client = InterProClient(requests_per_second=1000)
    pages = iter(
        [
            ({"count": 2, "next": "https://example.test/page2", "results": [{"id": 1}]}, type("R", (), {"headers": {"InterPro-Version": "110.0"}})()),
            ({"count": 2, "next": None, "results": [{"id": 2}]}, type("R", (), {"headers": {"InterPro-Version": "110.0"}})()),
        ]
    )
    visited = []

    def fake_get(url):
        visited.append(url)
        return next(pages)

    monkeypatch.setattr(client, "_get_json", fake_get)
    payload, release = client.fetch_protein_matches("P0C062")

    assert release == "110.0"
    assert [item["id"] for item in payload["results"]] == [1, 2]
    assert visited[-1] == "https://example.test/page2"