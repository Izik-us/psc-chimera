import numpy as np
import pytest

from data.annotations import (
    DomainAnnotation,
    EvidenceClass,
    EvidenceRecord,
    ModuleAnnotation,
    build_antismash_annotation,
    build_ligand_context,
    parse_mibig_entry,
)
from data.structures import LigandRecord


def test_mibig_products_are_curated_and_domains_are_not_invented():
    annotation = parse_mibig_entry(
        {"cluster": {"genes": [{"gene": "geneA"}], "compounds": [{"compound": "example-product"}]}},
        accession="BGC0000001",
        release="4.0",
        verification_date="2026-10-04",
    )

    assert annotation.evidence.evidence_class == EvidenceClass.DATABASE_CURATED
    assert annotation.products[0].substrate_name == "example-product"
    assert annotation.products[0].evidence.evidence_class == EvidenceClass.DATABASE_CURATED
    assert annotation.domains == ()
    assert annotation.modules == ()


def test_antismash_domain_and_module_evidence_stays_computational():
    evidence = EvidenceRecord(
        "antiSMASH", "cluster-1", "8.0", "2026-10-04", EvidenceClass.DATABASE_COMPUTED
    )
    domains = (
        DomainAnnotation("A1", "A", "protein-1", 0, 100, evidence, (10, 20)),
        DomainAnnotation("T1", "T/PCP", "protein-1", 100, 160, evidence, ppant_attachment_residue=130, ppant_evidence=evidence),
    )
    module = ModuleAnnotation("module-1", 1, ("A1", "T1"), "A1", "T1", None, None, evidence)
    annotation = build_antismash_annotation(
        cluster_id="cluster-1",
        genes=["protein-1"],
        domains=domains,
        modules=[module],
        tool_version="8.0",
        verification_date="2026-10-04",
    )
    annotation.validate({"protein-1": 200})

    assert annotation.evidence.evidence_class == EvidenceClass.DATABASE_COMPUTED
    assert annotation.domains[0].selectivity_positions == (10, 20)
    assert annotation.domains[1].ppant_attachment_residue == 130


def test_antiSMASH_predictions_cannot_be_relabelled_as_experimental():
    evidence = EvidenceRecord(
        "antiSMASH", "cluster-1", "8.0", "2026-10-04", EvidenceClass.EXPERIMENTAL
    )
    domain = DomainAnnotation("A1", "A", "protein-1", 0, 100, evidence)

    with pytest.raises(ValueError, match="relabeled as experimental"):
        build_antismash_annotation(
            cluster_id="cluster-1",
            genes=["protein-1"],
            domains=[domain],
            modules=[],
            tool_version="8.0",
            verification_date="2026-10-04",
        )


def test_pdb_ligand_context_keeps_role_unassigned_and_maps_binding_domain():
    ligand = LigandRecord(
        chemical_component_id="ATP",
        label_asym_id="A",
        auth_asym_id="A",
        auth_seq_id="501",
        coordinates=np.asarray([[1.0, 2.0, 3.0]], dtype=np.float32),
        binding_residue_indices=[4, 5],
        minimum_contact_distance=2.1,
    )
    context = build_ligand_context(
        structure_id="1ABC",
        ligand_id="1ABC:A:ATP:501",
        ligand=ligand,
        chain_entity_id="1",
        canonical_chemical_ids={"PubChem": "5957"},
        domains=[("A-domain-1", 0, 20)],
        experimental_method="X-RAY DIFFRACTION",
        resolution_angstrom=2.0,
        verification_date="2026-10-04",
    )

    assert context.chemical_component_id == "ATP"
    assert context.binding_domain_ids == ["A-domain-1"]
    assert context.role == "UNASSIGNED"
    assert context.evidence.evidence_class == EvidenceClass.EXPERIMENTAL