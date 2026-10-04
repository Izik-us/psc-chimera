"""Evidence-typed NRPS and substrate/ligand annotation contracts."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from enum import Enum
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import numpy as np
from Bio import SeqIO

from .structures import LigandRecord
from .ncbi import NCBIProteinSequence


class EvidenceClass(str, Enum):
    EXPERIMENTAL = "EXPERIMENTAL"
    DATABASE_CURATED = "DATABASE_CURATED"
    DATABASE_COMPUTED = "DATABASE_COMPUTED"
    SEQUENCE_INFERRED = "SEQUENCE_INFERRED"
    STRUCTURE_INFERRED = "STRUCTURE_INFERRED"
    HEURISTIC = "HEURISTIC"


@dataclass(frozen=True)
class EvidenceRecord:
    source: str
    source_accession: str
    source_version: str
    verification_date: str
    evidence_class: EvidenceClass
    evidence_detail: str | None = None


@dataclass(frozen=True)
class DomainAnnotation:
    domain_id: str
    domain_type: str
    protein_accession: str
    start_aa: int
    end_aa: int
    boundary_evidence: EvidenceRecord
    selectivity_positions: tuple[int, ...] = ()
    selectivity_prediction: str | None = None
    selectivity_evidence: EvidenceRecord | None = None
    ppant_attachment_residue: int | None = None
    ppant_evidence: EvidenceRecord | None = None
    additional_evidence: tuple[EvidenceRecord, ...] = ()

    def validate(self, protein_length: int) -> None:
        if not 0 <= self.start_aa < self.end_aa <= protein_length:
            raise ValueError(f"invalid domain span {self.start_aa}:{self.end_aa}")
        if self.domain_type not in {"A", "T/PCP", "C", "TE", "LINKER", "E", "Cy", "MT", "OTHER"}:
            raise ValueError(f"unsupported NRPS domain type: {self.domain_type}")
        if any(position < self.start_aa or position >= self.end_aa for position in self.selectivity_positions):
            raise ValueError("selectivity positions must fall within the annotated domain")
        if self.ppant_attachment_residue is not None and not self.start_aa <= self.ppant_attachment_residue < self.end_aa:
            raise ValueError("PPant attachment residue must fall within the carrier domain")


@dataclass(frozen=True)
class SubstrateAssociation:
    substrate_name: str
    canonical_identifiers: Mapping[str, str]
    evidence: EvidenceRecord
    confidence: float | None = None
    association_type: str = "biosynthetic_product_or_monomer"

    def __post_init__(self) -> None:
        if self.confidence is not None and not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")


@dataclass(frozen=True)
class ModuleAnnotation:
    module_id: str
    order: int
    domain_ids: tuple[str, ...]
    adenylation_domain_id: str | None
    carrier_domain_id: str | None
    condensation_domain_id: str | None
    substrate_association: SubstrateAssociation | None
    evidence: EvidenceRecord
    incomplete: bool = False


@dataclass(frozen=True)
class NRPSClusterAnnotation:
    cluster_id: str
    genes: tuple[str, ...]
    domains: tuple[DomainAnnotation, ...]
    modules: tuple[ModuleAnnotation, ...]
    products: tuple[SubstrateAssociation, ...]
    evidence: EvidenceRecord
    raw_metadata: Mapping[str, Any] = field(default_factory=dict)

    def validate(self, protein_lengths: Mapping[str, int]) -> None:
        domain_ids = set()
        for domain in self.domains:
            if domain.domain_id in domain_ids:
                raise ValueError(f"duplicate domain id {domain.domain_id}")
            domain_ids.add(domain.domain_id)
            if domain.protein_accession not in protein_lengths:
                raise ValueError(f"missing protein length for {domain.protein_accession}")
            domain.validate(protein_lengths[domain.protein_accession])
        module_orders = [module.order for module in self.modules]
        if len(module_orders) != len(set(module_orders)):
            raise ValueError("module order values must be unique")
        for module in self.modules:
            unknown = set(module.domain_ids) - domain_ids
            if unknown:
                raise ValueError(f"module {module.module_id} refers to unknown domains: {sorted(unknown)}")
            if module.adenylation_domain_id and module.carrier_domain_id is None:
                raise ValueError("A-domain module must explicitly represent its T/PCP relationship")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MIBiGProteinLinkage:
    cluster_accession: str
    nucleotide_accession: str
    protein_accession: str
    locus_tag: str | None
    gene: str | None
    product: str | None
    target_protein_accession: str
    sequence_identity: float
    coverage: float
    mismatches: int
    gaps: int
    evidence: EvidenceRecord


@dataclass
class LigandContext:
    structure_id: str
    ligand_id: str
    chemical_component_id: str
    canonical_chemical_ids: dict[str, str]
    coordinates: np.ndarray
    binding_residue_indices: list[int]
    binding_residue_refs: list[dict[str, Any]]
    binding_chain_ids: list[str]
    binding_domain_ids: list[str]
    minimum_contact_distance: float | None
    experimental_context: dict[str, str | None]
    evidence: EvidenceRecord
    confidence: float | None = None
    role: str = "UNASSIGNED"

    def to_dict(self, *, include_coordinates: bool = False) -> dict[str, Any]:
        payload = asdict(self)
        if include_coordinates:
            payload["coordinates"] = self.coordinates.tolist()
        else:
            payload.pop("coordinates")
        return payload


def parse_mibig_entry(
    payload: Mapping[str, Any],
    *,
    accession: str,
    release: str,
    verification_date: str,
) -> NRPSClusterAnnotation:
    """Normalize MIBiG cluster/product metadata without inventing domain evidence."""
    cluster = payload.get("cluster", payload)
    if not isinstance(cluster, Mapping):
        raise ValueError("MIBiG entry must contain a cluster object")
    quality = cluster.get("quality", "UNKNOWN")
    status = cluster.get("status", "UNKNOWN")
    evidence = EvidenceRecord(
        source="MIBiG",
        source_accession=accession,
        source_version=release,
        verification_date=verification_date,
        evidence_class=EvidenceClass.DATABASE_CURATED,
        evidence_detail=f"MIBiG release entry; quality={quality}; status={status}",
    )
    products: list[SubstrateAssociation] = []
    for product in cluster.get("compounds", []) or []:
        if not isinstance(product, Mapping):
            continue
        name = product.get("compound") or product.get("name")
        if not name:
            continue
        products.append(
            SubstrateAssociation(
                substrate_name=str(name),
                canonical_identifiers={
                    key: str(value)
                    for key, value in product.items()
                    if key.lower() in {"pubchem", "chebi", "inchikey", "inchi", "smiles"} and value
                },
                evidence=EvidenceRecord(
                    source="MIBiG",
                    source_accession=accession,
                    source_version=release,
                    verification_date=verification_date,
                    evidence_class=EvidenceClass.DATABASE_CURATED,
                    evidence_detail=(
                        "MIBiG compound listing; explicit evidence methods: "
                        + ", ".join(
                            str(item.get("method", "UNKNOWN"))
                            for item in product.get("evidence", []) or []
                            if isinstance(item, Mapping)
                        )
                        if product.get("evidence")
                        else "MIBiG lists this compound but records no explicit compound-evidence method."
                    ),
                ),
                confidence=None,
                association_type="cluster_product",
            )
        )
    genes = cluster.get("genes", []) or []
    gene_ids = tuple(
        str(gene.get("gene", gene.get("locus_tag", gene.get("id", "UNKNOWN"))))
        for gene in genes
        if isinstance(gene, Mapping)
    )
    if not gene_ids and isinstance(genes, (list, tuple)):
        gene_ids = tuple(str(gene) for gene in genes)
    biosynthesis = cluster.get("biosynthesis", {}) or {}
    biosynthetic_classes = [
        str(item.get("class"))
        for item in biosynthesis.get("classes", [])
        if isinstance(item, Mapping) and item.get("class")
    ]
    return NRPSClusterAnnotation(
        cluster_id=accession,
        genes=gene_ids,
        domains=(),
        modules=(),
        products=tuple(products),
        evidence=evidence,
        raw_metadata={
            "biosynthetic_classes": biosynthetic_classes or cluster.get("biosyn_class", []),
            "organism_name": cluster.get("organism_name") or cluster.get("taxonomy", {}).get("name"),
            "taxonomy_id": cluster.get("taxonomy", {}).get("ncbiTaxId"),
            "quality": quality,
            "status": status,
            "loci": cluster.get("loci", []),
            "mibig_version": release,
        },
    )


def link_mibig_cluster_protein(
    annotation: NRPSClusterAnnotation,
    *,
    nucleotide_accession: str,
    target_protein_accession: str,
    target_sequence: str,
    genbank_proteins: Sequence[NCBIProteinSequence],
    verification_date: str,
    minimum_identity: float = 0.8,
    minimum_coverage: float = 0.8,
) -> tuple[NRPSClusterAnnotation, MIBiGProteinLinkage | None]:
    """Link a MIBiG locus to a protein only after alignment to translated GenBank CDS."""
    from .sequence_linkage import _alignment_metrics

    matches = []
    for protein in genbank_proteins:
        identity, coverage, mismatches, gaps, _mapping = _alignment_metrics(
            target_sequence.upper(), protein.sequence.upper()
        )
        if identity >= minimum_identity and coverage >= minimum_coverage:
            matches.append((identity, coverage, protein, mismatches, gaps))
    if not matches:
        return annotation, None
    identity, coverage, protein, mismatches, gaps = max(
        matches,
        key=lambda item: (item[0], item[1], item[2].accession),
    )
    linkage_evidence = EvidenceRecord(
        source="MIBiG locus + NCBI GenBank translated CDS",
        source_accession=nucleotide_accession,
        source_version="accession.version pinned in GenBank record",
        verification_date=verification_date,
        evidence_class=EvidenceClass.SEQUENCE_INFERRED,
        evidence_detail=(
            f"global amino-acid alignment identity={identity:.6f}, coverage={coverage:.6f}, "
            f"mismatches={mismatches}, gaps={gaps}"
        ),
    )
    linkage = MIBiGProteinLinkage(
        cluster_accession=annotation.cluster_id,
        nucleotide_accession=nucleotide_accession,
        protein_accession=protein.accession,
        locus_tag=protein.locus_tag,
        gene=protein.gene,
        product=protein.product,
        target_protein_accession=target_protein_accession,
        sequence_identity=identity,
        coverage=coverage,
        mismatches=mismatches,
        gaps=gaps,
        evidence=linkage_evidence,
    )
    metadata = dict(annotation.raw_metadata)
    metadata.setdefault("sequence_linkages", []).append(asdict(linkage))
    metadata["sequence_linked_gene_count"] = len(metadata["sequence_linkages"])
    updated = replace(
        annotation,
        genes=tuple(dict.fromkeys((*annotation.genes, protein.locus_tag or protein.accession))),
        raw_metadata=metadata,
    )
    return updated, linkage


def build_antismash_annotation(
    *,
    cluster_id: str,
    genes: Sequence[str],
    domains: Sequence[DomainAnnotation],
    modules: Sequence[ModuleAnnotation],
    products: Sequence[SubstrateAssociation] = (),
    tool_version: str,
    verification_date: str,
) -> NRPSClusterAnnotation:
    """Build an antiSMASH annotation with all predictions explicitly computed evidence."""
    evidence = EvidenceRecord(
        source="antiSMASH",
        source_accession=cluster_id,
        source_version=tool_version,
        verification_date=verification_date,
        evidence_class=EvidenceClass.DATABASE_COMPUTED,
        evidence_detail="antiSMASH gene-cluster/domain/module prediction",
    )
    for domain in domains:
        if domain.boundary_evidence.source != "antiSMASH":
            raise ValueError("antiSMASH domain boundaries must retain antiSMASH evidence provenance")
        if domain.boundary_evidence.evidence_class != EvidenceClass.DATABASE_COMPUTED:
            raise ValueError("antiSMASH predictions must not be relabeled as experimental")
    return NRPSClusterAnnotation(
        cluster_id=cluster_id,
        genes=tuple(genes),
        domains=tuple(domains),
        modules=tuple(modules),
        products=tuple(products),
        evidence=evidence,
    )


def build_ligand_context(
    *,
    structure_id: str,
    ligand_id: str,
    ligand: LigandRecord,
    chain_entity_id: str,
    canonical_chemical_ids: Mapping[str, str] | None = None,
    chain_id: str | None = None,
    domains: Sequence[tuple[str, int, int]] = (),
    experimental_method: str | None = None,
    resolution_angstrom: float | None = None,
    verification_date: str,
) -> LigandContext:
    """Attach experimentally observed ligand coordinates without asserting substrate role."""
    residue_refs = [
        ref for ref in ligand.binding_residue_refs
        if chain_id is None or ref["label_asym_id"] == chain_id
    ]
    binding_indices = [
        int(ref["canonical_residue_index"]) for ref in residue_refs
    ] if residue_refs else list(ligand.binding_residue_indices)
    binding_domains = sorted(
        domain_id
        for domain_id, start, end in domains
        if any(start <= residue_index < end for residue_index in binding_indices)
    )
    evidence = EvidenceRecord(
        source="RCSB PDB / wwPDB",
        source_accession=structure_id,
        source_version="entry_revision_metadata",
        verification_date=verification_date,
        evidence_class=EvidenceClass.EXPERIMENTAL,
        evidence_detail="ligand coordinates and protein contacts are observed in the deposited structure; biochemical role unassigned",
    )
    return LigandContext(
        structure_id=structure_id,
        ligand_id=ligand_id,
        chemical_component_id=ligand.chemical_component_id,
        canonical_chemical_ids=dict(canonical_chemical_ids or {}),
        coordinates=ligand.coordinates.copy(),
        binding_residue_indices=binding_indices,
        binding_residue_refs=residue_refs,
        binding_chain_ids=sorted({ref["label_asym_id"] for ref in residue_refs}),
        binding_domain_ids=binding_domains,
        minimum_contact_distance=ligand.minimum_contact_distance,
        experimental_context={
            "entity_id": chain_entity_id,
            "experimental_method": experimental_method,
            "resolution_angstrom": None if resolution_angstrom is None else str(resolution_angstrom),
        },
        evidence=evidence,
        role="UNASSIGNED",
    )


def _evidence_class_from_eco(evidences: Sequence[Mapping[str, Any]]) -> EvidenceClass:
    codes = {str(item.get("evidenceCode", "")) for item in evidences}
    if "ECO:0000269" in codes:
        return EvidenceClass.EXPERIMENTAL
    if "ECO:0000250" in codes:
        return EvidenceClass.SEQUENCE_INFERRED
    if any(code in codes for code in {"ECO:0000255", "ECO:0000256", "ECO:0007829"}):
        return EvidenceClass.DATABASE_COMPUTED
    return EvidenceClass.DATABASE_CURATED


def parse_uniprot_nrps_features(
    payload: Mapping[str, Any],
    *,
    accession: str,
    release: str,
    verification_date: str,
) -> NRPSClusterAnnotation:
    """Parse UniProt evidence-coded carrier/PPant features without inventing A-domain spans."""
    sequence = str(payload.get("sequence", {}).get("value", ""))
    if not sequence:
        raise ValueError("UniProt protein entry has no sequence")
    entry_accession = str(payload.get("primaryAccession", accession))
    genes = tuple(
        str(item.get("geneName", {}).get("value"))
        for item in payload.get("genes", [])
        if item.get("geneName", {}).get("value")
    )
    evidence_by_feature = {
        index: EvidenceRecord(
            source="UniProtKB",
            source_accession=entry_accession,
            source_version=release,
            verification_date=verification_date,
            evidence_class=_evidence_class_from_eco(feature.get("evidences", [])),
            evidence_detail="; ".join(
                str(item.get("source", item.get("evidenceCode", "")))
                for item in feature.get("evidences", [])
            ) or None,
        )
        for index, feature in enumerate(payload.get("features", []))
    }
    carrier_domains: list[dict[str, Any]] = []
    ppant_sites: list[tuple[int, EvidenceRecord]] = []
    for index, feature in enumerate(payload.get("features", [])):
        feature_type = str(feature.get("type", "")).lower()
        description = str(feature.get("description", ""))
        start = feature.get("location", {}).get("start", {}).get("value")
        end = feature.get("location", {}).get("end", {}).get("value")
        if start is None or end is None:
            continue
        start_aa = int(start) - 1
        end_aa = int(end)
        evidence = evidence_by_feature[index]
        if feature_type == "domain" and any(word in description.lower() for word in ("carrier", "pcp", "thiolation")):
            carrier_domains.append(
                {
                    "feature_id": feature.get("featureId", f"carrier-{start}-{end}"),
                    "start": start_aa,
                    "end": end_aa,
                    "evidence": evidence,
                }
            )
        if feature_type == "modified residue" and any(word in description.lower() for word in ("pantetheine", "phosphopantetheine")):
            ppant_sites.append((start_aa, evidence))

    domains: list[DomainAnnotation] = []
    for carrier in carrier_domains:
        site = next(
            (position for position, _evidence in ppant_sites if carrier["start"] <= position < carrier["end"]),
            None,
        )
        site_evidence = next(
            (evidence for position, evidence in ppant_sites if position == site),
            None,
        )
        domains.append(
            DomainAnnotation(
                domain_id=f"{entry_accession}:{carrier['feature_id']}",
                domain_type="T/PCP",
                protein_accession=entry_accession,
                start_aa=carrier["start"],
                end_aa=carrier["end"],
                boundary_evidence=carrier["evidence"],
                ppant_attachment_residue=site,
                ppant_evidence=site_evidence,
            )
        )
    cluster_evidence = EvidenceRecord(
        source="UniProtKB",
        source_accession=entry_accession,
        source_version=release,
        verification_date=verification_date,
        evidence_class=EvidenceClass.DATABASE_CURATED,
        evidence_detail="reviewed UniProtKB feature annotations; individual feature evidence is retained",
    )
    return NRPSClusterAnnotation(
        cluster_id=f"UniProt:{entry_accession}",
        genes=genes,
        domains=tuple(domains),
        modules=(),
        products=(),
        evidence=cluster_evidence,
        raw_metadata={
            "protein_name": payload.get("proteinDescription", {}).get("recommendedName", {}).get("fullName", {}).get("value"),
            "organism": payload.get("organism", {}).get("scientificName"),
            "taxonomy_id": payload.get("organism", {}).get("taxonId"),
            "protein_length": len(sequence),
            "unmapped_ppant_residues": [
                position for position, _evidence in ppant_sites
                if not any(domain.start_aa <= position < domain.end_aa for domain in domains)
            ],
        },
    )


def parse_interpro_domain_matches(
    payload: Mapping[str, Any],
    *,
    accession: str,
    release: str,
    verification_date: str,
) -> NRPSClusterAnnotation:
    """Normalize InterPro protein-domain coordinates as database-computed evidence."""
    metadata = payload.get("metadata", {})
    entry_accession = str(metadata.get("accession", accession))
    domains: list[DomainAnnotation] = []
    for result in payload.get("results", []):
        entry_metadata = result.get("metadata", {})
        entry_type = str(entry_metadata.get("type", ""))
        if entry_type not in {"domain", "family"}:
            continue
        proteins = result.get("proteins", [])
        protein = next(
            (item for item in proteins if str(item.get("accession", "")).upper() == entry_accession.upper()),
            None,
        )
        if protein is None:
            continue
        name = str(entry_metadata.get("name", ""))
        normalized_name = name.lower()
        if "amino acid adenylation" in normalized_name or entry_metadata.get("accession") == "IPR010071":
            domain_type = "A"
        elif "phosphopantetheine binding" in normalized_name or "acyl-carrier" in normalized_name:
            domain_type = "T/PCP"
        elif "condensation" in normalized_name:
            domain_type = "C"
        elif "thioesterase" in normalized_name:
            domain_type = "TE"
        elif "epimerization" in normalized_name:
            domain_type = "E"
        else:
            domain_type = "OTHER"
        location_groups = protein.get("entry_protein_locations", [])
        for location_index, location in enumerate(location_groups):
            for fragment_index, fragment in enumerate(location.get("fragments", [])):
                source_start = int(fragment["start"])
                source_end = int(fragment["end"])
                if source_start < 1 or source_end < source_start:
                    continue
                evidence = EvidenceRecord(
                    source="InterPro",
                    source_accession=str(entry_metadata.get("accession", "UNKNOWN")),
                    source_version=release,
                    verification_date=verification_date,
                    evidence_class=EvidenceClass.DATABASE_COMPUTED,
                    evidence_detail=(
                        f"{name}; model={location.get('model')}; score={location.get('score')}; "
                        f"match_status={fragment.get('dc-status')}"
                    ),
                )
                domains.append(
                    DomainAnnotation(
                        domain_id=(
                            f"InterPro:{entry_metadata.get('accession')}:{entry_accession}:"
                            f"{location_index}:{fragment_index}"
                        ),
                        domain_type=domain_type,
                        protein_accession=entry_accession,
                        start_aa=source_start - 1,
                        end_aa=source_end,
                        boundary_evidence=evidence,
                    )
                )
    cluster_evidence = EvidenceRecord(
        source="InterPro",
        source_accession=entry_accession,
        source_version=release,
        verification_date=verification_date,
        evidence_class=EvidenceClass.DATABASE_COMPUTED,
        evidence_detail="InterPro integrated signatures and domain model matches",
    )
    return NRPSClusterAnnotation(
        cluster_id=f"InterPro:{entry_accession}",
        genes=(str(metadata.get("gene")),) if metadata.get("gene") else (),
        domains=tuple(domains),
        modules=(),
        products=(),
        evidence=cluster_evidence,
        raw_metadata={
            "protein_length": metadata.get("length"),
            "organism": metadata.get("source_organism", {}).get("scientificName"),
            "taxonomy_id": metadata.get("source_organism", {}).get("taxId"),
            "interpro_release": release,
        },
    )


def combine_uniprot_interpro_nrps(
    interpro_annotation: NRPSClusterAnnotation,
    uniprot_payload: Mapping[str, Any],
    *,
    uniprot_release: str,
    interpro_release: str,
    verification_date: str,
) -> NRPSClusterAnnotation:
    """Join UniProt feature evidence and InterPro spans without flattening evidence classes."""
    accession = str(uniprot_payload.get("primaryAccession", "UNKNOWN"))
    uniprot_annotation = parse_uniprot_nrps_features(
        uniprot_payload,
        accession=accession,
        release=uniprot_release,
        verification_date=verification_date,
    )
    domains = list(interpro_annotation.domains)
    for uniprot_domain in uniprot_annotation.domains:
        matched_index = next(
            (
                index for index, domain in enumerate(domains)
                if domain.protein_accession == uniprot_domain.protein_accession
                and domain.domain_type == uniprot_domain.domain_type
                and max(domain.start_aa, uniprot_domain.start_aa) < min(domain.end_aa, uniprot_domain.end_aa)
            ),
            None,
        )
        if matched_index is None:
            domains.append(uniprot_domain)
            continue
        existing = domains[matched_index]
        extra_evidence = [*existing.additional_evidence, uniprot_domain.boundary_evidence]
        if uniprot_domain.ppant_evidence is not None:
            extra_evidence.append(uniprot_domain.ppant_evidence)
        domains[matched_index] = replace(
            existing,
            ppant_attachment_residue=uniprot_domain.ppant_attachment_residue,
            ppant_evidence=uniprot_domain.ppant_evidence,
            additional_evidence=tuple(extra_evidence),
        )

    module_comment = None
    substrate_associations: list[SubstrateAssociation] = []
    for comment in uniprot_payload.get("comments", []):
        comment_type = str(comment.get("commentType", ""))
        if comment_type == "DOMAIN":
            text = " ".join(item.get("value", "") for item in comment.get("texts", []))
            if "module-bearing" in text.lower() and "adenylation" in text.lower() and "thiolation" in text.lower():
                module_comment = comment
        if comment_type == "CATALYTIC ACTIVITY":
            reaction = comment.get("reaction", {})
            reaction_name = str(reaction.get("name", ""))
            reactants = reaction_name.split("=", 1)[0]
            substrate = next(
                (item.strip() for item in reactants.split("+") if "phenylalanine" in item.lower()),
                None,
            )
            if substrate:
                evidence = EvidenceRecord(
                    source="UniProtKB",
                    source_accession=accession,
                    source_version=uniprot_release,
                    verification_date=verification_date,
                    evidence_class=_evidence_class_from_eco(comment.get("evidences", [])),
                    evidence_detail="curated catalytic-activity reaction; UniProt evidence codes retained",
                )
                substrate_associations.append(
                    SubstrateAssociation(
                        substrate_name=substrate,
                        canonical_identifiers={},
                        evidence=evidence,
                        association_type="enzyme_reaction_substrate",
                    )
                )

    modules: list[ModuleAnnotation] = []
    if module_comment is not None:
        comment_text = " ".join(item.get("value", "") for item in module_comment.get("texts", []))
        module_evidence = EvidenceRecord(
            source="UniProtKB",
            source_accession=accession,
            source_version=uniprot_release,
            verification_date=verification_date,
            evidence_class=_evidence_class_from_eco(module_comment.get("texts", [{}])[0].get("evidences", [])),
            evidence_detail="UniProt domain-architecture comment: " + comment_text,
        )
        adenylation = next((domain for domain in domains if domain.domain_type == "A"), None)
        carrier = next((domain for domain in domains if domain.domain_type == "T/PCP"), None)
        if adenylation is not None and carrier is not None:
            modules.append(
                ModuleAnnotation(
                    module_id=f"{accession}:module-1",
                    order=1,
                    domain_ids=(adenylation.domain_id, carrier.domain_id),
                    adenylation_domain_id=adenylation.domain_id,
                    carrier_domain_id=carrier.domain_id,
                    condensation_domain_id=None,
                    substrate_association=substrate_associations[0] if substrate_associations else None,
                    evidence=module_evidence,
                    incomplete=False,
                )
            )

    cluster_evidence = EvidenceRecord(
        source="UniProtKB + InterPro",
        source_accession=accession,
        source_version=f"UniProt:{uniprot_release};InterPro:{interpro_release}",
        verification_date=verification_date,
        evidence_class=EvidenceClass.DATABASE_CURATED,
        evidence_detail="source-specific evidence remains attached to every domain/module/substrate annotation",
    )
    return NRPSClusterAnnotation(
        cluster_id=f"UniProt/InterPro:{accession}",
        genes=uniprot_annotation.genes,
        domains=tuple(domains),
        modules=tuple(modules),
        products=tuple(substrate_associations),
        evidence=cluster_evidence,
        raw_metadata={
            **dict(interpro_annotation.raw_metadata),
            **dict(uniprot_annotation.raw_metadata),
            "source_releases": {"UniProtKB": uniprot_release, "InterPro": interpro_release},
        },
    )
def _qualifier_values(qualifiers: Mapping[str, Any], names: Sequence[str]) -> list[str]:
    for name in names:
        value = qualifiers.get(name)
        if value is None:
            continue
        if isinstance(value, (list, tuple)):
            return [str(item) for item in value]
        return [str(value)]
    return []


def _antismash_domain_type(description: str) -> str:
    value = description.lower()
    if "adenyl" in value or "amp-binding" in value or value == "a":
        return "A"
    if "carrier" in value or "pcp" in value or "thiolation" in value:
        return "T/PCP"
    if "condensation" in value or value.startswith("c-"):
        return "C"
    if "thioesterase" in value or value == "te":
        return "TE"
    if "epimerization" in value or value == "e":
        return "E"
    if "heterocycl" in value or value in {"cy", "cyclization"}:
        return "Cy"
    if "methyltransferase" in value or value in {"mt", "nmt", "omt"}:
        return "MT"
    if "linker" in value:
        return "LINKER"
    return "OTHER"


def parse_antismash_genbank(
    path: str | Path,
    *,
    cluster_id: str,
    tool_version: str,
    verification_date: str,
) -> NRPSClusterAnnotation:
    """Import antiSMASH annotated GenBank features with computed evidence labels.

    CDS translations are used only for amino-acid coordinate conversion; domain
    calls and predicted substrate specificity remain antiSMASH-computed evidence.
    """
    cluster_evidence = EvidenceRecord(
        source="antiSMASH",
        source_accession=cluster_id,
        source_version=tool_version,
        verification_date=verification_date,
        evidence_class=EvidenceClass.DATABASE_COMPUTED,
        evidence_detail="antiSMASH annotated GenBank record",
    )
    genes: list[str] = []
    protein_lengths: dict[str, int] = {}
    domains: list[DomainAnnotation] = []
    modules: list[ModuleAnnotation] = []
    products: list[SubstrateAssociation] = []
    raw_region_metadata: list[dict[str, Any]] = []

    for record in SeqIO.parse(str(path), "genbank"):
        cds_by_locus: dict[str, tuple[Any, int]] = {}
        domain_features: list[tuple[Any, str, str, int, int, EvidenceRecord, str | None]] = []
        module_features: list[Any] = []
        for feature in record.features:
            qualifiers = feature.qualifiers
            feature_type = feature.type.lower()
            locus_values = _qualifier_values(qualifiers, ("locus_tag", "protein_id", "gene"))
            locus = locus_values[0] if locus_values else f"{record.id}:{int(feature.location.start)}"
            if feature_type == "cds":
                translation = _qualifier_values(qualifiers, ("translation",))
                if translation:
                    cds_by_locus[locus] = (feature.location, len(translation[0].replace(" ", "")))
                    protein_lengths[locus] = len(translation[0].replace(" ", ""))
                    genes.append(locus)
            if feature_type in {"asdomain", "asdomaIn".lower()}:
                description_values = _qualifier_values(
                    qualifiers,
                    ("aSDomain", "domain", "domain_type", "description", "note"),
                )
                description = description_values[0] if description_values else "OTHER"
                if locus not in cds_by_locus:
                    matching_cds = [
                        (cds_locus, item)
                        for cds_locus, item in cds_by_locus.items()
                        if int(item[0].start) <= int(feature.location.start)
                        and int(feature.location.end) <= int(item[0].end)
                    ]
                    if matching_cds:
                        locus, _cds = matching_cds[0]
                cds = cds_by_locus.get(locus)
                if cds is None:
                    continue
                cds_location, protein_length = cds
                if int(cds_location.strand or 1) >= 0:
                    nucleotide_start = int(feature.location.start) - int(cds_location.start)
                    nucleotide_end = int(feature.location.end) - int(cds_location.start)
                else:
                    nucleotide_start = int(cds_location.end) - int(feature.location.end)
                    nucleotide_end = int(cds_location.end) - int(feature.location.start)
                start_aa = max(0, nucleotide_start // 3)
                end_aa = min(protein_length, (nucleotide_end + 2) // 3)
                if start_aa >= end_aa:
                    continue
                domain_number = len(domain_features) + 1
                domain_id_values = _qualifier_values(qualifiers, ("domain_id", "aSDomain_id"))
                domain_id = domain_id_values[0] if domain_id_values else f"{locus}:domain-{domain_number}"
                specificity_values = _qualifier_values(
                    qualifiers,
                    ("specificity", "substrate_specificity", "consensus"),
                )
                specificity = "; ".join(specificity_values) if specificity_values else None
                domain_evidence = EvidenceRecord(
                    source="antiSMASH",
                    source_accession=cluster_id,
                    source_version=tool_version,
                    verification_date=verification_date,
                    evidence_class=EvidenceClass.DATABASE_COMPUTED,
                    evidence_detail="NRPS/PKS domain feature; source span retained from GenBank coordinates",
                )
                domain_features.append(
                    (feature, domain_id, locus, start_aa, end_aa, domain_evidence, specificity)
                )
                domains.append(
                    DomainAnnotation(
                        domain_id=domain_id,
                        domain_type=_antismash_domain_type(description),
                        protein_accession=locus,
                        start_aa=start_aa,
                        end_aa=end_aa,
                        boundary_evidence=domain_evidence,
                        selectivity_prediction=specificity,
                        selectivity_evidence=domain_evidence if specificity else None,
                    )
                )
            elif feature_type == "asmodule":
                module_features.append(feature)
            elif feature_type in {"region", "cand_cluster", "protocluster"}:
                products.extend(
                    SubstrateAssociation(
                        substrate_name=value,
                        canonical_identifiers={},
                        evidence=cluster_evidence,
                        confidence=None,
                        association_type="antiSMASH_predicted_region_product",
                    )
                    for value in _qualifier_values(qualifiers, ("product",))
                )
                raw_region_metadata.append(
                    {
                        "feature_type": feature.type,
                        "start": int(feature.location.start),
                        "end": int(feature.location.end),
                        "product": _qualifier_values(qualifiers, ("product",)),
                        "region_number": _qualifier_values(qualifiers, ("region_number",)),
                    }
                )

        for module_index, feature in enumerate(module_features, start=1):
            qualifiers = feature.qualifiers
            references = _qualifier_values(qualifiers, ("domains", "domain_ids"))
            selected_domains = [
                item for item in domain_features
                if (references and item[1] in references)
                or (not references and int(feature.location.start) <= int(item[0].location.start) and int(item[0].location.end) <= int(feature.location.end))
            ]
            domain_ids = tuple(item[1] for item in selected_domains)
            types_by_id = {domain.domain_id: domain.domain_type for domain in domains}
            adenylation = next((domain_id for domain_id in domain_ids if types_by_id.get(domain_id) == "A"), None)
            carrier = next((domain_id for domain_id in domain_ids if types_by_id.get(domain_id) == "T/PCP"), None)
            condensation = next((domain_id for domain_id in domain_ids if types_by_id.get(domain_id) == "C"), None)
            order_values = _qualifier_values(qualifiers, ("module_number", "module_id"))
            try:
                order = int(order_values[0]) if order_values else module_index
            except ValueError:
                order = module_index
            module_name_values = _qualifier_values(qualifiers, ("module_id", "module_number"))
            modules.append(
                ModuleAnnotation(
                    module_id=module_name_values[0] if module_name_values else f"{record.id}:module-{module_index}",
                    order=order,
                    domain_ids=domain_ids,
                    adenylation_domain_id=adenylation,
                    carrier_domain_id=carrier,
                    condensation_domain_id=condensation,
                    substrate_association=None,
                    evidence=cluster_evidence,
                    incomplete=not bool(adenylation and carrier),
                )
            )

    annotation = NRPSClusterAnnotation(
        cluster_id=cluster_id,
        genes=tuple(dict.fromkeys(genes)),
        domains=tuple(domains),
        modules=tuple(sorted(modules, key=lambda module: module.order)),
        products=tuple(products),
        evidence=cluster_evidence,
        raw_metadata={"region_features": raw_region_metadata, "tool": "antiSMASH"},
    )
    if protein_lengths:
        annotation.validate(protein_lengths)
    return annotation