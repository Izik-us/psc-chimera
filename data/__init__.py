"""Data sourcing and preprocessing utilities for the PSC pipeline."""

from .dataset_engine import (
    AcquisitionManifest,
    DataQualityResult,
    DatasetSplit,
    DatasetVersion,
    ExternalSourceInventory,
    ProvenanceRecord,
    StructuralManifest,
    build_acquisition_manifest,
    build_dataset_version,
    stable_hash,
    summarize_records,
)
from .real_acquisition import acquire_rcsb_structure, run_external_seed_pipeline
from .rcsb import AcquisitionState, AcquisitionSummary, RCSBAcquirer, RCSBQuery
from .annotations import (
    DomainAnnotation,
    EvidenceClass,
    EvidenceRecord,
    LigandContext,
    ModuleAnnotation,
    NRPSClusterAnnotation,
    MIBiGProteinLinkage,
    SubstrateAssociation,
    parse_antismash_genbank,
    parse_interpro_domain_matches,
    parse_mibig_entry,
    parse_uniprot_nrps_features,
    combine_uniprot_interpro_nrps,
    link_mibig_cluster_protein,
)
from .augmentations import AugmentationConfig, augment_sample
from .dataset_api import (
    SQLiteChimeraDataset,
    build_dataset_sample,
    collate_chimera_samples,
    make_dataloader,
    write_immutable_dataset,
)
from .geometry import GeometryRecord, derive_geometry
from .interpro import InterProClient
from .leakage_splits import SplitConfig, SplitManifest, generate_leakage_safe_splits
from .msa import MSAConfig, MSARecord, build_msa_record
from .ncbi import NCBIClient, NCBIProteinSequence, parse_genbank_protein_sequences
from .sequence_linkage import SequenceCandidate, SequenceLinkage, link_structure_sequence
from .structures import CanonicalChain, CanonicalStructure, LigandRecord, StructuralQCConfig, parse_mmcif_structure
from .uniprot import RetrievedProtein, UniProtClient, align_sequences_to_query

__all__ = [
    "AcquisitionManifest",
    "DataQualityResult",
    "DatasetSplit",
    "DatasetVersion",
    "ExternalSourceInventory",
    "ProvenanceRecord",
    "StructuralManifest",
    "build_acquisition_manifest",
    "build_dataset_version",
    "AcquisitionSummary",
    "acquire_rcsb_structure",
    "run_external_seed_pipeline",
    "AcquisitionState",
    "RCSBAcquirer",
    "RCSBQuery",
    "DomainAnnotation",
    "EvidenceClass",
    "EvidenceRecord",
    "LigandContext",
    "ModuleAnnotation",
    "NRPSClusterAnnotation",
    "MIBiGProteinLinkage",
    "SubstrateAssociation",
    "AugmentationConfig",
    "augment_sample",
    "parse_antismash_genbank",
    "parse_interpro_domain_matches",
    "parse_mibig_entry",
    "parse_uniprot_nrps_features",
    "combine_uniprot_interpro_nrps",
    "link_mibig_cluster_protein",
    "SQLiteChimeraDataset",
    "build_dataset_sample",
    "collate_chimera_samples",
    "make_dataloader",
    "write_immutable_dataset",
    "GeometryRecord",
    "derive_geometry",
    "InterProClient",
    "SplitConfig",
    "SplitManifest",
    "generate_leakage_safe_splits",
    "MSAConfig",
    "MSARecord",
    "build_msa_record",
    "NCBIClient",
    "NCBIProteinSequence",
    "parse_genbank_protein_sequences",
    "SequenceCandidate",
    "SequenceLinkage",
    "link_structure_sequence",
    "CanonicalChain",
    "CanonicalStructure",
    "LigandRecord",
    "StructuralQCConfig",
    "parse_mmcif_structure",
    "RetrievedProtein",
    "UniProtClient",
    "align_sequences_to_query",
    "stable_hash",
    "summarize_records",
]
