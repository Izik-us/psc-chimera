"""Rate-limited UniProt/UniRef sequence linkage and controlled MSA retrieval."""

from __future__ import annotations

import json
import random
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from threading import Lock
from typing import Any, Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

from Bio.Align import PairwiseAligner

from .msa import MSAConfig, MSARecord, build_msa_record
from .sequence_linkage import SequenceCandidate, SequenceLinkage, link_structure_sequence
from .annotations import NRPSClusterAnnotation, parse_uniprot_nrps_features

UNIPROT_REST = "https://rest.uniprot.org"


class _RateLimiter:
    def __init__(self, requests_per_second: float):
        if requests_per_second <= 0:
            raise ValueError("requests_per_second must be positive")
        self.interval = 1.0 / requests_per_second
        self.next_request = 0.0
        self.lock = Lock()

    def wait(self) -> None:
        with self.lock:
            now = time.monotonic()
            wait_time = max(0.0, self.next_request - now)
            self.next_request = max(now, self.next_request) + self.interval
        if wait_time:
            time.sleep(wait_time)


@dataclass(frozen=True)
class RetrievedProtein:
    accession: str
    sequence: str
    organism_taxonomy_id: str | None
    organism_name: str | None
    release: str
    entry_version: int | None

    def to_candidate(self) -> SequenceCandidate:
        return SequenceCandidate(
            accession=self.accession,
            sequence=self.sequence,
            source="UniProtKB",
            source_version=self.release,
        )


def _format_date_or_version(response, payload: dict[str, Any]) -> str:
    release = response.headers.get("x-uniprot-release")
    if release:
        return release
    audit = payload.get("entryAudit", {})
    if audit.get("entryVersion") is not None:
        return f"entryVersion:{audit['entryVersion']}"
    return "UNKNOWN"


def align_sequences_to_query(
    query_sequence: str,
    proteins: Iterable[RetrievedProtein],
) -> tuple[str, list[dict[str, Any]]]:
    """Project authentic unaligned sequences onto query columns deterministically.

    Insertions relative to the structure query are omitted; deletions are gaps.
    The generation method is recorded in MSA metadata so this is not presented
    as a full de novo multiple alignment.
    """
    query = query_sequence.upper()
    if not query or any(character not in "ACDEFGHIKLMNPQRSTVWYBXZUO" for character in query):
        raise ValueError("query_sequence must be a non-empty protein sequence")
    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.match_score = 2.0
    aligner.mismatch_score = -1.0
    aligner.open_gap_score = -5.0
    aligner.extend_gap_score = -0.5
    aligned_rows = [query]
    source_records: list[dict[str, Any]] = []
    seen_sequences = {query}
    for protein in proteins:
        sequence = protein.sequence.upper()
        if not sequence or sequence in seen_sequences:
            continue
        alignment = aligner.align(query, sequence)[0]
        projected = ["-"] * len(query)
        coordinates = alignment.coordinates
        for step in range(coordinates.shape[1] - 1):
            query_start, query_end = map(int, coordinates[0, step:step + 2])
            sequence_start, sequence_end = map(int, coordinates[1, step:step + 2])
            query_width = query_end - query_start
            sequence_width = sequence_end - sequence_start
            if query_width and sequence_width:
                for offset in range(min(query_width, sequence_width)):
                    projected[query_start + offset] = sequence[sequence_start + offset]
        aligned_rows.append("".join(projected))
        seen_sequences.add(sequence)
        source_records.append(
            {
                "accession": protein.accession,
                "sequence": protein.sequence,
                "taxonomy_id": protein.organism_taxonomy_id,
                "organism": protein.organism_name,
            }
        )
    text = "".join(
        f">{record['accession']}\n{row}\n"
        for record, row in zip(source_records, aligned_rows[1:])
    )
    query_id = "target-query"
    return f">{query_id}\n{query}\n{text}", source_records


class UniProtClient:
    """Small, serial-by-default client with retry and cache support."""

    def __init__(
        self,
        *,
        requests_per_second: float = 1.0,
        timeout_seconds: float = 45.0,
        retries: int = 3,
        email: str | None = None,
    ):
        if retries < 0:
            raise ValueError("retries must be non-negative")
        self.rate_limiter = _RateLimiter(requests_per_second)
        self.timeout_seconds = timeout_seconds
        self.retries = retries
        self.email = email

    def _get(self, url: str) -> tuple[bytes, Any]:
        for attempt in range(self.retries + 1):
            try:
                self.rate_limiter.wait()
                headers = {"Accept": "application/json", "User-Agent": "CHIMERA-data-pipeline/0.2"}
                if self.email:
                    headers["From"] = self.email
                with urlopen(Request(url, headers=headers), timeout=self.timeout_seconds) as response:
                    return response.read(), response
            except HTTPError as exc:
                if attempt >= self.retries or exc.code not in {429, 500, 502, 503, 504}:
                    raise
                retry_after = exc.headers.get("Retry-After")
                delay = float(retry_after) if retry_after and retry_after.isdigit() else min(2.0 ** attempt, 8.0)
                time.sleep(delay)
            except (URLError, TimeoutError):
                if attempt >= self.retries:
                    raise
                time.sleep(min(2.0 ** attempt, 8.0))
        raise RuntimeError("UniProt retry loop exhausted")

    def fetch_uniprot_entry(self, accession: str) -> tuple[dict[str, Any], Any]:
        safe_accession = quote(accession, safe="")
        content, response = self._get(f"{UNIPROT_REST}/uniprotkb/{safe_accession}.json")
        return json.loads(content.decode("utf-8")), response

    def fetch_nrps_annotation(
        self,
        accession: str,
        *,
        verification_date: str | None = None,
    ) -> NRPSClusterAnnotation:
        payload, response = self.fetch_uniprot_entry(accession)
        if verification_date is None:
            verification_date = datetime.now(timezone.utc).date().isoformat()
        return parse_uniprot_nrps_features(
            payload,
            accession=accession,
            release=_format_date_or_version(response, payload),
            verification_date=verification_date,
        )

    def _protein_from_entry(self, accession: str, payload: dict[str, Any], response) -> RetrievedProtein | None:
        sequence = payload.get("sequence", {}).get("value")
        primary_accession = payload.get("primaryAccession", accession)
        if not sequence:
            return None
        organism = payload.get("organism", {})
        taxonomy_id = organism.get("taxonId")
        audit = payload.get("entryAudit", {})
        return RetrievedProtein(
            accession=str(primary_accession),
            sequence=str(sequence).upper(),
            organism_taxonomy_id=None if taxonomy_id is None else str(taxonomy_id),
            organism_name=organism.get("scientificName"),
            release=_format_date_or_version(response, payload),
            entry_version=audit.get("entryVersion"),
        )

    def resolve_accession(self, accession: str) -> tuple[list[str], dict[str, Any]]:
        payload, _response = self.fetch_uniprot_entry(accession)
        inactive = payload.get("inactiveReason", {})
        replacements = inactive.get("mergeDemergeTo", []) or []
        if replacements:
            return [str(item) for item in replacements], payload
        primary = payload.get("primaryAccession")
        return ([str(primary)] if primary else [accession]), payload

    def fetch_proteins(self, accessions: Iterable[str]) -> list[RetrievedProtein]:
        proteins: list[RetrievedProtein] = []
        seen: set[str] = set()
        for accession in accessions:
            if accession in seen:
                continue
            seen.add(accession)
            try:
                payload, response = self.fetch_uniprot_entry(accession)
            except HTTPError as exc:
                if exc.code == 404:
                    continue
                raise
            protein = self._protein_from_entry(accession, payload, response)
            if protein is not None:
                proteins.append(protein)
        return proteins

    def fetch_uniref_cluster(self, cluster_id: str) -> tuple[dict[str, Any], str]:
        if not re.fullmatch(r"UniRef(?:50|90|100)_[A-Z0-9]+", cluster_id):
            raise ValueError("invalid UniRef cluster identifier")
        content, response = self._get(f"{UNIPROT_REST}/uniref/{quote(cluster_id, safe='_')}.json")
        payload = json.loads(content.decode("utf-8"))
        updated = payload.get("updated") or _format_date_or_version(response, payload)
        return payload, str(updated)

    def find_uniref_cluster(self, accession: str, identity: int = 90) -> str | None:
        if identity not in {50, 90, 100}:
            raise ValueError("UniRef identity must be 50, 90, or 100")
        candidates, _entry = self.resolve_accession(accession)
        for candidate in candidates:
            cluster_id = f"UniRef{identity}_{candidate}"
            try:
                self.fetch_uniref_cluster(cluster_id)
                return cluster_id
            except HTTPError as exc:
                if exc.code != 404:
                    raise
        return None

    def retrieve_msa(
        self,
        *,
        target_sequence: str,
        source_accessions: Iterable[str],
        target_sequence_id: str,
        structure_id: str = "",
        chain_id: str = "",
        entity_id: str = "",
        msa_family_id: str | None = None,
        max_members: int = 128,
        config: MSAConfig | None = None,
    ) -> tuple[MSARecord, list[SequenceLinkage]]:
        """Retrieve authentic UniRef90 homologs, resolve IDs, and align to a structural target."""
        if max_members < 1:
            raise ValueError("max_members must be positive")
        reported_accessions = list(dict.fromkeys(str(item) for item in source_accessions))
        current_accessions: list[str] = []
        for accession in reported_accessions:
            resolved, _entry = self.resolve_accession(accession)
            current_accessions.extend(resolved)
        current_accessions = list(dict.fromkeys(current_accessions))

        family_payload: dict[str, Any] | None = None
        family_id: str | None = None
        family_release = "UNKNOWN"
        for accession in current_accessions:
            proposed = f"UniRef90_{accession}"
            try:
                family_payload, family_release = self.fetch_uniref_cluster(proposed)
                family_id = proposed
                break
            except HTTPError as exc:
                if exc.code != 404:
                    raise

        if family_payload is None or family_id is None:
            raw_alignment, source_records = align_sequences_to_query(target_sequence, [])
            msa = build_msa_record(
                raw_alignment,
                msa_family_id=msa_family_id or f"single:{target_sequence_id}",
                target_sequence_id=target_sequence_id,
                source="RCSB PDB polymer entity sequence",
                source_version="entry_revision_metadata",
                target_sequence=target_sequence,
                raw_sequence_records=source_records,
                generation_method="single-sequence-fallback",
                config=config,
            )
            return msa, []

        member_accessions: list[str] = []
        for member in family_payload.get("members", []):
            if member.get("memberIdType") != "UniProtKB ID":
                continue
            member_accessions.extend(str(value) for value in member.get("accessions", []) or [])
            if len(set(member_accessions)) >= max_members:
                break
        member_accessions = list(dict.fromkeys(member_accessions))[:max_members]
        proteins = self.fetch_proteins(member_accessions)
        candidates = [protein.to_candidate() for protein in proteins]
        linkages = [
            link_structure_sequence(
                structure_id=structure_id,
                chain_id=chain_id,
                entity_id=entity_id,
                sequence=target_sequence,
                candidates=candidates,
                reported_accession=reported_accessions[0] if reported_accessions else None,
            )
        ] if candidates else []
        raw_alignment, source_records = align_sequences_to_query(target_sequence, proteins)
        taxonomy_by_row = {
            index + 1: protein.organism_taxonomy_id
            for index, protein in enumerate(proteins)
            if protein.organism_taxonomy_id is not None
        }
        msa = build_msa_record(
            raw_alignment,
            msa_family_id=msa_family_id or family_id,
            target_sequence_id=target_sequence_id,
            source="UniRef90 / UniProtKB",
            source_version=family_release,
            target_sequence=target_sequence,
            taxonomy_by_row=taxonomy_by_row,
            raw_sequence_records=source_records,
            generation_method="uniref90-target-guided-global-projection",
            generation_version="uniref90-projection-v0.1",
            config=config,
        )
        return msa, linkages