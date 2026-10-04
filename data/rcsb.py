"""Resumable, provenance-aware acquisition from the authoritative RCSB archive."""

from __future__ import annotations

import hashlib
import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Iterable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from Bio.PDB.MMCIF2Dict import MMCIF2Dict

from .structures import parse_mmcif_structure

RCSB_SEARCH_ENDPOINT = "https://search.rcsb.org/rcsbsearch/v2/query"
RCSB_MMCIF_URL = "https://files.rcsb.org/download/{structure_id}.cif"


class AcquisitionState(str, Enum):
    REQUESTED = "REQUESTED"
    DISCOVERED = "DISCOVERED"
    DOWNLOADING = "DOWNLOADING"
    DOWNLOADED = "DOWNLOADED"
    CHECKSUM_COMPUTED = "CHECKSUM_COMPUTED"
    CHECKSUM_VERIFIED = "CHECKSUM_VERIFIED"
    PARSED = "PARSED"
    QC_PASSED = "QC_PASSED"
    QC_FAILED = "QC_FAILED"
    ACCEPTED = "ACCEPTED"
    REJECTED = "REJECTED"
    QUARANTINED = "QUARANTINED"


@dataclass(frozen=True)
class RCSBQuery:
    release_date_from: str | None = None
    release_date_to: str | None = None
    experimental_methods: tuple[str, ...] = ()
    maximum_resolution_angstrom: float | None = None
    minimum_sequence_length: int | None = None
    maximum_sequence_length: int | None = None
    taxonomy_ids: tuple[int, ...] = ()
    text_terms: tuple[str, ...] = ()
    nrps_targeted: bool = False
    page_size: int = 100

    def __post_init__(self) -> None:
        for value in (self.release_date_from, self.release_date_to):
            if value is not None and not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
                raise ValueError("release-date filters must use YYYY-MM-DD")
        if self.maximum_resolution_angstrom is not None and self.maximum_resolution_angstrom <= 0:
            raise ValueError("maximum_resolution_angstrom must be positive")
        if self.minimum_sequence_length is not None and self.minimum_sequence_length < 1:
            raise ValueError("minimum_sequence_length must be positive")
        if self.maximum_sequence_length is not None and self.maximum_sequence_length < 1:
            raise ValueError("maximum_sequence_length must be positive")
        if (
            self.minimum_sequence_length is not None
            and self.maximum_sequence_length is not None
            and self.minimum_sequence_length > self.maximum_sequence_length
        ):
            raise ValueError("minimum_sequence_length exceeds maximum_sequence_length")
        if not 1 <= self.page_size <= 1000:
            raise ValueError("page_size must be between 1 and 1000")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AcquisitionSummary:
    requested: int = 0
    discovered: int = 0
    downloading: int = 0
    downloaded: int = 0
    sha256_computed: int = 0
    checksum_verified: int = 0
    parsed: int = 0
    qc_passed: int = 0
    qc_failed: int = 0
    accepted: int = 0
    rejected: int = 0
    quarantined: int = 0
    canonical_chain_count: int = 0
    failure_reasons: list[str] = field(default_factory=list)
    records: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class _RateLimiter:
    def __init__(self, requests_per_second: float):
        if requests_per_second <= 0:
            raise ValueError("requests_per_second must be positive")
        self.interval = 1.0 / requests_per_second
        self.next_request = 0.0
        self.lock = threading.Lock()

    def wait(self) -> None:
        with self.lock:
            now = time.monotonic()
            delay = max(0.0, self.next_request - now)
            self.next_request = max(now, self.next_request) + self.interval
        if delay:
            time.sleep(delay)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _terminal(attribute: str, operator: str, value: Any) -> dict[str, Any]:
    return {
        "type": "terminal",
        "service": "text" if isinstance(value, str) else "text",
        "parameters": {"attribute": attribute, "operator": operator, "value": value},
    }


def build_rcsb_query_payload(query: RCSBQuery, *, start: int = 0) -> dict[str, Any]:
    """Build a documented RCSB Search API query from explicit entry criteria."""
    nodes: list[dict[str, Any]] = []

    def add_or_group(terms: list[dict[str, Any]]) -> None:
        if terms:
            nodes.append({"type": "group", "logical_operator": "or", "nodes": terms})

    if query.release_date_from:
        nodes.append(_terminal("rcsb_accession_info.initial_release_date", "greater_or_equal", query.release_date_from))
    if query.release_date_to:
        nodes.append(_terminal("rcsb_accession_info.initial_release_date", "less_or_equal", query.release_date_to))
    if query.maximum_resolution_angstrom is not None:
        nodes.append(_terminal("rcsb_entry_info.resolution_combined", "less_or_equal", query.maximum_resolution_angstrom))
    add_or_group([
        _terminal("exptl.method", "exact_match", method)
        for method in query.experimental_methods
    ])
    if query.minimum_sequence_length is not None:
        nodes.append(_terminal("entity_poly.rcsb_sample_sequence_length", "greater_or_equal", query.minimum_sequence_length))
    if query.maximum_sequence_length is not None:
        nodes.append(_terminal("entity_poly.rcsb_sample_sequence_length", "less_or_equal", query.maximum_sequence_length))
    add_or_group([
        _terminal("rcsb_entity_source_organism.taxonomy_lineage.id", "exact_match", str(taxonomy_id))
        for taxonomy_id in query.taxonomy_ids
    ])
    terms = list(query.text_terms)
    if query.nrps_targeted:
        terms.extend(("nonribosomal peptide synthetase", "NRPS", "adenylation domain"))
    add_or_group([
        _terminal("struct.title", "contains_words", term)
        for term in dict.fromkeys(terms)
    ])
    group: dict[str, Any] = {
        "type": "group",
        "logical_operator": "and",
        "nodes": nodes,
    }
    return {
        "query": group,
        "return_type": "entry",
        "request_options": {
            "paginate": {"start": start, "rows": query.page_size},
            "results_content_type": ["experimental"],
        },
    }


class RCSBAcquirer:
    def __init__(
        self,
        *,
        requests_per_second: float = 2.0,
        maximum_workers: int = 4,
        retries: int = 3,
        request_timeout_seconds: float = 60.0,
    ):
        if maximum_workers < 1 or retries < 0:
            raise ValueError("maximum_workers must be positive and retries non-negative")
        self.rate_limiter = _RateLimiter(requests_per_second)
        self.maximum_workers = maximum_workers
        self.retries = retries
        self.request_timeout_seconds = request_timeout_seconds

    def discover(self, query: RCSBQuery) -> list[str]:
        identifiers: list[str] = []
        start = 0
        while True:
            payload = build_rcsb_query_payload(query, start=start)
            request = Request(
                RCSB_SEARCH_ENDPOINT,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json", "Accept": "application/json"},
                method="POST",
            )
            self.rate_limiter.wait()
            with urlopen(request, timeout=self.request_timeout_seconds) as response:
                data = json.loads(response.read().decode("utf-8"))
            result_set = data.get("result_set", [])
            identifiers.extend(
                str(item["identifier"]).split("_")[0].upper()
                for item in result_set
                if item.get("identifier")
            )
            total = int(data.get("total_count", start + len(result_set)))
            start += len(result_set)
            if not result_set or start >= total:
                break
        return sorted(set(identifiers))

    def _download(self, structure_id: str, destination: Path) -> tuple[bool, str | None]:
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".part")
        for attempt in range(self.retries + 1):
            try:
                self.rate_limiter.wait()
                request = Request(
                    RCSB_MMCIF_URL.format(structure_id=structure_id.lower()),
                    headers={"User-Agent": "CHIMERA-data-pipeline/0.2", "Accept": "text/plain"},
                )
                with urlopen(request, timeout=self.request_timeout_seconds) as response:
                    payload = response.read()
                if not payload:
                    raise ValueError("RCSB returned an empty mmCIF file")
                temporary.write_bytes(payload)
                temporary.replace(destination)
                return True, None
            except (HTTPError, URLError, TimeoutError, OSError, ValueError) as exc:
                temporary.unlink(missing_ok=True)
                if attempt >= self.retries:
                    return False, f"DOWNLOAD_ERROR:{type(exc).__name__}:{exc}"
                time.sleep(min(2.0 ** attempt, 8.0))
        return False, "DOWNLOAD_ERROR:retry_limit"

    @staticmethod
    def _record_state(record: dict[str, Any], state: AcquisitionState) -> None:
        record.setdefault("state_history", []).append({"state": state.value, "timestamp": _now()})

    def acquire_id(
        self,
        structure_id: str,
        output_dir: str | Path,
        *,
        expected_sha256: str | None = None,
        refresh: bool = False,
        assembly_id: str | None = None,
    ) -> dict[str, Any]:
        structure_id = structure_id.upper().strip()
        if not re.fullmatch(r"[A-Z0-9]{4,12}", structure_id):
            raise ValueError(f"invalid RCSB structure identifier: {structure_id!r}")
        if assembly_id is not None:
            assembly_id = str(assembly_id)
            if not assembly_id.isdigit() or int(assembly_id) < 1:
                raise ValueError("assembly_id must be a positive integer")
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        assembly_suffix = f"-assembly{assembly_id}" if assembly_id is not None else ""
        raw_path = output / f"{structure_id}{assembly_suffix}.cif"
        download_url = (
            f"https://files.rcsb.org/download/{structure_id.lower()}-assembly{assembly_id}.cif"
            if assembly_id is not None
            else RCSB_MMCIF_URL.format(structure_id=structure_id.lower())
        )
        record: dict[str, Any] = {
            "structure_id": structure_id,
            "requested_timestamp": _now(),
            "source": "RCSB PDB / wwPDB",
            "download_url": download_url,
            "assembly_id": assembly_id or "asymmetric_unit",
            "raw_path": str(raw_path),
            "expected_sha256": expected_sha256,
            "checksum_verified": False,
            "checksum_verification_source": None,
            "parsed": False,
            "canonical_chains": [],
            "failure_reason": None,
        }
        self._record_state(record, AcquisitionState.REQUESTED)
        self._record_state(record, AcquisitionState.DISCOVERED)

        cached = raw_path.is_file() and raw_path.stat().st_size > 0 and not refresh
        if not cached:
            self._record_state(record, AcquisitionState.DOWNLOADING)
            success, failure = self._download(structure_id, raw_path)
            if not success:
                record.update(status="rejected", failure_reason=failure, download_status="failed")
                self._record_state(record, AcquisitionState.REJECTED)
                return record
        record["cached"] = cached
        record["download_status"] = "downloaded"
        record["download_timestamp"] = _now()
        self._record_state(record, AcquisitionState.DOWNLOADED)

        digest = hashlib.sha256()
        with raw_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        actual_sha256 = digest.hexdigest()
        checksum_verified = expected_sha256 is not None and actual_sha256.lower() == expected_sha256.lower()
        checksum_status = "VERIFIED" if checksum_verified else (
            "MISMATCH" if expected_sha256 is not None else "UNVERIFIED"
        )
        record.update(
            sha256=actual_sha256,
            sha256_computed=True,
            expected_sha256=expected_sha256,
            checksum_verified=checksum_verified,
            checksum_status=checksum_status,
            checksum_verification_source="caller_supplied_manifest" if expected_sha256 else None,
        )
        self._record_state(record, AcquisitionState.CHECKSUM_COMPUTED)
        if checksum_verified:
            self._record_state(record, AcquisitionState.CHECKSUM_VERIFIED)
        elif expected_sha256 is not None:
            record.update(status="quarantined", failure_reason="CHECKSUM_FAILURE")
            self._record_state(record, AcquisitionState.QUARANTINED)
            return record

        try:
            structure = parse_mmcif_structure(
                raw_path,
                structure_id=structure_id,
                assembly_id=assembly_id or "asymmetric_unit",
            )
        except Exception as exc:
            record.update(
                status="rejected",
                failure_reason="INVALID_MMCIF",
                parse_error=str(exc),
            )
            self._record_state(record, AcquisitionState.REJECTED)
            return record

        self._record_state(record, AcquisitionState.PARSED)
        chain_records = []
        for chain in structure.chains:
            chain_records.append(
                {
                    "label_asym_id": chain.label_asym_id,
                    "auth_asym_id": chain.auth_asym_id,
                    "entity_id": chain.entity_id,
                    "sequence": chain.sequence,
                    "sequence_sha256": chain.sequence_sha256,
                    "residue_count": len(chain.sequence),
                    "observed_residue_count": int(chain.residue_mask.sum()),
                    "quality": chain.quality.to_dict(),
                    "canonical": chain.to_dict(include_arrays=True),
                }
            )
        qc_passed = sum(chain.quality.status == "accepted" for chain in structure.chains)
        qc_failed = len(structure.chains) - qc_passed
        self._record_state(record, AcquisitionState.QC_PASSED if qc_failed == 0 else AcquisitionState.QC_FAILED)
        accepted = qc_passed > 0
        status = "accepted" if accepted and qc_failed == 0 else "quarantined"
        if accepted:
            self._record_state(record, AcquisitionState.ACCEPTED)
        else:
            status = "quarantined"
            self._record_state(record, AcquisitionState.QUARANTINED)
        record.update(
            status=status,
            parsed=True,
            canonical_chains=chain_records,
            canonical_chain_count=len(chain_records),
            chain_qc_passed=qc_passed,
            chain_qc_failed=qc_failed,
            experimental_method=structure.experimental_method,
            resolution_angstrom=structure.resolution_angstrom,
            release_date=structure.release_date,
            assembly_ids=structure.assembly_ids,
            ligands=[ligand.to_dict() for ligand in structure.ligands],
            source_accessions=structure.source_accessions,
            manifest_version="CHIMERA-DATASET-v0.1",
        )
        return record

    def acquire_many(
        self,
        structure_ids: Iterable[str],
        output_dir: str | Path,
        *,
        expected_checksums: dict[str, str] | None = None,
        refresh: bool = False,
        assembly_id: str | None = None,
    ) -> dict[str, Any]:
        identifiers = sorted({str(value).upper().strip() for value in structure_ids})
        summary = AcquisitionSummary(requested=len(identifiers), discovered=len(identifiers))
        expected_checksums = {key.upper(): value for key, value in (expected_checksums or {}).items()}
        with ThreadPoolExecutor(max_workers=self.maximum_workers) as executor:
            results = list(
                executor.map(
                    lambda structure_id: self.acquire_id(
                        structure_id,
                        output_dir,
                        expected_sha256=expected_checksums.get(structure_id),
                        refresh=refresh,
                        assembly_id=assembly_id,
                    ),
                    identifiers,
                )
            )
        summary.records = results
        for result in results:
            summary.downloading += any(item["state"] == AcquisitionState.DOWNLOADING.value for item in result["state_history"])
            summary.downloaded += result.get("download_status") == "downloaded"
            summary.sha256_computed += bool(result.get("sha256_computed"))
            summary.checksum_verified += bool(result.get("checksum_verified"))
            summary.parsed += bool(result.get("parsed"))
            summary.canonical_chain_count += int(result.get("canonical_chain_count", 0))
            summary.qc_passed += int(result.get("chain_qc_passed", 0))
            summary.qc_failed += int(result.get("chain_qc_failed", 0))
            status = result.get("status")
            if status == "accepted":
                summary.accepted += 1
            elif status == "rejected":
                summary.rejected += 1
                summary.failure_reasons.append(str(result.get("failure_reason")))
            elif status == "quarantined":
                summary.quarantined += 1
                summary.failure_reasons.append(str(result.get("failure_reason", "STRUCTURAL_QC_FAILURE")))

        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        manifest_path = output / "acquisition_manifest.jsonl"
        with manifest_path.open("w", encoding="utf-8", newline="\n") as handle:
            for result in results:
                handle.write(json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n")
        return summary.to_dict()

    def acquire_query(self, query: RCSBQuery, output_dir: str | Path) -> dict[str, Any]:
        identifiers = self.discover(query)
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        query_manifest = {
            "source": "RCSB PDB Search API",
            "endpoint": RCSB_SEARCH_ENDPOINT,
            "query": query.to_dict(),
            "verification_date": datetime.now(timezone.utc).date().isoformat(),
            "structure_ids": identifiers,
        }
        (output / "discovery_manifest.json").write_text(
            json.dumps(query_manifest, indent=2, sort_keys=True), encoding="utf-8"
        )
        return self.acquire_many(identifiers, output)