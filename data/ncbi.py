"""Rate-limited NCBI E-utilities access for accessioned cluster loci."""

from __future__ import annotations

import io
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from threading import Lock
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from Bio import SeqIO

NCBI_EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


@dataclass(frozen=True)
class NCBIProteinSequence:
    accession: str
    locus_tag: str | None
    gene: str | None
    product: str | None
    sequence: str
    source_accession: str
    nucleotide_start: int
    nucleotide_end: int
    strand: int | None


class NCBIClient:
    def __init__(
        self,
        *,
        requests_per_second: float = 2.0,
        timeout_seconds: float = 60.0,
        retries: int = 3,
        email: str | None = None,
        tool: str = "psc-chimera",
    ):
        if requests_per_second <= 0 or retries < 0:
            raise ValueError("requests_per_second must be positive and retries non-negative")
        self.interval = 1.0 / requests_per_second
        self.next_request = 0.0
        self.lock = Lock()
        self.timeout_seconds = timeout_seconds
        self.retries = retries
        self.email = email
        self.tool = tool

    def fetch_genbank_region(
        self,
        accession: str,
        start: int,
        end: int,
    ) -> tuple[str, dict[str, Any]]:
        if start < 1 or end < start:
            raise ValueError("GenBank region must use valid one-based inclusive coordinates")
        parameters = {
            "db": "nuccore",
            "id": accession,
            "seq_start": str(start),
            "seq_stop": str(end),
            "rettype": "gb",
            "retmode": "text",
            "tool": self.tool,
        }
        if self.email:
            parameters["email"] = self.email
        url = f"{NCBI_EUTILS}/efetch.fcgi?{urlencode(parameters)}"
        for attempt in range(self.retries + 1):
            with self.lock:
                now = time.monotonic()
                delay = max(0.0, self.next_request - now)
                self.next_request = max(now, self.next_request) + self.interval
            if delay:
                time.sleep(delay)
            try:
                request = Request(url, headers={"User-Agent": self.tool, "Accept": "text/plain"})
                with urlopen(request, timeout=self.timeout_seconds) as response:
                    text = response.read().decode("utf-8")
                    metadata = {
                        "source": "NCBI E-utilities",
                        "source_accession": accession,
                        "requested_start": start,
                        "requested_end": end,
                        "retrieved_at": datetime.now(timezone.utc).isoformat(),
                        "response_content_type": response.headers.get("Content-Type"),
                    }
                    return text, metadata
            except HTTPError as exc:
                if attempt >= self.retries or exc.code not in {429, 500, 502, 503, 504}:
                    raise
                retry_after = exc.headers.get("Retry-After")
                time.sleep(float(retry_after) if retry_after and retry_after.isdigit() else min(2.0**attempt, 8.0))
            except (URLError, TimeoutError):
                if attempt >= self.retries:
                    raise
                time.sleep(min(2.0**attempt, 8.0))
        raise RuntimeError("NCBI retry loop exhausted")


def parse_genbank_protein_sequences(
    genbank_text: str,
    *,
    source_accession: str,
) -> list[NCBIProteinSequence]:
    record = SeqIO.read(io.StringIO(genbank_text), "genbank")
    proteins: list[NCBIProteinSequence] = []
    for feature in record.features:
        if feature.type != "CDS":
            continue
        qualifiers = feature.qualifiers
        locus_tag = qualifiers.get("locus_tag", [None])[0]
        protein_accession = qualifiers.get("protein_id", [None])[0]
        translation = "".join(qualifiers.get("translation", [])).replace(" ", "")
        if not translation:
            try:
                translation = str(feature.extract(record.seq).translate(to_stop=True))
            except Exception:
                continue
        if not translation:
            continue
        proteins.append(
            NCBIProteinSequence(
                accession=str(protein_accession or locus_tag or "UNKNOWN"),
                locus_tag=locus_tag,
                gene=qualifiers.get("gene", [None])[0],
                product=qualifiers.get("product", [None])[0],
                sequence=translation.upper(),
                source_accession=source_accession,
                nucleotide_start=int(feature.location.start),
                nucleotide_end=int(feature.location.end),
                strand=feature.location.strand,
            )
        )
    return proteins