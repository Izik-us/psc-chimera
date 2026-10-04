"""Rate-limited InterPro REST access for protein family/domain records."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from threading import Lock
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen
from typing import Any

from .annotations import NRPSClusterAnnotation, parse_interpro_domain_matches

INTERPRO_REST = "https://www.ebi.ac.uk/interpro/api"


class InterProClient:
    def __init__(
        self,
        *,
        requests_per_second: float = 1.0,
        timeout_seconds: float = 45.0,
        retries: int = 3,
    ):
        if requests_per_second <= 0 or retries < 0:
            raise ValueError("requests_per_second must be positive and retries non-negative")
        self.interval = 1.0 / requests_per_second
        self.next_request = 0.0
        self.lock = Lock()
        self.timeout_seconds = timeout_seconds
        self.retries = retries

    def _get_json(self, url: str) -> tuple[dict[str, Any], Any]:
        for attempt in range(self.retries + 1):
            with self.lock:
                now = time.monotonic()
                delay = max(0.0, self.next_request - now)
                self.next_request = max(now, self.next_request) + self.interval
            if delay:
                time.sleep(delay)
            try:
                request = Request(
                    url,
                    headers={"Accept": "application/json", "User-Agent": "CHIMERA-data-pipeline/0.2"},
                )
                with urlopen(request, timeout=self.timeout_seconds) as response:
                    return json.loads(response.read().decode("utf-8")), response
            except HTTPError as exc:
                if attempt >= self.retries or exc.code not in {429, 500, 502, 503, 504}:
                    raise
                retry_after = exc.headers.get("Retry-After")
                wait_seconds = float(retry_after) if retry_after and retry_after.isdigit() else min(2.0**attempt, 8.0)
                time.sleep(wait_seconds)
            except (URLError, TimeoutError):
                if attempt >= self.retries:
                    raise
                time.sleep(min(2.0**attempt, 8.0))
        raise RuntimeError("InterPro retry loop exhausted")

    def fetch_protein_matches(
        self,
        accession: str,
        *,
        database: str = "interpro",
    ) -> tuple[dict[str, Any], str]:
        if database not in {"interpro", "pfam"}:
            raise ValueError("database must be interpro or pfam")
        safe_accession = quote(accession, safe="")
        url = f"{INTERPRO_REST}/entry/{database}/protein/uniprot/{safe_accession}/"
        first_payload, response = self._get_json(url)
        payload = dict(first_payload)
        results = list(first_payload.get("results", []))
        next_url = first_payload.get("next")
        while next_url:
            page, _response = self._get_json(next_url)
            results.extend(page.get("results", []))
            next_url = page.get("next")
        payload["results"] = results
        release = response.headers.get("InterPro-Version", "UNKNOWN")
        payload.setdefault("metadata", {}).setdefault("release", release)
        return payload, release

    def fetch_nrps_annotation(
        self,
        accession: str,
        *,
        verification_date: str | None = None,
    ) -> NRPSClusterAnnotation:
        payload, release = self.fetch_protein_matches(accession)
        if verification_date is None:
            verification_date = datetime.now(timezone.utc).date().isoformat()
        return parse_interpro_domain_matches(
            payload,
            accession=accession,
            release=release,
            verification_date=verification_date,
        )