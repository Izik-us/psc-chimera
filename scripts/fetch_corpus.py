#!/usr/bin/env python3
"""CHIMERA corpus fetcher and assembly-line driver.

The repository already contains the individual stations of a dataset assembly line
(RCSB acquisition, UniProt/InterPro clients, MSA normalisation, leakage-safe splits,
immutable SQLite/NPZ storage) but nothing that runs them end to end at corpus scale.
This script is that conveyor. It reuses the repo's own clients and builders and adds
what was missing: bulk downloads with resume and checksums, a real homology-search
stage (the UniRef90-cluster MSA path yields Neff ~ 1), external sequence-cluster
labels for splitting, and sharded dataset writing.

    python scripts/fetch_corpus.py plan                    # bill of materials, no network
    python scripts/fetch_corpus.py plan --live             # + live RCSB entry counts
    python scripts/fetch_corpus.py verify                  # check every URL / API is alive
    python scripts/fetch_corpus.py fetch --root corpus --stages light --yes
    python scripts/fetch_corpus.py fetch --root corpus --stages structures --profiles nrps-pfam
    python scripts/fetch_corpus.py msa --root corpus --uniref-db /scratch/uniref90_db
    python scripts/fetch_corpus.py assemble --root corpus --dataset-version CHIMERA-DATASET-v0.2
    python scripts/fetch_corpus.py status --root corpus

Network endpoints are the ones recorded in data/external_sources.json (verified by the
repo on 2026-10-04) plus a few well-known bulk files marked ``unverified``; run
``verify`` once on your machine before a long job. Sizes in ``plan`` are estimates.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import http.client
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

USER_AGENT = "CHIMERA-corpus/0.1"
GiB = 1024 ** 3


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# --------------------------------------------------------------------------------------
# Bill of materials (declarative). ``status``: "repo-inventory" = recorded and verified in
# data/external_sources.json; "unverified" = believed correct, check with ``verify``.
# --------------------------------------------------------------------------------------
SOURCES: list[dict[str, Any]] = [
    dict(
        id="rcsb_structures", stage="structures", kind="rcsb", status="repo-inventory",
        title="RCSB PDB experimental structures (mmCIF, asymmetric unit)",
        regimes=["flow", "sequence", "constraint", "objective(stability geometry)", "CHIMERAFold"],
        est="nrps-pfam: expected low hundreds of entries (<1 GB; run plan --live for the true count). Whole PDB is ~2e5 "
            "entries, ~60 GB as .cif.gz (AlphaFold3 setup docs); this script stores uncompressed .cif, several times larger",
        license="CC0 1.0 (wwPDB)", rps=2.0,
        notes="Discovery via the repo's RCSBAcquirer (text) or a Pfam-annotation search (this script). "
              "Acquisition records QC state, SHA-256 (computed, not externally verified).",
    ),
    dict(
        id="rcsb_clusters", stage="clusters", kind="static", status="verified-docs",
        title="RCSB sequence clusters by entity (30% and 90% identity)",
        urls=[("clusters-by-entity-30.txt", "https://cdn.rcsb.org/resources/sequence/clusters/clusters-by-entity-30.txt"),
              ("clusters-by-entity-90.txt", "https://cdn.rcsb.org/resources/sequence/clusters/clusters-by-entity-90.txt")],
        regimes=["splits (sequence_cluster_id)", "folding calibration"],
        est="tens of MB (plain text, one cluster per line, entity IDs)", license="CC0 1.0", rps=1.0,
        notes="URLs confirmed on RCSB's File Download Services page (updated 2026-02-10). Weekly clusters; files are "
              "now produced with DIAMOND. The repo requires external cluster labels for non-trivial split builds.",
    ),
    dict(
        id="wwpdb_ccd", stage="ccd", kind="static", status="repo-inventory",
        title="wwPDB Chemical Component Dictionary",
        urls=[("components.cif.gz", "https://files.wwpdb.org/pub/pdb/data/monomers/components.cif.gz")],
        regimes=["constraint (substrate/ligand context)"], est="~0.5 GB", license="CC0 1.0", rps=1.0,
        notes="Ligand identity only; a CCD match does not establish biochemical substrate role.",
    ),
    dict(
        id="sifts", stage="sifts", kind="static", status="verified-docs",
        title="SIFTS PDB-chain to UniProt and Pfam mappings",
        urls=[("pdb_chain_uniprot.tsv.gz", "https://ftp.ebi.ac.uk/pub/databases/msd/sifts/flatfiles/tsv/pdb_chain_uniprot.tsv.gz"),
              ("pdb_chain_pfam.tsv.gz", "https://ftp.ebi.ac.uk/pub/databases/msd/sifts/flatfiles/tsv/pdb_chain_pfam.tsv.gz")],
        regimes=["splits (protein_family_id)", "linkage at scale", "offline NRPS discovery (Pfam)"], est="tens to ~100 MB",
        license="EMBL-EBI terms", rps=1.0,
        notes="File names confirmed on PDBe's SIFTS page. pdb_chain_pfam.tsv.gz drives offline discovery of PF00668 entries, "
              "so the unverified RCSB Pfam query is only a fallback.",
    ),
    dict(
        id="mibig", stage="mibig", kind="static", status="verified-docs",
        title="MIBiG 4.0.1 curated BGC records (JSON, GenBank, protein FASTA) from Zenodo",
        urls=[("mibig_json_4.0.tar.gz", "https://zenodo.org/records/14835872/files/mibig_json_4.0.tar.gz?download=1",
               {"md5": "ce1736f4fac7168d85dae3b37c77d3d2"}),
              ("mibig_gbk_4.0.tar.gz", "https://zenodo.org/records/14835872/files/mibig_gbk_4.0.tar.gz?download=1",
               {"md5": "4e905ea01e4d8990e24c36161aea5fbd"}),
              ("mibig_prot_seqs_4.0.fasta", "https://zenodo.org/records/14835872/files/mibig_prot_seqs_4.0.fasta?download=1",
               {"md5": "1dd73356e63d6ba3507f9199cfd22e1c"})],
        regimes=["objective(substrate_selectivity)", "constraint", "splits (nrps_family, substrate)"],
        est="125.5 MB total (10.0 + 83.7 + 31.8)", license="CC BY 4.0", rps=1.0,
        notes="Zenodo record 14835872 (doi 10.5281/zenodo.14835872), 3,059 curated entries; names, sizes and MD5s "
              "taken from the record page. Downloads are MD5-verified against Zenodo.",
    ),
    dict(
        id="uniprot_nrps", stage="nrps_uniprot", kind="uniprot_stream", status="repo-inventory",
        title="UniProtKB NRPS-family proteins (InterPro IPR010071 adenylation, IPR001242 condensation)",
        query="(xref:interpro-IPR010071) OR (xref:interpro-IPR001242)",
        regimes=["representation (NRPS-specific MSAs)", "homolog sets"], est="unknown until counted (run verify); likely hundreds of MB to low GB (FASTA, gz)",
        license="CC BY 4.0", rps=1.0,
        notes="Stream endpoint is resource-intensive; one serial request, compressed.",
    ),
    dict(
        id="uniref90", stage="uniref90", kind="static", status="unverified", huge=True,
        title="UniRef90 FASTA (homology-search target database)",
        urls=[("uniref90.fasta.gz", "https://ftp.uniprot.org/pub/databases/uniprot/uniref/uniref90/uniref90.fasta.gz")],
        regimes=["representation (deep MSAs via MMseqs2)"], est="~13-55 GB compressed depending on release (third-party docs; current release not checked); "
            "roughly 60-140 GB unpacked; budget >200 GB scratch for an indexed MMseqs2 DB",
        license="CC BY 4.0", rps=1.0,
        notes="Only needed for general (non-NRPS) deep MSAs. For NRPS-focused work, search the UniProt NRPS-family FASTA "
              "instead (msa --target nrps, the default): hundreds of MB instead of hundreds of GB. The repo's UniRef90 "
              "*cluster-member* MSA collapses to Neff ~ 1.",
    ),
    dict(
        id="codon_in_repo", stage=None, kind="in_repo", status="repo",
        title="Codon-usage sets: Fath 2011 (measured) and human-expression proxy",
        regimes=["objective(expression_efficiency)", "codon optimizer"],
        est="588 rows already in data/", license="see data/external_sources and the source papers", rps=0,
        notes="Fath 2011 is shipped. The human proxy needs the Pouyet supplementary table and "
              "scripts/fetch_human_expression_cds.py <summary.csv> (Ensembl REST).",
    ),
    dict(
        id="afdb_predicted", stage=None, kind="manual", status="unverified",
        title="AlphaFold DB predicted structures (animal NRPS with no experimental structure)",
        regimes=["flow", "sequence", "representation (structure-supervised)"],
        est="small (one model per accession)", license="CC BY 4.0", rps=1.0,
        notes="GET https://alphafold.ebi.ac.uk/api/prediction/{UniProt} then the returned cifUrl. The repo keeps "
              "'predicted' provenance separate from 'experimental'; do not merge silently.",
    ),
    dict(
        id="openproteinset", stage=None, kind="manual", status="unverified",
        title="OpenProteinSet (precomputed AlphaFold-style MSAs for PDB chains and UniClust30 clusters)",
        regimes=["representation (bootstrap deep MSAs)"],
        est="hundreds of GB to TB", license="CC BY 4.0 (check the dataset card)", rps=0,
        notes="aws s3 sync s3://openfold/ <dir> --no-sign-request --exclude '*' --include 'pdb/*' (check bucket layout "
              "first). a3m files convert with the msa-stage helper a3m_to_aligned_fasta().",
    ),
    dict(
        id="stability_labels", stage=None, kind="manual", status="unverified",
        title="Experimental stability measurements (e.g. mega-scale cDNA-display proteolysis, Tsuboyama 2023, Zenodo)",
        regimes=["objective(structural_stability)"], est="~1 GB", license="check Zenodo record", rps=0,
        notes="Small domains only (~40-72 aa); label kind should be validated_surrogate at best for 1000-residue NRPS modules.",
    ),
    dict(
        id="selectivity_labels", stage=None, kind="manual", status="unverified",
        title="A-domain substrate-specificity labels (MIBiG module annotations; Stachelhaus-code sets)",
        regimes=["objective(substrate_selectivity)", "constraint"], est="small", license="mixed", rps=0,
        notes="MIBiG is fetched; extra curated sets must be mapped to sequences by accession, never by name.",
    ),
    dict(
        id="proteus_rounds", stage=None, kind="wet_lab", status="n/a",
        title="PROTEUS directed-evolution rounds (preference pairs)",
        regimes=["preference (DPO)"], est="produced in the lab", license="yours", rps=0,
        notes="Cannot be downloaded. The PREFERENCE regime stays unusable until real rounds exist.",
    ),
]

PFAM_ANCHORS = {
    "nrps-pfam": ["PF00668"],                      # condensation domain: NRPS-specific anchor
    "adenylate-family": ["PF00501", "PF13193"],    # AMP-binding; includes non-NRPS acyl-CoA synthetases
}
STAGE_GROUPS = {
    "light": ["clusters", "sifts", "ccd", "mibig", "nrps_uniprot", "structures"],
    "all": ["clusters", "sifts", "ccd", "mibig", "nrps_uniprot", "structures", "uniref90"],
}


# --------------------------------------------------------------------------------------
# Download machinery
# --------------------------------------------------------------------------------------
@dataclass
class Ctx:
    root: Path
    email: str | None = None
    refresh: bool = False
    dry_run: bool = False
    yes: bool = False
    log: Any = print
    manifest: Path = field(init=False)

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        self.manifest = self.root / "download_manifest.jsonl"


def _headers(ctx: Ctx | None, extra: dict[str, str] | None = None) -> dict[str, str]:
    headers = {"User-Agent": USER_AGENT + (f" ({ctx.email})" if ctx and ctx.email else "")}
    if ctx and ctx.email:
        headers["From"] = ctx.email
    headers.update(extra or {})
    return headers


def _open(url: str, ctx: Ctx | None, *, timeout: float = 60.0, headers: dict[str, str] | None = None,
          method: str | None = None, data: bytes | None = None):
    return urlopen(Request(url, data=data, method=method, headers=_headers(ctx, headers)), timeout=timeout)


def _retry(fn, *, retries: int = 3, what: str = "request"):
    for attempt in range(retries + 1):
        try:
            return fn()
        except HTTPError as exc:
            if attempt >= retries or exc.code not in {429, 500, 502, 503, 504}:
                raise
            retry_after = exc.headers.get("Retry-After") if exc.headers else None
            time.sleep(float(retry_after) if retry_after and retry_after.isdigit() else min(2.0 ** attempt, 30.0))
        except (URLError, TimeoutError, ConnectionError, http.client.HTTPException):
            if attempt >= retries:
                raise
            time.sleep(min(2.0 ** attempt, 30.0))
    raise RuntimeError(f"{what} retry loop exhausted")


def read_manifest(path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                records[row["path"]] = row
    return records


def download(url: str, dest: Path, ctx: Ctx, *, source_id: str, rps: float = 1.0,
             expected_sha256: str | None = None, expected_md5: str | None = None,
             retries: int = 3, chunk: int = 1 << 20) -> dict[str, Any]:
    """Resumable, atomic, checksummed download. Re-running skips completed files."""
    dest = Path(dest)
    known = read_manifest(ctx.manifest).get(str(dest))
    if dest.exists() and known and not ctx.refresh:
        return {**known, "skipped": True}
    if ctx.dry_run:
        ctx.log(f"[dry-run] would download {url} -> {dest}")
        return {"path": str(dest), "url": url, "dry_run": True}
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    if ctx.refresh and part.exists():
        part.unlink()
    time.sleep(1.0 / rps if rps else 0.0)

    def attempt() -> dict[str, Any]:
        offset = part.stat().st_size if part.exists() else 0
        extra = {"Range": f"bytes={offset}-"} if offset else {}
        with _open(url, ctx, timeout=120, headers=extra) as response:
            status = getattr(response, "status", 200)
            if offset and status != 206:      # server ignored Range: restart cleanly
                offset = 0
            mode = "ab" if offset else "wb"
            declared = response.headers.get("Content-Length")
            received = 0
            with part.open(mode) as handle:
                while True:
                    block = response.read(chunk)
                    if not block:
                        break
                    handle.write(block)
                    received += len(block)
            # http.client does not raise when a body is shorter than Content-Length, so check it here;
            # raising triggers a retry that resumes from the bytes already on disk.
            if declared is not None and declared.isdigit() and received != int(declared):
                raise ConnectionError(f"incomplete body: got {received} of {declared} bytes")
            return {"etag": response.headers.get("ETag"), "last_modified": response.headers.get("Last-Modified")}

    meta = _retry(attempt, retries=retries, what=f"download {url}")
    digest, md5 = hashlib.sha256(), hashlib.md5()
    with part.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
            md5.update(block)
    sha = digest.hexdigest()
    if expected_sha256 and sha != expected_sha256.lower():
        part.rename(dest.with_name(dest.name + ".CHECKSUM_MISMATCH"))
        raise ValueError(f"checksum mismatch for {url}: expected {expected_sha256}, got {sha}")
    if expected_md5 and md5.hexdigest() != expected_md5.lower():
        part.rename(dest.with_name(dest.name + ".CHECKSUM_MISMATCH"))
        raise ValueError(f"checksum mismatch (md5) for {url}: expected {expected_md5}, got {md5.hexdigest()}")
    if dest.suffix == ".gz" and part.stat().st_size <= GiB:     # full gunzip check for files up to 1 GiB;
        with gzip.open(part, "rb") as probe:                      # larger files rely on the length check above
            while probe.read(1 << 20):
                pass
    part.replace(dest)
    record = {
        "source_id": source_id, "url": url, "path": str(dest), "bytes": dest.stat().st_size, "sha256": sha,
        "checksum_verified_against_source": bool(expected_sha256 or expected_md5),
        "md5": md5.hexdigest(), "retrieved_utc": utc_now(), **meta,
    }
    ctx.manifest.parent.mkdir(parents=True, exist_ok=True)
    with ctx.manifest.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
    ctx.log(f"downloaded {dest.name} ({record['bytes'] / 1e6:.1f} MB)")
    return record


def probe(url: str, ctx: Ctx | None = None) -> dict[str, Any]:
    """HEAD (falling back to a 1-byte ranged GET) without downloading the body."""
    for method, extra in (("HEAD", None), ("GET", {"Range": "bytes=0-0"})):
        try:
            with _open(url, ctx, timeout=30, method=method, headers=extra) as response:
                length = response.headers.get("Content-Range", "").split("/")[-1] or response.headers.get("Content-Length")
                return {"ok": True, "status": response.status, "bytes": int(length) if length and length.isdigit() else None,
                        "last_modified": response.headers.get("Last-Modified")}
        except HTTPError as exc:
            last = {"ok": False, "status": exc.code, "error": str(exc)}
            if exc.code not in {403, 405, 501}:
                return last
        except (URLError, TimeoutError, ConnectionError) as exc:
            return {"ok": False, "status": None, "error": str(exc)}
    return last


# --------------------------------------------------------------------------------------
# RCSB discovery (repo clients + a Pfam-annotation search the repo lacks)
# --------------------------------------------------------------------------------------
def rcsb_pfam_payload(pfam_ids: Iterable[str], *, max_resolution: float | None, start: int, rows: int) -> dict[str, Any]:
    def terminal(attribute: str, operator: str, value: Any) -> dict[str, Any]:
        return {"type": "terminal", "service": "text",
                "parameters": {"attribute": attribute, "operator": operator, "value": value}}

    nodes: list[dict[str, Any]] = [{
        "type": "group", "logical_operator": "or",
        "nodes": [terminal("rcsb_polymer_entity_annotation.annotation_id", "exact_match", pid) for pid in pfam_ids],
    }]
    if max_resolution is not None:
        nodes.append(terminal("rcsb_entry_info.resolution_combined", "less_or_equal", max_resolution))
    return {
        "query": {"type": "group", "logical_operator": "and", "nodes": nodes},
        "return_type": "entry",
        "request_options": {"paginate": {"start": start, "rows": rows}, "results_content_type": ["experimental"]},
    }


def rcsb_search(payload: dict[str, Any], ctx: Ctx | None, endpoint: str | None = None) -> dict[str, Any]:
    from data.rcsb import RCSB_SEARCH_ENDPOINT
    body = json.dumps(payload).encode("utf-8")

    def call() -> dict[str, Any]:
        with _open(endpoint or RCSB_SEARCH_ENDPOINT, ctx, timeout=60, method="POST", data=body,
                   headers={"Content-Type": "application/json"}) as response:
            raw = response.read()
            return json.loads(raw.decode("utf-8")) if raw else {"total_count": 0, "result_set": []}

    return _retry(call, what="RCSB search")


def pfam_entries_from_sifts(path: Path, pfam_ids: Iterable[str]) -> list[str]:
    """Entry IDs with any chain annotated with one of the Pfam IDs (offline; SIFTS pdb_chain_pfam.tsv.gz)."""
    wanted, found = set(pfam_ids), set()
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith(("#", "PDB")) or not line.strip():
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) > 3 and parts[3] in wanted:
                found.add(parts[0].upper())
    return sorted(found)


def cluster_representatives(ids: Iterable[str], clusters_path: Path) -> list[str]:
    """One entry per sequence cluster: the first cluster member that is in ``ids``."""
    wanted, chosen = {value.upper() for value in ids}, set()
    with open(clusters_path, "rt", encoding="utf-8") as handle:
        for line in handle:
            for member in line.split():
                entry = member.split("_")[0].upper()
                if entry in wanted:
                    chosen.add(entry)
                    break
    return sorted(chosen)


def discover_ids(profile: str, ctx: Ctx, *, count_only: bool = False) -> list[str] | int:
    from data.rcsb import RCSBAcquirer, RCSBQuery, build_rcsb_query_payload
    pfam_profiles = PFAM_ANCHORS
    sifts_path = ctx.root / "raw" / "sifts" / "pdb_chain_pfam.tsv.gz"
    if profile in pfam_profiles and sifts_path.exists():
        ids = pfam_entries_from_sifts(sifts_path, pfam_profiles[profile])
        return len(ids) if count_only else ids
    if profile in pfam_profiles:
        ids: list[str] = []
        start, rows = 0, 1 if count_only else 1000
        while True:
            page = rcsb_search(rcsb_pfam_payload(pfam_profiles[profile], max_resolution=4.0, start=start, rows=rows), ctx)
            total = int(page.get("total_count", 0))
            if count_only:
                return total
            batch = [item["identifier"] for item in page.get("result_set", [])]
            ids.extend(batch)
            start += rows
            if not batch or start >= total:
                return sorted(set(ids))
            time.sleep(0.5)
    queries = {
        "nrps-text": RCSBQuery(nrps_targeted=True, maximum_resolution_angstrom=4.0, minimum_sequence_length=40, page_size=500,
                               experimental_methods=("X-RAY DIFFRACTION", "ELECTRON MICROSCOPY")),
        "general": RCSBQuery(maximum_resolution_angstrom=2.5, minimum_sequence_length=50, maximum_sequence_length=1500,
                             experimental_methods=("X-RAY DIFFRACTION",), page_size=1000),
    }
    if profile not in queries:
        raise SystemExit(f"unknown profile {profile!r}; choose from {sorted([*pfam_profiles, *queries])}")
    if count_only:
        payload = build_rcsb_query_payload(queries[profile], start=0)
        payload.setdefault("request_options", {})["paginate"] = {"start": 0, "rows": 1}
        return int(rcsb_search(payload, ctx).get("total_count", 0))
    return RCSBAcquirer(requests_per_second=2.0).discover(queries[profile])


# --------------------------------------------------------------------------------------
# Stages
# --------------------------------------------------------------------------------------
def source_by_stage(stage: str) -> dict[str, Any]:
    return next(item for item in SOURCES if item.get("stage") == stage)


def stage_static(stage: str, ctx: Ctx) -> None:
    source = source_by_stage(stage)
    for item in source["urls"]:
        name, url = item[0], item[1]
        extra = item[2] if len(item) > 2 else {}
        download(url, ctx.root / "raw" / source["id"] / name, ctx, source_id=source["id"], rps=source["rps"],
                 expected_md5=extra.get("md5"), expected_sha256=extra.get("sha256"))


def stage_nrps_uniprot(ctx: Ctx) -> None:
    from urllib.parse import quote
    source = source_by_stage("nrps_uniprot")
    url = ("https://rest.uniprot.org/uniprotkb/stream?format=fasta&compressed=true&query=" + quote(source["query"]))
    download(url, ctx.root / "raw" / "uniprot_nrps" / "nrps_family.fasta.gz", ctx, source_id="uniprot_nrps", rps=1.0)


def stage_structures(ctx: Ctx, profiles: list[str], *, max_structures: int | None, batch_size: int, assembly: str | None,
                     cluster_identity: int | None = None) -> None:
    from data.rcsb import RCSBAcquirer
    acquirer = RCSBAcquirer(requests_per_second=2.0, maximum_workers=4)
    for profile in profiles:
        out = ctx.root / "structures" / profile
        discovery = out / "discovery_manifest.json"
        if discovery.exists() and not ctx.refresh:
            ids = json.loads(discovery.read_text(encoding="utf-8"))["structure_ids"]
        else:
            try:
                ids = discover_ids(profile, ctx)
            except (URLError, TimeoutError, ConnectionError, HTTPError) as exc:
                message = f"RCSB discovery for profile {profile!r} failed ({type(exc).__name__}: {exc})"
                if ctx.dry_run:
                    ctx.log(f"[dry-run] {message}; skipping (discovery needs network)")
                    continue
                raise SystemExit(message + ". Check connectivity, then re-run (finished work is cached).") from exc
            if ctx.dry_run:
                ctx.log(f"[dry-run] profile {profile}: {len(ids)} entries discovered")
                continue
            out.mkdir(parents=True, exist_ok=True)
            discovery.write_text(json.dumps({"profile": profile, "retrieved_utc": utc_now(), "structure_ids": ids}, indent=1),
                                 encoding="utf-8")
        if cluster_identity:
            clusters_file = ctx.root / "raw" / "rcsb_clusters" / f"clusters-by-entity-{cluster_identity}.txt"
            if not clusters_file.exists():
                raise SystemExit(f"{clusters_file} missing; run the 'clusters' stage first (it is in the default order).")
            before = len(ids)
            ids = cluster_representatives(ids, clusters_file)
            ctx.log(f"profile {profile}: {before} entries -> {len(ids)} cluster representatives at {cluster_identity}%")
        if max_structures:
            ids = ids[:max_structures]
        ctx.log(f"profile {profile}: {len(ids)} entries in {-(-len(ids) // batch_size)} batch(es)")
        for index in range(0, len(ids), batch_size):
            batch_dir = out / f"batch_{index // batch_size:05d}"
            if (batch_dir / "acquisition_manifest.jsonl").exists() and not ctx.refresh:
                continue                                    # resume: finished batches are skipped
            if ctx.dry_run:
                ctx.log(f"[dry-run] would acquire {len(ids[index:index + batch_size])} entries into {batch_dir}")
                continue
            summary = acquirer.acquire_many(ids[index:index + batch_size], batch_dir, assembly_id=assembly)
            ctx.log(f"  {batch_dir.name}: accepted={summary.get('accepted')} rejected={summary.get('rejected')} "
                    f"quarantined={summary.get('quarantined')}")


# ---- MSA (MMseqs2) --------------------------------------------------------------------
def a3m_to_aligned_fasta(text: str) -> str:
    """Drop a3m insertions (lowercase) so every row has the query length; strip NULs."""
    out, name, seq = [], None, []
    for line in text.replace("\x00", "").splitlines():
        if line.startswith(">"):
            if name is not None:
                out.append(f"{name}\n{''.join(seq)}")
            name, seq = line.rstrip(), []
        elif line.strip():
            seq.append("".join(ch for ch in line.strip() if not ch.islower()))
    if name is not None:
        out.append(f"{name}\n{''.join(seq)}")
    return "\n".join(out) + "\n"


def accepted_chains(root: Path, *, all_chains: bool = False, min_len: int = 30, max_len: int = 1500) -> Iterator[dict[str, Any]]:
    for manifest in sorted((root / "structures").glob("**/acquisition_manifest.jsonl")):
        profile = manifest.relative_to(root / "structures").parts[0]
        for line in manifest.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("status") != "accepted":
                continue
            seen_entities: set[str] = set()
            for chain in record.get("canonical_chains", []):
                quality = (chain.get("quality") or {}).get("status")
                sequence = chain.get("sequence", "")
                if quality != "accepted" or not min_len <= len(sequence) <= max_len:
                    continue
                if not all_chains and chain["entity_id"] in seen_entities:
                    continue
                seen_entities.add(chain["entity_id"])
                yield {"profile": profile, "structure_id": record["structure_id"], "raw_path": record["raw_path"],
                       "label_asym_id": chain["label_asym_id"], "entity_id": chain["entity_id"], "sequence": sequence}


def seq_key(sequence: str) -> str:
    return "SEQ_" + hashlib.sha1(sequence.encode("ascii")).hexdigest()[:16]


def run(cmd: list[str], ctx: Ctx, **kwargs: Any) -> None:
    ctx.log("$ " + " ".join(cmd))
    if not ctx.dry_run:
        subprocess.run(cmd, check=True, **kwargs)


def stage_msa(ctx: Ctx, *, mmseqs: str, uniref_fasta: Path | None, uniref_db: Path | None, threads: int,
              sensitivity: float, iterations: int, max_seqs: int, all_chains: bool, target: str = "nrps") -> None:
    binary = shutil.which(mmseqs)
    if binary is None and not ctx.dry_run:
        raise SystemExit("mmseqs not found. Install MMseqs2 (https://github.com/soedinglab/MMseqs2) or pass --mmseqs PATH.")
    binary = binary or mmseqs
    work = ctx.root / "msa"
    out_dir = work / "a3m"
    if uniref_db is None:
        if uniref_fasta is None:
            uniref_fasta = (ctx.root / "raw" / "uniprot_nrps" / "nrps_family.fasta.gz" if target == "nrps"
                            else ctx.root / "raw" / "uniref90" / "uniref90.fasta.gz")
        uniref_db = work / f"{target}_db"
        if not (uniref_db.with_name(uniref_db.name + ".dbtype")).exists():
            run([binary, "createdb", str(uniref_fasta), str(uniref_db)], ctx)
            run([binary, "createindex", str(uniref_db), str(work / "tmp_index"), "--threads", str(threads)], ctx)
    queries: dict[str, str] = {}
    for chain in accepted_chains(ctx.root, all_chains=all_chains):
        queries.setdefault(seq_key(chain["sequence"]), chain["sequence"])        # one search per unique sequence
    todo = {key: seq for key, seq in queries.items() if not (out_dir / f"{key}.a3m").exists()}
    ctx.log(f"{len(queries)} unique sequences, {len(todo)} still to search")
    if not todo or ctx.dry_run:
        return
    work.mkdir(parents=True, exist_ok=True)
    query_fasta = work / "queries.fasta"
    query_fasta.write_text("".join(f">{key}\n{seq}\n" for key, seq in sorted(todo.items())), encoding="ascii")
    for leftover in work.glob("qdb*"), work.glob("res*"), work.glob("msa_db*"):
        for item in leftover:
            item.unlink()
    shutil.rmtree(work / "tmp", ignore_errors=True)
    run([binary, "createdb", str(query_fasta), str(work / "qdb"), "-v", "1"], ctx)
    run([binary, "search", str(work / "qdb"), str(uniref_db), str(work / "res"), str(work / "tmp"),
         "-s", str(sensitivity), "--num-iterations", str(iterations), "--max-seqs", str(max_seqs),
         "-e", "1e-3", "--threads", str(threads), "-v", "1"], ctx)
    run([binary, "result2msa", str(work / "qdb"), str(uniref_db), str(work / "res"), str(work / "msa_db"),
         "--msa-format-mode", "6", "--threads", str(threads), "-v", "1"], ctx)
    unpack = work / "unpacked"
    shutil.rmtree(unpack, ignore_errors=True)
    run([binary, "unpackdb", str(work / "msa_db"), str(unpack), "--unpack-name-mode", "1", "--unpack-suffix", ".a3m", "-v", "1"], ctx)
    out_dir.mkdir(parents=True, exist_ok=True)
    produced = 0
    for path in sorted(unpack.glob("*.a3m")):
        text = path.read_text(encoding="ascii", errors="replace").replace("\x00", "")
        first = text.split("\n", 1)[0].lstrip(">").split()[0] if text.startswith(">") else ""
        if first in todo:                                     # query name is the first a3m header
            (out_dir / f"{first}.a3m").write_text(text, encoding="ascii")
            produced += 1
    ctx.log(f"wrote {produced} a3m files to {out_dir}")


# ---- Assembly line --------------------------------------------------------------------
def load_rcsb_clusters(path: Path) -> dict[str, str]:
    """entity id ('1AMU_1') -> cluster label. Each line of the RCSB file is one cluster."""
    mapping: dict[str, str] = {}
    digest = hashlib.sha256(path.read_bytes()).hexdigest()[:10]
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            for member in line.split():
                mapping[member.upper()] = f"rcsb:{path.stem}:{digest}:{index}"
    return mapping


def load_sifts(path: Path, *, column: int) -> dict[tuple[str, str], str]:
    mapping: dict[tuple[str, str], str] = {}
    if not path.exists():
        return mapping
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith(("#", "PDB")) or not line.strip():
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) > column:
                mapping.setdefault((parts[0].upper(), parts[1]), parts[column])
    return mapping


def cached_json(path: Path, fetch):
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    value = fetch()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return value


def stage_assemble(ctx: Ctx, *, dataset_version: str, annotate: str, all_chains: bool, max_chains: int | None,
                   shard_size: int, split_fractions: tuple[float, float, float, float], ood_dimension: str | None,
                   uniprot_client=None, interpro_client=None, qc_config=None) -> dict[str, Any]:
    from data.annotations import combine_uniprot_interpro_nrps, parse_interpro_domain_matches
    from data.dataset_api import build_dataset_sample, write_immutable_dataset
    from data.interpro import InterProClient
    from data.leakage_splits import SplitConfig, generate_leakage_safe_splits
    from data.msa import MSAConfig, build_msa_record
    from data.sequence_linkage import link_structure_sequence
    from data.structures import StructuralQCConfig, parse_mmcif_structure
    from data.uniprot import UniProtClient

    root = ctx.root
    clusters: dict[str, str] = {}
    for candidate in sorted((root / "raw" / "rcsb_clusters").glob("clusters-by-entity-30.txt*")):
        clusters = load_rcsb_clusters(candidate)
    sifts_pfam = load_sifts(root / "raw" / "sifts" / "pdb_chain_pfam.tsv.gz", column=3)
    uniprot = uniprot_client or UniProtClient(requests_per_second=1.0, email=ctx.email)
    interpro = interpro_client or InterProClient(requests_per_second=1.0)
    cache = root / "cache"
    today = datetime.now(timezone.utc).date().isoformat()
    qc = qc_config or StructuralQCConfig()

    pass1: list[dict[str, Any]] = []
    stats = {"considered": 0, "unparsed": 0, "annotated": 0, "with_deep_msa": 0, "single_sequence_msa": 0, "unlinked": 0}
    for chain in accepted_chains(root, all_chains=all_chains):
        if max_chains and len(pass1) >= max_chains:
            break
        stats["considered"] += 1
        try:
            structure = parse_mmcif_structure(chain["raw_path"], structure_id=chain["structure_id"], config=qc)
            parsed = next(item for item in structure.chains if item.label_asym_id == chain["label_asym_id"])
        except Exception as exc:                          # noqa: BLE001 - record and continue
            stats["unparsed"] += 1
            ctx.log(f"skip {chain['structure_id']}:{chain['label_asym_id']}: {type(exc).__name__}: {exc}")
            continue
        reported = next((item["accession"] for item in structure.source_accessions), None)
        record_id = f"{chain['structure_id']}:{parsed.label_asym_id}:{parsed.entity_id}"
        candidates, annotation, taxonomy, release = [], None, "", "UNKNOWN"
        do_annotate = annotate == "all" or (annotate == "nrps" and chain["profile"].startswith("nrps"))
        if reported and do_annotate:
            try:
                resolved, _ = uniprot.resolve_accession(reported)
                candidates = [p.to_candidate() for p in uniprot.fetch_proteins(resolved)]
                primary = resolved[0]
                entry = cached_json(cache / "uniprot" / f"{primary}.json", lambda: uniprot.fetch_uniprot_entry(primary)[0])
                taxonomy = str(entry.get("organism", {}).get("taxonId", ""))
                release = entry.get("entryAudit", {}).get("lastAnnotationUpdateDate", "UNKNOWN")
                payload = cached_json(cache / "interpro" / f"{primary}.json",
                                      lambda: interpro.fetch_protein_matches(primary)[0])
                matches = parse_interpro_domain_matches(payload, accession=primary, release="UNKNOWN", verification_date=today)
                annotation = combine_uniprot_interpro_nrps(matches, entry, uniprot_release=str(release),
                                                           interpro_release="UNKNOWN", verification_date=today)
                stats["annotated"] += 1
            except Exception as exc:                      # noqa: BLE001
                ctx.log(f"annotation unavailable for {record_id} ({reported}): {type(exc).__name__}: {exc}")
        linkage = link_structure_sequence(
            structure_id=structure.structure_id, chain_id=parsed.label_asym_id, entity_id=parsed.entity_id,
            sequence=parsed.sequence, candidates=candidates, reported_accession=reported,
        )
        stats["unlinked"] += linkage.external_accession is None
        a3m = root / "msa" / "a3m" / f"{seq_key(parsed.sequence)}.a3m"
        msa = None
        if a3m.exists():
            msa = build_msa_record(
                a3m_to_aligned_fasta(a3m.read_text(encoding="ascii", errors="replace")),
                msa_family_id=f"mmseqs:{seq_key(parsed.sequence)}", target_sequence_id=record_id,
                source="MMseqs2 search against UniRef90", source_version="local-db", target_sequence=parsed.sequence,
                generation_method="mmseqs2-iterative-search", generation_version="fetch_corpus-v0.1",
                config=MSAConfig(minimum_coverage=0.4, maximum_gap_fraction=0.7, maximum_depth=512, identity_cluster_threshold=0.9),
            )
            stats["with_deep_msa"] += 1
        else:
            stats["single_sequence_msa"] += 1
        domains = annotation.domains if annotation else ()
        pass1.append({
            "record_id": record_id, "chain": chain, "linkage": linkage, "annotation": annotation, "msa": msa,
            "split_record": {
                "record_id": record_id, "sequence": parsed.sequence,
                "sequence_cluster_id": clusters.get(f"{chain['structure_id']}_{parsed.entity_id}".upper(), f"singleton:{record_id}"),
                "protein_family_id": sifts_pfam.get((chain["structure_id"].upper(), parsed.label_asym_id), "UNKNOWN"),
                "nrps_family_id": f"UniProt:{linkage.external_accession}" if annotation and linkage.external_accession else "NONE",
                "taxonomy_id": taxonomy or "UNKNOWN",
                "substrate_ids": sorted({p.substrate_name for p in annotation.products if p.substrate_name}) if annotation else [],
                "module_composition": "-".join(d.domain_type for d in domains) if domains else "NONE",
            },
        })
    if not pass1:
        raise SystemExit("no accepted chains found; run the structures stage first")

    train, validation, test, ood = split_fractions
    manifest = generate_leakage_safe_splits(
        [item["split_record"] for item in pass1], dataset_version=dataset_version,
        config=SplitConfig(train_fraction=train, validation_fraction=validation, test_fraction=test, ood_fraction=ood,
                           ood_dimension=ood_dimension if ood > 0 else None),
    )
    out = root / "dataset"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{dataset_version}.split_manifest.json").write_text(json.dumps(manifest.to_dict(), indent=1, default=str), encoding="utf-8")

    shard_paths: list[str] = []
    buffer: list[dict[str, Any]] = []

    def flush() -> None:
        if not buffer:
            return
        path = out / f"{dataset_version}-shard{len(shard_paths):04d}.sqlite"
        write_immutable_dataset(path, buffer, dataset_version=dataset_version, split_manifest_sha256=manifest.manifest_sha256)
        shard_paths.append(str(path))
        buffer.clear()

    for item in pass1:                                    # pass 2: build tensors, one shard at a time
        chain = item["chain"]
        structure = parse_mmcif_structure(chain["raw_path"], structure_id=chain["structure_id"], config=qc)
        parsed = next(c for c in structure.chains if c.label_asym_id == chain["label_asym_id"])
        buffer.append(build_dataset_sample(
            structure, parsed, msa=item["msa"], sequence_linkage=item["linkage"], nrps_annotation=item["annotation"],
            dataset_version=dataset_version, split=manifest.assignments[item["record_id"]],
        ))
        if len(buffer) >= shard_size:
            flush()
    flush()
    split_counts: dict[str, int] = {}
    for value in manifest.assignments.values():
        split_counts[value] = split_counts.get(value, 0) + 1
    report = {"dataset_version": dataset_version, "samples": len(pass1), "shards": shard_paths, "splits": split_counts,
              "split_manifest_sha256": manifest.manifest_sha256, **stats}
    (out / f"{dataset_version}.report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    return report


# --------------------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------------------
def cmd_plan(args: argparse.Namespace) -> int:
    ctx = Ctx(root=Path("."), email=args.email)
    live: dict[str, Any] = {}
    if args.live:
        for profile in ("nrps-pfam", "adenylate-family", "nrps-text", "general"):
            try:
                live[profile] = discover_ids(profile, ctx, count_only=True)
            except Exception as exc:                      # noqa: BLE001
                live[profile] = f"unavailable ({type(exc).__name__}: {exc})"
    if args.json:
        print(json.dumps({"sources": SOURCES, "rcsb_live_counts": live}, indent=2))
        return 0
    print(f"{'stage':14} {'source':44} {'status':15} est. size")
    for source in SOURCES:
        print(f"{str(source.get('stage') or '-'):14} {source['title'][:44]:44} {source['status']:15} {source['est']}")
    if live:
        print("\nLive RCSB entry counts:", json.dumps(live))
    print("\nstage groups:", json.dumps(STAGE_GROUPS))
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    ctx = Ctx(root=Path("/nonexistent"), email=args.email)   # no local SIFTS file: verify must test the live API
    failures = 0
    for source in SOURCES:
        if source["kind"] == "static":
            for item in source["urls"]:
                name, url = item[0], item[1]
                result = probe(url, ctx)
                failures += not result["ok"]
                size = f"{result['bytes'] / 1e6:.0f} MB" if result.get("bytes") else "?"
                print(f"{'OK ' if result['ok'] else 'FAIL'} {source['id']:16} {name:30} {result.get('status')} {size}")
        elif source["kind"] == "uniprot_stream":
            result = probe("https://rest.uniprot.org/uniprotkb/search?size=1&format=fasta&query=" + "xref%3Ainterpro-IPR010071", ctx)
            failures += not result["ok"]
            print(f"{'OK ' if result['ok'] else 'FAIL'} {source['id']:16} {'search':30} {result.get('status')}")
    for profile in PFAM_ANCHORS:
        try:
            total = discover_ids(profile, ctx, count_only=True)
            print(f"OK   rcsb_search     {profile:30} {total} entries")
            failures += total == 0
        except Exception as exc:                          # noqa: BLE001
            failures += 1
            print(f"FAIL rcsb_search     {profile:30} {type(exc).__name__}: {exc}")
    return 1 if failures else 0


def cmd_status(args: argparse.Namespace) -> int:
    root = Path(args.root)
    downloads = read_manifest(root / "download_manifest.jsonl")
    print(f"downloads: {len(downloads)} files, {sum(r['bytes'] for r in downloads.values()) / GiB:.2f} GiB")
    for profile_dir in sorted((root / "structures").glob("*")) if (root / "structures").exists() else []:
        accepted = rejected = 0
        for manifest in profile_dir.glob("**/acquisition_manifest.jsonl"):
            for line in manifest.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    status = json.loads(line).get("status")
                    accepted += status == "accepted"
                    rejected += status != "accepted"
        print(f"structures/{profile_dir.name}: accepted={accepted} not_accepted={rejected}")
    a3m = len(list((root / "msa" / "a3m").glob("*.a3m"))) if (root / "msa" / "a3m").exists() else 0
    print(f"msa: {a3m} a3m files")
    for report in sorted((root / "dataset").glob("*.report.json")) if (root / "dataset").exists() else []:
        print(f"dataset report {report.name}: {report.read_text(encoding='utf-8')}")
    return 0


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------
def expand_stages(names: list[str]) -> list[str]:
    stages: list[str] = []
    for name in names:
        stages.extend(STAGE_GROUPS.get(name, [name]))
    return list(dict.fromkeys(stages))


def estimate_bytes(stages: list[str]) -> str:
    return ", ".join(f"{s}: {source_by_stage(s)['est']}" for s in stages if any(x.get("stage") == s for x in SOURCES))


def cmd_fetch(args: argparse.Namespace) -> int:
    ctx = Ctx(root=Path(args.root), email=args.email, refresh=args.refresh, dry_run=args.dry_run, yes=args.yes)
    stages = expand_stages(args.stages)
    valid = {item["stage"] for item in SOURCES if item.get("stage")}
    unknown = [s for s in stages if s not in valid]
    if unknown:
        raise SystemExit(f"unknown stage(s) {unknown}; valid: {sorted(valid)} or groups {sorted(STAGE_GROUPS)}")
    big = [s for s in stages if source_by_stage(s).get("huge")]
    if big and not (ctx.yes or ctx.dry_run):
        raise SystemExit(f"stage(s) {big} are very large ({estimate_bytes(big)}). Re-run with --yes to confirm.")
    ctx.log(f"stages: {stages}\nroot: {ctx.root.resolve()}")
    for stage in stages:
        ctx.log(f"== {stage} ==")
        if stage == "structures":
            stage_structures(ctx, args.profiles, max_structures=args.max_structures, batch_size=args.batch_size,
                             assembly=args.assembly, cluster_identity=args.cluster_representatives)
        elif stage == "nrps_uniprot":
            stage_nrps_uniprot(ctx)
        else:
            stage_static(stage, ctx)
    return 0


def cmd_msa(args: argparse.Namespace) -> int:
    ctx = Ctx(root=Path(args.root), dry_run=args.dry_run)
    stage_msa(ctx, mmseqs=args.mmseqs, uniref_fasta=args.uniref_fasta, uniref_db=args.uniref_db, threads=args.threads,
              sensitivity=args.sensitivity, iterations=args.iterations, max_seqs=args.max_seqs, all_chains=args.all_chains,
              target=args.target)
    return 0


def cmd_assemble(args: argparse.Namespace) -> int:
    ctx = Ctx(root=Path(args.root), email=args.email, dry_run=args.dry_run)
    fractions = tuple(float(x) for x in args.split.split(","))
    if len(fractions) != 4 or abs(sum(fractions) - 1.0) > 1e-8:
        raise SystemExit("--split must be four comma-separated fractions train,val,test,ood summing to 1")
    report = stage_assemble(ctx, dataset_version=args.dataset_version, annotate=args.annotate, all_chains=args.all_chains,
                            max_chains=args.max_chains, shard_size=args.shard_size, split_fractions=fractions,
                            ood_dimension=args.ood_dimension)
    print(json.dumps(report, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--email", help="contact address sent in the User-Agent/From headers (recommended by NCBI/UniProt)")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("plan", help="print the bill of materials")
    p.add_argument("--live", action="store_true", help="query RCSB for live entry counts")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_plan)

    p = sub.add_parser("verify", help="probe every URL and API once (no bulk download)")
    p.set_defaults(func=cmd_verify)

    p = sub.add_parser("fetch", help="download source data (resumable)")
    p.add_argument("--root", required=True)
    p.add_argument("--stages", nargs="+", default=["light"], help="stage names or groups: light, all")
    p.add_argument("--profiles", nargs="+", default=["nrps-pfam", "nrps-text"],
                   help="structure profiles: nrps-pfam, nrps-text, adenylate-family, general")
    p.add_argument("--max-structures", type=int)
    p.add_argument("--cluster-representatives", type=int, choices=[30, 90],
                   help="keep one entry per RCSB sequence cluster at this identity (needs the clusters stage); "
                        "cuts the general profile several-fold")
    p.add_argument("--batch-size", type=int, default=500)
    p.add_argument("--assembly", help="also fetch biological assembly N (default: asymmetric unit only)")
    p.add_argument("--refresh", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--yes", action="store_true", help="confirm very large downloads")
    p.set_defaults(func=cmd_fetch)

    p = sub.add_parser("msa", help="homology search (MMseqs2) for every unique chain sequence")
    p.add_argument("--root", required=True)
    p.add_argument("--mmseqs", default="mmseqs")
    p.add_argument("--uniref-db", type=Path, help="existing MMseqs2 database (skips createdb/createindex)")
    p.add_argument("--uniref-fasta", type=Path, help="FASTA(.gz) to build the database from")
    p.add_argument("--target", choices=["nrps", "uniref90"], default="nrps",
                   help="search the small UniProt NRPS-family set (default) or the full UniRef90")
    p.add_argument("--threads", type=int, default=os.cpu_count() or 4)
    p.add_argument("--sensitivity", type=float, default=7.5)
    p.add_argument("--iterations", type=int, default=3)
    p.add_argument("--max-seqs", type=int, default=10000)
    p.add_argument("--all-chains", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_msa)

    p = sub.add_parser("assemble", help="run the repo's assembly line and write immutable, sharded datasets")
    p.add_argument("--root", required=True)
    p.add_argument("--dataset-version", required=True)
    p.add_argument("--annotate", choices=["none", "nrps", "all"], default="nrps",
                   help="UniProt/InterPro API annotation (1 req/s each); 'nrps' = NRPS profiles only")
    p.add_argument("--split", default="0.7,0.1,0.1,0.1", help="train,validation,test,ood fractions")
    p.add_argument("--ood-dimension", default="nrps_family")
    p.add_argument("--shard-size", type=int, default=2000)
    p.add_argument("--max-chains", type=int)
    p.add_argument("--all-chains", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_assemble)

    p = sub.add_parser("status", help="summarise a corpus directory")
    p.add_argument("--root", required=True)
    p.set_defaults(func=cmd_status)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
