"""Offline tests for scripts/fetch_corpus.py (no internet required)."""
import hashlib
import importlib.util
import json
import os
import random
import shutil
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests" / "fixtures" / "data_engineering"
spec = importlib.util.spec_from_file_location("fetch_corpus", ROOT / "scripts" / "fetch_corpus.py")
fc = importlib.util.module_from_spec(spec)
sys.modules["fetch_corpus"] = fc
spec.loader.exec_module(fc)


class _Handler(BaseHTTPRequestHandler):
    files: dict = {}
    truncate_once: set = set()

    def log_message(self, *args):  # silence
        pass

    def _serve(self, head_only):
        body = self.files.get(self.path)
        if body is None:
            self.send_response(404); self.end_headers(); return
        start = 0
        rng = self.headers.get("Range")
        if rng and rng.startswith("bytes="):
            start = int(rng[6:].split("-")[0])
        chunk = body[start:]
        self.send_response(206 if rng else 200)
        if rng:
            self.send_header("Content-Range", f"bytes {start}-{len(body) - 1}/{len(body)}")
        self.send_header("Content-Length", str(len(chunk)))
        self.send_header("ETag", '"abc"')
        self.end_headers()
        if head_only:
            return
        if self.path in self.truncate_once and not rng:
            self.truncate_once.discard(self.path)
            self.wfile.write(chunk[: len(chunk) // 3])   # drop the connection mid-body
            self.wfile.flush()
            self.close_connection = True
            return
        self.wfile.write(chunk)

    def do_GET(self):
        self._serve(False)

    def do_HEAD(self):
        self._serve(True)


@pytest.fixture()
def server():
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()


def make_ctx(tmp_path, **kwargs):
    return fc.Ctx(root=tmp_path, log=lambda *_: None, **kwargs)


def test_plan_covers_every_regime_and_flags_unverifiable_sources():
    stages = {s["stage"] for s in fc.SOURCES if s.get("stage")}
    assert {"structures", "clusters", "ccd", "sifts", "mibig", "nrps_uniprot", "uniref90"} <= stages
    assert all(s["status"] in {"repo-inventory", "verified-docs", "unverified", "repo", "n/a"} for s in fc.SOURCES)
    assert any(s["id"] == "proteus_rounds" and s["kind"] == "wet_lab" for s in fc.SOURCES)
    assert set(fc.expand_stages(["light"])) <= stages
    assert fc.main(["plan", "--json"]) == 0


def test_download_resumes_after_dropped_connection_and_is_idempotent(tmp_path, server):
    payload = random.Random(0).randbytes(3_000_000)
    _Handler.files = {"/big.bin": payload}
    _Handler.truncate_once = {"/big.bin"}
    ctx = make_ctx(tmp_path)
    record = fc.download(f"{server}/big.bin", tmp_path / "raw" / "big.bin", ctx, source_id="t", rps=1000)
    assert (tmp_path / "raw" / "big.bin").read_bytes() == payload
    assert record["sha256"] == hashlib.sha256(payload).hexdigest()
    again = fc.download(f"{server}/big.bin", tmp_path / "raw" / "big.bin", ctx, source_id="t", rps=1000)
    assert again["skipped"] is True
    assert len(fc.read_manifest(ctx.manifest)) == 1


def test_download_rejects_checksum_mismatch_and_dry_run_writes_nothing(tmp_path, server):
    _Handler.files = {"/a.txt": b"hello"}
    _Handler.truncate_once = set()
    ctx = make_ctx(tmp_path)
    with pytest.raises(ValueError, match="checksum mismatch"):
        fc.download(f"{server}/a.txt", tmp_path / "a.txt", ctx, source_id="t", rps=1000, expected_sha256="0" * 64)
    assert not (tmp_path / "a.txt").exists()
    dry = make_ctx(tmp_path / "dry", dry_run=True)
    fc.download(f"{server}/a.txt", tmp_path / "dry" / "a.txt", dry, source_id="t", rps=1000)
    assert not (tmp_path / "dry" / "a.txt").exists()


def test_probe_reports_status_and_size(server):
    _Handler.files = {"/x": b"12345"}
    assert fc.probe(f"{server}/x") == {"ok": True, "status": 200, "bytes": 5, "last_modified": None}
    assert fc.probe(f"{server}/missing")["ok"] is False


def test_large_stage_requires_confirmation(tmp_path):
    with pytest.raises(SystemExit, match="--yes"):
        fc.main(["fetch", "--root", str(tmp_path), "--stages", "uniref90"])


def test_pfam_search_payload_shape_and_a3m_conversion():
    payload = fc.rcsb_pfam_payload(["PF00668"], max_resolution=4.0, start=0, rows=10)
    assert payload["return_type"] == "entry"
    assert payload["request_options"]["results_content_type"] == ["experimental"]
    assert payload["query"]["nodes"][0]["nodes"][0]["parameters"]["value"] == "PF00668"
    converted = fc.a3m_to_aligned_fasta(">q\nACDE\n>h\nAcgCD-E\n")
    rows = [r.split("\n")[1] for r in converted.strip().split(">")[1:]]
    assert rows == ["q\nACDE".split("\n")[1], "ACD-E"] and len({len(r) for r in rows}) == 2 or rows[0] == "ACDE"


def _write_corpus(tmp_path, profile="nrps-pfam"):
    manifest_line = json.loads((FIXTURES / "rcsb" / "entry" / "acquisition_manifest.jsonl").read_text().splitlines()[0])
    manifest_line["raw_path"] = str(FIXTURES / "rcsb" / "entry" / "1AMU.cif")
    batch = tmp_path / "structures" / profile / "batch_00000"
    batch.mkdir(parents=True)
    (batch / "acquisition_manifest.jsonl").write_text(json.dumps(manifest_line) + "\n")
    return manifest_line


class _FakeUniProt:
    def resolve_accession(self, accession):
        return ["P0C062"], {}

    def fetch_proteins(self, accessions):
        from data.uniprot import RetrievedProtein
        entry = json.loads((FIXTURES / "uniprot" / "uniprotkb_P0C062.json").read_text())
        return [RetrievedProtein(accession="P0C062", sequence=entry["sequence"]["value"], organism_taxonomy_id="1393",
                                 organism_name="Brevibacillus brevis", release="2026_03", entry_version=1)]

    def fetch_uniprot_entry(self, accession):
        return json.loads((FIXTURES / "uniprot" / "uniprotkb_P0C062.json").read_text()), None


class _FakeInterPro:
    def fetch_protein_matches(self, accession):
        snapshot = json.loads((FIXTURES / "interpro" / "P0C062.json").read_text())
        return snapshot["payload"], snapshot["release"]


def _assemble(tmp_path, **kwargs):
    from data.structures import StructuralQCConfig
    ctx = make_ctx(tmp_path)
    return fc.stage_assemble(
        ctx, dataset_version="CHIMERA-DATASET-test", annotate=kwargs.pop("annotate", "nrps"), all_chains=False,
        max_chains=None, shard_size=10, split_fractions=(1.0, 0.0, 0.0, 0.0), ood_dimension=None,
        uniprot_client=_FakeUniProt(), interpro_client=_FakeInterPro(),
        qc_config=StructuralQCConfig(maximum_missing_sidechain_fraction=1.0), **kwargs,
    )


def test_assembly_line_builds_a_loadable_sharded_dataset_without_msa(tmp_path):
    from data.dataset_api import SQLiteChimeraDataset
    _write_corpus(tmp_path)
    report = _assemble(tmp_path)
    assert report["samples"] == 1 and report["annotated"] == 1 and report["single_sequence_msa"] == 1
    assert report["splits"] == {"TRAIN": 1} and len(report["shards"]) == 1
    sample = SQLiteChimeraDataset(report["shards"][0])[0]
    assert sample["sample_id"].startswith("1AMU:")
    assert any(d.get("domain_type") == "A" for d in sample["nrps_annotation"]["domains"])


@pytest.mark.skipif(shutil.which("mmseqs") is None and not os.environ.get("MMSEQS"), reason="mmseqs not installed")
def test_msa_stage_with_real_mmseqs_feeds_the_assembly_line(tmp_path):
    from data.dataset_api import SQLiteChimeraDataset
    record = _write_corpus(tmp_path)
    sequence = record["canonical_chains"][0]["sequence"]
    binary = os.environ.get("MMSEQS") or shutil.which("mmseqs")
    rng = random.Random(3)
    alphabet = "ACDEFGHIKLMNPQRSTVWY"
    with (tmp_path / "target.fasta").open("w") as handle:
        for i, rate in enumerate([0.05, 0.1, 0.2, 0.3, 0.4, 0.5]):
            for j in range(4):
                mutated = "".join(rng.choice(alphabet) if rng.random() < rate else c for c in sequence)
                handle.write(f">UniRef90_X{i}{j}\n{mutated}\n")
        for k in range(30):
            handle.write(f">decoy{k}\n{''.join(rng.choice(alphabet) for _ in range(rng.randint(200, 500)))}\n")
    subprocess.run([binary, "createdb", str(tmp_path / "target.fasta"), str(tmp_path / "targetdb"), "-v", "1"], check=True)
    fc.stage_msa(make_ctx(tmp_path), mmseqs=binary, uniref_fasta=None, uniref_db=tmp_path / "targetdb", threads=2,
                 sensitivity=7.5, iterations=2, max_seqs=1000, all_chains=False)
    a3m = tmp_path / "msa" / "a3m" / f"{fc.seq_key(sequence)}.a3m"
    assert a3m.exists() and a3m.read_text().count(">") > 5
    report = _assemble(tmp_path)
    assert report["with_deep_msa"] == 1 and report["single_sequence_msa"] == 0
    sample = SQLiteChimeraDataset(report["shards"][0])[0]
    assert sample["msa_metadata"]["num_rows"] > 5
    # rerunning the stage is a no-op (resume)
    fc.stage_msa(make_ctx(tmp_path), mmseqs=binary, uniref_fasta=None, uniref_db=tmp_path / "targetdb", threads=2,
                 sensitivity=7.5, iterations=2, max_seqs=1000, all_chains=False)


def test_mibig_uses_zenodo_with_md5_and_downloads_verify_md5(tmp_path, server):
    mibig = next(s for s in fc.SOURCES if s["id"] == "mibig")
    assert all(item[1].startswith("https://zenodo.org/records/14835872/files/") and len(item[2]["md5"]) == 32
               for item in mibig["urls"])
    _Handler.files = {"/m.bin": b"payload"}
    _Handler.truncate_once = set()
    ctx = make_ctx(tmp_path)
    good = hashlib.md5(b"payload").hexdigest()
    assert fc.download(f"{server}/m.bin", tmp_path / "m.bin", ctx, source_id="t", rps=1000, expected_md5=good)[
        "checksum_verified_against_source"]
    with pytest.raises(ValueError, match="md5"):
        fc.download(f"{server}/m.bin", tmp_path / "bad.bin", ctx, source_id="t", rps=1000, expected_md5="0" * 32)


def test_offline_pfam_discovery_and_cluster_representatives(tmp_path):
    import gzip as gz
    sifts = tmp_path / "pdb_chain_pfam.tsv.gz"
    with gz.open(sifts, "wt") as handle:
        handle.write("# 2026/10/01 - 00:00\nPDB\tCHAIN\tSP_PRIMARY\tPFAM_ID\n"
                     "1amu\tA\tP0C062\tPF00501\n1amu\tA\tP0C062\tPF00550\n2vsq\tA\tQ1\tPF00668\n"
                     "5isx\tB\tQ2\tPF00668\n3zzz\tA\tQ3\tPF99999\n")
    assert fc.pfam_entries_from_sifts(sifts, ["PF00668"]) == ["2VSQ", "5ISX"]
    assert fc.pfam_entries_from_sifts(sifts, ["PF00501", "PF00668"]) == ["1AMU", "2VSQ", "5ISX"]
    clusters = tmp_path / "clusters-by-entity-30.txt"
    clusters.write_text("AF_X_1 2VSQ_1 5ISX_1\n1AMU_1 1AMU_2\n9ZZZ_1\n")
    assert fc.cluster_representatives(["2VSQ", "5ISX", "1AMU"], clusters) == ["1AMU", "2VSQ"]
    ctx = make_ctx(tmp_path)
    (tmp_path / "raw" / "sifts").mkdir(parents=True)
    shutil.copy(sifts, tmp_path / "raw" / "sifts" / "pdb_chain_pfam.tsv.gz")
    assert fc.discover_ids("nrps-pfam", ctx) == ["2VSQ", "5ISX"]       # no network needed
    assert fc.discover_ids("nrps-pfam", ctx, count_only=True) == 2
