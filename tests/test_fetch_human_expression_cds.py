from __future__ import annotations

import json
from pathlib import Path

from scripts import fetch_human_expression_cds as downloader


def test_fetch_cds_batch_uses_cache_and_batch_results(tmp_path, monkeypatch):
    cds = "ATG" + "GCT" * 30
    calls = []

    def fake_request(request, attempts=5):
        calls.append(request)
        return json.dumps([{"id": "ENST00000000001", "seq": cds}]).encode()

    monkeypatch.setattr(downloader, "_request_bytes", fake_request)
    result, cache_hits, failures = downloader.fetch_cds_batch(
        ["ENST00000000001"], tmp_path
    )
    assert result == {"ENST00000000001": cds}
    assert cache_hits == 0
    assert failures == 0
    assert len(calls) == 1
    assert (tmp_path / "ENST00000000001.txt").read_text().strip() == cds

    result_again, cache_hits_again, failures_again = downloader.fetch_cds_batch(
        ["ENST00000000001"], tmp_path
    )
    assert result_again == result
    assert cache_hits_again == 1
    assert failures_again == 0
    assert len(calls) == 1


def test_build_records_checkpoints_and_resumes(tmp_path, monkeypatch):
    summary = tmp_path / "summary.txt"
    summary.write_text(
        "Transcript Exp_Meiosis Ensembl.Gene.ID gene.symbol\n"
        "ENST00000000001 10 ENSG00000000001 GENE1\n"
        "ENST00000000002 5 ENSG00000000002 GENE2\n",
        encoding="utf-8",
    )
    output = tmp_path / "records.jsonl"
    cache = tmp_path / "cache"
    cds_by_id = {
        "ENST00000000001": "ATG" + "GCT" * 30,
        "ENST00000000002": "ATG" + "GCC" * 30,
    }
    calls = []

    def fake_batch(transcripts, cache_dir):
        calls.append(list(transcripts))
        return ({key: cds_by_id[key] for key in transcripts}, 0, 0)

    monkeypatch.setattr(downloader, "fetch_cds_batch", fake_batch)
    records = downloader.build_records(summary, cache, output, 10, "Exp_Meiosis", 2)
    assert len(records) == 2
    assert len(output.read_text(encoding="utf-8").splitlines()) == 2
    assert calls == [["ENST00000000001", "ENST00000000002"]]

    resumed = downloader.build_records(summary, cache, output, 10, "Exp_Meiosis", 2)
    assert len(resumed) == 2
    assert len(calls) == 1


def test_retry_after_numeric_header_is_honored():
    class Headers(dict):
        pass

    assert downloader._retry_delay(Headers({"Retry-After": "7"}), 0) == 7.0
