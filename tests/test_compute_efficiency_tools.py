from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import torch

from chimera.training import gradient_flow_report


ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _record(offset: int):
    return {
        "msa_tokens": [[[1 + offset, 2, 3], [1, 4, 5]]],
        "pair_features": [[[[0.0] * 16 for _ in range(3)] for _ in range(3)]],
        "residue_mask": [[True, True, True]],
    }


def test_streaming_jsonl_loader_batches_records_without_loading_all_rows(tmp_path):
    runner = _load_script("train_canonical.py")
    path = tmp_path / "train.jsonl"
    path.write_text(
        "\n".join(json.dumps(_record(index)) for index in range(4)) + "\n",
        encoding="utf-8",
    )

    batches = list(runner._iter_batches(path, batch_size=2))

    assert len(batches) == 2
    assert batches[0].msa_tokens.shape == (2, 2, 3)
    assert batches[0].pair_features.shape == (2, 3, 3, 16)
    assert batches[1].msa_tokens[0, 0, 0].item() == 3


def test_missing_pair_features_are_materialized_from_model_width():
    runner = _load_script("train_canonical.py")
    record = {"msa_tokens": [[[1, 2, 3], [1, 4, 5]]], "residue_mask": [[True, True, True]]}

    batch = runner._batch(record, pair_width=16)

    assert batch.pair_features.shape == (1, 3, 3, 16)
    assert torch.count_nonzero(batch.pair_features).item() == 0


def test_distributed_jsonl_sharding_yields_equal_rank_batches(tmp_path):
    runner = _load_script("train_canonical.py")
    path = tmp_path / "train.jsonl"
    path.write_text(
        "\n".join(json.dumps(_record(index)) for index in range(8)) + "\n",
        encoding="utf-8",
    )

    rank_zero = list(runner._iter_batches(path, 1, rank=0, world_size=2))
    rank_one = list(runner._iter_batches(path, 1, rank=1, world_size=2))

    assert len(rank_zero) == len(rank_one) == 4
    assert rank_zero[0].msa_tokens[0, 0, 0].item() == 1
    assert rank_one[0].msa_tokens[0, 0, 0].item() == 2


def test_fast_gradient_report_preserves_component_contract_without_parameter_rows():
    module = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.Linear(8, 2))
    module(torch.ones(3, 4)).sum().backward()

    report = gradient_flow_report(module, detailed=False)

    assert report["parameters"] == []
    assert report["components"]["0"]["gradient_parameters"] > 0
    assert report["components"]["1"]["gradient_parameters"] > 0
    assert report["components"]["0"]["all_gradients_finite"] is True


def test_compute_estimator_counts_pair_representation_and_dominant_macs():
    profiler = _load_script("profile_chimera_compute.py")
    result = profiler.estimate(
        batch=1, msa_depth=128, length=600, d_msa=256, d_pair=128,
        c_hidden_opm=32, c_hidden_triangle=128, n_evo_blocks=138,
        dtype_bytes=4,
    )

    assert result["representation_tensors"]["pair"]["elements"] == 600 * 600 * 128
    assert result["dominant_operations"]["combined_MACs_per_example_per_block"] > 100_000_000_000
