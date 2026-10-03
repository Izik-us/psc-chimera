import math
from importlib.metadata import version
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from chimera import __version__
from chimera.model_store import asset_path, cache_root, verify_asset


def _verified_asset_or_skip(asset_id: str) -> Path:
    verification = verify_asset(asset_id)
    if verification["state"] == "MISSING":
        pytest.skip(f"SKIP: required verified artifact unavailable: {asset_id}")
    assert verification["state"] == "INTEGRITY_VERIFIED", verification
    return asset_path(asset_id)


def _deny_network(*_args, **_kwargs):
    pytest.fail("cached native model inference attempted network access")


def _source_revision() -> dict[str, object]:
    root = Path(__file__).resolve().parents[1]
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        text=True,
    ).strip()
    changes = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=root,
        text=True,
    )
    return {
        "commit": commit,
        "worktree_clean": not bool(changes.strip()),
    }


@pytest.mark.requires_weights
def test_esm2_native_checkpoint_runs_through_codon_optimizer(monkeypatch):
    checkpoint = _verified_asset_or_skip("esm2_t30_150m")
    from chimera import model_store
    from chimera.codon_optimizer import AA_TO_IDX, CodonOptimizer

    monkeypatch.setattr(
        model_store.urllib.request,
        "urlopen",
        _deny_network,
    )
    sequence = "MKTAYIAK"
    optimizer = CodonOptimizer(
        d_model=32,
        n_heads=4,
        n_dec_layers=1,
        dim_ff=64,
        dropout=0.0,
        esm_model_path=str(checkpoint),
        max_protein_length=16,
        max_codons=16,
    ).eval()
    assert optimizer.esm_model is not None
    assert optimizer.esm_model.num_layers == 30
    assert optimizer.esm_model.embed_dim == 640
    assert (
        140_000_000
        <= sum(parameter.numel() for parameter in optimizer.esm_model.parameters())
        <= 160_000_000
    )
    assert all(
        optimizer.esm_alphabet.get_idx(amino_acid) >= 0 for amino_acid in AA_TO_IDX
    )

    tokens = torch.tensor(
        [[AA_TO_IDX[amino_acid] for amino_acid in sequence]],
        dtype=torch.long,
    )
    with torch.no_grad():
        memory, padding_mask = optimizer.encode_protein(
            tokens,
            protein_sequences=[sequence],
        )
        repeated_memory, repeated_mask = optimizer.encode_protein(
            tokens,
            protein_sequences=[sequence],
        )

    assert memory.shape == (1, len(sequence), 32)
    assert padding_mask.shape == (1, len(sequence))
    assert not padding_mask.any()
    assert torch.isfinite(memory).all()
    assert torch.equal(padding_mask, repeated_mask)
    assert torch.allclose(memory, repeated_memory, rtol=1e-5, atol=1e-6)
    evidence = {
        "result": "PASS",
        "test": "test_esm2_native_checkpoint_runs_through_codon_optimizer",
        "software": {
            "version": __version__,
            **_source_revision(),
        },
        "upstream_revision": model_store.DEPENDENCIES["esm2-t30-150m-ur50d"][
            "upstream_revision"
        ],
        "runtime": {
            "fair_esm": version("fair-esm"),
            "torch": torch.__version__,
            "python": sys.version,
        },
        "device": "cpu",
        "input_length": len(sequence),
        "representation_layer": optimizer.esm_model.num_layers,
        "model_embedding_dimension": optimizer.esm_model.embed_dim,
        "chimera_output_shape": list(memory.shape),
        "repeat_tolerance": {"rtol": 1e-5, "atol": 1e-6},
    }
    for asset_id in (
        "esm2_t30_150m",
        "esm2_t30_150m_contact_regression",
    ):
        result = model_store.record_native_status(
            asset_id,
            status="NATIVE_VERIFIED",
            detail="Real fair-esm checkpoint inference passed through the CHIMERA CodonOptimizer encoder.",
            evidence=evidence,
        )
        assert result["native_status"] == "NATIVE_VERIFIED"
        assert result["compatibility"] == "VALID"


def _helical_backbone(length: int) -> torch.Tensor:
    coords = torch.empty((1, length, 4, 3), dtype=torch.float32)
    for index in range(length):
        angle = index * 1.745329252
        ca = torch.tensor([2.3 * math.cos(angle), 2.3 * math.sin(angle), 1.5 * index])
        tangent = torch.tensor([-math.sin(angle), math.cos(angle), 0.0])
        radial = torch.tensor([math.cos(angle), math.sin(angle), 0.0])
        coords[0, index, 0] = (
            ca - 1.2 * radial - 0.2 * tangent - torch.tensor([0.0, 0.0, 0.15])
        )
        coords[0, index, 1] = ca
        coords[0, index, 2] = (
            ca + 1.3 * radial + 0.2 * tangent + torch.tensor([0.0, 0.0, 0.35])
        )
        coords[0, index, 3] = coords[0, index, 2] + 0.8 * torch.tensor(
            [math.cos(angle + 0.5), math.sin(angle + 0.5), 0.2]
        )
    return coords


@pytest.mark.requires_weights
def test_proteinmpnn_native_checkpoint_runs_through_adapter(monkeypatch):
    checkpoint = _verified_asset_or_skip("proteinmpnn_v48_020")
    upstream_source = cache_root().parent / "upstream" / "ProteinMPNN"
    if not (upstream_source / "protein_mpnn_utils.py").is_file():
        pytest.skip("SKIP: exact pinned ProteinMPNN source checkout unavailable")
    from chimera import model_store
    from chimera.adapters import ProteinMPNNAdapter
    from chimera.codon_optimizer import AA_TO_IDX

    monkeypatch.setattr(
        model_store.urllib.request,
        "urlopen",
        _deny_network,
    )
    adapter = ProteinMPNNAdapter(
        checkpoint,
        proteinmpnn_repo=upstream_source,
        temperature=0.5,
    )
    assert len(adapter.model.encoder_layers) == 3
    assert len(adapter.model.decoder_layers) == 3

    length = 8
    fixed_positions = torch.zeros((1, length), dtype=torch.bool)
    fixed_positions[0, 0] = True
    fixed_aas = torch.zeros((1, length), dtype=torch.long)
    sequence, log_probs = adapter.design(
        _helical_backbone(length),
        fixed_positions=fixed_positions,
        fixed_aas=fixed_aas,
    )

    assert sequence.shape == (1, length)
    assert log_probs.shape == (1, length, 21)
    assert sequence[0, 0].item() == AA_TO_IDX["A"]
    assert torch.isfinite(log_probs).all()
    probabilities = log_probs.exp()
    assert torch.allclose(
        probabilities[~fixed_positions].sum(dim=-1),
        torch.ones(length - 1),
        rtol=1e-4,
        atol=1e-5,
    )
    result = model_store.record_native_status(
        "proteinmpnn_v48_020",
        status="NATIVE_VERIFIED",
        detail="The exact pinned ProteinMPNN runtime and checkpoint passed adapter sequence-design inference.",
        evidence={
            "result": "PASS",
            "test": "test_proteinmpnn_native_checkpoint_runs_through_adapter",
            "software": {
                "version": __version__,
                **_source_revision(),
            },
            "upstream_revision": model_store.DEPENDENCIES["proteinmpnn-v-48-020"][
                "upstream_revision"
            ],
            "runtime": {"torch": torch.__version__, "python": sys.version},
            "device": "cpu",
            "input_shape": list(_helical_backbone(length).shape),
            "sequence_shape": list(sequence.shape),
            "log_probabilities_shape": list(log_probs.shape),
            "fixed_position_count": int(fixed_positions.sum()),
        },
    )
    assert result["native_status"] == "NATIVE_VERIFIED"
    assert result["compatibility"] == "VALID"
