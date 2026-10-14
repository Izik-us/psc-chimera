import numpy as np
import pytest
import torch

from chimera.retrieval import StructuralRetriever


def _metadata(n):
    return [
        {
            "pdb_id": f"TEST{i}",
            "substrate": "ALA",
            "stachelhaus_code": "CODE1234",
            "pocket_coords": np.zeros((10, 3), dtype=np.float32).tolist(),
        }
        for i in range(n)
    ]


def test_retrieval_index_round_trip(tmp_path):
    retriever = StructuralRetriever(d_embed=8, d_context=16, n_retrieve=2)
    embeddings = np.arange(24, dtype=np.float32).reshape(3, 8)
    retriever.build_index(embeddings, _metadata(3), encoder_identity="encoder-v1")
    path = tmp_path / "retrieval.npz"
    digest = retriever.save_index(path)

    restored = StructuralRetriever(d_embed=8, d_context=16, n_retrieve=2)
    loaded_digest = restored.load_index(path, expected_encoder_identity="encoder-v1")
    assert digest == loaded_digest
    assert restored.index_identity == retriever.index_identity
    metadata, coords = restored.retrieve(torch.tensor(embeddings[:1]))
    assert len(metadata[0]) == 2
    assert coords.shape == (1, 2, 10, 3)


def test_retrieval_rejects_encoder_identity_mismatch(tmp_path):
    retriever = StructuralRetriever(d_embed=8, d_context=16)
    retriever.build_index(np.ones((1, 8), dtype=np.float32), _metadata(1), encoder_identity="encoder-v1")
    path = tmp_path / "retrieval.npz"
    retriever.save_index(path)
    restored = StructuralRetriever(d_embed=8, d_context=16)
    with pytest.raises(RuntimeError, match="encoder identity"):
        restored.load_index(path, expected_encoder_identity="encoder-v2")


def test_forward_is_fail_closed_without_aligned_query():
    retriever = StructuralRetriever(d_embed=8, d_context=16)
    retriever.build_index(np.ones((1, 8), dtype=np.float32), _metadata(1), encoder_identity="encoder-v1")
    with pytest.raises(RuntimeError, match="aligned query embedding"):
        retriever(
            torch.zeros(1, 4, 16),
            torch.zeros(1, 1, dtype=torch.long),
        )
