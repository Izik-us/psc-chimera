"""Optional structural retrieval with explicit embedding/index provenance."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn


class StructuralRetriever(nn.Module):
    """Retrieval-Augmented Generation for protein structure design.

    The retriever owns the runtime index, but deliberately refuses to pretend
    that arbitrary structural embeddings are compatible with its substrate
    query encoder. Query/index alignment must be trained and identified by the
    caller before retrieval is enabled in the canonical model.
    """

    INDEX_SCHEMA_VERSION = 1

    def __init__(
        self,
        d_embed: int = 128,
        d_context: int = 512,
        n_retrieve: int = 5,
    ):
        super().__init__()
        if d_embed <= 0 or d_context <= 0:
            raise ValueError("d_embed and d_context must be positive")
        if n_retrieve <= 0:
            raise ValueError("n_retrieve must be positive")
        self.d_embed = d_embed
        self.n_retrieve = n_retrieve
        self.substrate_encoder = nn.Sequential(
            nn.Embedding(25, 64),
            nn.LSTM(64, d_embed // 2, batch_first=True, bidirectional=True),
        )
        self.context_encoder = nn.Sequential(
            nn.Linear(10 * 3, d_embed),
            nn.LayerNorm(d_embed),
            nn.GELU(),
            nn.Linear(d_embed, d_context),
        )
        self.retrieval_cross_attn = nn.MultiheadAttention(
            embed_dim=d_context,
            num_heads=8,
            batch_first=True,
        )
        self.retrieval_norm = nn.LayerNorm(d_context)
        self.index = None
        self.index_embs = None
        self.index_meta: list[dict] = []
        self.index_identity: str | None = None
        self.index_encoder_identity: str | None = None

    @staticmethod
    def _metadata_identity(metadata: List[dict]) -> str:
        encoded = json.dumps(metadata, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def build_index(
        self,
        structure_embeddings: np.ndarray,
        metadata: List[dict],
        *,
        encoder_identity: str | None = None,
    ):
        """Build an index from precomputed embeddings and auditable metadata."""
        structure_embeddings = np.asarray(structure_embeddings, dtype=np.float32)
        if structure_embeddings.ndim != 2 or structure_embeddings.shape[1] != self.d_embed:
            raise ValueError(f"structure_embeddings must have shape (N, {self.d_embed})")
        if len(metadata) != structure_embeddings.shape[0]:
            raise ValueError("metadata count must match structure embedding count")
        if not np.isfinite(structure_embeddings).all():
            raise ValueError("structure_embeddings must be finite")
        for index, item in enumerate(metadata):
            if not isinstance(item, dict):
                raise ValueError(f"metadata[{index}] must be a mapping")
            for key in ("pdb_id", "substrate", "stachelhaus_code", "pocket_coords"):
                if key not in item:
                    raise ValueError(f"metadata[{index}] is missing required field {key!r}")
            coords = np.asarray(item["pocket_coords"], dtype=np.float32)
            if coords.shape != (10, 3) or not np.isfinite(coords).all():
                raise ValueError(f"metadata[{index}].pocket_coords must have finite shape (10,3)")
        self.index = None
        self.index_embs = structure_embeddings
        self.index_meta = [dict(item) for item in metadata]
        self.index_encoder_identity = encoder_identity
        self.index_identity = self._metadata_identity(self.index_meta + [
            {"embedding_sha256": hashlib.sha256(structure_embeddings.tobytes()).hexdigest()}
        ])
        if structure_embeddings.shape[0] == 0:
            return
        try:
            import faiss
            self.index = faiss.IndexFlatL2(self.d_embed)
            self.index.add(structure_embeddings)
        except ImportError:
            self.index_embs = torch.tensor(structure_embeddings)

    def save_index(self, path: str | Path) -> str:
        """Persist embeddings, metadata, and provenance as one deterministic artifact."""
        if self.index_embs is None:
            raise RuntimeError("cannot save an unconfigured retrieval index")
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        manifest = {
            "schema_version": self.INDEX_SCHEMA_VERSION,
            "d_embed": self.d_embed,
            "n_retrieve": self.n_retrieve,
            "index_identity": self.index_identity,
            "encoder_identity": self.index_encoder_identity,
        }
        with target.open("wb") as stream:
            np.savez_compressed(
                stream,
                embeddings=np.asarray(self.index_embs, dtype=np.float32),
                metadata=np.asarray(json.dumps(self.index_meta, sort_keys=True), dtype=np.str_),
                manifest=np.asarray(json.dumps(manifest, sort_keys=True), dtype=np.str_),
            )
        return hashlib.sha256(target.read_bytes()).hexdigest()

    def load_index(self, path: str | Path, *, expected_encoder_identity: str | None = None) -> str:
        """Load a persisted index and reject incompatible encoder provenance."""
        source = Path(path)
        if not source.exists():
            raise FileNotFoundError(source)
        with np.load(source, allow_pickle=False) as archive:
            embeddings = np.asarray(archive["embeddings"], dtype=np.float32)
            manifest = json.loads(str(archive["manifest"].item()))
            metadata = json.loads(str(archive["metadata"].item()))
        if manifest.get("schema_version") != self.INDEX_SCHEMA_VERSION:
            raise RuntimeError("unsupported retrieval index schema")
        if int(manifest.get("d_embed", -1)) != self.d_embed:
            raise RuntimeError("retrieval index embedding dimension is incompatible")
        encoder_identity = manifest.get("encoder_identity")
        if expected_encoder_identity is not None and encoder_identity != expected_encoder_identity:
            raise RuntimeError("retrieval index was built by a different query/index encoder identity")
        self.build_index(embeddings, metadata, encoder_identity=encoder_identity)
        if manifest.get("index_identity") != self.index_identity:
            raise RuntimeError("retrieval index integrity identity mismatch")
        return hashlib.sha256(source.read_bytes()).hexdigest()

    def retrieve(self, query_embedding: torch.Tensor) -> Tuple[List[List[dict]], torch.Tensor | None]:
        """Retrieve K nearest A-domain structures and their pocket coordinates."""
        if query_embedding.ndim != 2 or query_embedding.shape[1] != self.d_embed:
            raise ValueError(f"query_embedding must have shape (B, {self.d_embed})")
        if not torch.isfinite(query_embedding).all():
            raise ValueError("query_embedding must be finite")
        batch = query_embedding.shape[0]
        if batch == 0:
            return [], None
        if self.index is not None:
            database_size = int(self.index.ntotal)
        elif self.index_embs is not None:
            database_size = int(self.index_embs.shape[0])
        else:
            return [[] for _ in range(batch)], None
        if database_size == 0:
            return [[] for _ in range(batch)], None
        if database_size != len(self.index_meta):
            raise RuntimeError("retrieval index metadata count does not match database size")
        k = min(self.n_retrieve, database_size)
        query_np = query_embedding.detach().cpu().numpy().astype(np.float32)
        if self.index is not None:
            _, indices = self.index.search(query_np, k)
        else:
            distances = torch.cdist(query_embedding.cpu(), self.index_embs.float())
            indices = distances.topk(k, dim=-1, largest=False).indices.numpy()
        if np.any(indices < 0):
            raise RuntimeError("retrieval backend returned an invalid negative index")
        all_meta = []
        all_coords = []
        for row in indices:
            batch_meta = [self.index_meta[int(i)] for i in row]
            batch_coords = torch.stack([torch.tensor(meta["pocket_coords"]) for meta in batch_meta])
            all_meta.append(batch_meta)
            all_coords.append(batch_coords)
        coords_tensor = torch.stack(all_coords)
        return all_meta, coords_tensor.to(query_embedding.device)

    def encode_query(self, substrate_tokens: torch.Tensor) -> torch.Tensor:
        """Encode substrate tokens; caller must prove alignment with the index."""
        if substrate_tokens.ndim != 2:
            raise ValueError("substrate_tokens must have shape (B, S)")
        emb = self.substrate_encoder[0](substrate_tokens)
        _, (hidden, _) = self.substrate_encoder[1](emb)
        return hidden.permute(1, 0, 2).reshape(substrate_tokens.shape[0], -1)[:, : self.d_embed]

    def forward(
        self,
        current_design_repr: torch.Tensor,
        substrate_tokens: torch.Tensor,
        *,
        query_embedding: torch.Tensor | None = None,
        encoder_identity: str | None = None,
    ) -> torch.Tensor:
        """Condition a design representation on retrieved structures.

        Retrieval is fail-closed. The provisional substrate encoder cannot be
        used against an arbitrary index. Production callers must supply an
        aligned query embedding and matching encoder identity.
        """
        if current_design_repr.ndim != 3:
            raise ValueError("current_design_repr must have shape (B,L,d_context)")
        if substrate_tokens.ndim != 2 or substrate_tokens.shape[0] != current_design_repr.shape[0]:
            raise ValueError("substrate_tokens must have shape (B,S) matching current_design_repr")
        if self.index_embs is None or len(self.index_meta) == 0:
            raise RuntimeError("RAG_UNAVAILABLE: structural retrieval index is not configured")
        if query_embedding is None:
            raise RuntimeError("RAG_UNAVAILABLE: an explicitly aligned query embedding is required")
        if self.index_encoder_identity is None or encoder_identity != self.index_encoder_identity:
            raise RuntimeError("RAG_UNAVAILABLE: query/index encoder identity mismatch")
        _, pocket_coords = self.retrieve(query_embedding)
        if pocket_coords is None:
            raise RuntimeError("RAG_UNAVAILABLE: no structures were retrieved")
        batch, k, _, _ = pocket_coords.shape
        context = self.context_encoder(pocket_coords.reshape(batch, k, -1))
        attended, _ = self.retrieval_cross_attn(current_design_repr, context, context)
        return self.retrieval_norm(current_design_repr + attended)
