"""Optional structural retrieval; embedding alignment is a caller responsibility."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn


class StructuralRetriever(nn.Module):
    """
    Retrieval-Augmented Generation for protein structure design.

    Maintains an embedding index of all known A-domain structures.
    At inference, queries by substrate type → retrieves K most similar
    known A-domain binding pockets → conditions CHIMERA generation on them.

    This dramatically reduces the generation task: instead of hallucinating
    a novel A-domain geometry from scratch, CHIMERA adapts the closest
    known structure toward the desired substrate and mammalian compatibility.

    Embedding-space limitation:
        ``encode_query`` currently uses a substrate-token Embedding + LSTM.
        Externally supplied structural vectors may come from unrelated
        encoders, and no trained alignment is provided here. Retrieval is
        therefore a provisional API only; callers must establish compatible
        query/index embeddings before interpreting nearest neighbors.
    """

    def __init__(
        self,
        d_embed: int = 256,
        d_context: int = 256,
        n_retrieve: int = 5,
    ):
        super().__init__()
        if d_embed <= 0 or d_context <= 0:
            raise ValueError("d_embed and d_context must be positive")
        if n_retrieve <= 0:
            raise ValueError("n_retrieve must be positive")
        self.d_embed = d_embed
        self.n_retrieve = n_retrieve

        # Provisional query encoder. It is not aligned to arbitrary structure vectors.
        self.substrate_encoder = nn.Sequential(
            nn.Embedding(25, 64),  # 20 AA + 5 non-standard
            nn.LSTM(64, d_embed // 2, batch_first=True, bidirectional=True),
        )

        # Context encoder: retrieved structure → conditioning vector
        # Takes backbone coordinates of retrieved A-domain pocket (K=10 key residues)
        self.context_encoder = nn.Sequential(
            nn.Linear(10 * 3, d_embed),  # 10 Stachelhaus positions × 3D coords
            nn.LayerNorm(d_embed),
            nn.GELU(),
            nn.Linear(d_embed, d_context),
        )

        # Cross-attention: current design queries retrieved structures
        self.retrieval_cross_attn = nn.MultiheadAttention(
            embed_dim=d_context,
            num_heads=8,
            batch_first=True,
        )
        self.retrieval_norm = nn.LayerNorm(d_context)

        # The actual index lives in CPU memory (FAISS)
        self.index = None  # set via build_index()
        self.index_embs = None  # (N_structures, d_embed) numpy array
        self.index_meta = []  # list of dicts: pdb_id, substrate, coords

    def build_index(self, structure_embeddings: np.ndarray, metadata: List[dict]):
        """
        Build FAISS flat L2 index from pre-computed structure embeddings.
        Call this once during setup after processing all PDB A-domain structures.

        Args:
            structure_embeddings: (N, d_embed) numpy array
            metadata: list of {pdb_id, substrate, stachelhaus_code, pocket_coords}
        """
        structure_embeddings = np.asarray(structure_embeddings, dtype=np.float32)
        if structure_embeddings.ndim != 2 or structure_embeddings.shape[1] != self.d_embed:
            raise ValueError(
                f"structure_embeddings must have shape (N, {self.d_embed})"
            )
        if len(metadata) != structure_embeddings.shape[0]:
            raise ValueError("metadata count must match structure embedding count")
        if not np.isfinite(structure_embeddings).all():
            raise ValueError("structure_embeddings must be finite")
        self.index = None
        self.index_embs = structure_embeddings
        self.index_meta = metadata
        if structure_embeddings.shape[0] == 0:
            return

        try:
            import faiss

            self.index = faiss.IndexFlatL2(self.d_embed)
            self.index.add(structure_embeddings)
            print(f"[StructuralRetriever] Index built: {len(metadata)} structures")
        except ImportError:
            print("[StructuralRetriever] FAISS not installed — using brute force")
            self.index_embs = torch.tensor(structure_embeddings)

    def retrieve(
        self,
        query_embedding: torch.Tensor,  # (B, d_embed) substrate query
    ) -> Tuple[List[List[dict]], torch.Tensor]:
        """
        Retrieve K most similar known A-domain structures for each batch element.
        Returns metadata list + pocket coordinate tensors for cross-attention.
        """
        if query_embedding.ndim != 2 or query_embedding.shape[1] != self.d_embed:
            raise ValueError(f"query_embedding must have shape (B, {self.d_embed})")
        if not torch.isfinite(query_embedding).all():
            raise ValueError("query_embedding must be finite")
        B = query_embedding.shape[0]
        if B == 0:
            return [], None
        q_np = query_embedding.detach().cpu().numpy().astype(np.float32)

        if self.index is not None:
            database_size = int(self.index.ntotal)
        elif self.index_embs is not None:
            database_size = int(self.index_embs.shape[0])
        else:
            return [[] for _ in range(B)], None
        if database_size == 0:
            return [[] for _ in range(B)], None
        if database_size != len(self.index_meta):
            raise RuntimeError("retrieval index metadata count does not match database size")
        k = min(self.n_retrieve, database_size)

        if self.index is not None:
            import faiss

            _, indices = self.index.search(q_np, k)
        elif self.index_embs is not None:
            # Brute force fallback
            dists = torch.cdist(query_embedding.cpu(), self.index_embs.float())
            indices = dists.topk(k, dim=-1, largest=False).indices.numpy()

        if np.any(indices < 0):
            raise RuntimeError("retrieval backend returned an invalid negative index")

        # Gather retrieved pocket coordinates
        all_meta = []
        all_coords = []
        for b in range(B):
            batch_meta = [self.index_meta[i] for i in indices[b]]
            batch_coords = torch.stack(
                [torch.tensor(meta["pocket_coords"]) for meta in batch_meta]
            )  # (K, 10, 3)
            all_meta.append(batch_meta)
            all_coords.append(batch_coords)

        coords_tensor = torch.stack(all_coords)  # (B, K, 10, 3)
        return all_meta, coords_tensor.to(query_embedding.device)

    def encode_query(self, substrate_tokens: torch.Tensor) -> torch.Tensor:
        """Encode substrate tokens into the index embedding space."""
        if substrate_tokens.ndim != 2:
            raise ValueError("substrate_tokens must have shape (B, S)")
        emb = self.substrate_encoder[0](substrate_tokens)
        _, (hidden, _) = self.substrate_encoder[1](emb)
        return hidden.permute(1, 0, 2).reshape(substrate_tokens.shape[0], -1)[
            :, : self.d_embed
        ]

    def forward(
        self,
        current_design_repr: torch.Tensor,  # (B, L, d_context)
        substrate_tokens: torch.Tensor,  # (B, S) amino acid tokens for substrate
    ) -> torch.Tensor:
        """
        Enrich current design representation with retrieved structural analogs.
        Returns: (B, L, d_context) enriched representation
        """
        del current_design_repr, substrate_tokens
        if self.index_embs is None or self.index_embs.shape[0] == 0:
            raise RuntimeError("RAG_UNAVAILABLE: structural retrieval index is not configured")
        raise RuntimeError(
            "RAG_UNAVAILABLE: substrate query encoder is not trained into the supplied index embedding space"
        )
