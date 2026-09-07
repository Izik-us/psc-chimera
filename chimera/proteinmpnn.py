"""
CHIMERA — ProteinMPNN Module
=============================
Evolutionary-context-aware message-passing neural network for sequence design.

Extends standard ProteinMPNN (Dauparas et al. 2022) with:
  - EvoFormer single_repr concatenated to node features at every position
  - This makes each residue's sequence decision jointly informed by:
    (a) local backbone geometry (existing ProteinMPNN capability)
    (b) evolutionary co-conservation patterns from the animal NRPS family
        (new from CHIMERA's EvoFormer → node_projection connector)

For the PSC NRPS design task:
  - Backbone geometry tells ProteinMPNN what fold the residue is in
  - Evolutionary embedding tells it which residues are catalytically essential
    (conserved across animal NRPS family) vs tunable for new substrate specificity

References:
  Dauparas et al. 2022 (ProteinMPNN) — Science 378:49–56
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional


# ── Protein Graph Construction ────────────────────────────────────────────────


def get_protein_graph(
    t_coords: torch.Tensor, R_frames: torch.Tensor, k_neighbors: int = 32
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build a k-NN graph from Cα coordinates.

    Returns:
        edge_index:   (B, L, K) indices of k nearest neighbors for each residue
        edge_features:(B, L, K, 16) geometric edge features
        edge_mask:    (B, L, K) valid edges (for padding handling)
    """
    B, L, _ = t_coords.shape
    device = t_coords.device

    # Pairwise distances
    diff = t_coords.unsqueeze(2) - t_coords.unsqueeze(1)  # (B, L, L, 3)
    dist = diff.norm(dim=-1)  # (B, L, L)

    # Self-loop distance = inf (exclude self)
    dist_no_self = dist + torch.eye(L, device=device).unsqueeze(0) * 1e9
    # Clamp k to L-1 so short sequences (L < k_neighbors+1) don't crash topk.
    k = max(1, min(k_neighbors, L - 1))
    _, top_k_idx = dist_no_self.topk(k, dim=-1, largest=False)

    # Geometric edge features (ProteinMPNN convention)
    features = _compute_edge_features(t_coords, R_frames, top_k_idx)
    mask = dist_no_self.gather(-1, top_k_idx) < 20.0  # within 20Å

    return top_k_idx, features, mask


def _compute_edge_features(
    t: torch.Tensor, R: torch.Tensor, k_idx: torch.Tensor
) -> torch.Tensor:
    """
    Compute 16-dim edge features between residue pairs:
      - RBF encoding of Cα-Cα distance  (16 Gaussian basis functions)
      - Relative orientation features (from local frames)
    """
    B, L, K = k_idx.shape
    device = t.device

    # Gather neighbor coordinates
    t_j = t.unsqueeze(2).expand(-1, -1, K, -1)
    idx = k_idx.unsqueeze(-1).expand(-1, -1, -1, 3)
    t_j = t.unsqueeze(1).expand(-1, L, -1, -1).gather(2, idx)

    # Distance
    diff = t_j - t.unsqueeze(2).expand(-1, -1, K, -1)  # (B, L, K, 3)
    dist = diff.norm(dim=-1)  # (B, L, K)

    # RBF encoding (16 Gaussian basis functions from 0 to 20Å)
    centers = torch.linspace(0, 20, 16, device=device)
    rbf = torch.exp(-((dist.unsqueeze(-1) - centers) ** 2) / 2.0)
    return rbf  # (B, L, K, 16)


# ── Node and Edge Update Layers ─────────────────────────────────────────────


class NodeMPNN(nn.Module):
    """Update node features by aggregating messages from all neighbors."""

    def __init__(self, c_node: int = 128, c_edge: int = 128):
        super().__init__()
        self.norm = nn.LayerNorm(c_node)
        self.msg = nn.Sequential(
            nn.Linear(c_node * 2 + c_edge, c_node * 2),
            nn.GELU(),
            nn.Linear(c_node * 2, c_node),
        )
        self.gate = nn.Linear(c_node * 2 + c_edge, c_node)
        self.ff = nn.Sequential(
            nn.LayerNorm(c_node),
            nn.Linear(c_node, c_node * 4),
            nn.GELU(),
            nn.Linear(c_node * 4, c_node),
        )

    def forward(
        self,
        node: torch.Tensor,
        edge: torch.Tensor,
        k_idx: torch.Tensor,
        edge_mask: torch.Tensor,
    ) -> torch.Tensor:
        B, L, K, _ = edge.shape
        node_n = self.norm(node)

        # Gather neighbor node features
        idx_expanded = k_idx.unsqueeze(-1).expand(-1, -1, -1, node_n.shape[-1])
        node_j = node_n.unsqueeze(1).expand(-1, L, -1, -1).gather(2, idx_expanded)

        # Compute messages
        msg_input = torch.cat(
            [
                node_n.unsqueeze(2).expand(-1, -1, K, -1),
                node_j,
                edge,
            ],
            dim=-1,
        )
        messages = self.msg(msg_input) * torch.sigmoid(self.gate(msg_input))
        messages = messages * edge_mask.float().unsqueeze(-1)

        # Aggregate (mean over neighbors)
        agg = messages.sum(2) / edge_mask.float().sum(-1, keepdim=True).clamp(1)
        node = node + agg
        node = node + self.ff(node)
        return node


class EdgeMPNN(nn.Module):
    """Update edge features from node features."""

    def __init__(self, c_node: int = 128, c_edge: int = 128):
        super().__init__()
        self.update = nn.Sequential(
            nn.LayerNorm(c_node * 2 + c_edge),
            nn.Linear(c_node * 2 + c_edge, c_edge * 2),
            nn.GELU(),
            nn.Linear(c_edge * 2, c_edge),
        )

    def forward(
        self, node: torch.Tensor, edge: torch.Tensor, k_idx: torch.Tensor
    ) -> torch.Tensor:
        B, L, K, _ = edge.shape
        idx_exp = k_idx.unsqueeze(-1).expand(-1, -1, -1, node.shape[-1])
        node_j = node.unsqueeze(1).expand(-1, L, -1, -1).gather(2, idx_exp)
        node_i = node.unsqueeze(2).expand(-1, -1, K, -1)
        edge_in = torch.cat([node_i, node_j, edge], dim=-1)
        return edge + self.update(edge_in)


# ── Autoregressive Sequence Decoder ─────────────────────────────────────────


class SequenceDecoder(nn.Module):
    """Autoregressive decoder with optional fixed-residue constraints."""

    def __init__(self, c_node: int = 128, vocab_size: int = 20):
        super().__init__()
        # Amino acid embedding; vocab_size is reserved as the masked token.
        self.aa_embed = nn.Embedding(vocab_size + 1, c_node)

        self.decoder_layers = nn.ModuleList(
            [
                nn.TransformerDecoderLayer(
                    d_model=c_node,
                    nhead=4,
                    dim_feedforward=512,
                    dropout=0.1,
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(3)
            ]
        )
        self.to_logits = nn.Linear(c_node, vocab_size)
        self.vocab_size = vocab_size

    def forward(
        self,
        node_features: torch.Tensor,
        sequence_so_far: Optional[torch.Tensor] = None,
        fixed_positions: Optional[torch.Tensor] = None,
        fixed_aas: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        node_features:    (B, L, c_node) from ProteinMPNN graph network
        sequence_so_far:  (B, L) optional known sequence context
        fixed_positions:  (B, L) bool, True = position is fixed
        fixed_aas:        (B, L) integer amino-acid IDs at fixed positions
        Returns logits:   (B, L, 20)

        Fixed residues are both kept out of the designed-position context and
        hard-constrained in the returned logits. Designed positions therefore
        cannot accidentally learn from or overwrite fixed residues.
        """
        B, L, _ = node_features.shape
        device = node_features.device

        if sequence_so_far is None:
            seq_in = torch.full((B, L), self.vocab_size, dtype=torch.long, device=device)
        else:
            seq_in = sequence_so_far.clone()

        # If fixed residues are supplied, expose only fixed amino acids as
        # teacher-forced context. Designed positions receive the masked token.
        if fixed_positions is not None:
            if fixed_positions.shape != (B, L) or fixed_positions.dtype != torch.bool:
                raise ValueError("fixed_positions must have shape (B, L) and dtype bool")
            if fixed_aas is None or fixed_aas.shape != (B, L):
                raise ValueError("fixed_aas must have shape (B, L) when fixed_positions is provided")
            if torch.any(fixed_positions & ((fixed_aas < 0) | (fixed_aas >= self.vocab_size))):
                raise ValueError("fixed_aas contains invalid amino-acid IDs at fixed positions")
            masked = torch.full_like(seq_in, self.vocab_size)
            seq_in = torch.where(fixed_positions, fixed_aas, masked)

        seq_emb = self.aa_embed(seq_in)
        tgt = node_features + seq_emb

        causal_mask = nn.Transformer.generate_square_subsequent_mask(L, device=device)

        for layer in self.decoder_layers:
            tgt = layer(tgt=tgt, memory=node_features, tgt_mask=causal_mask)

        logits = self.to_logits(tgt)

        # Fixed residues must remain exactly fixed. Give their prescribed amino
        # acid a finite zero logit and every alternative -inf so downstream
        # softmax/argmax cannot mutate a constrained position.
        if fixed_positions is not None:
            fixed_logits = torch.full_like(logits, float("-inf"))
            fixed_logits.scatter_(
                -1, fixed_aas.unsqueeze(-1), torch.zeros((), device=device, dtype=logits.dtype)
            )
            logits = torch.where(fixed_positions.unsqueeze(-1), fixed_logits, logits)

        return logits


# ── Full ProteinMPNN ────────────────────────────────────────────────────────


class ProteinMPNN(nn.Module):
    """CHIMERA sequence design module."""

    def __init__(
        self,
        c_node: int = 128,
        c_edge: int = 128,
        n_mp_layers: int = 3,
        k_neighbors: int = 32,
        vocab_size: int = 20,
    ):
        super().__init__()
        self.node_embed = nn.Sequential(
            nn.Linear(6, c_node),
            nn.GELU(),
            nn.Linear(c_node, c_node),
        )
        self.edge_embed = nn.Sequential(
            nn.Linear(16, c_edge),
            nn.GELU(),
            nn.Linear(c_edge, c_edge),
        )

        self.node_layers = nn.ModuleList(
            [NodeMPNN(c_node, c_edge) for _ in range(n_mp_layers)]
        )
        self.edge_layers = nn.ModuleList(
            [EdgeMPNN(c_node, c_edge) for _ in range(n_mp_layers)]
        )

        self.decoder = SequenceDecoder(c_node, vocab_size)

    def forward(
        self,
        t_coords: torch.Tensor,
        R_frames: torch.Tensor,
        evol_node_features: torch.Tensor,
        fixed_positions: Optional[torch.Tensor] = None,
        fixed_aas: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        t_coords:           (B, L, 3)       Cα coordinates
        R_frames:            (B, L, 3, 3)    Backbone frames
        evol_node_features: (B, L, 128)     EvoFormer → node projection
        fixed_positions:    (B, L) bool     True = position is fixed
        fixed_aas:          (B, L) int      Amino acids at fixed positions
        """
        B, L, _ = t_coords.shape

        k_idx, edge_geom, edge_mask = get_protein_graph(t_coords, R_frames, self.decoder.vocab_size)
        # Use the configured graph neighborhood independently of vocabulary size.
        k_idx, edge_geom, edge_mask = get_protein_graph(t_coords, R_frames)

        t_centered = t_coords - t_coords.mean(dim=1, keepdim=True)
        torsions = torch.zeros(B, L, 3, device=t_coords.device, dtype=t_coords.dtype)
        node_geom = torch.cat([t_centered, torsions], dim=-1)

        node = self.node_embed(node_geom) + evol_node_features
        edge = self.edge_embed(edge_geom)

        for node_layer, edge_layer in zip(self.node_layers, self.edge_layers):
            node = node_layer(node, edge, k_idx, edge_mask)
            edge = edge_layer(node, edge, k_idx)

        return self.decoder(
            node,
            fixed_positions=fixed_positions,
            fixed_aas=fixed_aas,
        )
