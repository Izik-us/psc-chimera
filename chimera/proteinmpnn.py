"""
CHIMERA — ProteinMPNN Module
=============================
Evolutionary-context-aware geometric message-passing sequence designer.

This is a ProteinMPNN-inspired implementation, not the original pretrained
ProteinMPNN weights. Geometry is represented with rigid-frame invariants:
relative positions are expressed in the query residue's local frame and
relative rotations are represented by R_i^T R_j.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional


def get_protein_graph(
    t_coords: torch.Tensor, R_frames: torch.Tensor, k_neighbors: int = 32
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build a k-NN graph from Cα coordinates with explicit edge validity."""
    if t_coords.ndim != 3 or t_coords.shape[-1] != 3:
        raise ValueError("t_coords must have shape (B, L, 3)")
    if R_frames.shape != (t_coords.shape[0], t_coords.shape[1], 3, 3):
        raise ValueError("R_frames must have shape (B, L, 3, 3)")
    if k_neighbors < 1:
        raise ValueError("k_neighbors must be at least 1")
    B, L, _ = t_coords.shape
    if L < 2:
        raise ValueError("Protein graph construction requires at least two residues")
    device = t_coords.device

    diff = t_coords.unsqueeze(2) - t_coords.unsqueeze(1)
    dist = diff.norm(dim=-1)
    dist_no_self = dist.masked_fill(
        torch.eye(L, device=device, dtype=torch.bool).unsqueeze(0), float("inf")
    )
    k = min(k_neighbors, L - 1)
    _, top_k_idx = dist_no_self.topk(k, dim=-1, largest=False)
    features = _compute_edge_features(t_coords, R_frames, top_k_idx)
    mask = dist_no_self.gather(-1, top_k_idx) < 20.0
    return top_k_idx, features, mask


def _compute_edge_features(
    t: torch.Tensor, R: torch.Tensor, k_idx: torch.Tensor
) -> torch.Tensor:
    """Compute invariant local-frame geometry: 16 RBF + 3 position + 9 rotation features."""
    B, L, K = k_idx.shape
    idx = k_idx.unsqueeze(-1).expand(-1, -1, -1, 3)
    t_j = t.unsqueeze(1).expand(-1, L, -1, -1).gather(2, idx)
    delta = t_j - t.unsqueeze(2)
    dist = delta.norm(dim=-1)
    centers = torch.linspace(0, 20, 16, device=t.device, dtype=t.dtype)
    rbf = torch.exp(-((dist.unsqueeze(-1) - centers) ** 2) / 2.0)

    R_i = R.unsqueeze(2).expand(-1, -1, K, -1, -1)
    local_delta = torch.einsum(
        "blkij,blkj->blki", R_i.transpose(-1, -2), delta
    )
    local_delta = local_delta / dist.clamp_min(1e-6).unsqueeze(-1)

    idx_r = k_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, 3, 3)
    R_j = R.unsqueeze(1).expand(-1, L, -1, -1, -1).gather(2, idx_r)
    relative_R = torch.matmul(R_i.transpose(-1, -2), R_j).reshape(B, L, K, 9)
    return torch.cat([rbf, local_delta, relative_R], dim=-1)


class NodeMPNN(nn.Module):
    """Update node features by aggregating masked geometric messages."""

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

    def forward(self, node, edge, k_idx, edge_mask):
        B, L, K, _ = edge.shape
        node_n = self.norm(node)
        idx_expanded = k_idx.unsqueeze(-1).expand(-1, -1, -1, node_n.shape[-1])
        node_j = node_n.unsqueeze(1).expand(-1, L, -1, -1).gather(2, idx_expanded)
        node_i = node_n.unsqueeze(2).expand(-1, -1, K, -1)
        msg_input = torch.cat([node_i, node_j, edge], dim=-1)
        messages = self.msg(msg_input) * torch.sigmoid(self.gate(msg_input))
        messages = messages * edge_mask.to(messages.dtype).unsqueeze(-1)
        denom = edge_mask.to(messages.dtype).sum(-1, keepdim=True).clamp_min(1.0)
        agg = messages.sum(2) / denom
        return node + agg + self.ff(node + agg)


class EdgeMPNN(nn.Module):
    """Update edge features from invariant node features."""

    def __init__(self, c_node: int = 128, c_edge: int = 128):
        super().__init__()
        self.update = nn.Sequential(
            nn.LayerNorm(c_node * 2 + c_edge),
            nn.Linear(c_node * 2 + c_edge, c_edge * 2),
            nn.GELU(),
            nn.Linear(c_edge * 2, c_edge),
        )

    def forward(self, node, edge, k_idx):
        B, L, K, _ = edge.shape
        idx_exp = k_idx.unsqueeze(-1).expand(-1, -1, -1, node.shape[-1])
        node_j = node.unsqueeze(1).expand(-1, L, -1, -1).gather(2, idx_exp)
        node_i = node.unsqueeze(2).expand(-1, -1, K, -1)
        return edge + self.update(torch.cat([node_i, node_j, edge], dim=-1))


class SequenceDecoder(nn.Module):
    """Autoregressive decoder with hard fixed-residue constraints."""

    def __init__(self, c_node: int = 128, vocab_size: int = 20):
        super().__init__()
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

    def forward(self, node_features, sequence_so_far=None, fixed_positions=None, fixed_aas=None):
        B, L, _ = node_features.shape
        device = node_features.device
        if sequence_so_far is None:
            seq_in = torch.full((B, L), self.vocab_size, dtype=torch.long, device=device)
        else:
            if sequence_so_far.shape != (B, L):
                raise ValueError("sequence_so_far must have shape (B, L)")
            seq_in = sequence_so_far.clone()

        if fixed_positions is not None:
            if fixed_positions.shape != (B, L) or fixed_positions.dtype != torch.bool:
                raise ValueError("fixed_positions must have shape (B, L) and dtype bool")
            if fixed_aas is None or fixed_aas.shape != (B, L):
                raise ValueError("fixed_aas must have shape (B, L) when fixed_positions is provided")
            if torch.any(fixed_positions & ((fixed_aas < 0) | (fixed_aas >= self.vocab_size))):
                raise ValueError("fixed_aas contains invalid amino-acid IDs at fixed positions")
            masked = torch.full_like(seq_in, self.vocab_size)
            seq_in = torch.where(fixed_positions, fixed_aas, masked)

        tgt = node_features + self.aa_embed(seq_in)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(L, device=device)
        for layer in self.decoder_layers:
            tgt = layer(tgt=tgt, memory=node_features, tgt_mask=causal_mask)
        logits = self.to_logits(tgt)

        if fixed_positions is not None:
            fixed_logits = torch.full_like(logits, float("-inf"))
            fixed_logits.scatter_(-1, fixed_aas.unsqueeze(-1), 0.0)
            logits = torch.where(fixed_positions.unsqueeze(-1), fixed_logits, logits)
        return logits


class ProteinMPNN(nn.Module):
    """ProteinMPNN-inspired geometric sequence design module."""

    def __init__(self, c_node=128, c_edge=128, n_mp_layers=3, k_neighbors=32, vocab_size=20):
        super().__init__()
        self.k_neighbors = k_neighbors
        self.node_embed = nn.Sequential(
            nn.Linear(6, c_node), nn.GELU(), nn.Linear(c_node, c_node)
        )
        self.edge_embed = nn.Sequential(
            nn.Linear(28, c_edge), nn.GELU(), nn.Linear(c_edge, c_edge)
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
        t_coords,
        R_frames,
        evol_node_features,
        fixed_positions=None,
        fixed_aas=None,
    ):
        B, L, _ = t_coords.shape
        if evol_node_features.shape[:2] != (B, L):
            raise ValueError("evol_node_features must have shape (B, L, d)")
        k_idx, edge_geom, edge_mask = get_protein_graph(
            t_coords, R_frames, self.k_neighbors
        )

        # Six invariant node descriptors, avoiding absolute/global-frame coordinates.
        centered = t_coords - t_coords.mean(dim=1, keepdim=True)
        all_dist = torch.cdist(t_coords, t_coords)
        mean_dist = all_dist.mean(dim=-1, keepdim=True)
        std_dist = all_dist.std(dim=-1, keepdim=True)
        kth_dist = (
            all_dist.masked_fill(
                torch.eye(L, device=t_coords.device, dtype=torch.bool).unsqueeze(0),
                float("inf"),
            )
            .topk(min(self.k_neighbors, L - 1), dim=-1, largest=False)
            .values[..., -1:]
            .to(t_coords.dtype)
        )
        frame_err = (
            R_frames.transpose(-1, -2) @ R_frames
            - torch.eye(3, device=t_coords.device, dtype=t_coords.dtype)
        ).square().mean(dim=(-1, -2)).unsqueeze(-1)
        center_radius = centered.norm(dim=-1, keepdim=True)
        node_geom = torch.cat(
            [center_radius, mean_dist, std_dist, kth_dist, frame_err, torch.ones_like(center_radius)],
            dim=-1,
        )

        node = self.node_embed(node_geom) + evol_node_features
        edge = self.edge_embed(edge_geom)
        for node_layer, edge_layer in zip(self.node_layers, self.edge_layers):
            node = node_layer(node, edge, k_idx, edge_mask)
            edge = edge_layer(node, edge, k_idx)
        return self.decoder(node, fixed_positions=fixed_positions, fixed_aas=fixed_aas)
