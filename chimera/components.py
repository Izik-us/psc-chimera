"""Local model components used by canonical and legacy CHIMERA compositions.

The canonical MSA backbone implements the AlphaFold-2 EvoFormer core; the
structural connector and ProteinMPNN-inspired trunk remain CHIMERA-specific.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from .domain_schema import NRPSConstraints
from .evoformer_stack import EvoFormerStack

class TriangularPairUpdateConnector(nn.Module):
    """Adapt EvoFormer pairs to the structural-conditioning channel width.

    The canonical EvoFormer performs evolutionary pair reasoning at its native
    pair width. This downstream connector projects that representation into
    the flow model's conditioning width, applies task-specific refinements, and
    can fuse explicitly aligned retrieved context. Its updates do not enforce
    triangle inequalities or feed back into the EvoFormer stream.
    """

    def __init__(
        self,
        evo_pair_dim: int = 128,
        out_dim: int = 512,
        n_heads: int = 4,
    ):
        super().__init__()
        # Linear projection: 128→512
        self.input_proj = nn.Linear(evo_pair_dim, out_dim)
        self.input_norm = nn.LayerNorm(out_dim)

        # Aggregate pair information over an intermediate residue index.
        self.tri_attn_out = TriangularAttention(out_dim, n_heads, mode="outgoing")
        self.tri_attn_in = TriangularAttention(out_dim, n_heads, mode="incoming")

        self.tri_mult_out = TriangularMultiplicativeUpdate(out_dim, mode="outgoing")
        self.tri_mult_in = TriangularMultiplicativeUpdate(out_dim, mode="incoming")

        # Transition FFN
        self.transition = nn.Sequential(
            nn.LayerNorm(out_dim),
            nn.Linear(out_dim, out_dim * 4),
            nn.ReLU(),
            nn.Linear(out_dim * 4, out_dim),
        )

        # Optional context attention; callers must establish embedding alignment.
        self.retrieval_cross = nn.MultiheadAttention(
            embed_dim=out_dim,
            num_heads=n_heads,
            batch_first=True,
        )
        self.out_norm = nn.LayerNorm(out_dim)

        self._init_to_identity()

    def _init_to_identity(self):
        """Initialize the overlapping input-projection dimensions to identity."""
        nn.init.zeros_(self.input_proj.bias)
        nn.init.eye_(
            self.input_proj.weight[:128, :128]
            if self.input_proj.weight.shape[1] >= 128
            else self.input_proj.weight
        )

    def forward(
        self,
        pair_repr: torch.Tensor,  # (B, L, L, 128)
        retrieved_context: Optional[torch.Tensor] = None,  # (B, K, 256) from RAG
    ) -> torch.Tensor:  # (B, L, L, 512)

        z = self.input_norm(self.input_proj(pair_repr))  # (B, L, L, 256)

        # Learned pair-feature updates; these are not geometric constraints.
        z = z + self.tri_mult_out(z)  # cheaper: no attention
        z = z + self.tri_mult_in(z)
        z = z + self.tri_attn_out(z)  # richer: with attention
        z = z + self.tri_attn_in(z)
        z = z + self.transition(z)

        # Add optional retrieved context when supplied by the caller.
        if retrieved_context is not None:
            B, L, _, d = z.shape
            z_flat = z.reshape(B, L * L, d)
            ctx_attn, _ = self.retrieval_cross(
                z_flat, retrieved_context, retrieved_context
            )
            z = z + self.out_norm(ctx_attn.reshape(B, L, L, d))

        return z

class TriangularAttention(nn.Module):
    """
    Triangular self-attention on pair representation.
    mode='outgoing':  (i,j) attends over k using (i,k) as keys/queries
    mode='incoming':  (i,j) attends over k using (k,j) as keys/queries
    """

    def __init__(self, d: int, n_heads: int, mode: str):
        super().__init__()
        self.mode = mode
        self.norm = nn.LayerNorm(d)
        self.q = nn.Linear(d, d, bias=False)
        self.k = nn.Linear(d, d, bias=False)
        self.v = nn.Linear(d, d, bias=False)
        self.b = nn.Linear(d, n_heads, bias=False)  # pair bias
        self.gate = nn.Linear(d, d)
        self.out = nn.Linear(d, d)
        self.n_heads = n_heads
        self.scale = (d // n_heads) ** -0.5

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        B, L, _, d = z.shape
        z = self.norm(z)

        if self.mode == "outgoing":
            Q = self.q(z)  # (B, L, L, d)
            K = self.k(z)  # (B, L, L, d)
            V = self.v(z)  # (B, L, L, d)
            b = self.b(z)  # (B, L, L, n_heads)
            dh = d // self.n_heads

            Q = Q.view(B, L, L, self.n_heads, dh).permute(0, 1, 3, 2, 4)
            K = K.view(B, L, L, self.n_heads, dh).permute(0, 1, 3, 2, 4)
            V = V.view(B, L, L, self.n_heads, dh).permute(0, 1, 3, 2, 4)

            # scores[b, i, h, j, k] = query(i, j) dot key(i, k)
            attn = torch.einsum("bihjd,bihkd->bihjk", Q, K) * self.scale
            bias = b.permute(0, 1, 3, 2).unsqueeze(-1).expand(-1, -1, -1, -1, L)
            attn = (attn + bias).softmax(dim=-1)
            out = torch.einsum("bihjk,bihkd->bihjd", attn, V)
            out = out.permute(0, 1, 3, 2, 4).reshape(B, L, L, d)

        else:  # incoming
            z_T = z.transpose(1, 2)
            Q = self.q(z_T)
            K = self.k(z_T)
            V = self.v(z_T)
            b = self.b(z_T)
            dh = d // self.n_heads

            Q = Q.view(B, L, L, self.n_heads, dh).permute(0, 1, 3, 2, 4)
            K = K.view(B, L, L, self.n_heads, dh).permute(0, 1, 3, 2, 4)
            V = V.view(B, L, L, self.n_heads, dh).permute(0, 1, 3, 2, 4)

            attn = torch.einsum("bihjd,bihkd->bihjk", Q, K) * self.scale
            bias = b.permute(0, 1, 3, 2).unsqueeze(-1).expand(-1, -1, -1, -1, L)
            attn = (attn + bias).softmax(dim=-1)
            out = torch.einsum("bihjk,bihkd->bihjd", attn, V)
            out = out.permute(0, 1, 3, 2, 4).reshape(B, L, L, d).transpose(1, 2)

        # Gate
        g = torch.sigmoid(self.gate(z))
        out = self.out(out * g)
        return out

class TriangularMultiplicativeUpdate(nn.Module):
    """
    Cheaper alternative: multiplicative pair update via gating (no attention).
    mode='outgoing': z[i,j] updated by z[i,k] and z[k,j] products
    mode='incoming': z[i,j] updated by z[k,i] and z[j,k] products
    """

    def __init__(self, d: int, mode: str):
        super().__init__()
        self.mode = mode
        self.norm = nn.LayerNorm(d)
        self.norm_o = nn.LayerNorm(d)
        self.left_in = nn.Linear(d, d)
        self.left_g = nn.Linear(d, d)
        self.right_in = nn.Linear(d, d)
        self.right_g = nn.Linear(d, d)
        self.gate = nn.Linear(d, d)
        self.out = nn.Linear(d, d)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z = self.norm(z)
        l = torch.sigmoid(self.left_g(z)) * self.left_in(z)  # (B,L,L,d)
        r = torch.sigmoid(self.right_g(z)) * self.right_in(z)

        if self.mode == "outgoing":
            # p[i,j] = sum_k l[i,k] * r[k,j]
            p = torch.einsum("bild,bljd->bijd", l, r)
        else:
            # p[i,j] = sum_k l[k,i] * r[j,k]
            p = torch.einsum("blid,bjld->bijd", l, r)

        g = torch.sigmoid(self.gate(z))
        return self.out(self.norm_o(p) * g)

class EvolCrossAttentionConnector(nn.Module):
    """Combine structural-node and evolutionary features through attention.

    The learned gate conditions one attention path on the supplied flow time.
    Its values are learned and are not constrained to represent a specific
    noise schedule or to monotonically vary with flow time.
    """

    def __init__(self, d_se3: int = 768, d_evo: int = 256, n_heads: int = 8):
        super().__init__()
        # Main cross-attention: SE3 queries attend to evolutionary memory
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_se3,
            num_heads=n_heads,
            kdim=d_evo,
            vdim=d_evo,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(d_se3)

        # Learned flow-time-conditioned gate for the first attention path.
        self.noise_gate = nn.Sequential(
            nn.Linear(1, 64),
            nn.SiLU(),
            nn.Linear(64, d_se3),
            nn.Sigmoid(),
        )

        # A second attention path over the evolutionary representation.
        self.direct_attn = nn.MultiheadAttention(
            embed_dim=d_se3,
            num_heads=n_heads,
            kdim=d_evo,
            vdim=d_evo,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(d_se3)

        # Combine the two learned feature paths.
        self.combine = nn.Sequential(
            nn.Linear(d_se3 * 2, d_se3),
            nn.LayerNorm(d_se3),
        )

    def forward(
        self,
        se3_node: torch.Tensor,  # (B, L, d_se3)
        evo_single: torch.Tensor,  # (B, L, d_evo)
        noise_level: torch.Tensor,  # (B,) flow time t ∈ [0,1]
    ) -> torch.Tensor:
        gate = self.noise_gate(noise_level.unsqueeze(-1))  # (B, d_se3)
        gate = gate.unsqueeze(1)  # (B, 1, d_se3)

        # Pathway 1: standard cross-attention
        path1, _ = self.cross_attn(se3_node, evo_single, evo_single)
        path1 = self.norm1(se3_node + gate * path1)

        # A second attention path over the same evolutionary memory.
        path2, _ = self.direct_attn(se3_node, evo_single, evo_single)
        path2 = self.norm2(se3_node + path2)

        return self.combine(torch.cat([path1, path2], dim=-1))

class NodeProjectionConnector(nn.Module):
    """Project evolutionary residue features into the local sequence-model width.

    An optional pocket mask adds a learned feature bias at marked positions;
    it does not encode residue identity or perform attention.
    """

    def __init__(self, evo_dim: int = 256, mpnn_dim: int = 512):
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(evo_dim),
            nn.Linear(evo_dim, evo_dim),
            nn.GELU(),
            nn.Linear(evo_dim, mpnn_dim * 2),
            nn.GELU(),
            nn.Linear(mpnn_dim * 2, mpnn_dim),
            nn.LayerNorm(mpnn_dim),
        )
        # Learned additive bias for caller-marked pocket positions.
        self.pocket_scale = nn.Linear(mpnn_dim, mpnn_dim)

    def forward(
        self,
        evo_single: torch.Tensor,  # (B, L, 256)
        pocket_mask: Optional[torch.Tensor] = None,  # (B, L) bool: near pocket
    ) -> torch.Tensor:
        projected = self.proj(evo_single)
        if pocket_mask is not None:
            pocket_bias = self.pocket_scale(projected)
            projected = projected + pocket_mask.float().unsqueeze(-1) * pocket_bias
        return projected

class NRPSConstraintEncoder(nn.Module):
    """Encode NRPS annotations as conditioning features for the flow model.

    The encoding exposes constraints to the learned model; it does not itself
    impose hard geometric constraints on generated structures.
    """

    def __init__(self, d: int = 512, max_len: int = 2000):
        super().__init__()
        # Domain type embeddings (A, T, C, TE, linker = 5 types)
        self.domain_emb = nn.Embedding(5, d)
        # PPant attachment site marker
        self.ppt_marker = nn.Parameter(torch.randn(d))
        # Stachelhaus position marker
        self.stach_marker = nn.Parameter(torch.randn(d))
        # Position encoding
        self.pos_enc = nn.Embedding(max_len, d)

    def forward(
        self,
        L: int,
        constraints: NRPSConstraints,
        device: torch.device,
        batch_size: int = 1,
    ) -> torch.Tensor:  # (B, L, d) positional constraint map
        pos = torch.arange(L, device=device)
        c_map = self.pos_enc(pos).unsqueeze(0).expand(batch_size, -1, -1).clone()

        # Mark Stachelhaus selectivity code positions
        for idx in constraints.stachelhaus_positions:
            if idx < L:
                c_map[:, idx] = c_map[:, idx] + self.stach_marker.unsqueeze(0)

        # Mark PPant serine
        ppt = constraints.ppt_serine_position
        if ppt < L:
            c_map[:, ppt] = c_map[:, ppt] + self.ppt_marker.unsqueeze(0)

        # Mark domain types independently for every batch element.
        for d_idx in range(constraints.domain_boundaries.shape[1]):
            for batch_idx in range(batch_size):
                s = int(constraints.domain_boundaries[batch_idx, d_idx, 0].clamp(0, L))
                e = int(constraints.domain_boundaries[batch_idx, d_idx, 1].clamp(0, L))
                if e > s:
                    domain_type = (
                        int(constraints.domain_types[batch_idx, d_idx].clamp(0, 4))
                        if constraints.domain_types is not None
                        else min(d_idx, 4)
                    )
                    domain_emb = self.domain_emb(torch.tensor(domain_type, device=device))
                    c_map[batch_idx, s:e] = c_map[batch_idx, s:e] + domain_emb

        return c_map

class MSARepresentationBackbone(EvoFormerStack):
    """Canonical two-stream MSA/pair EvoFormer; ``True`` mask values are padding."""

    def __init__(
        self,
        d_single: int = 256,
        d_pair: int = 128,
        n_blocks: int = 48,
        **configuration,
    ):
        super().__init__(
            d_msa=d_single,
            d_pair=d_pair,
            d_single=d_single,
            n_blocks=n_blocks,
            **configuration,
        )

    def masked_reconstruction_loss(
        self,
        msa_tokens: torch.Tensor,
        msa_padding_mask: Optional[torch.Tensor] = None,
        mask_probability: float = 0.15,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Compatibility helper for masked-MSA-only diagnostics."""
        batch, _, n_res = msa_tokens.shape
        pair_features = msa_tokens.new_zeros(
            batch, n_res, n_res, self.d_pair, dtype=self.token_embed.weight.dtype
        )
        total, _ = self.representation_loss(
            msa_tokens,
            pair_features,
            msa_padding_mask=msa_padding_mask,
            mask_probability=mask_probability,
            pair_loss_weight=0.0,
            generator=generator,
        )
        return total


EvoFormerBackbone = MSARepresentationBackbone

class ProteinMPNNBackbone(nn.Module):
    """CHIMERA geometric residue trunk for structure-conditioned sequence design.

    Backbone atoms define residue-local frames, a masked k-NN graph supplies
    rigid-transform-invariant edge geometry, and alternating node/edge
    message-passing layers condition evolutionary residue states on the
    generated 3-D backbone. Sequence logits remain owned by the downstream
    CHIMERA sequence-design stack.
    """

    def __init__(self, node_features: int = 128, edge_features: int = 128,
                 max_neighbors: int = 32, n_mp_layers: int = 3):
        super().__init__()
        if node_features < 1 or edge_features < 1:
            raise ValueError("node_features and edge_features must be positive")
        if max_neighbors < 1:
            raise ValueError("max_neighbors must be positive")
        if n_mp_layers < 1:
            raise ValueError("n_mp_layers must be positive")
        self.node_features = node_features
        self.edge_features = edge_features
        self.max_neighbors = max_neighbors
        self.n_mp_layers = n_mp_layers
        self.node_geometry_dim = 6
        self.edge_geometry_dim = 28
        self.node_geometry_encoder = nn.Sequential(
            nn.LayerNorm(self.node_geometry_dim),
            nn.Linear(self.node_geometry_dim, node_features),
            nn.GELU(),
            nn.Linear(node_features, node_features),
        )
        self.edge_geometry_encoder = nn.Sequential(
            nn.Linear(self.edge_geometry_dim, edge_features),
            nn.GELU(),
            nn.Linear(edge_features, edge_features),
        )
        from .proteinmpnn import EdgeMPNN, NodeMPNN
        self.node_layers = nn.ModuleList(
            [NodeMPNN(node_features, edge_features) for _ in range(n_mp_layers)]
        )
        self.edge_layers = nn.ModuleList(
            [EdgeMPNN(node_features, edge_features) for _ in range(n_mp_layers)]
        )
        self.output_norm = nn.LayerNorm(node_features)

    @staticmethod
    def _derive_local_frames(backbone_coords: torch.Tensor, residue_mask: torch.Tensor):
        """Construct N-CA-C frames that transform equivariantly under rigid motions."""
        n_atom = backbone_coords[..., 0, :]
        ca_atom = backbone_coords[..., 1, :]
        c_atom = backbone_coords[..., 2, :]
        ca_to_c = c_atom - ca_atom
        ca_to_n = n_atom - ca_atom
        cross = torch.cross(ca_to_c, ca_to_n, dim=-1)
        finite = torch.isfinite(torch.stack([n_atom, ca_atom, c_atom], dim=-2)).all(dim=(-1, -2))
        valid = (residue_mask & finite &
                 (ca_to_c.norm(dim=-1) > 1e-6) &
                 (ca_to_n.norm(dim=-1) > 1e-6) &
                 (cross.norm(dim=-1) > 1e-6))
        x_axis = F.normalize(ca_to_c, dim=-1, eps=1e-6)
        z_axis = F.normalize(cross, dim=-1, eps=1e-6)
        y_axis = F.normalize(torch.cross(z_axis, x_axis, dim=-1), dim=-1, eps=1e-6)
        frames = torch.stack([x_axis, y_axis, z_axis], dim=-1)
        identity = torch.eye(3, dtype=backbone_coords.dtype, device=backbone_coords.device)
        identity = identity.view(1, 1, 3, 3).expand_as(frames)
        frames = torch.where(valid.unsqueeze(-1).unsqueeze(-1), frames, identity)
        return frames, valid

    def _node_geometry(self, ca_coords: torch.Tensor, backbone_coords: torch.Tensor, residue_mask: torch.Tensor):
        """Return scalar invariants with no dependence on the global coordinate frame."""
        mask_f = residue_mask.to(ca_coords.dtype)
        count = mask_f.sum(dim=1, keepdim=True).clamp_min(1.0)
        center = (ca_coords * mask_f.unsqueeze(-1)).sum(dim=1, keepdim=True) / count.unsqueeze(-1)
        center_radius = (ca_coords - center).norm(dim=-1, keepdim=True)
        pairwise = torch.cdist(ca_coords, ca_coords)
        pair_mask = residue_mask.unsqueeze(1) & residue_mask.unsqueeze(2)
        pair_mask_f = pair_mask.to(ca_coords.dtype)
        pair_count = pair_mask_f.sum(dim=-1, keepdim=True).clamp_min(1.0)
        mean_dist = (pairwise * pair_mask_f).sum(dim=-1, keepdim=True) / pair_count
        variance = ((pairwise - mean_dist).square() * pair_mask_f).sum(dim=-1, keepdim=True) / pair_count
        std_dist = variance.clamp_min(0).sqrt()
        self_mask = torch.eye(ca_coords.shape[1], dtype=torch.bool, device=ca_coords.device).unsqueeze(0)
        nearest = pairwise.masked_fill(~pair_mask | self_mask, float("inf")).min(dim=-1, keepdim=True).values
        nearest = torch.where(torch.isfinite(nearest), nearest, torch.zeros_like(nearest))
        n_atom = backbone_coords[..., 0, :]
        ca_atom = backbone_coords[..., 1, :]
        c_atom = backbone_coords[..., 2, :]
        n_ca = (n_atom - ca_atom).norm(dim=-1, keepdim=True)
        ca_c = (c_atom - ca_atom).norm(dim=-1, keepdim=True)
        geometry = torch.cat([center_radius, mean_dist, std_dist, nearest, n_ca, ca_c], dim=-1)
        return geometry * residue_mask.unsqueeze(-1).to(geometry.dtype)

    def forward(self, backbone_coords: torch.Tensor, node_features: torch.Tensor,
                residue_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Encode evolutionary residue features conditioned on 3-D backbone geometry."""
        if backbone_coords.ndim != 4 or backbone_coords.shape[-2:] not in ((4, 3), (5, 3)):
            raise ValueError("backbone_coords must have shape (B, L, 4|5, 3)")
        if node_features.ndim != 3:
            raise ValueError("node_features must have shape (B, L, node_features)")
        if node_features.shape[:2] != backbone_coords.shape[:2]:
            raise ValueError("node_features and backbone_coords must agree on batch and length")
        if node_features.shape[-1] != self.node_features:
            raise ValueError(f"node_features must have shape (B, L, {self.node_features})")
        batch_size, length = backbone_coords.shape[:2]
        device = backbone_coords.device
        if residue_mask is None:
            residue_mask = torch.ones(batch_size, length, dtype=torch.bool, device=device)
        elif residue_mask.shape != (batch_size, length) or residue_mask.dtype != torch.bool:
            raise ValueError("residue_mask must have shape (B, L) and dtype bool")
        if not torch.isfinite(backbone_coords).all() and residue_mask.all():
            raise ValueError("backbone_coords contains non-finite values without a masking residue")
        safe_coords = torch.nan_to_num(backbone_coords, nan=0.0, posinf=0.0, neginf=0.0)
        safe_coords = safe_coords * residue_mask.unsqueeze(-1).unsqueeze(-1).to(safe_coords.dtype)
        ca_coords = safe_coords[..., 1, :]
        frames, frame_valid = self._derive_local_frames(safe_coords, residue_mask)
        effective_mask = residue_mask & frame_valid
        node_geometry = self._node_geometry(ca_coords, safe_coords, effective_mask)
        node = node_features * effective_mask.unsqueeze(-1).to(node_features.dtype)
        node = node + self.node_geometry_encoder(node_geometry)
        from .proteinmpnn import get_protein_graph
        neighbor_index, edge_geometry, edge_mask = get_protein_graph(ca_coords, frames, k_neighbors=self.max_neighbors)
        neighbor_valid = residue_mask.unsqueeze(1).expand(-1, length, -1).gather(2, neighbor_index)
        edge_mask = edge_mask & effective_mask.unsqueeze(-1) & neighbor_valid
        edge = self.edge_geometry_encoder(edge_geometry)
        edge = edge * edge_mask.unsqueeze(-1).to(edge.dtype)
        for node_layer, edge_layer in zip(self.node_layers, self.edge_layers):
            node = node_layer(node, edge, neighbor_index, edge_mask)
            node = node * effective_mask.unsqueeze(-1).to(node.dtype)
            edge = edge_layer(node, edge, neighbor_index)
            edge = edge * edge_mask.unsqueeze(-1).to(edge.dtype)
        return self.output_norm(node) * effective_mask.unsqueeze(-1).to(node.dtype)
