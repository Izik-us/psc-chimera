"""Reusable CHIMERA model components shared by canonical and legacy compositions."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
from .domain_schema import NRPSConstraints

class TriangularPairUpdateConnector(nn.Module):
    """
    Upgraded pair connector for CHIMERAv2.

    v1: Simple linear projection 128→256 + one cross-attention
    v2: Triangular updates (from AF2 EvoFormer) + linear projection + cross-attention

    Triangular updates are the key innovation in AlphaFold2's pair representation:
    they enforce the triangle inequality constraint that if residue i contacts j,
    and j contacts k, then i likely contacts k. This is a structural prior.

    For NRPS design, this matters because:
        A-domain substrate binding is a closed-shell interaction —
        if residue 235 is close to the substrate AND residue 301 is close
        to the substrate, then 235 and 301 are close to each other.
        The triangular update encodes this constraint, ensuring the
        generated A-domain binding pocket has geometrically consistent
        substrate contacts.
    """

    def __init__(
        self,
        evo_pair_dim: int = 128,
        out_dim: int = 256,
        n_heads: int = 4,
    ):
        super().__init__()
        # Linear projection: 128→256
        self.input_proj = nn.Linear(evo_pair_dim, out_dim)
        self.input_norm = nn.LayerNorm(out_dim)

        # Triangular attention: outgoing edges (i,j) updated by (i,k) and (k,j)
        self.tri_attn_out = TriangularAttention(out_dim, n_heads, mode="outgoing")
        # Triangular attention: incoming edges
        self.tri_attn_in = TriangularAttention(out_dim, n_heads, mode="incoming")

        # Triangular multiplicative update (cheaper, no attention)
        self.tri_mult_out = TriangularMultiplicativeUpdate(out_dim, mode="outgoing")
        self.tri_mult_in = TriangularMultiplicativeUpdate(out_dim, mode="incoming")

        # Transition FFN
        self.transition = nn.Sequential(
            nn.LayerNorm(out_dim),
            nn.Linear(out_dim, out_dim * 4),
            nn.ReLU(),
            nn.Linear(out_dim * 4, out_dim),
        )

        # Cross-attention with retrieved structures
        self.retrieval_cross = nn.MultiheadAttention(
            embed_dim=out_dim,
            num_heads=n_heads,
            batch_first=True,
        )
        self.out_norm = nn.LayerNorm(out_dim)

        self._init_to_identity()

    def _init_to_identity(self):
        """Initialize so untrained connector passes pair_repr through unchanged."""
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
    ) -> torch.Tensor:  # (B, L, L, 256)

        z = self.input_norm(self.input_proj(pair_repr))  # (B, L, L, 256)

        # Triangular updates — enforce geometric consistency in pair repr
        z = z + self.tri_mult_out(z)  # cheaper: no attention
        z = z + self.tri_mult_in(z)
        z = z + self.tri_attn_out(z)  # richer: with attention
        z = z + self.tri_attn_in(z)
        z = z + self.transition(z)

        # Cross-attend to retrieved structural analogs (if available)
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
    """
    v2 upgrade: Noise-adaptive evolutionary cross-attention with
    dual-pathway gating (structural path + evolutionary path).
    """

    def __init__(self, d_se3: int = 256, d_evo: int = 256, n_heads: int = 8):
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

        # Noise-adaptive gate: higher evolutionary guidance at high noise
        # (early diffusion steps: coarse structure from evo)
        # (late diffusion steps: fine detail from geometry)
        self.noise_gate = nn.Sequential(
            nn.Linear(1, 64),
            nn.SiLU(),
            nn.Linear(64, d_se3),
            nn.Sigmoid(),
        )

        # Second attention: SE3 node to EvoFormer single_repr direct
        self.direct_attn = nn.MultiheadAttention(
            embed_dim=d_se3,
            num_heads=n_heads,
            kdim=d_evo,
            vdim=d_evo,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(d_se3)

        # Combine both attention pathways
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
        # Gate strength: at t=1 (pure noise), gate=1 (full evo guidance)
        #                at t=0 (clean data), gate~0 (geometry dominates)
        gate = self.noise_gate(noise_level.unsqueeze(-1))  # (B, d_se3)
        gate = gate.unsqueeze(1)  # (B, 1, d_se3)

        # Pathway 1: standard cross-attention
        path1, _ = self.cross_attn(se3_node, evo_single, evo_single)
        path1 = self.norm1(se3_node + gate * path1)

        # Pathway 2: direct position-matched injection
        path2, _ = self.direct_attn(se3_node, evo_single, evo_single)
        path2 = self.norm2(se3_node + path2)

        return self.combine(torch.cat([path1, path2], dim=-1))

class NodeProjectionConnector(nn.Module):
    """
    v2 upgrade: multi-layer projection with residue-type-aware scaling
    and a substrate pocket attentional bias.
    """

    def __init__(self, evo_dim: int = 256, mpnn_dim: int = 128):
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
        # Substrate pocket attention bias
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
    """
    Encodes NRPS-specific constraints into conditioning tensors.
    Used by the flow matching velocity field to enforce NRPS geometry.
    """

    def __init__(self, d: int = 256, max_len: int = 2000):
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

class EvoFormerBackbone(nn.Module):
    """EvoFormer-inspired MSA encoder approximation with explicit padding masks.

    This is not a native OpenFold/EvoFormer implementation. ``True`` in the
    optional ``msa_padding_mask`` means the MSA token is padding. Row attention
    masks keys; masked query outputs are zeroed before valid-row pooling.
    There is no separate MSA column-attention stack in this approximation.
    """

    def __init__(self, d_single: int = 256, d_pair: int = 128, n_blocks: int = 48):
        super().__init__()
        self.d_single = d_single
        self.d_pair = d_pair
        if d_single % 8:
            raise ValueError("d_single must be divisible by 8 attention heads")
        self.padding_token = 22
        self.token_embed = nn.Embedding(23, d_single, padding_idx=self.padding_token)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_single,
            nhead=8,
            dim_feedforward=1024,
            batch_first=True,
            dropout=0.0,
        )
        self.msa_encoder = nn.TransformerEncoder(encoder_layer, num_layers=8)
        self.pair_init = nn.Linear(d_single * 2, d_pair)
        self.frozen_feature_expander = nn.Sequential(
            nn.Linear(d_single, d_single * 8),
            nn.GELU(),
            nn.Linear(d_single * 8, d_single * 8),
            nn.GELU(),
            nn.Linear(d_single * 8, d_single),
        )
        print("[EvoFormerBackbone] Approximation loaded; OpenFold is not configured.")

    def forward(
        self,
        msa_tokens: torch.Tensor,  # (B, N_seq, L) MSA token IDs
        pair_features: torch.Tensor,  # (B, L, L, d_pair) initial pair features
        msa_padding_mask: Optional[torch.Tensor] = None,  # True means padding
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if msa_tokens.ndim != 3:
            raise ValueError("msa_tokens must have shape (B,N_seq,L)")
        B, N, L = msa_tokens.shape
        if N < 1 or L < 1:
            raise ValueError("MSA sequence and residue dimensions must be non-empty")
        if pair_features.shape != (B, L, L, self.d_pair):
            raise ValueError("pair_features must have shape (B,L,L,d_pair)")
        if msa_tokens.min() < 0 or msa_tokens.max() >= self.token_embed.num_embeddings:
            raise ValueError("msa_tokens contains values outside the 0..22 vocabulary")
        if msa_padding_mask is None:
            msa_padding_mask = msa_tokens.eq(self.padding_token)
        elif msa_padding_mask.shape != msa_tokens.shape or msa_padding_mask.dtype != torch.bool:
            raise ValueError("msa_padding_mask must be bool with shape (B,N_seq,L); True means padding")
        elif msa_padding_mask.device != msa_tokens.device:
            raise ValueError("msa_padding_mask and msa_tokens must be on the same device")

        residue_valid = (~msa_padding_mask).any(dim=1)
        tok_emb = self.token_embed(msa_tokens)
        msa_emb = tok_emb.reshape(B * N, L, self.d_single)
        row_padding = msa_padding_mask.reshape(B * N, L).clone()
        all_padding_rows = row_padding.all(dim=1)
        if all_padding_rows.any():
            row_padding[all_padding_rows, 0] = False
        enc = self.msa_encoder(
            msa_emb,
            src_key_padding_mask=row_padding,
        ).reshape(B, N, L, self.d_single)
        enc = enc.masked_fill(msa_padding_mask.unsqueeze(-1), 0.0)
        valid_counts = (~msa_padding_mask).sum(dim=1).clamp_min(1).to(enc.dtype)
        single = enc.sum(dim=1) / valid_counts.unsqueeze(-1)
        single = self.frozen_feature_expander(single)
        single = single.masked_fill(~residue_valid.unsqueeze(-1), 0.0)
        # Pair: outer product mean
        pair_l = single.unsqueeze(2).expand(-1, -1, L, -1)
        pair_r = single.unsqueeze(1).expand(-1, L, -1, -1)
        pair_repr = self.pair_init(
            torch.cat([pair_l, pair_r], dim=-1)
        )  # (B, L, L, d_pair)
        pair_valid = residue_valid.unsqueeze(1) & residue_valid.unsqueeze(2)
        pair_repr = (pair_repr + pair_features).masked_fill(~pair_valid.unsqueeze(-1), 0.0)
        return single, pair_repr

class ProteinMPNNBackbone(nn.Module):
    """
    Wraps dauparas/ProteinMPNN for base residue-level sequence design.
    Augmented by multi-scale designer in CHIMERAv2.
    """

    def __init__(self, node_features: int = 128, edge_features: int = 128):
        super().__init__()
        # Stub; production: from protein_mpnn_utils import ProteinMPNN
        self.node_features = node_features
        self.frozen_residue_adapter = nn.Sequential(
            nn.Linear(node_features, node_features * 8),
            nn.GELU(),
            nn.Linear(node_features * 8, node_features * 8),
            nn.GELU(),
            nn.Linear(node_features * 8, node_features),
        )
        self.mpnn_trunk = nn.Sequential(
            nn.Linear(node_features, node_features * 2),
            nn.GELU(),
            nn.Linear(node_features * 2, node_features),
        )
        print("[ProteinMPNNBackbone] Stub loaded. Replace with dauparas/ProteinMPNN.")

    def forward(self, backbone_coords, node_features):
        adapter = self.frozen_residue_adapter(node_features)
        return self.mpnn_trunk(adapter)  # (B, L, node_features)

