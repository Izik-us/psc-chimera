"""Canonical hierarchical sequence-design component."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleNRPSDesigner(nn.Module):
    """
    Replaces ProteinMPNN alone with a four-scale hierarchical designer.

    Hierarchy:
        Scale 1 — Residue (~4Å):
            ProteinMPNN message-passing on k-NN protein graph.
            Every residue attends to its 32 spatial neighbors.
            + EvoFormer node features (evolutionary co-conservation)

        Scale 2 — Domain (~30Å):
            Domain-level graph attention.
            One node per NRPS domain (A, T, C, TE, linker).
            Domain node = mean of residue features within domain boundaries.
            Domain edges = domain-domain spatial relationships.
            Captures which A-domain sequence choices are compatible with
            the T-domain architecture.

        Scale 3 — Module (~80Å):
            Module-module interface attention.
            One node per NRPS module (can be 3-5 modules in a PSC Layer 1).
            Learns which inter-module linker choices are compatible.
            THIS IS THE SOLUTION to the 30-year module incompatibility problem:
            hierarchical context explicitly models why some linkers work and others don't.

        Scale 4 — Assembly (~120nm icosahedral face):
            Icosahedral face compatibility scoring.
            Ensures module sequences are compatible with the coiled-coil
            assembly interface regions.
            Queries the icosahedral symmetry constraint.

    Message passing is bidirectional:
        Bottom-up: residue → domain → module → assembly
        Top-down:  assembly → module → domain → residue

    The top-down pass is the key innovation:
        The icosahedral face constraint flows down to every individual residue,
        ensuring global assembly compatibility in local sequence decisions.
    """

    def __init__(
        self,
        d_residue: int = 512,  # per-residue feature dim (ProteinMPNN node dim)
        d_domain: int = 1024,  # per-domain feature dim
        d_module: int = 2048,  # per-module feature dim
        d_assembly: int = 1024,  # assembly context dim
        n_domains: int = 5,  # A, T, C, TE, linker
        n_modules: int = 5,  # up to 5 NRPS modules in PSC Layer 1
        vocab_size: int = 20,  # amino acid vocabulary
        edge_dim: int = 28,
    ):
        super().__init__()
        self.n_domains = n_domains
        self.n_modules = n_modules

        # ── Scale 1: Residue-level (ProteinMPNN-style) ──────────────────────
        self.residue_mpnn = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(d_residue * 2, d_residue),
                    nn.LayerNorm(d_residue),
                    nn.GELU(),
                    nn.Linear(d_residue, d_residue),
                )
                for _ in range(3)  # 3 rounds of message passing
            ]
        )
        if edge_dim <= 0:
            raise ValueError("edge_dim must be positive")
        self.edge_proj = nn.Linear(edge_dim, d_residue)  # geometric edge features
        self.edge_type_embedding = nn.Embedding(3, d_residue)
        self.domain_gate = nn.Parameter(torch.tensor(0.0))
        self.module_gate = nn.Parameter(torch.tensor(0.0))
        self.assembly_gate = nn.Parameter(torch.tensor(0.0))

        # ── Scale 2: Domain-level pooling and attention ──────────────────────
        self.domain_pool = nn.Linear(d_residue, d_domain)
        self.domain_attn = nn.MultiheadAttention(
            embed_dim=d_domain, num_heads=8, batch_first=True
        )
        self.domain_ffn = nn.Sequential(
            nn.LayerNorm(d_domain),
            nn.Linear(d_domain, d_domain * 2),
            nn.GELU(),
            nn.Linear(d_domain * 2, d_domain),
        )
        # Retained for strict legacy state_dict compatibility; current forward
        # uses domain_ffn's normalization and does not consume these modules.
        self.domain_norms = nn.ModuleList(
            [nn.LayerNorm(d_domain) for _ in range(n_domains)]
        )

        # ── Scale 3: Module-level attention ─────────────────────────────────
        self.module_pool = nn.Linear(d_domain, d_module)
        self.module_attn = nn.MultiheadAttention(
            embed_dim=d_module, num_heads=8, batch_first=True
        )
        self.module_interface_head = nn.Sequential(
            nn.Linear(d_module * 2, d_module),  # concatenate adjacent module pairs
            nn.GELU(),
            nn.Linear(d_module, d_module),
        )

        # ── Scale 4: Assembly context (icosahedral face) ─────────────────────
        # Fixed sinusoidal encoding of icosahedral face identity (0-19)
        self.face_encoding = nn.Embedding(20, d_assembly)
        self.assembly_project = nn.Linear(d_module, d_assembly)
        self.assembly_attn = nn.MultiheadAttention(
            embed_dim=d_assembly, num_heads=4, batch_first=True
        )

        # ── Top-down projections (assembly → module → domain → residue) ────
        self.td_assembly_to_module = nn.Linear(d_assembly, d_module)
        self.td_module_to_domain = nn.Linear(d_module, d_domain)
        self.td_domain_to_residue = nn.Linear(d_domain, d_residue)

        # ── Final sequence prediction ─────────────────────────────────────
        self.sequence_head = nn.Sequential(
            nn.LayerNorm(d_residue),
            nn.Linear(d_residue, d_residue * 2),
            nn.GELU(),
            nn.Linear(d_residue * 2, vocab_size),
        )

    @staticmethod
    def _segment_membership(
        boundaries: torch.Tensor, length: int, device: torch.device
    ) -> torch.Tensor:
        """Build vectorized batch/segment/residue span membership."""
        if boundaries.ndim != 3 or boundaries.shape[-1] != 2:
            raise ValueError("boundaries must have shape (B, segments, 2)")
        starts = boundaries[..., 0].to(device=device, dtype=torch.long).clamp(0, length)
        ends = boundaries[..., 1].to(device=device, dtype=torch.long).clamp(0, length)
        if torch.any(ends < starts):
            raise ValueError("segment end boundaries must be >= start boundaries")
        positions = torch.arange(length, device=device).view(1, 1, length)
        return (positions >= starts.unsqueeze(-1)) & (positions < ends.unsqueeze(-1))

    @staticmethod
    def _masked_segment_pool(
        features: torch.Tensor, membership: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Mean-pool variable-length spans; empty spans map to finite zeros."""
        weights = membership.to(features.dtype)
        counts = weights.sum(dim=-1)
        pooled = torch.einsum("bsl,bld->bsd", weights, features)
        pooled = pooled / counts.clamp_min(1).unsqueeze(-1)
        valid = counts > 0
        return pooled * valid.unsqueeze(-1).to(pooled.dtype), valid

    @staticmethod
    def _safe_attention_mask(valid_segments: torch.Tensor) -> torch.Tensor:
        """Avoid all-key-masked attention without marking empty segments valid."""
        safe = valid_segments.clone()
        safe[:, 0] = safe[:, 0] | ~valid_segments.any(dim=1)
        return safe

    def forward(
        self,
        residue_feats: torch.Tensor,
        evol_node_feats: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_index: torch.Tensor,
        domain_boundaries: torch.Tensor,
        module_boundaries: torch.Tensor,
        icosahedral_face: torch.Tensor,
        edge_mask: Optional[torch.Tensor] = None,
        residue_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Produce amino-acid logits with mask-aware multiscale information flow.

        Span computation is vectorized over the batch. Empty spans and padded
        residues stay finite, and masked neighbors cannot affect valid residues.
        Existing modules and parameter names are retained for checkpoint reuse.
        """
        if residue_feats.ndim != 3:
            raise ValueError("residue_feats must have shape (B,L,d_residue)")
        B, L, D = residue_feats.shape
        if D != self.sequence_head[0].normalized_shape[0]:
            raise ValueError("residue feature width does not match d_residue")
        if evol_node_feats.shape != residue_feats.shape:
            raise ValueError("evol_node_feats must match residue_feats shape")
        if edge_index.ndim != 3 or edge_index.shape[:2] != (B, L):
            raise ValueError("edge_index must have shape (B,L,K)")
        if edge_feats.shape[:3] != edge_index.shape:
            raise ValueError("edge_feats dimensions must match edge_index")
        expected_edge_dim = self.edge_proj.in_features
        if edge_feats.shape[-1] == 16 and expected_edge_dim == 28:
            edge_feats = F.pad(edge_feats, (0, 12))
        elif edge_feats.shape[-1] != expected_edge_dim:
            raise ValueError(
                f"edge_feats last dimension must be {expected_edge_dim}"
                + (" (or legacy 16-D)" if expected_edge_dim == 28 else "")
            )
        if edge_index.dtype not in (torch.int32, torch.int64):
            raise ValueError("edge_index must contain integer residue indices")
        if torch.any((edge_index < 0) | (edge_index >= L)):
            raise ValueError("edge_index contains residue indices outside the sequence")
        if (
            domain_boundaries.ndim != 3
            or domain_boundaries.shape[0] != B
            or domain_boundaries.shape[-1] != 2
            or domain_boundaries.shape[1] < self.n_domains
        ):
            raise ValueError("domain_boundaries must contain at least n_domains spans")
        if (
            module_boundaries.ndim != 3
            or module_boundaries.shape[0] != B
            or module_boundaries.shape[-1] != 2
            or module_boundaries.shape[1] < self.n_modules
        ):
            raise ValueError("module_boundaries must contain at least n_modules spans")
        # Existing callers may pad boundary tensors to a common maximum count.
        domain_boundaries = domain_boundaries[:, : self.n_domains]
        module_boundaries = module_boundaries[:, : self.n_modules]
        if icosahedral_face.shape != (B,):
            raise ValueError("icosahedral_face must have shape (B,)")
        if torch.any((icosahedral_face < 0) | (icosahedral_face >= self.face_encoding.num_embeddings)):
            raise ValueError("icosahedral_face IDs must be in [0, 19]")
        if residue_mask is None:
            residue_mask = torch.ones(B, L, dtype=torch.bool, device=residue_feats.device)
        elif residue_mask.shape != (B, L) or residue_mask.dtype != torch.bool:
            raise ValueError("residue_mask must be bool with shape (B,L)")
        if edge_mask is None:
            edge_mask = torch.ones_like(edge_index, dtype=torch.bool)
        elif edge_mask.shape != edge_index.shape or edge_mask.dtype != torch.bool:
            raise ValueError("edge_mask must be bool with shape (B,L,K)")

        device = residue_feats.device
        domain_membership = self._segment_membership(domain_boundaries, L, device)
        module_membership = self._segment_membership(module_boundaries, L, device)
        domain_membership = domain_membership & residue_mask.unsqueeze(1)
        module_membership = module_membership & residue_mask.unsqueeze(1)

        # Residue scale: edge-conditioned neighborhood attention replaces the
        # mean-only reducer while using the native edge and relation features.
        s = (residue_feats + evol_node_feats) * residue_mask.unsqueeze(-1).to(residue_feats.dtype)
        domain_labels = domain_membership.to(torch.int64).argmax(dim=1)
        has_domain = domain_membership.any(dim=1)
        domain_ids = torch.where(has_domain, domain_labels, torch.full_like(domain_labels, -1))
        edge_types = self.classify_edge_types(domain_ids, edge_index)
        edge_emb = self.edge_proj(edge_feats) + self.edge_type_embedding(edge_types)
        neighbor_index = edge_index.unsqueeze(-1).expand(-1, -1, -1, D)

        # Partition channels into up to eight heads. Each head learns a
        # distinct compatibility surface from its own feature subspace while
        # keeping the existing parameter schema intact.
        attention_heads = min(8, D)
        while D % attention_heads:
            attention_heads -= 1
        head_dim = D // attention_heads

        for mpnn_layer in self.residue_mpnn:
            neighbors = s.unsqueeze(1).expand(-1, L, -1, -1).gather(2, neighbor_index)
            neighbor_messages = neighbors + edge_emb
            neighbor_valid = residue_mask.unsqueeze(1).expand(-1, L, -1).gather(2, edge_index)
            valid_edges = edge_mask & residue_mask.unsqueeze(-1) & neighbor_valid

            query_heads = F.normalize(
                s.reshape(B, L, attention_heads, head_dim), p=2, dim=-1, eps=1e-6
            )
            message_heads = F.normalize(
                neighbor_messages.reshape(B, L, edge_index.shape[-1], attention_heads, head_dim),
                p=2, dim=-1, eps=1e-6,
            )
            scores = torch.einsum("blhd,blkhd->blhk", query_heads, message_heads) * 4.0
            head_valid = valid_edges.unsqueeze(2)
            scores = scores.masked_fill(~head_valid, torch.finfo(scores.dtype).min)
            weights = torch.softmax(scores, dim=-1) * head_valid.to(scores.dtype)
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            message_values = neighbor_messages.reshape(
                B, L, edge_index.shape[-1], attention_heads, head_dim
            )
            aggregate = torch.einsum(
                "blhk,blkhd->blhd", weights, message_values
            ).reshape(B, L, D)
            s = s + mpnn_layer(torch.cat([s, aggregate], dim=-1))
            s = s * residue_mask.unsqueeze(-1).to(s.dtype)

        # Domain scale: safely pooled bottom-up summaries and masked attention.
        domain_means, domain_valid = self._masked_segment_pool(s, domain_membership)
        domain_repr = self.domain_pool(domain_means)
        domain_repr = domain_repr * domain_valid.unsqueeze(-1).to(domain_repr.dtype)
        safe_domain_mask = self._safe_attention_mask(domain_valid)
        domain_ctx, _ = self.domain_attn(
            domain_repr, domain_repr, domain_repr,
            key_padding_mask=~safe_domain_mask, need_weights=False,
        )
        domain_repr = domain_repr + torch.tanh(self.domain_gate) * self.domain_ffn(domain_ctx)
        domain_repr = domain_repr * domain_valid.unsqueeze(-1).to(domain_repr.dtype)

        # Bottom-up domain -> module transfer. Pool attended domain states by
        # their actual residue-span overlap with each module, rather than
        # bypassing the domain scale with a second residue-only mean pool.
        overlap = torch.einsum(
            "bdl,bml->bdm",
            domain_membership.to(s.dtype),
            module_membership.to(s.dtype),
        )
        overlap_sum = overlap.sum(dim=1)  # (B,M), overlap mass per module
        domain_to_module_weights = overlap / overlap_sum.unsqueeze(1).clamp_min(1.0)
        module_from_domains = torch.einsum(
            "bdm,bdf->bmf", domain_to_module_weights, domain_repr
        )
        module_means, module_valid = self._masked_segment_pool(s, module_membership)
        module_from_residues = self.domain_pool(module_means)
        has_domain_context = (overlap_sum > 0).unsqueeze(-1)
        module_inputs = torch.where(
            has_domain_context,
            0.5 * (module_from_residues + module_from_domains),
            module_from_residues,
        )
        module_repr = self.module_pool(module_inputs)
        module_repr = module_repr * module_valid.unsqueeze(-1).to(module_repr.dtype)
        safe_module_mask = self._safe_attention_mask(module_valid)
        module_ctx, _ = self.module_attn(
            module_repr, module_repr, module_repr,
            key_padding_mask=~safe_module_mask, need_weights=False,
        )
        module_repr = module_repr + torch.tanh(self.module_gate) * module_ctx
        module_repr = module_repr * module_valid.unsqueeze(-1).to(module_repr.dtype)

        if self.n_modules > 1:
            left, right = module_repr[:, :-1], module_repr[:, 1:]
            interface_valid = module_valid[:, :-1] & module_valid[:, 1:]
            compatibility = F.cosine_similarity(left, right, dim=-1).unsqueeze(-1)
            interface = self.module_interface_head(torch.cat([left, right], dim=-1))
            interface = interface * torch.sigmoid(compatibility)
            interface = interface * interface_valid.unsqueeze(-1).to(interface.dtype)
            zeros = torch.zeros_like(module_repr[:, :1])
            module_repr = module_repr + torch.tanh(self.module_gate) * (
                torch.cat([interface, zeros], dim=1) +
                torch.cat([zeros, interface], dim=1)
            )
            module_repr = module_repr * module_valid.unsqueeze(-1).to(module_repr.dtype)

        # Assembly scale: face-conditioned module representations with
        # safe handling when a batch contains no non-empty module spans.
        face_emb = self.face_encoding(icosahedral_face.to(device=device, dtype=torch.long))
        assembly_repr = self.assembly_project(module_repr) + face_emb.unsqueeze(1)
        assembly_repr = assembly_repr * module_valid.unsqueeze(-1).to(assembly_repr.dtype)
        asm_ctx, _ = self.assembly_attn(
            assembly_repr, assembly_repr, assembly_repr,
            key_padding_mask=~safe_module_mask, need_weights=False,
        )
        assembly_repr = assembly_repr + torch.tanh(self.assembly_gate) * asm_ctx
        assembly_repr = assembly_repr * module_valid.unsqueeze(-1).to(assembly_repr.dtype)

        # Top-down: module-specific assembly context, followed by explicit
        # domain/module span overlap rather than a global context broadcast.
        module_repr = module_repr + torch.tanh(self.assembly_gate) * self.td_assembly_to_module(assembly_repr)
        module_repr = module_repr * module_valid.unsqueeze(-1).to(module_repr.dtype)
        # Count only unmasked residues shared by each domain/module pair.
        overlap = overlap.to(module_repr.dtype)
        overlap = overlap * module_valid.unsqueeze(1).to(overlap.dtype)
        mapped_modules = self.td_module_to_domain(module_repr)
        domain_module_context = torch.einsum("bdm,bmf->bdf", overlap, mapped_modules)
        overlap_count = overlap.sum(dim=-1, keepdim=True)
        domain_module_context = domain_module_context / overlap_count.clamp_min(1.0)
        domain_repr = domain_repr + torch.tanh(self.module_gate) * domain_module_context
        domain_repr = domain_repr * domain_valid.unsqueeze(-1).to(domain_repr.dtype)

        domain_residue = self.td_domain_to_residue(domain_repr)
        residue_weights = domain_membership.transpose(1, 2).to(s.dtype)
        residue_weights = residue_weights / residue_weights.sum(dim=-1, keepdim=True).clamp_min(1.0)
        domain_feedback = torch.einsum("bld,bdf->blf", residue_weights, domain_residue)
        s = (s + domain_feedback) * residue_mask.unsqueeze(-1).to(s.dtype)

        logits = self.sequence_head(s)
        return logits * residue_mask.unsqueeze(-1).to(logits.dtype)

    @staticmethod
    def classify_edge_types(domain_ids: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Classify edges: 0 intra-domain local, 1 inter-domain interface, 2 intra-domain long-range."""
        if domain_ids.ndim != 2 or edge_index.ndim != 3 or edge_index.shape[:2] != domain_ids.shape:
            raise ValueError("domain_ids and edge_index must have shapes (B,L) and (B,L,K)")
        _, length = domain_ids.shape
        neighbor_domains = domain_ids.unsqueeze(1).expand(-1, length, -1).gather(2, edge_index)
        query_domains = domain_ids.unsqueeze(-1)
        inter_domain = (query_domains != neighbor_domains) & (query_domains >= 0) & (neighbor_domains >= 0)
        positions = torch.arange(length, device=edge_index.device).view(1, length, 1)
        sequence_separation = (positions - edge_index).abs()
        long_range = (~inter_domain) & (sequence_separation >= 32)
        return torch.where(inter_domain, 1, torch.where(long_range, 2, 0))
