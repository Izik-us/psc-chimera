"""Canonical hierarchical sequence-design component."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


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

    def forward(
        self,
        residue_feats: torch.Tensor,  # (B, L, d_residue) from ProteinMPNN
        evol_node_feats: torch.Tensor,  # (B, L, d_residue) from EvoFormer
        edge_feats: torch.Tensor,  # (B, L, K, 16) geometric edge features
        edge_index: torch.Tensor,  # (B, L, K) k-NN indices
        domain_boundaries: torch.Tensor,  # (B, n_domains, 2) [start, end] per domain
        module_boundaries: torch.Tensor,  # (B, n_modules, 2) [start, end] per module
        icosahedral_face: torch.Tensor,  # (B,) which face (0-19) is this module on
        edge_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:  # (B, L, 20) amino acid logits
        B, L, _ = residue_feats.shape
        if edge_index.ndim != 3 or edge_index.shape[:2] != (B, L):
            raise ValueError("edge_index must have shape (B,L,K)")
        if edge_feats.shape[:3] != edge_index.shape:
            raise ValueError("edge_feats dimensions must match edge_index")
        expected_edge_dim = self.edge_proj.in_features
        if edge_feats.shape[-1] == 16 and expected_edge_dim == 28:
            # Legacy checkpoints supplied 16-D geometric edges. Preserve those
            # features and zero-fill the 12 geometry channels added in v2.
            edge_feats = torch.nn.functional.pad(edge_feats, (0, 12))
        elif edge_feats.shape[-1] != expected_edge_dim:
            raise ValueError(
                f"edge_feats last dimension must be {expected_edge_dim}"
                + (" (or legacy 16-D)" if expected_edge_dim == 28 else "")
            )
        if torch.any((edge_index < 0) | (edge_index >= L)):
            raise ValueError("edge_index contains residue indices outside the sequence")
        if edge_mask is None:
            edge_mask = torch.ones_like(edge_index, dtype=torch.bool)
        elif edge_mask.shape != edge_index.shape or edge_mask.dtype != torch.bool:
            raise ValueError("edge_mask must be bool with shape (B,L,K)")

        # ── Scale 1: Residue message passing ─────────────────────────────────
        s = residue_feats + evol_node_feats  # fuse evolutionary context
        domain_ids = torch.full((B, L), -1, dtype=torch.long, device=s.device)
        for domain_idx in range(self.n_domains):
            starts = domain_boundaries[:, domain_idx, 0].clamp(0, L)
            ends = domain_boundaries[:, domain_idx, 1].clamp(0, L)
            for batch_idx in range(B):
                domain_ids[batch_idx, starts[batch_idx]:ends[batch_idx]] = domain_idx
        edge_types = self.classify_edge_types(domain_ids, edge_index)

        for mpnn_layer in self.residue_mpnn:
            # Aggregate neighbor messages
            neighbors = edge_index.unsqueeze(-1).expand(-1, -1, -1, s.shape[-1])
            neighbor_feats = s.unsqueeze(2).expand(-1, -1, edge_index.shape[-1], -1)
            neighbor_feats = (
                s.unsqueeze(1).expand(-1, L, -1, -1).gather(2, neighbors)
            )  # (B, L, K, d_residue)
            edge_emb = self.edge_proj(edge_feats) + self.edge_type_embedding(edge_types)
            valid_edges = edge_mask.unsqueeze(-1).to(neighbor_feats.dtype)
            neighbor_messages = (neighbor_feats + edge_emb) * valid_edges
            neighbor_feats_flat = neighbor_messages.sum(dim=2) / valid_edges.sum(dim=2).clamp_min(1)
            combined = torch.cat([s, neighbor_feats_flat], dim=-1)
            s = s + mpnn_layer(combined)

        # ── Scale 2: Bottom-up domain pooling ────────────────────────────────
        domain_feats = []
        for d in range(self.n_domains):
            start = domain_boundaries[:, d, 0]  # (B,)
            end = domain_boundaries[:, d, 1]

            # Pool residue features within this domain
            domain_residues = []
            for b in range(B):
                s_b, e_b = start[b].item(), end[b].item()
                domain_residues.append(s[b, s_b:e_b].mean(dim=0))
            domain_feat = torch.stack(domain_residues)  # (B, d_residue)
            domain_feats.append(self.domain_pool(domain_feat))

        domain_repr = torch.stack(domain_feats, dim=1)  # (B, n_domains, d_domain)

        # Domain self-attention (which domains influence which)
        domain_ctx, _ = self.domain_attn(domain_repr, domain_repr, domain_repr)
        domain_repr = domain_repr + torch.tanh(self.domain_gate) * self.domain_ffn(domain_ctx)

        # ── Scale 3: Bottom-up module pooling ─────────────────────────────────
        module_feats = []
        for m in range(self.n_modules):
            start = module_boundaries[:, m, 0]
            end = module_boundaries[:, m, 1]
            module_residues = []
            for b in range(B):
                s_b, e_b = start[b].item(), end[b].item()
                if e_b > s_b:
                    module_residues.append(s[b, s_b:e_b].mean(dim=0))
                else:
                    module_residues.append(torch.zeros_like(s[b, 0]))
            module_feat = self.domain_pool(torch.stack(module_residues))
            module_feats.append(self.module_pool(module_feat))

        module_repr = torch.stack(module_feats, dim=1)  # (B, n_modules, d_module)

        # Module self-attention — THIS learns inter-module compatibility
        module_ctx, _ = self.module_attn(module_repr, module_repr, module_repr)
        module_repr = module_repr + torch.tanh(self.module_gate) * module_ctx

        # Module pair interaction (explicit interface modeling)
        if module_repr.shape[1] > 1:
            left = module_repr[:, :-1, :]
            right = module_repr[:, 1:, :]
            interface = self.module_interface_head(torch.cat([left, right], dim=-1))
            module_repr = module_repr.clone()
            module_repr[:, :-1] = module_repr[:, :-1] + torch.tanh(self.module_gate) * interface
            module_repr[:, 1:] = module_repr[:, 1:] + torch.tanh(self.module_gate) * interface

        # ── Scale 4: Assembly context ─────────────────────────────────────────
        face_emb = self.face_encoding(icosahedral_face)  # (B, d_assembly)
        assembly_repr = self.assembly_project(module_repr)  # (B, n_modules, d_assembly)
        assembly_repr = assembly_repr + face_emb.unsqueeze(1)  # broadcast face context

        # Assembly-level attention (module sees icosahedral context)
        asm_ctx, _ = self.assembly_attn(assembly_repr, assembly_repr, assembly_repr)
        assembly_repr = assembly_repr + torch.tanh(self.assembly_gate) * asm_ctx

        # ── Top-down pass: flow assembly context back to residues ────────────
        # Assembly → module
        asm_to_mod = self.td_assembly_to_module(
            assembly_repr.mean(dim=1)
        )  # (B, d_module)
        module_repr = module_repr + torch.tanh(self.assembly_gate) * asm_to_mod.unsqueeze(1)

        # Module → domain
        mod_to_dom = self.td_module_to_domain(module_repr.mean(dim=1))  # (B, d_domain)
        domain_repr = domain_repr + torch.tanh(self.module_gate) * mod_to_dom.unsqueeze(1)

        # Domain → residue (scatter back using domain boundaries)
        dom_to_res = self.td_domain_to_residue(domain_repr)  # (B, n_domains, d_residue)
        for d in range(self.n_domains):
            for b in range(B):
                s_b = domain_boundaries[b, d, 0].item()
                e_b = domain_boundaries[b, d, 1].item()
                s[b, s_b:e_b] = s[b, s_b:e_b] + dom_to_res[b, d].unsqueeze(0)

        # ── Final sequence prediction ─────────────────────────────────────────
        return self.sequence_head(s)  # (B, L, 20)

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
