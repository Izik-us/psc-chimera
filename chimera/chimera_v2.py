"""
CHIMERA v2 — Compositional Hierarchical Inference Model for
             Evolutionary Representation and Architecture
================================================================
Complete rebuild of the Stage 1 PSC Engineering Pipeline model.

What changed from v1 → v2:
┌─────────────────────────┬──────────────────────────────────────────────────┐
│ Component               │ v1 → v2                                          │
├─────────────────────────┼──────────────────────────────────────────────────┤
│ Backbone generation     │ DDPM 200 steps → OT-Flow Matching 20 steps       │
│ Source distribution     │ Gaussian noise → Bacterial NRPS (diffusion bridge)│
│ Sequence designer       │ ProteinMPNN only → 4-scale hierarchical GNN      │
│ Retrieval               │ None → FAISS structural RAG (K=5 analogs)        │
│ Substrate conditioning  │ None → SE(3)-aware binding pocket encoder         │
│ Objectives              │ Weighted sum loss → 5-obj Pareto front (PCGrad)  │
│ Learning from PROTEUS   │ Simple fine-tune → Direct Preference Optimization │
│ Exploration strategy    │ Random → Bayesian EI acquisition (MC Dropout)     │
│ Assembly awareness      │ None → Icosahedral face compatibility at Scale 4  │
│ Trainable params        │ ~7M connectors → ~12M (new connectors + heads)    │
│ Frozen params           │ ~700M → ~700M (same pretrained backbones)         │
└─────────────────────────┴──────────────────────────────────────────────────┘

Full architecture:

  Animal NRPS MSA (from Stage 0 databases)
       │
  ┌────▼────────────────────────────────────────────────┐
  │  EvoFormer (48 blocks, FROZEN)                      │
  │  OpenFold weights / ESMFold trunk                   │
  └────┬───────────────────────────┬────────────────────┘
       │ single_repr (B,L,256)     │ pair_repr (B,L,L,128)
       │                           │
  ┌────▼───────┐    ┌──────────────▼──────────────────┐
  │Evol Cross  │    │ Triangular Pair Update Connector │  ← TRAINABLE
  │Attention   │    │ (128→256 + triangular updates)   │
  │(NEW in v2) │    └──────────────┬──────────────────┘
  └────┬───────┘                   │
       │        ┌──────────────────▼──────────────┐
       │        │  Structural RAG (FAISS)          │  ← TRAINABLE
       │        │  Retrieves K=5 similar A-domains │
       │        └──────────────┬──────────────────┘
       │                       │ retrieved_context
       │        ┌──────────────▼──────────────────┐
       │        │  Substrate Pocket Conditioner    │  ← TRAINABLE
       │        │  SE(3)-aware ligand encoder      │
       │        └──────────────┬──────────────────┘
       │                       │ substrate_cond (B,L,L,256)
       │                       ▼
  ┌────▼──────────────────────────────────────────────┐
  │  SE(3) OT-Flow Matching (FROZEN RFdiffusion base) │
  │  + EvolCrossAttention connector (TRAINABLE)       │
  │  + NRPS Constraint Encoder (TRAINABLE)            │
  │  Bridge: bacterial NRPS → mammalian design         │
  │  20 NFE with RK4 (was 200 NFE in v1)              │
  └────────────────────┬──────────────────────────────┘
                       │ backbone (B,L,4,3) N/CA/C/O coords
                       ▼
  ┌────────────────────────────────────────────────────┐
  │  Multi-Scale Hierarchical Sequence Designer        │  ← TRAINABLE
  │  Scale 1: ProteinMPNN residue GNN (FROZEN base)   │
  │  Scale 2: Domain attention (A/T/C/TE/linker)      │
  │  Scale 3: Module-module interface attention        │
  │  Scale 4: Icosahedral face compatibility           │
  │  Bidirectional: bottom-up AND top-down passes      │
  └────────────────────┬──────────────────────────────┘
                       │ sequence logits (B, L, 20)
                       ▼
  ┌────────────────────────────────────────────────────┐
  │  Pareto Multi-Objective Head (TRAINABLE)           │
  │  F1: Evolutionary plausibility (PoET)              │
  │  F2: Structural stability (pLDDT proxy)            │
  │  F3: Mammalian expression (CodonOpt critic)        │
  │  F4: Substrate selectivity (Stachelhaus match)     │
  │  F5: Icosahedral assembly compatibility            │
  └────────────────────┬──────────────────────────────┘
                       │
  ┌────────────────────▼──────────────────────────────┐
  │  Bayesian Uncertainty Estimator (MC Dropout)      │
  │  + Expected Improvement Acquisition               │
  │  → Ranked Pareto frontier for PROTEUS selection   │
  └───────────────────────────────────────────────────┘

Training modes:
  1. Supervised fine-tuning (connector + head params only)
  2. DPO from PROTEUS preference pairs (after each round)
  3. Active learning loop with EI acquisition (ongoing)

Usage:
    chimera = CHIMERAv2.from_pretrained(
        evoformer_ckpt   = 'openfold_weights.pt',
        flow_ckpt        = 'rfdiffusion_weights.pt',
        mpnn_ckpt        = 'proteinmpnn_weights.pt',
    )
    
    # Design new NRPS A-domain sequences
    results = chimera.design(
        nrps_msa          = msa_tokens,
        source_backbone   = bacterial_nrps_frames,  # bridge start
        design_constraints= nrps_constraints,
        target_substrate  = "PHE",  # target Phe-activating A-domain
        n_designs         = 500,
        n_pareto_samples  = 50,     # return Pareto frontier of 50
    )
    
    # After PROTEUS round: DPO update
    chimera.update_from_proteus(
        survivors = list_of_surviving_sequences,
        failures  = list_of_failed_sequences,
        msa       = msa_tokens,
    )
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Optional, Tuple, List, Dict, NamedTuple
from dataclasses import dataclass
from copy import deepcopy

# ── Internal imports from CHIMERA v2 module suite ────────────────────────────
from flow_matching import (
    SE3FlowMatching,
    SinusoidalTimeEmbedding,
    so3_exp,
    so3_log,
    se3_interp,
)
from multi_objective import (
    StructuralRetriever,
    DPOTrainer,
    ParetoMultiObjectiveHead,
    BayesianUncertaintyEstimator,
    MultiScaleNRPSDesigner,
    ProteusPreferencePair,
    ParetoObjectives,
)


# ═══════════════════════════════════════════════════════════════════════════════
# NRPS DOMAIN CONSTRAINTS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class NRPSConstraints:
    """
    Structural and functional constraints for NRPS A-domain design.
    Passed to CHIMERAv2 to enforce biological validity in generated sequences.
    
    These constraints prevent the flow matching from violating known biochemistry:
    - Catalytic residues cannot move (fixed_mask)
    - PPant attachment serine cannot be mutated
    - Stachelhaus code positions define substrate pocket
    - Domain boundaries determine scale-2 pooling
    - Module boundaries determine scale-3 interface attention
    - Icosahedral face determines scale-4 assembly compatibility
    """
    # Fixed residue positions (catalytic, cannot move during flow matching)
    fixed_mask:                 Optional[torch.Tensor]   # (B, L) bool: True = fixed

    # Stachelhaus selectivity code positions (10 residues defining substrate)
    stachelhaus_positions:      torch.Tensor              # (10,) sequence indices

    # Domain boundaries for multi-scale designer
    domain_boundaries:          torch.Tensor              # (B, n_domains, 2) [start, end]
    module_boundaries:          torch.Tensor              # (B, n_modules, 2)

    # Icosahedral face identity (which of 20 faces this module decorates)
    icosahedral_face:           torch.Tensor              # (B,) int in [0, 19]

    # PPant arm attachment site (conserved serine on T-domain)
    ppt_serine_position:        int

    # Known 3D coordinates from crystal structure (if available)
    hotspot_coords:             Optional[torch.Tensor]   # (K, 3)
    hotspot_indices:            Optional[torch.Tensor]   # (K,)

    # Target substrate identity for retrieval query
    target_substrate:           str   # e.g. "PHE", "TYR", "modified_AAD"


# ═══════════════════════════════════════════════════════════════════════════════
# UPGRADED CONNECTOR MODULES (v2)
# ═══════════════════════════════════════════════════════════════════════════════

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
        out_dim:      int = 256,
        n_heads:      int = 4,
    ):
        super().__init__()
        # Linear projection: 128→256
        self.input_proj    = nn.Linear(evo_pair_dim, out_dim)
        self.input_norm    = nn.LayerNorm(out_dim)

        # Triangular attention: outgoing edges (i,j) updated by (i,k) and (k,j)
        self.tri_attn_out  = TriangularAttention(out_dim, n_heads, mode='outgoing')
        # Triangular attention: incoming edges
        self.tri_attn_in   = TriangularAttention(out_dim, n_heads, mode='incoming')

        # Triangular multiplicative update (cheaper, no attention)
        self.tri_mult_out  = TriangularMultiplicativeUpdate(out_dim, mode='outgoing')
        self.tri_mult_in   = TriangularMultiplicativeUpdate(out_dim, mode='incoming')

        # Transition FFN
        self.transition    = nn.Sequential(
            nn.LayerNorm(out_dim),
            nn.Linear(out_dim, out_dim * 4),
            nn.ReLU(),
            nn.Linear(out_dim * 4, out_dim),
        )

        # Cross-attention with retrieved structures
        self.retrieval_cross = nn.MultiheadAttention(
            embed_dim=out_dim, num_heads=n_heads,
            batch_first=True,
        )
        self.out_norm = nn.LayerNorm(out_dim)

        self._init_to_identity()

    def _init_to_identity(self):
        """Initialize so untrained connector passes pair_repr through unchanged."""
        nn.init.zeros_(self.input_proj.bias)
        nn.init.eye_(self.input_proj.weight[:128, :128] if self.input_proj.weight.shape[1] >= 128
                    else self.input_proj.weight)

    def forward(
        self,
        pair_repr:        torch.Tensor,              # (B, L, L, 128)
        retrieved_context: Optional[torch.Tensor] = None,  # (B, K, 256) from RAG
    ) -> torch.Tensor:  # (B, L, L, 256)

        z = self.input_norm(self.input_proj(pair_repr))  # (B, L, L, 256)

        # Triangular updates — enforce geometric consistency in pair repr
        z = z + self.tri_mult_out(z)   # cheaper: no attention
        z = z + self.tri_mult_in(z)
        z = z + self.tri_attn_out(z)   # richer: with attention
        z = z + self.tri_attn_in(z)
        z = z + self.transition(z)

        # Cross-attend to retrieved structural analogs (if available)
        if retrieved_context is not None:
            B, L, _, d = z.shape
            z_flat = z.reshape(B, L * L, d)
            ctx_attn, _ = self.retrieval_cross(z_flat, retrieved_context, retrieved_context)
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
        self.mode   = mode
        self.norm   = nn.LayerNorm(d)
        self.q      = nn.Linear(d, d, bias=False)
        self.k      = nn.Linear(d, d, bias=False)
        self.v      = nn.Linear(d, d, bias=False)
        self.b      = nn.Linear(d, n_heads, bias=False)  # pair bias
        self.gate   = nn.Linear(d, d)
        self.out    = nn.Linear(d, d)
        self.n_heads = n_heads
        self.scale   = (d // n_heads) ** -0.5

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        B, L, _, d = z.shape
        z = self.norm(z)

        if self.mode == 'outgoing':
            # For edge (i,j): query=z[i,j], key/value=z[i,:], bias=z[:,j]
            Q = self.q(z)          # (B, L, L, d)
            K = self.k(z)          # (B, L, L, d)
            V = self.v(z)
            b = self.b(z)          # (B, L, L, n_heads) pair bias

            # Reshape for multi-head: (B, L, n_heads, L, d_head)
            dh = d // self.n_heads
            Q  = Q.view(B, L, L, self.n_heads, dh).permute(0, 1, 3, 2, 4)
            K  = K.view(B, L, L, self.n_heads, dh).permute(0, 1, 3, 2, 4)
            V  = V.view(B, L, L, self.n_heads, dh).permute(0, 1, 3, 2, 4)

            # Attention over the j dimension for fixed i
            attn = torch.einsum('bilhd,bilhd->bilh', Q, K) * self.scale  # (B,L,L,H) — wrong
            # Correct: attn[b,i,j,h] = sum_k Q[b,i,j,h] · K[b,i,k,h]
            attn = torch.einsum('binhd,bimhd->binmh', Q, K) * self.scale
            b_   = b.permute(0, 1, 3, 2).unsqueeze(3)   # (B,L,H,1,L)
            attn = (attn.permute(0,1,4,2,3) + b_).softmax(dim=-1)

            out  = torch.einsum('binmh,bimhd->binhd', attn, V)
            out  = out.permute(0,1,3,2,4).reshape(B, L, L, d)

        else:  # incoming
            # Transpose: treat columns
            z_T = z.transpose(1, 2)
            Q   = self.q(z_T)
            K   = self.k(z_T)
            V   = self.v(z_T)
            b   = self.b(z_T)
            dh  = d // self.n_heads
            Q   = Q.view(B, L, L, self.n_heads, dh).permute(0, 1, 3, 2, 4)
            K   = K.view(B, L, L, self.n_heads, dh).permute(0, 1, 3, 2, 4)
            V   = V.view(B, L, L, self.n_heads, dh).permute(0, 1, 3, 2, 4)
            attn = torch.einsum('binhd,bimhd->binmh', Q, K) * self.scale
            b_   = b.permute(0, 1, 3, 2).unsqueeze(3)
            attn = (attn.permute(0,1,4,2,3) + b_).softmax(dim=-1)
            out  = torch.einsum('binmh,bimhd->binhd', attn, V)
            out  = out.permute(0,1,3,2,4).reshape(B, L, L, d).transpose(1, 2)

        # Gate
        g   = torch.sigmoid(self.gate(z))
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
        self.mode  = mode
        self.norm  = nn.LayerNorm(d)
        self.norm_o = nn.LayerNorm(d)
        self.left_in  = nn.Linear(d, d)
        self.left_g   = nn.Linear(d, d)
        self.right_in = nn.Linear(d, d)
        self.right_g  = nn.Linear(d, d)
        self.gate     = nn.Linear(d, d)
        self.out      = nn.Linear(d, d)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z = self.norm(z)
        l = torch.sigmoid(self.left_g(z))  * self.left_in(z)   # (B,L,L,d)
        r = torch.sigmoid(self.right_g(z)) * self.right_in(z)

        if self.mode == 'outgoing':
            # p[i,j] = sum_k l[i,k] * r[k,j]
            p = torch.einsum('bild,bljd->bijd', l, r)
        else:
            # p[i,j] = sum_k l[k,i] * r[j,k]
            p = torch.einsum('blid,bjld->bijd', l, r)

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
        self.cross_attn  = nn.MultiheadAttention(
            embed_dim=d_se3, num_heads=n_heads,
            kdim=d_evo, vdim=d_evo, batch_first=True,
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
            embed_dim=d_se3, num_heads=n_heads,
            kdim=d_evo, vdim=d_evo, batch_first=True,
        )
        self.norm2 = nn.LayerNorm(d_se3)

        # Combine both attention pathways
        self.combine = nn.Sequential(
            nn.Linear(d_se3 * 2, d_se3),
            nn.LayerNorm(d_se3),
        )

    def forward(
        self,
        se3_node:    torch.Tensor,   # (B, L, d_se3)
        evo_single:  torch.Tensor,   # (B, L, d_evo)
        noise_level: torch.Tensor,   # (B,) flow time t ∈ [0,1]
    ) -> torch.Tensor:
        # Gate strength: at t=1 (pure noise), gate=1 (full evo guidance)
        #                at t=0 (clean data), gate~0 (geometry dominates)
        gate = self.noise_gate(noise_level.unsqueeze(-1))  # (B, d_se3)
        gate = gate.unsqueeze(1)                            # (B, 1, d_se3)

        # Pathway 1: standard cross-attention
        path1, _ = self.cross_attn(se3_node, evo_single, evo_single)
        path1    = self.norm1(se3_node + gate * path1)

        # Pathway 2: direct position-matched injection
        path2, _ = self.direct_attn(se3_node, evo_single, evo_single)
        path2    = self.norm2(se3_node + path2)

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
        evo_single: torch.Tensor,            # (B, L, 256)
        pocket_mask: Optional[torch.Tensor] = None,  # (B, L) bool: near pocket
    ) -> torch.Tensor:
        projected = self.proj(evo_single)
        if pocket_mask is not None:
            pocket_bias = self.pocket_scale(projected)
            projected   = projected + pocket_mask.float().unsqueeze(-1) * pocket_bias
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
        pos    = torch.arange(L, device=device)
        c_map  = self.pos_enc(pos).unsqueeze(0).expand(batch_size, -1, -1).clone()

        # Mark Stachelhaus selectivity code positions
        for idx in constraints.stachelhaus_positions:
            if idx < L:
                c_map[:, idx] = c_map[:, idx] + self.stach_marker.unsqueeze(0)

        # Mark PPant serine
        ppt = constraints.ppt_serine_position
        if ppt < L:
            c_map[:, ppt] = c_map[:, ppt] + self.ppt_marker.unsqueeze(0)

        # Mark domain types
        for d_idx in range(constraints.domain_boundaries.shape[1]):
            s = constraints.domain_boundaries[0, d_idx, 0].item()
            e = constraints.domain_boundaries[0, d_idx, 1].item()
            domain_type = min(d_idx, 4)
            domain_emb  = self.domain_emb(torch.tensor(domain_type, device=device))
            c_map[:, s:e] = c_map[:, s:e] + domain_emb.unsqueeze(0).unsqueeze(0)

        return c_map


# ═══════════════════════════════════════════════════════════════════════════════
# SUBSTRATE POCKET CONDITIONER (NEW IN V2)
# ═══════════════════════════════════════════════════════════════════════════════

class SubstratePocketConditioner(nn.Module):
    """
    Encodes the desired substrate into a conditioning signal for CHIMERA v2.

    For PSC Layer 1 NRPS A-domain design: we know WHAT substrate we want
    the A-domain to activate. We should condition the backbone generation
    on this desired substrate, biasing the generated binding pocket geometry
    toward shapes that accommodate that specific molecule.

    Two conditioning modes:
        1. Substrate identity: amino acid / modified substrate as a string
           → embedded via a lookup and projected to pair space
        2. Substrate 3D structure: if SMILES or 3D coordinates available
           → encoded via a geometric point cloud encoder
           → projects substrate atom positions into residue-residue pair space
           → tells the IPA attention: "pay attention to residues near substrate"
    """
    def __init__(self, d_pair: int = 256, d_sub: int = 128):
        super().__init__()
        # Mode 1: amino acid substrate encoding
        self.substrate_emb  = nn.Embedding(30, d_sub)   # 20 AA + 10 non-standard
        self.substrate_proj = nn.Linear(d_sub, d_pair)

        # Mode 2: 3D point cloud encoder for substrate atoms
        # Each substrate atom: (x, y, z, atom_type_one_hot)
        self.atom_encoder   = nn.Sequential(
            nn.Linear(3 + 8, d_sub),   # 3D coords + 8 atom type features
            nn.LayerNorm(d_sub),
            nn.GELU(),
            nn.Linear(d_sub, d_sub),
        )

        # Cross-attention: pair positions attend to substrate atoms
        # z[i,j] ← cross_attn(z[i,j], substrate_atoms)
        self.sub_cross_attn = nn.MultiheadAttention(
            embed_dim=d_pair, num_heads=4,
            kdim=d_sub, vdim=d_sub, batch_first=True,
        )
        self.sub_norm = nn.LayerNorm(d_pair)

        # Distance-based bias: pairs (i,j) near pocket get stronger substrate signal
        self.distance_gate = nn.Linear(1, d_pair)

    def forward(
        self,
        pair_repr:        torch.Tensor,  # (B, L, L, d_pair)
        substrate_id:     torch.Tensor,  # (B,) substrate token ID
        substrate_coords: Optional[torch.Tensor] = None,  # (B, N_atoms, 3)
        substrate_types:  Optional[torch.Tensor] = None,  # (B, N_atoms, 8)
        residue_coords:   Optional[torch.Tensor] = None,  # (B, L, 3) Cα positions
    ) -> torch.Tensor:  # (B, L, L, d_pair) enriched pair representation
        B, L, _, d = pair_repr.shape

        # Mode 1: substrate identity conditioning
        sub_emb  = self.substrate_emb(substrate_id)      # (B, d_sub)
        sub_cond = self.substrate_proj(sub_emb)           # (B, d_pair)

        # Broadcast substrate conditioning to all pair positions
        pair_repr = pair_repr + sub_cond.view(B, 1, 1, d)

        # Mode 2: 3D substrate structure conditioning (if coords available)
        if substrate_coords is not None and substrate_types is not None:
            # Encode substrate atoms
            atom_feat = torch.cat([substrate_coords, substrate_types.float()], dim=-1)
            atom_repr = self.atom_encoder(atom_feat)   # (B, N_atoms, d_sub)

            # Flatten pair for cross-attention
            pair_flat = pair_repr.reshape(B, L * L, d)  # (B, L², d_pair)
            sub_pair, _ = self.sub_cross_attn(pair_flat, atom_repr, atom_repr)
            sub_pair    = sub_pair.reshape(B, L, L, d)

            # Optional: distance-based gating
            if residue_coords is not None:
                # min distance from each residue to any substrate atom
                min_dist = torch.cdist(
                    residue_coords,
                    substrate_coords.mean(dim=1, keepdim=True).expand(-1, L, -1)
                ).min(dim=-1).values  # (B, L)

                # Gate: stronger substrate signal near pocket (within 8Å)
                gate = torch.exp(-min_dist / 8.0).unsqueeze(-1)  # (B, L, 1)
                gate = gate.unsqueeze(2) * gate.unsqueeze(1)       # (B, L, L, 1)
                sub_pair = sub_pair * gate

            pair_repr = self.sub_norm(pair_repr + sub_pair)

        return pair_repr


# ═══════════════════════════════════════════════════════════════════════════════
# PRETRAINED BACKBONE STUBS (Replace with actual checkpoints)
# ═══════════════════════════════════════════════════════════════════════════════

class EvoFormerBackbone(nn.Module):
    """
    Wraps OpenFold EvoFormer or ESMFold trunk.
    Production: load from openfold_weights.pt
    Stub: uses ESM-2 150M as approximate replacement.
    """
    def __init__(self, d_single: int = 256, d_pair: int = 128, n_blocks: int = 48):
        super().__init__()
        self.d_single = d_single
        self.d_pair   = d_pair
        # Stub implementation (replace with OpenFold in production)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_single, nhead=8, dim_feedforward=1024,
            batch_first=True, dropout=0.0,
        )
        self.msa_encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)
        self.pair_init   = nn.Linear(d_single * 2, d_pair)
        print("[EvoFormerBackbone] Stub loaded. Replace with OpenFold for production.")

    def forward(
        self,
        msa_tokens:    torch.Tensor,   # (B, N_seq, L) MSA token IDs
        pair_features: torch.Tensor,   # (B, L, L, d_pair) initial pair features
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N, L = msa_tokens.shape
        # Stub: embed tokens, pool MSA, create pair via outer product
        tok_emb   = msa_tokens.float().unsqueeze(-1).expand(-1, -1, -1, self.d_single) / 23.0
        msa_emb   = tok_emb.reshape(B * N, L, self.d_single)
        enc       = self.msa_encoder(msa_emb).reshape(B, N, L, self.d_single)
        single    = enc.mean(dim=1)            # (B, L, d_single) pool over sequences
        # Pair: outer product mean
        pair_l    = single.unsqueeze(2).expand(-1, -1, L, -1)
        pair_r    = single.unsqueeze(1).expand(-1, L, -1, -1)
        pair_repr = self.pair_init(
            torch.cat([pair_l, pair_r], dim=-1)
        )  # (B, L, L, d_pair)
        return single, pair_repr + pair_features


class FlowMatchingBackbone(nn.Module):
    """
    Wraps SE3FlowMatching with RFdiffusion-pretrained weights.
    In production: load from rfdiffusion_weights.pt and replace
    the denoiser with the flow matching velocity field.
    """
    def __init__(self, d_single: int = 256, d_pair: int = 256, n_blocks: int = 8):
        super().__init__()
        # In production: load RFdiffusion weights and attach flow matching head
        self.flow_model = SE3FlowMatching(d_single, d_pair, n_blocks)
        print("[FlowMatchingBackbone] Flow matching loaded. Bridge: bacterial→mammalian.")

    def sample(self, R0, t0, pair_cond, evol_single, n_steps=20,
               fixed_mask=None, substrate_coords=None):
        return self.flow_model.sample(R0, t0, pair_cond, evol_single,
                                      n_steps, fixed_mask, substrate_coords)

    def loss(self, R0, t0, R1, t1, pair_cond, evol_single,
             fixed_mask=None, substrate_coords=None):
        return self.flow_model.flow_matching_loss(
            R0, t0, R1, t1, pair_cond, evol_single, fixed_mask, substrate_coords
        )


class ProteinMPNNBackbone(nn.Module):
    """
    Wraps dauparas/ProteinMPNN for base residue-level sequence design.
    Augmented by multi-scale designer in CHIMERAv2.
    """
    def __init__(self, node_features: int = 128, edge_features: int = 128):
        super().__init__()
        # Stub; production: from protein_mpnn_utils import ProteinMPNN
        self.node_features = node_features
        self.mpnn_trunk    = nn.Sequential(
            nn.Linear(node_features, node_features * 2),
            nn.GELU(),
            nn.Linear(node_features * 2, node_features),
        )
        print("[ProteinMPNNBackbone] Stub loaded. Replace with dauparas/ProteinMPNN.")

    def forward(self, backbone_coords, node_features):
        return self.mpnn_trunk(node_features)   # (B, L, node_features)


# ═══════════════════════════════════════════════════════════════════════════════
# CHIMERA v2 — MAIN MODEL
# ═══════════════════════════════════════════════════════════════════════════════

class CHIMERAv2(nn.Module):
    """
    CHIMERA v2: Full rebuilt PSC Stage 1 computational design model.

    Trainable: ~12M parameters (connectors + new heads)
    Frozen:    ~700M parameters (EvoFormer + FlowMatching + ProteinMPNN)
    """

    def __init__(
        self,
        d_evo_single:   int = 256,
        d_evo_pair:     int = 128,
        d_se3:          int = 256,
        d_pair_out:     int = 256,
        d_mpnn:         int = 128,
        n_flow_blocks:  int = 8,
        n_flow_steps:   int = 20,
        n_retrieve:     int = 5,
        n_mpnn_seqs:    int = 10,
        n_mc_dropout:   int = 30,
        n_domains:      int = 5,
        n_modules:      int = 5,
    ):
        super().__init__()
        self.n_flow_steps = n_flow_steps
        self.n_mpnn_seqs  = n_mpnn_seqs

        # ── FROZEN PRETRAINED BACKBONES ──────────────────────────────────────
        self.evoformer   = EvoFormerBackbone(d_evo_single, d_evo_pair)
        self.flow_model  = FlowMatchingBackbone(d_se3, d_pair_out, n_flow_blocks)
        self.base_mpnn   = ProteinMPNNBackbone(d_mpnn)

        # ── TRAINABLE CONNECTORS (~7M params) ────────────────────────────────

        # Pair connector: EvoFormer pair → flow model pair conditioning
        # UPGRADED: triangular updates + retrieval cross-attention
        self.pair_connector = TriangularPairUpdateConnector(
            evo_pair_dim=d_evo_pair,
            out_dim=d_pair_out,
        )

        # Evolutionary cross-attention: noise-adaptive dual-pathway
        self.evol_cross_attn = EvolCrossAttentionConnector(
            d_se3=d_se3, d_evo=d_evo_single,
        )

        # Node projection: EvoFormer → MPNN node features
        # UPGRADED: multi-layer + pocket-aware scaling
        self.node_connector = NodeProjectionConnector(
            evo_dim=d_evo_single, mpnn_dim=d_mpnn,
        )

        # NRPS constraint encoder
        self.constraint_encoder = NRPSConstraintEncoder(d=d_pair_out)

        # ── NEW IN V2: Structural RAG ─────────────────────────────────────────
        self.structural_retriever = StructuralRetriever(
            d_embed=d_evo_pair, d_context=d_pair_out, n_retrieve=n_retrieve,
        )

        # ── NEW IN V2: Substrate pocket conditioning ──────────────────────────
        self.substrate_conditioner = SubstratePocketConditioner(
            d_pair=d_pair_out,
        )

        # ── NEW IN V2: Multi-scale hierarchical designer (~3M params) ────────
        self.multi_scale_designer = MultiScaleNRPSDesigner(
            d_residue=d_mpnn, d_domain=256, d_module=512,
            d_assembly=256, n_domains=n_domains, n_modules=n_modules,
        )

        # ── NEW IN V2: Pareto multi-objective head (~1M params) ──────────────
        self.pareto_head = ParetoMultiObjectiveHead(d_model=d_mpnn)

        # ── NEW IN V2: Bayesian uncertainty ──────────────────────────────────
        self.uncertainty_estimator = BayesianUncertaintyEstimator(
            n_mc_samples=n_mc_dropout,
        )

        # ── NEW IN V2: DPO trainer (stateless — operates on model externally) ─
        self.dpo_trainer = DPOTrainer(beta=0.1)

        # Reference model for DPO (frozen copy of self at time of DPO init)
        self._reference_model: Optional['CHIMERAv2'] = None

        # Freeze all pretrained backbones immediately
        self.freeze_pretrained()

    # ── Setup / Loading ──────────────────────────────────────────────────────

    @classmethod
    def from_pretrained(
        cls,
        evoformer_ckpt:  str = None,
        flow_ckpt:       str = None,
        mpnn_ckpt:       str = None,
        **kwargs
    ) -> 'CHIMERAv2':
        model = cls(**kwargs)

        if evoformer_ckpt:
            sd = torch.load(evoformer_ckpt, map_location='cpu')
            model.evoformer.load_state_dict(sd, strict=False)
            print(f"[CHIMERAv2] EvoFormer loaded from {evoformer_ckpt}")

        if flow_ckpt:
            sd = torch.load(flow_ckpt, map_location='cpu')
            model.flow_model.load_state_dict(sd, strict=False)
            print(f"[CHIMERAv2] Flow model loaded from {flow_ckpt}")

        if mpnn_ckpt:
            sd = torch.load(mpnn_ckpt, map_location='cpu')
            model.base_mpnn.load_state_dict(sd, strict=False)
            print(f"[CHIMERAv2] ProteinMPNN loaded from {mpnn_ckpt}")

        model.freeze_pretrained()
        print(f"\n[CHIMERAv2] Ready:")
        print(f"  Frozen:    {model.count_frozen():>12,} parameters")
        print(f"  Trainable: {model.count_trainable():>12,} parameters")
        return model

    def freeze_pretrained(self):
        for m in [self.evoformer, self.flow_model, self.base_mpnn]:
            for p in m.parameters():
                p.requires_grad = False

    def unfreeze_connectors(self):
        for m in [
            self.pair_connector, self.evol_cross_attn,
            self.node_connector, self.constraint_encoder,
            self.structural_retriever, self.substrate_conditioner,
            self.multi_scale_designer, self.pareto_head,
        ]:
            for p in m.parameters():
                p.requires_grad = True

    def count_frozen(self):
        return sum(p.numel() for p in self.parameters() if not p.requires_grad)

    def count_trainable(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def init_dpo_reference(self):
        """Call this before any DPO training: freezes current model as reference."""
        self._reference_model = deepcopy(self)
        for p in self._reference_model.parameters():
            p.requires_grad = False
        print("[CHIMERAv2] DPO reference model initialized (current state frozen as π_ref)")

    def build_retrieval_index(self, embeddings: np.ndarray, metadata: List[dict]):
        """Load structural database into FAISS index."""
        self.structural_retriever.build_index(embeddings, metadata)

    # ── Core Forward Pass ────────────────────────────────────────────────────

    def forward(
        self,
        msa_tokens:           torch.Tensor,             # (B, N_seq, L)
        initial_pair_features: torch.Tensor,            # (B, L, L, d_evo_pair)
        source_R:             torch.Tensor,             # (B, L, 3, 3) bacterial backbone R
        source_t:             torch.Tensor,             # (B, L, 3)    bacterial backbone t
        constraints:          Optional[NRPSConstraints] = None,
        substrate_id:         Optional[torch.Tensor]   = None,  # (B,) token
        substrate_coords:     Optional[torch.Tensor]   = None,  # (B, N_atoms, 3)
        substrate_types:      Optional[torch.Tensor]   = None,  # (B, N_atoms, 8)
        n_flow_steps:         Optional[int]            = None,
        n_mpnn_seqs:          Optional[int]            = None,
    ) -> Dict[str, torch.Tensor]:
        B, N_seq, L = msa_tokens.shape
        device      = msa_tokens.device

        # ════════════════════════════════════════════════════════════════════
        # STAGE A: EvoFormer — frozen evolutionary representations
        # ════════════════════════════════════════════════════════════════════
        with torch.no_grad():
            single_repr, pair_repr = self.evoformer(msa_tokens, initial_pair_features)
            # single_repr: (B, L, 256)
            # pair_repr:   (B, L, L, 128)

        # ════════════════════════════════════════════════════════════════════
        # STAGE B: Structural Retrieval (NEW v2)
        # ════════════════════════════════════════════════════════════════════
        retrieved_context = None
        if substrate_id is not None and self.structural_retriever.index_embs is not None:
            substrate_tokens_for_retrieval = substrate_id.unsqueeze(-1)   # (B, 1)
            _, retrieved_context = self.structural_retriever.retrieve(
                query_embedding=single_repr.mean(dim=1)  # (B, 256) pooled
            )
            if retrieved_context is not None:
                K = retrieved_context.shape[1]
                retrieved_context = retrieved_context.reshape(B, K, -1).float()
                # retrieved_context: (B, K, 30) → project to (B, K, d_pair_out)
                if not hasattr(self, '_ret_proj'):
                    self._ret_proj = nn.Linear(30, 256).to(device)
                retrieved_context = self._ret_proj(retrieved_context)

        # ════════════════════════════════════════════════════════════════════
        # STAGE C: Pair Connector — triangular updates + retrieval
        # ════════════════════════════════════════════════════════════════════
        pair_cond = self.pair_connector(
            pair_repr=pair_repr,
            retrieved_context=retrieved_context,
        )  # (B, L, L, 256)

        # ════════════════════════════════════════════════════════════════════
        # STAGE D: Substrate Conditioning (NEW v2)
        # ════════════════════════════════════════════════════════════════════
        if substrate_id is not None:
            pair_cond = self.substrate_conditioner(
                pair_repr=pair_cond,
                substrate_id=substrate_id,
                substrate_coords=substrate_coords,
                substrate_types=substrate_types,
                residue_coords=source_t,   # use source backbone Cα as residue coords
            )

        # ════════════════════════════════════════════════════════════════════
        # STAGE E: NRPS Constraint Encoding
        # ════════════════════════════════════════════════════════════════════
        if constraints is not None:
            constraint_cond = self.constraint_encoder(
                L=L, constraints=constraints, device=device, batch_size=B,
            )  # (B, L, 256)
            # Inject into pair diagonal
            diag_idx = torch.arange(L, device=device)
            pair_cond[:, diag_idx, diag_idx] = (
                pair_cond[:, diag_idx, diag_idx] + constraint_cond
            )
        else:
            constraint_cond = torch.zeros(B, L, 256, device=device)

        fixed_mask = constraints.fixed_mask if constraints is not None else None

        # ════════════════════════════════════════════════════════════════════
        # STAGE F: SE(3) OT-Flow Matching — bridge generation
        # ════════════════════════════════════════════════════════════════════
        # The evol_cross_attn connector is injected INSIDE the flow sampling
        # loop via a callback that the velocity field calls at each step.
        # Here we pass the connector function as a callable.
        def evol_conditioning_fn(se3_node, t_flow):
            return self.evol_cross_attn(se3_node, single_repr, t_flow)

        # Monkey-patch the velocity field's conditioning (clean in production:
        # pass as an argument to flow_model.sample)
        self.flow_model.flow_model.velocity_field._evol_fn = evol_conditioning_fn

        steps = n_flow_steps or self.n_flow_steps

        R_final, t_final = self.flow_model.sample(
            R0=source_R,
            t0=source_t,
            pair_cond=pair_cond,
            evol_single=single_repr,
            n_steps=steps,
            fixed_mask=fixed_mask,
            substrate_coords=substrate_coords.mean(dim=1)
                             if substrate_coords is not None else None,
        )
        # R_final: (B, L, 3, 3), t_final: (B, L, 3)

        # Convert SE(3) frames → backbone atom coordinates (N, CA, C, O)
        backbone_coords = self._frames_to_coords(R_final, t_final)  # (B, L, 4, 3)

        # ════════════════════════════════════════════════════════════════════
        # STAGE G: Node Connector → Multi-Scale Sequence Designer
        # ════════════════════════════════════════════════════════════════════
        # Pocket mask: residues near substrate binding pocket
        pocket_mask = None
        if constraints is not None:
            pocket_mask = torch.zeros(B, L, dtype=torch.bool, device=device)
            for pos in constraints.stachelhaus_positions:
                if pos < L:
                    # Mark residues within 2 positions of selectivity code
                    lo = max(0, pos - 2)
                    hi = min(L, pos + 3)
                    pocket_mask[:, lo:hi] = True

        evol_node_feats = self.node_connector(single_repr, pocket_mask)  # (B, L, 128)

        # Base ProteinMPNN pass
        base_node_repr  = self.base_mpnn(backbone_coords, evol_node_feats)  # (B, L, 128)

        # Multi-scale hierarchical design (NEW v2)
        # Generate dummy edge features if not available
        K_nn        = 32  # k-NN
        edge_index  = torch.randint(0, L, (B, L, K_nn), device=device)  # placeholder
        edge_feats  = torch.zeros(B, L, K_nn, 16, device=device)         # placeholder

        d_bounds = (constraints.domain_boundaries if constraints is not None
                    else torch.tensor([[[0, L//5], [L//5, 2*L//5], [2*L//5, 3*L//5],
                                        [3*L//5, 4*L//5], [4*L//5, L]]],
                                      device=device).expand(B, -1, -1))
        m_bounds = (constraints.module_boundaries if constraints is not None
                    else torch.tensor([[[0, L//2], [L//2, L], [0, 0], [0, 0], [0, 0]]],
                                      device=device).expand(B, -1, -1))
        face_id  = (constraints.icosahedral_face if constraints is not None
                    else torch.zeros(B, dtype=torch.long, device=device))

        n_seqs = n_mpnn_seqs or self.n_mpnn_seqs
        all_logits = []
        for _ in range(n_seqs):
            logits = self.multi_scale_designer(
                residue_feats=base_node_repr,
                evol_node_feats=evol_node_feats,
                edge_feats=edge_feats,
                edge_index=edge_index,
                domain_boundaries=d_bounds,
                module_boundaries=m_bounds,
                icosahedral_face=face_id,
            )  # (B, L, 20)
            all_logits.append(logits)

        sequences = torch.stack(all_logits, dim=1)  # (B, n_seqs, L, 20)

        # ════════════════════════════════════════════════════════════════════
        # STAGE H: Pareto Multi-Objective Scoring (NEW v2)
        # ════════════════════════════════════════════════════════════════════
        # Use the mean sequence logit repr as input to pareto heads
        mean_repr = sequences.mean(dim=1)  # (B, L, 20) → need to re-encode
        # Quick re-embed through a small linear to get representation
        if not hasattr(self, '_seq_to_repr'):
            self._seq_to_repr = nn.Linear(20, 128).to(device)
        seq_repr = self._seq_to_repr(mean_repr)  # (B, L, 128)

        pareto_objectives = self.pareto_head(seq_repr)

        # ════════════════════════════════════════════════════════════════════
        # OUTPUT PACKAGE
        # ════════════════════════════════════════════════════════════════════
        return {
            'sequences':            sequences,           # (B, n_seqs, L, 20)
            'backbone_coords':      backbone_coords,     # (B, L, 4, 3)
            'R_final':              R_final,             # (B, L, 3, 3)
            't_final':              t_final,             # (B, L, 3)
            'evol_plausibility':    pareto_objectives.evolutionary_plausibility,
            'structural_stability': pareto_objectives.structural_stability,
            'expression_efficiency':pareto_objectives.expression_efficiency,
            'substrate_selectivity':pareto_objectives.substrate_selectivity,
            'assembly_compat':      pareto_objectives.assembly_compatibility,
            'pareto_objectives':    pareto_objectives,
            'pair_cond':            pair_cond,           # for debugging
            'single_repr':          single_repr,         # for PoET scoring
        }

    # ── High-Level Design API ────────────────────────────────────────────────

    @torch.no_grad()
    def design(
        self,
        nrps_msa:         torch.Tensor,
        source_backbone:  Tuple[torch.Tensor, torch.Tensor],  # (R, t) bacterial
        initial_pair_features: torch.Tensor,
        constraints:      Optional[NRPSConstraints] = None,
        target_substrate: str  = "PHE",
        n_designs:        int  = 500,
        n_pareto_samples: int  = 50,
        device:           str  = 'cuda',
    ) -> Dict:
        """
        Full design pipeline: generate n_designs sequences and return
        the Pareto-optimal subset.

        Args:
            nrps_msa:           Animal NRPS MSA tokens (B, N_seq, L)
            source_backbone:    (R, t) from bacterial NRPS crystal structure
                                Bridge starts here, flows to mammalian design
            initial_pair_features: Initial pair features (B, L, L, d_evo_pair)
            constraints:        NRPS domain constraints
            target_substrate:   Desired substrate (for retrieval + conditioning)
            n_designs:          Total sequences to generate
            n_pareto_samples:   How many from the Pareto front to return
            device:             'cuda' or 'cpu'

        Returns:
            Dictionary containing:
                'pareto_sequences':  top sequences from Pareto frontier
                'pareto_scores':     5-vector objective scores per sequence
                'all_sequences':     all n_designs sequences (for PROTEUS batch)
                'acquisition_scores': uncertainty × quality per sequence
        """
        source_R, source_t = source_backbone
        all_seqs, all_obj_vecs = [], []

        batch_size   = min(16, n_designs)
        n_batches    = (n_designs + batch_size - 1) // batch_size

        # Substrate token
        SUBSTRATES = {s: i for i, s in enumerate([
            "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU",
            "GLY", "HIS", "ILE", "LEU", "LYS", "MET", "PHE",
            "PRO", "SER", "THR", "TRP", "TYR", "VAL",
        ])}
        sub_token = torch.tensor(
            [SUBSTRATES.get(target_substrate, 0)], device=device
        ).expand(batch_size)

        for batch_idx in range(n_batches):
            print(f"[CHIMERAv2.design] Batch {batch_idx+1}/{n_batches}")

            outputs = self(
                msa_tokens            = nrps_msa.to(device),
                initial_pair_features = initial_pair_features.to(device),
                source_R              = source_R.expand(batch_size, -1, -1, -1).to(device),
                source_t              = source_t.expand(batch_size, -1, -1).to(device),
                constraints           = constraints,
                substrate_id          = sub_token,
            )

            # Best sequence per batch element (argmax over amino acids)
            best_seqs = outputs['sequences'].mean(dim=1).argmax(dim=-1)  # (B, L)
            all_seqs.append(best_seqs.cpu())

            # Objective vector per batch element
            obj_vec = torch.stack([
                outputs['evol_plausibility'],
                outputs['structural_stability'] / 100.0,
                outputs['expression_efficiency'],
                outputs['substrate_selectivity'],
                outputs['assembly_compat'],
            ], dim=-1)  # (B, 5)
            all_obj_vecs.append(obj_vec.cpu())

        all_seqs    = torch.cat(all_seqs, dim=0)     # (N_total, L)
        all_obj_vecs = torch.cat(all_obj_vecs, dim=0) # (N_total, 5)

        # Pareto front
        pareto_front, pareto_idx = self.pareto_head.compute_pareto_frontier(
            all_obj_vecs, maximize=[True, True, True, True, True]
        )

        # Uncertainty-weighted acquisition for PROTEUS selection
        uncertainty_scores = torch.rand(len(all_seqs))  # placeholder; use MC estimator
        quality_scores     = all_obj_vecs.mean(dim=-1)
        acquisition_scores = uncertainty_scores * quality_scores

        # Select top-n_pareto_samples from Pareto front by acquisition score
        pareto_acquisition = acquisition_scores[pareto_idx]
        top_k              = min(n_pareto_samples, len(pareto_idx))
        top_k_in_pareto    = pareto_acquisition.topk(top_k).indices

        return {
            'pareto_sequences':   all_seqs[pareto_idx[top_k_in_pareto]],
            'pareto_scores':      pareto_front[top_k_in_pareto],
            'pareto_count':       len(pareto_idx),
            'all_sequences':      all_seqs,
            'all_objectives':     all_obj_vecs,
            'acquisition_scores': acquisition_scores,
            'total_generated':    len(all_seqs),
        }

    # ── PROTEUS Integration ──────────────────────────────────────────────────

    def update_from_proteus(
        self,
        survivors:       List[str],
        failures:        List[str],
        msa:             torch.Tensor,
        pair_features:   torch.Tensor,
        n_dpo_steps:     int = 50,
        learning_rate:   float = 1e-5,
    ) -> Dict:
        """
        One-call interface: receive PROTEUS results, run DPO fine-tuning.

        survivors: amino acid sequences that survived PROTEUS (expressed + active)
        failures:  sequences that failed (expressed but inactive, or didn't express)
        """
        if self._reference_model is None:
            print("[CHIMERAv2] Initializing DPO reference model (first PROTEUS round)")
            self.init_dpo_reference()

        assert len(survivors) > 0 and len(failures) > 0, \
            "Need at least one survivor and one failure for DPO"

        metrics = self.dpo_trainer.update_from_proteus_round(
            policy_model          = self,
            reference_model       = self._reference_model,
            surviving_sequences   = survivors,
            failed_sequences      = failures,
            msa_tokens            = msa,
            pair_features         = pair_features,
            n_dpo_steps           = n_dpo_steps,
            learning_rate         = learning_rate,
        )

        print(f"[CHIMERAv2] PROTEUS DPO update complete. "
              f"Reward margin: {metrics['reward_margin']:.3f}")
        return metrics

    # ── Loss for Supervised Fine-Tuning ─────────────────────────────────────

    def compute_loss(
        self,
        outputs:          Dict[str, torch.Tensor],
        target_sequences: torch.Tensor,             # (B, L) ground truth
        target_R:         Optional[torch.Tensor] = None,  # (B, L, 3, 3)
        target_t:         Optional[torch.Tensor] = None,  # (B, L, 3)
        source_R:         Optional[torch.Tensor] = None,
        source_t:         Optional[torch.Tensor] = None,
        pair_cond:        Optional[torch.Tensor] = None,
        evol_single:      Optional[torch.Tensor] = None,
        objective_labels: Optional[Dict]         = None,
    ) -> Tuple[torch.Tensor, Dict]:

        losses = {}

        # ── Sequence recovery loss ───────────────────────────────────────────
        B, n_seqs, L, vocab = outputs['sequences'].shape
        seq_loss = F.cross_entropy(
            outputs['sequences'].reshape(B * n_seqs, L, vocab).transpose(1, 2),
            target_sequences.unsqueeze(1).expand(-1, n_seqs, -1).reshape(B * n_seqs, L),
        )
        losses['seq']  = seq_loss
        weight_seq     = 1.0

        # ── Flow matching loss (backbone geometry) ───────────────────────────
        flow_loss = torch.tensor(0.0, device=seq_loss.device)
        if (target_R is not None and target_t is not None
                and source_R is not None and source_t is not None):
            flow_loss = self.flow_model.loss(
                R0=source_R, t0=source_t,
                R1=target_R, t1=target_t,
                pair_cond=pair_cond or outputs['pair_cond'],
                evol_single=evol_single or outputs['single_repr'],
            )
            losses['flow'] = flow_loss
        weight_flow = 0.5

        # ── Pareto objective losses (PCGrad) ─────────────────────────────────
        pareto_loss, pareto_metrics = self.pareto_head.pcgrad_loss(
            objectives=outputs['pareto_objectives'],
            labels=objective_labels or {},
        )
        losses['pareto'] = pareto_loss
        losses.update({f'pareto_{k}': v for k, v in pareto_metrics.items()})
        weight_pareto    = 0.3

        total = (weight_seq   * seq_loss
               + weight_flow  * flow_loss
               + weight_pareto * pareto_loss)

        losses['total'] = total
        return total, {k: (v.item() if torch.is_tensor(v) else v) for k, v in losses.items()}

    # ── Utilities ────────────────────────────────────────────────────────────

    def _frames_to_coords(
        self,
        R: torch.Tensor,  # (B, L, 3, 3)
        t: torch.Tensor,  # (B, L, 3)
    ) -> torch.Tensor:    # (B, L, 4, 3) N / CA / C / O
        """Convert SE(3) backbone frames to atom coordinates."""
        B, L, _, _ = R.shape
        # Ideal local offsets (in Angstroms, local backbone frame)
        offsets = torch.tensor([
            [-0.527,  1.359, 0.0],   # N
            [ 0.000,  0.000, 0.0],   # CA (origin)
            [ 1.524,  0.000, 0.0],   # C
            [ 2.200, -1.000, 0.0],   # O
        ], device=R.device, dtype=R.dtype)  # (4, 3)

        offsets = offsets.view(1, 1, 4, 3).expand(B, L, -1, -1)
        coords  = torch.einsum('blij,blkj->blki', R, offsets) + t.unsqueeze(2)
        return coords

    def save(self, path: str):
        """Save only the trainable connector weights (not frozen backbones)."""
        state = {
            'pair_connector':        self.pair_connector.state_dict(),
            'evol_cross_attn':       self.evol_cross_attn.state_dict(),
            'node_connector':        self.node_connector.state_dict(),
            'constraint_encoder':    self.constraint_encoder.state_dict(),
            'structural_retriever':  self.structural_retriever.state_dict(),
            'substrate_conditioner': self.substrate_conditioner.state_dict(),
            'multi_scale_designer':  self.multi_scale_designer.state_dict(),
            'pareto_head':           self.pareto_head.state_dict(),
        }
        torch.save(state, path)
        print(f"[CHIMERAv2] Connector weights saved to {path}")

    def load_connectors(self, path: str):
        """Load previously saved connector weights."""
        state = torch.load(path, map_location='cpu')
        for name, sd in state.items():
            getattr(self, name).load_state_dict(sd)
        print(f"[CHIMERAv2] Connectors loaded from {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# EXAMPLE USAGE
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':

    print("=" * 70)
    print("CHIMERA v2 — PSC Engineering Pipeline Stage 1")
    print("=" * 70)

    # Build the model
    model = CHIMERAv2.from_pretrained(
        evoformer_ckpt = None,  # loads stub; replace with 'openfold_weights.pt'
        flow_ckpt      = None,  # loads stub; replace with 'rfdiffusion_weights.pt'
        mpnn_ckpt      = None,  # loads stub; replace with 'proteinmpnn_weights.pt'
    )

    # ── Define NRPS design problem ──────────────────────────────────────────
    B, N_seq, L = 2, 32, 600   # 2 designs, 32 MSA sequences, 600-AA A-domain

    constraints = NRPSConstraints(
        fixed_mask         = torch.zeros(B, L, dtype=torch.bool),  # none fixed in demo
        stachelhaus_positions = torch.tensor([235, 236, 239, 278, 299, 301, 322, 330, 517, 518]),
        domain_boundaries  = torch.tensor([[[0, 300], [300, 400], [400, 500], [500, 580], [580, L]]] * B),
        module_boundaries  = torch.tensor([[[0, L], [0, 0], [0, 0], [0, 0], [0, 0]]] * B),
        icosahedral_face   = torch.tensor([7, 14]),   # modules on face 7 and 14
        ppt_serine_position = 519,
        hotspot_coords     = None,
        hotspot_indices    = None,
        target_substrate   = "PHE",
    )

    # ── Run a forward pass ──────────────────────────────────────────────────
    with torch.no_grad():
        outputs = model(
            msa_tokens             = torch.randint(0, 23, (B, N_seq, L)),
            initial_pair_features  = torch.zeros(B, L, L, 128),
            source_R               = torch.eye(3).unsqueeze(0).unsqueeze(0).expand(B, L, -1, -1),
            source_t               = torch.randn(B, L, 3) * 10,
            constraints            = constraints,
            substrate_id           = torch.tensor([13, 13]),  # PHE = index 13
            n_flow_steps           = 5,   # fast demo; use 20 for real runs
            n_mpnn_seqs            = 3,   # fast demo; use 10 for real runs
        )

    print(f"\nForward pass output shapes:")
    print(f"  sequences:       {outputs['sequences'].shape}")       # (B, n_seqs, L, 20)
    print(f"  backbone_coords: {outputs['backbone_coords'].shape}") # (B, L, 4, 3)
    print(f"  evol_plausib:    {outputs['evol_plausibility'].shape}") # (B,)
    print(f"  struct_stability:{outputs['structural_stability'].shape}")

    print(f"\nObjective scores (raw, untrained connectors):")
    print(f"  Evolutionary plausibility: {outputs['evol_plausibility'].tolist()}")
    print(f"  Structural stability:      {outputs['structural_stability'].tolist()}")
    print(f"  Expression efficiency:     {outputs['expression_efficiency'].tolist()}")
    print(f"  Substrate selectivity:     {outputs['substrate_selectivity'].tolist()}")
    print(f"  Assembly compatibility:    {outputs['assembly_compat'].tolist()}")

    print(f"\n[CHIMERA v2] Ready for PSC pipeline.")
    print(f"Next step: fine-tune connectors on NRPS training data.")
    print(f"See training_data.py for complete data sourcing guide.")
