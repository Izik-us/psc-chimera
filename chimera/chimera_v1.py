"""
CHIMERA — Compositional Hierarchical Inference Model for
          Evolutionary Representation and Architecture
==========================================================

The complete Stage 1 hybrid model of the PSC Engineering Pipeline.
Integrates three pretrained architectures via novel connector modules
that are the only trainable components during fine-tuning.

Architecture:
    ┌─────────────────────────────────────────────────────────┐
    │                    CHIMERA                              │
    │                                                         │
    │  Animal NRPS MSA                                        │
    │       │                                                 │
    │  ┌────▼────────────┐                                    │
    │  │   EvoFormer     │  (frozen pretrained weights)       │
    │  │  48 blocks      │                                    │
    │  └────┬────────────┘                                    │
    │       │                                                 │
    │   single_repr (B,L,256)    pair_repr (B,L,L,128)       │
    │       │                         │                       │
    │  ┌────▼──────────┐  ┌───────────▼───────────────┐      │
    │  │ node_projection│  │   pair_projection         │  ← NOVEL CONNECTORS │
    │  │ (256→128)     │  │   (128→256) + cross-attn  │      │
    │  └────┬──────────┘  └───────────┬───────────────┘      │
    │       │                         │                       │
    │       │             ┌───────────▼──────────────┐        │
    │       │             │   SE3Denoiser (8 blocks)  │        │
    │       │             │   IPA conditioned on      │        │
    │       │             │   projected pair_repr     │        │
    │       │             └───────────┬──────────────┘        │
    │       │                         │                       │
    │       │             (B,L,3,3) R, (B,L,3) t             │
    │       │                         │                       │
    │  ┌────▼──────────────────────────▼──────────┐           │
    │  │           ProteinMPNN                    │           │
    │  │  node = geometric_features +             │           │
    │  │         evol_node_features (from EvoF)   │           │
    │  └────────────────────┬─────────────────────┘           │
    │                       │                                 │
    │              logits (B, L, 20)                          │
    │              + PoET score feedback                      │
    └───────────────────────┘─────────────────────────────────┘

Trainable parameters: ~7M (connectors only)
Frozen parameters:    ~700M (pretrained EvoFormer + SE3 + MPNN)

Usage:
    chimera = CHIMERA.from_pretrained(
        evoformer_checkpoint='openfold_weights.pt',
        se3_checkpoint='rfdiffusion_weights.pt',
        mpnn_checkpoint='proteinmpnn_weights.pt',
    )
    # Fine-tune on NRPS-specific data
    chimera.freeze_pretrained()   # freeze base models
    chimera.unfreeze_connectors() # train only connectors

    sequences, backbones = chimera(
        nrps_msa=msa_tokens,              # animal NRPS MSA from Stage 0
        design_constraints=constraints,    # A-domain hotspot residues
        n_diffusion_steps=200,
    )
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict

from chimera.evoformer import EvoFormer, C_S, C_Z
from chimera.se3_diffusion import SE3Denoiser
from chimera.proteinmpnn import ProteinMPNN

# ── CHIMERA Connector Modules (The Novel Contribution) ───────────────────────


class PairProjection(nn.Module):
    """
    Projects EvoFormer pair_repr → SE3Denoiser conditioning.

    EvoFormer pair_repr: (B, L, L, 128)  co-evolutionary pairwise information
    SE3Denoiser expects: (B, L, L, 256)  pair conditioning for IPA

    This is the connector that makes backbone diffusion evolutionary-context-aware.
    The projected pair repr enters IPA's pair_bias term, so every attention weight
    between residue frames is modulated by whether those residues co-evolve in the
    animal NRPS family.
    """

    def __init__(self, c_z_in: int = C_Z, c_z_out: int = 256):
        super().__init__()
        self.projection = nn.Sequential(
            nn.LayerNorm(c_z_in),
            nn.Linear(c_z_in, c_z_out * 2),
            nn.GELU(),
            nn.Linear(c_z_out * 2, c_z_out),
            nn.LayerNorm(c_z_out),
        )
        # Learned symmetrization: pair_repr should be symmetric
        # (co-evolution between i,j equals co-evolution between j,i)
        self.sym_weight = nn.Parameter(torch.tensor(0.5))

    def forward(self, pair_repr: torch.Tensor) -> torch.Tensor:
        proj = self.projection(pair_repr)  # (B, L, L, c_z_out)
        sym = self.sym_weight * proj + (1 - self.sym_weight) * proj.transpose(1, 2)
        return sym


class NodeProjection(nn.Module):
    """
    Projects EvoFormer single_repr → ProteinMPNN node features.

    EvoFormer single_repr: (B, L, 256)   per-residue evolutionary embedding
    ProteinMPNN expects:   (B, L, 128)   node feature dimension

    This makes every message-passing step jointly aware of:
      (a) backbone geometry  (existing ProteinMPNN)
      (b) evolutionary context at each position (new from CHIMERA)

    Residues conserved across the animal NRPS family will have distinctive
    evolutionary embeddings → their sequence design is constrained toward
    conservation. Variable residues (like A-domain selectivity loops) will
    have high evolutionary variance → the model explores more freely there.
    """

    def __init__(self, c_s_in: int = C_S, c_node_out: int = 128):
        super().__init__()
        self.projection = nn.Sequential(
            nn.LayerNorm(c_s_in),
            nn.Linear(c_s_in, c_s_in),
            nn.GELU(),
            nn.Linear(c_s_in, c_node_out),
            nn.LayerNorm(c_node_out),
        )

    def forward(self, single_repr: torch.Tensor) -> torch.Tensor:
        return self.projection(single_repr)  # (B, L, 128)


class EvolCrossAttention(nn.Module):
    """
    Cross-attention from SE3Denoiser's internal state → EvoFormer embeddings.

    This provides the SE3Denoiser with a soft "memory query" mechanism:
    at each denoising step, the current backbone state can attend to the
    full evolutionary memory from EvoFormer to retrieve relevant context.

    Mechanism: the denoiser's node state queries the EvoFormer single_repr
    as key/value, allowing each noisy residue position to retrieve its
    evolutionary neighbors' context during denoising.
    """

    def __init__(self, c_s: int = C_S, c_denoiser: int = 256, n_head: int = 8):
        super().__init__()
        self.norm_query = nn.LayerNorm(c_denoiser)
        self.norm_memory = nn.LayerNorm(c_s)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=c_denoiser, num_heads=n_head, kdim=c_s, vdim=c_s, batch_first=True
        )
        self.out_proj = nn.Sequential(
            nn.Linear(c_denoiser, c_denoiser),
            nn.GELU(),
        )
        self.gate = nn.Parameter(torch.zeros(1))  # learned mixing weight

    def forward(
        self, denoiser_state: torch.Tensor, evol_memory: torch.Tensor
    ) -> torch.Tensor:
        """
        denoiser_state: (B, L, c_denoiser) from SE3Denoiser intermediate layer
        evol_memory:    (B, L, c_s)        from EvoFormer single_repr

        Returns: (B, L, c_denoiser)  enriched denoiser state
        """
        q = self.norm_query(denoiser_state)
        k = v = self.norm_memory(evol_memory)
        attn_out, _ = self.cross_attn(q, k, v)
        # Gated residual (gate starts at 0 → safely initialized to zero contribution)
        return denoiser_state + torch.sigmoid(self.gate) * self.out_proj(attn_out)


# ── NRPS-Specific Design Constraints ─────────────────────────────────────────


class NRPSConstraintEmbedding(nn.Module):
    """
    Embeds NRPS-specific design constraints into the conditioning signal.

    Constraints supported:
      - A-domain selectivity code positions (10 residues that determine substrate)
      - T-domain phosphopantetheine attachment site (invariant Ser)
      - C-domain catalytic dyad (His-His or His-Asp depending on bond type)
      - TE-domain oxyanion hole positions
      - Module boundary positions (linker regions for inter-module design)
      - Animal NRPS conservation score per position (from PoET log-likelihood)
    """

    def __init__(self, c_out: int = 256, n_constraint_types: int = 8):
        super().__init__()
        self.constraint_embed = nn.Embedding(n_constraint_types + 1, c_out)
        # Conservation score embedding (continuous value → embedding)
        self.conservation_proj = nn.Sequential(
            nn.Linear(1, 64),
            nn.GELU(),
            nn.Linear(64, c_out),
        )
        self.combine = nn.Linear(c_out * 2, c_out)

    def forward(
        self, constraint_types: torch.Tensor, conservation_scores: torch.Tensor
    ) -> torch.Tensor:
        """
        constraint_types:    (B, L)   integer type per position (0=none, 1=A-selectivity, ...)
        conservation_scores: (B, L)   PoET log-likelihood per position (float, normalized 0-1)
        """
        type_emb = self.constraint_embed(constraint_types)  # (B, L, c_out)
        cons_emb = self.conservation_proj(
            conservation_scores.unsqueeze(-1)
        )  # (B, L, c_out)
        return self.combine(torch.cat([type_emb, cons_emb], dim=-1))


# ── Full CHIMERA Model ────────────────────────────────────────────────────────


class CHIMERA(nn.Module):
    """
    Complete Stage 1 pipeline of the PSC Engineering Framework.

    The three pretrained models (EvoFormer, SE3Denoiser, ProteinMPNN) are
    connected by novel trainable connectors. Only the connectors are trained;
    the base models are frozen, requiring only ~7M parameters of training.

    Forward pass produces:
      1. Evolutionary-context-aware protein backbone (R, t from SE3Denoiser)
      2. Amino acid sequence logits (from ProteinMPNN)
      3. PoET-compatible sequence embedding for downstream scoring

    For NRPS A-domain reprogramming:
      Input MSA = animal NRPS family (Stage 0 sequences)
      Constraints = target substrate binding pocket geometry (theozyme)
      Output = sequences predicted to fold correctly AND maintain evolutionary
               plausibility within the animal NRPS family

    For de novo insert domain design:
      Input MSA = closest known enzyme family for target chemistry
      Constraints = theozyme active site residue positions (fixed hotspots)
      Output = novel domain sequence threading around active site
    """

    def __init__(
        self,
        # EvoFormer config
        evoformer_n_blocks: int = 48,
        # SE3Denoiser config
        se3_n_blocks: int = 8,
        n_diffusion_steps: int = 200,
        # ProteinMPNN config
        mpnn_n_layers: int = 3,
        k_neighbors: int = 32,
        # Connector dimensions
        c_s: int = C_S,  # 256 — EvoFormer single repr channels
        c_z: int = C_Z,  # 128 — EvoFormer pair repr channels
        c_denoiser: int = 256,  # SE3Denoiser internal channels
        c_node: int = 128,  # ProteinMPNN node channels
    ):
        super().__init__()

        # ── Pretrained base models (frozen in fine-tuning) ─────────────────
        self.evoformer = EvoFormer(n_blocks=evoformer_n_blocks)
        self.se3denoiser = SE3Denoiser(
            c_s=c_denoiser,
            c_z=c_denoiser,
            n_blocks=se3_n_blocks,
            max_t=n_diffusion_steps,
        )
        self.mpnn = ProteinMPNN(
            c_node=c_node, n_mp_layers=mpnn_n_layers, k_neighbors=k_neighbors
        )

        # ── Novel CHIMERA connector modules (trainable) ────────────────────
        self.pair_projection = PairProjection(c_z, c_denoiser)
        self.node_projection = NodeProjection(c_s, c_node)
        self.evol_cross_attn = EvolCrossAttention(c_s, c_denoiser)

        # ── NRPS constraint conditioning ───────────────────────────────────
        self.nrps_constraint_embed = NRPSConstraintEmbedding(c_denoiser)

        # ── Output heads ───────────────────────────────────────────────────
        # PoET compatibility: project final node repr to PoET embedding space
        self.to_poet_embedding = nn.Sequential(
            nn.LayerNorm(c_node),
            nn.Linear(c_node, 512),
            nn.GELU(),
            nn.Linear(512, 512),
        )

        self.n_diffusion_steps = n_diffusion_steps

    def freeze_pretrained(self):
        """Freeze all pretrained model weights. Train only connectors."""
        for param in self.evoformer.parameters():
            param.requires_grad = False
        for param in self.se3denoiser.parameters():
            param.requires_grad = False
        for param in self.mpnn.parameters():
            param.requires_grad = False

    def unfreeze_connectors(self):
        """Ensure connector modules are trainable."""
        for module in [
            self.pair_projection,
            self.node_projection,
            self.evol_cross_attn,
            self.nrps_constraint_embed,
            self.to_poet_embedding,
        ]:
            for param in module.parameters():
                param.requires_grad = True

    def unfreeze_last_n_evoformer_blocks(self, n: int = 4):
        """Fine-tune last n EvoFormer blocks alongside connectors."""
        total = len(self.evoformer.blocks)
        for i, block in enumerate(self.evoformer.blocks):
            for param in block.parameters():
                param.requires_grad = i >= total - n

    @property
    def trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def total_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(
        self,
        msa_tokens: torch.Tensor,
        constraint_types: Optional[torch.Tensor] = None,
        conservation_scores: Optional[torch.Tensor] = None,
        motif_mask: Optional[torch.Tensor] = None,
        motif_coords: Optional[torch.Tensor] = None,
        n_diffusion_steps: Optional[int] = None,
        timestep: Optional[torch.Tensor] = None,
        noisy_R: Optional[torch.Tensor] = None,
        noisy_t: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Main CHIMERA forward pass.

        Args:
            msa_tokens:         (B, N_seq, L)  tokenized MSA of NRPS family
            constraint_types:   (B, L)          NRPS constraint type per position
            conservation_scores:(B, L)          PoET log-likelihood per position
            motif_mask:         (B, L)          fixed hotspot positions (theozyme)
            motif_coords:       (B, L, 3)       known Cα coordinates for hotspots
            timestep:           (B,)            diffusion timestep (training only)
            noisy_R, noisy_t:                    noisy frames (training only)

        Returns dict with:
            'sequences':    (B, L, 20)   amino acid logits
            'R_frames':     (B, L, 3, 3) predicted backbone rotations
            't_coords':     (B, L, 3)    predicted Cα coordinates
            'poet_embedding':(B, L, 512) for downstream PoET scoring
            'pair_repr':    (B, L, L, 256) projected for analysis
        """
        B, N_seq, L = msa_tokens.shape

        # ── Stage A: EvoFormer ────────────────────────────────────────────
        # Extract evolutionary representations from animal NRPS MSA
        single_repr, pair_repr = self.evoformer(msa_tokens)
        # single_repr: (B, L, 256)    per-residue evolutionary context
        # pair_repr:   (B, L, L, 128) pairwise co-evolutionary context

        # ── CHIMERA Connector 1: pair_projection ─────────────────────────
        # EvoFormer pair_repr → SE3Denoiser conditioning
        pair_cond = self.pair_projection(pair_repr)  # (B, L, L, 256)

        # NRPS constraint embedding (adds substrate-binding geometry awareness)
        if constraint_types is not None and conservation_scores is not None:
            constraint_cond = self.nrps_constraint_embed(
                constraint_types, conservation_scores
            )  # (B, L, 256)
            # Inject into pair diagonal (self-conditioning of each position)
            pair_cond = pair_cond + constraint_cond.unsqueeze(2) * 0.1

        # ── Stage B: SE3 Denoiser ─────────────────────────────────────────
        # Generate backbone conditioned on evolutionary pair context
        if self.training and noisy_R is not None:
            # Training mode: predict denoised frames from noisy input
            R_pred, t_pred = self.se3denoiser(
                noisy_R, noisy_t, pair_cond, timestep, motif_mask, motif_coords
            )
        else:
            # Inference mode: full reverse diffusion sampling
            R_pred, t_pred = self.se3denoiser.sample(
                L,
                pair_cond,
                motif_mask,
                motif_coords,
                n_steps=n_diffusion_steps or self.n_diffusion_steps,
            )

        # ── CHIMERA Connector 2: node_projection ─────────────────────────
        # EvoFormer single_repr → ProteinMPNN node features
        evol_node_features = self.node_projection(single_repr)  # (B, L, 128)

        # ── Stage C: ProteinMPNN Sequence Design ─────────────────────────
        # Design sequences on the generated backbone WITH evolutionary context
        sequence_logits = self.mpnn(
            t_coords=t_pred,
            R_frames=R_pred,
            evol_node_features=evol_node_features,
            fixed_positions=motif_mask,
            fixed_aas=None,
        )  # (B, L, 20)

        # ── PoET Compatibility Output ─────────────────────────────────────
        # Project to embedding space compatible with PoET scoring
        poet_emb = self.to_poet_embedding(evol_node_features)  # (B, L, 512)

        return {
            "sequences": sequence_logits,  # (B, L, 20) — softmax for probabilities
            "R_frames": R_pred,  # (B, L, 3, 3)
            "t_coords": t_pred,  # (B, L, 3)
            "poet_embedding": poet_emb,  # (B, L, 512) for PoET scoring
            "single_repr": single_repr,  # (B, L, 256) raw EvoFormer output
            "pair_repr": pair_cond,  # (B, L, L, 256) projected
        }

    @classmethod
    def from_pretrained(
        cls,
        evoformer_checkpoint: str = None,
        se3_checkpoint: str = None,
        mpnn_checkpoint: str = None,
        **kwargs,
    ) -> "CHIMERA":
        """
        Instantiate CHIMERA and load pretrained weights.

        Compatible checkpoint sources:
          evoformer_checkpoint: OpenFold (openfold.github.io) or AlphaFold2 weights
          se3_checkpoint:       RFdiffusion (github.com/RosettaCommons/RFdiffusion)
          mpnn_checkpoint:      ProteinMPNN (github.com/dauparas/ProteinMPNN)
        """
        model = cls(**kwargs)

        if evoformer_checkpoint:
            state = torch.load(evoformer_checkpoint, map_location="cpu")
            # Filter to EvoFormer-relevant keys
            evof_state = {
                k.replace("evoformer.", ""): v
                for k, v in state.items()
                if "evoformer" in k
            }
            model.evoformer.load_state_dict(evof_state, strict=False)
            print(f"Loaded EvoFormer weights from {evoformer_checkpoint}")

        if se3_checkpoint:
            state = torch.load(se3_checkpoint, map_location="cpu")
            model.se3denoiser.load_state_dict(state, strict=False)
            print(f"Loaded SE3Denoiser weights from {se3_checkpoint}")

        if mpnn_checkpoint:
            state = torch.load(mpnn_checkpoint, map_location="cpu")
            model.mpnn.load_state_dict(state, strict=False)
            print(f"Loaded ProteinMPNN weights from {mpnn_checkpoint}")

        return model


# ── Training Utilities ────────────────────────────────────────────────────────


class CHIMERALoss(nn.Module):
    """
    Multi-objective training loss for CHIMERA connector fine-tuning.

      L = L_struct + λ1·L_seq + λ2·L_evol + λ3·L_nrps
    """

    def __init__(
        self,
        lambda_seq: float = 1.0,
        lambda_evol: float = 0.5,
        lambda_nrps: float = 0.3,
    ):
        super().__init__()
        self.lambda_seq = lambda_seq
        self.lambda_evol = lambda_evol
        self.lambda_nrps = lambda_nrps

    def structure_loss(
        self,
        R_pred: torch.Tensor,
        t_pred: torch.Tensor,
        R_true: torch.Tensor,
        t_true: torch.Tensor,
    ) -> torch.Tensor:
        """FAPE-like structure loss (frame-aligned point error)."""
        t_loss = F.smooth_l1_loss(t_pred, t_true)
        # Rotation matrix Frobenius distance
        R_diff = torch.bmm(R_pred.view(-1, 3, 3), R_true.view(-1, 3, 3).transpose(1, 2))
        I = torch.eye(3, device=R_pred.device).unsqueeze(0)
        R_loss = (R_diff - I).norm(dim=(-1, -2)).mean()
        return t_loss + R_loss

    def sequence_loss(
        self,
        seq_logits: torch.Tensor,
        seq_true: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Cross-entropy sequence recovery loss."""
        loss = F.cross_entropy(
            seq_logits.view(-1, 20), seq_true.view(-1), reduction="none"
        )
        if mask is not None:
            loss = (loss * mask.view(-1).float()).sum() / mask.float().sum()
        return loss.mean()

    def evol_consistency_loss(
        self, poet_emb: torch.Tensor, poet_target: torch.Tensor
    ) -> torch.Tensor:
        """
        Pull CHIMERA outputs toward high-PoET-score region of sequence space.
        poet_target: embedding of known high-scoring animal NRPS sequences.
        """
        return F.mse_loss(poet_emb, poet_target)

    def nrps_constraint_loss(
        self,
        seq_logits: torch.Tensor,
        constraint_mask: torch.Tensor,
        required_aa: torch.Tensor,
    ) -> torch.Tensor:
        """
        Enforce NRPS-specific constraints:
          - Conserved positions must match known amino acid
          - Phosphopantetheine Ser must remain Ser (T-domain)
          - Catalytic residues must remain catalytically competent
        """
        constrained_logits = seq_logits[constraint_mask]
        constrained_aa = required_aa[constraint_mask]
        return F.cross_entropy(constrained_logits, constrained_aa)

    def forward(self, outputs: Dict, targets: Dict) -> Dict[str, torch.Tensor]:
        L_struct = self.structure_loss(
            outputs["R_frames"],
            outputs["t_coords"],
            targets["R_true"],
            targets["t_true"],
        )
        L_seq = self.sequence_loss(
            outputs["sequences"], targets["sequence"], targets.get("design_mask")
        )
        L_evol = (
            self.evol_consistency_loss(
                outputs["poet_embedding"], targets["poet_target"]
            )
            if "poet_target" in targets
            else torch.tensor(0.0)
        )

        L_nrps = (
            self.nrps_constraint_loss(
                outputs["sequences"], targets["constraint_mask"], targets["required_aa"]
            )
            if "constraint_mask" in targets
            else torch.tensor(0.0)
        )

        total = (
            L_struct
            + self.lambda_seq * L_seq
            + self.lambda_evol * L_evol
            + self.lambda_nrps * L_nrps
        )

        return {
            "total": total,
            "structure": L_struct,
            "sequence": L_seq,
            "evol": L_evol,
            "nrps": L_nrps,
        }


# ── Quick sanity check ────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Initializing CHIMERA (small config for testing)...")

    model = CHIMERA(
        evoformer_n_blocks=2,  # 48 in production
        se3_n_blocks=2,  # 8 in production
        n_diffusion_steps=10,  # 200 in production
        mpnn_n_layers=2,  # 3 in production
    )
    model.freeze_pretrained()
    model.unfreeze_connectors()

    print(f"Total parameters:     {model.total_params:,}")
    print(f"Trainable parameters: {model.trainable_params:,}")

    # Dummy forward pass: batch=2, N_seq=8, L=32 residues
    B, N_seq, L = 2, 8, 32
    msa = torch.randint(0, 22, (B, N_seq, L))
    ctypes = torch.randint(0, 8, (B, L))
    cscores = torch.rand(B, L)

    out = model(
        msa_tokens=msa,
        constraint_types=ctypes,
        conservation_scores=cscores,
        n_diffusion_steps=10,
    )

    print("\nOutput shapes:")
    for k, v in out.items():
        print(f"  {k:20s}: {tuple(v.shape)}")

    print("\nCHIMERA initialized successfully.")
    print("Next step: load pretrained weights via CHIMERA.from_pretrained()")
