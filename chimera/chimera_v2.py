"""Legacy CHIMERAv2 retained only for API and checkpoint compatibility.

This module is not the public canonical composition and is not production
ready. Its former architecture narrative incorrectly implied pretrained
OpenFold, RFdiffusion, and ProteinMPNN weights, enabled retrieval without a
validated shared embedding space, and described deterministic RK4 sampling.
The active public model is ``chimera.architecture.CanonicalCHIMERAv2``. This
legacy implementation must not be used to substantiate model capability or
training status.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Optional, Tuple, List, Dict, NamedTuple
from copy import deepcopy

# ── Internal imports from CHIMERA v2 module suite ────────────────────────────
from .flow_matching import (
    SE3FlowMatching,
    SinusoidalTimeEmbedding,
    so3_exp,
    so3_log,
    se3_interp,
)
from .proteinmpnn import get_protein_graph
from .geometry import validate_backbone
from .conditioning import SubstratePocketConditioner
from .domain_schema import NRPSConstraints
from .components import (
    TriangularPairUpdateConnector, TriangularAttention,
    TriangularMultiplicativeUpdate, EvolCrossAttentionConnector,
    NodeProjectionConnector, NRPSConstraintEncoder, EvoFormerBackbone,
    ProteinMPNNBackbone,
)
from .evaluators import BiologicalObjectiveEvaluator
from .multi_objective import (
    StructuralRetriever,
    DPOTrainer,
    ParetoMultiObjectiveHead,
    BayesianUncertaintyEstimator,
    MultiScaleNRPSDesigner,
    ProteusPreferencePair,
    ParetoObjectives,
    AutoregressiveSequencePolicy,
)

# ═══════════════════════════════════════════════════════════════════════════════
# LEGACY LOCAL APPROXIMATIONS (random initialization; no native weights)
# ═══════════════════════════════════════════════════════════════════════════════




class FlowMatchingBackbone(nn.Module):
    """Legacy local SE(3) flow approximation; it is not RFdiffusion."""

    def __init__(self, d_single: int = 256, d_pair: int = 256, n_blocks: int = 8):
        super().__init__()
        self.flow_model = SE3FlowMatching(d_single, d_pair, n_blocks)
        self.frozen_bridge = nn.Sequential(
            nn.Linear(d_single, d_single * 8),
            nn.GELU(),
            nn.Linear(d_single * 8, d_single * 8),
            nn.GELU(),
            nn.Linear(d_single * 8, d_single),
        )
        print("[LegacyFlowMatchingBackbone] Random local approximation; not RFdiffusion.")

    def sample(
        self,
        R0,
        t0,
        pair_cond,
        evol_single,
        n_steps=20,
        fixed_mask=None,
        substrate_coords=None,
        evol_conditioning_fn=None,
    ):
        evol_single = self.frozen_bridge(evol_single)
        return self.flow_model.sample(
            R0, t0, pair_cond, evol_single, n_steps, fixed_mask, substrate_coords,
            evol_conditioning_fn,
        )

    def loss(
        self,
        R0,
        t0,
        R1,
        t1,
        pair_cond,
        evol_single,
        fixed_mask=None,
        substrate_coords=None,
    ):
        evol_single = self.frozen_bridge(evol_single)
        return self.flow_model.flow_matching_loss(
            R0, t0, R1, t1, pair_cond, evol_single, fixed_mask, substrate_coords
        )




# ═══════════════════════════════════════════════════════════════════════════════
# CHIMERA v2 — MAIN MODEL
# ═══════════════════════════════════════════════════════════════════════════════


class CHIMERAv2(nn.Module):
    """Deprecated compatibility composition; use canonical ``chimera.CHIMERAv2``."""

    def __init__(
        self,
        d_evo_single: int = 256,
        d_evo_pair: int = 128,
        d_se3: int = 256,
        d_pair_out: int = 256,
        d_mpnn: int = 128,
        n_flow_blocks: int = 8,
        n_flow_steps: int = 20,
        n_retrieve: int = 5,
        n_mpnn_seqs: int = 10,
        n_mc_dropout: int = 30,
        n_domains: int = 5,
        n_modules: int = 5,
    ):
        super().__init__()
        self.n_flow_steps = n_flow_steps
        self.n_mpnn_seqs = n_mpnn_seqs
        self.best_observed: Optional[float] = None

        # Legacy local modules start random and remain trainable.
        self.evoformer = EvoFormerBackbone(d_evo_single, d_evo_pair)
        self.flow_model = FlowMatchingBackbone(d_se3, d_pair_out, n_flow_blocks)
        self.base_mpnn = ProteinMPNNBackbone(d_mpnn)

        # ── TRAINABLE CONNECTORS (~7M params) ────────────────────────────────

        # Pair connector: EvoFormer pair → flow model pair conditioning
        # UPGRADED: triangular updates + retrieval cross-attention
        self.pair_connector = TriangularPairUpdateConnector(
            evo_pair_dim=d_evo_pair,
            out_dim=d_pair_out,
        )

        # Evolutionary cross-attention: noise-adaptive dual-pathway
        self.evol_cross_attn = EvolCrossAttentionConnector(
            d_se3=d_se3,
            d_evo=d_evo_single,
        )

        # Node projection: EvoFormer → MPNN node features
        # UPGRADED: multi-layer + pocket-aware scaling
        self.node_connector = NodeProjectionConnector(
            evo_dim=d_evo_single,
            mpnn_dim=d_mpnn,
        )

        # NRPS constraint encoder
        self.constraint_encoder = NRPSConstraintEncoder(d=d_pair_out)

        # ── NEW IN V2: Structural RAG ─────────────────────────────────────────
        self.structural_retriever = StructuralRetriever(
            d_embed=d_evo_pair,
            d_context=d_pair_out,
            n_retrieve=n_retrieve,
        )

        # ── NEW IN V2: Substrate pocket conditioning ──────────────────────────
        self.substrate_conditioner = SubstratePocketConditioner(
            d_pair=d_pair_out,
        )

        # ── NEW IN V2: Multi-scale hierarchical designer (~3M params) ────────
        self.multi_scale_designer = MultiScaleNRPSDesigner(
            d_residue=d_mpnn,
            d_domain=256,
            d_module=512,
            d_assembly=256,
            n_domains=n_domains,
            n_modules=n_modules,
        )

        # ── NEW IN V2: Pareto multi-objective head (~1M params) ──────────────
        self.pareto_head = ParetoMultiObjectiveHead(d_model=d_mpnn)
        self.objective_evaluator = BiologicalObjectiveEvaluator()

        # Sequence representation input projection.
        self._seq_to_repr = nn.Linear(20, d_mpnn)
        self.sequence_policy = AutoregressiveSequencePolicy(d_mpnn)

        # ── RAG projection layer (registered here, not lazily in forward) ──────────
        # Projects retrieved context from FAISS embedding dim to pair conditioning dim.
        self.ret_proj = nn.Linear(30, d_pair_out)

        # ── NEW IN V2: Bayesian uncertainty ──────────────────────────────────
        self.uncertainty_estimator = BayesianUncertaintyEstimator(
            n_mc_samples=n_mc_dropout,
        )

        # ── NEW IN V2: DPO trainer (stateless — operates on model externally) ─
        self.dpo_trainer = DPOTrainer(beta=0.1)

        # Reference model for DPO (frozen copy of self at time of DPO init)
        self._reference_model: Optional["CHIMERAv2"] = None

        # No module is frozen without training/validation evidence.
        self.freeze_pretrained()

    # ── Setup / Loading ──────────────────────────────────────────────────────

    @classmethod
    def from_pretrained(
        cls,
        evoformer_ckpt: str = None,
        flow_ckpt: str = None,
        mpnn_ckpt: str = None,
        **kwargs,
    ) -> "CHIMERAv2":
        model = cls(**kwargs)

        def load_checkpoint(module, path, name):
            checkpoint = torch.load(path, map_location="cpu")
            state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
            try:
                module.load_state_dict(state_dict, strict=True)
            except (RuntimeError, TypeError) as exc:
                raise RuntimeError(
                    f"{name} checkpoint is incompatible with the configured architecture: {path}. "
                    "Use a checkpoint exported for this module, or provide an explicit adapter."
                ) from exc
            print(f"[CHIMERAv2] {name} loaded from {path}")

        if evoformer_ckpt:
            load_checkpoint(model.evoformer, evoformer_ckpt, "EvoFormer")

        if flow_ckpt:
            load_checkpoint(model.flow_model, flow_ckpt, "Flow model")

        if mpnn_ckpt:
            load_checkpoint(model.base_mpnn, mpnn_ckpt, "ProteinMPNN")

        model.freeze_pretrained()
        print("[LegacyCHIMERAv2] Strictly loaded local state dicts; training status is unverified.")
        return model

    def freeze_pretrained(self):
        """Deprecated no-op: this legacy class has no verified component-state contract."""

    def unfreeze_connectors(self):
        for m in [
            self.pair_connector,
            self.evol_cross_attn,
            self.node_connector,
            self.constraint_encoder,
            self.structural_retriever,
            self.substrate_conditioner,
            self.multi_scale_designer,
            self.pareto_head,
        ]:
            for p in m.parameters():
                p.requires_grad = True
        for module in (self._seq_to_repr, self.ret_proj, self.uncertainty_estimator):
            for p in module.parameters():
                p.requires_grad = True
        for p in self.sequence_policy.parameters():
            p.requires_grad = True

    def sequence_logprob(self, sequence_tokens, msa_tokens, pair_features, source_R, source_t):
        outputs = self(
            msa_tokens=msa_tokens,
            initial_pair_features=pair_features,
            source_R=source_R,
            source_t=source_t,
            n_mpnn_seqs=1,
        )
        context = self._seq_to_repr(outputs["sequences"][:, 0])
        return self.sequence_policy.logprob(context, sequence_tokens)

    def prepare_for_training(self):
        """Return trainable legacy parameters without freezing random local weights."""
        self.freeze_pretrained()
        self.unfreeze_connectors()
        return [p for p in self.parameters() if p.requires_grad]

    def count_frozen(self):
        return sum(p.numel() for p in self.parameters() if not p.requires_grad)

    def count_trainable(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def init_dpo_reference(self):
        raise RuntimeError(
            "LegacyCHIMERAv2 cannot establish supervised validation; use CanonicalTrainer's validated preference regime."
        )

    def build_retrieval_index(self, embeddings: np.ndarray, metadata: List[dict]):
        """Load structural database into FAISS index."""
        self.structural_retriever.build_index(embeddings, metadata)

    # ── Core Forward Pass ────────────────────────────────────────────────────

    def forward(
        self,
        msa_tokens: torch.Tensor,  # (B, N_seq, L)
        initial_pair_features: torch.Tensor,  # (B, L, L, d_evo_pair)
        source_R: torch.Tensor,  # (B, L, 3, 3) bacterial backbone R
        source_t: torch.Tensor,  # (B, L, 3)    bacterial backbone t
        constraints: Optional[NRPSConstraints] = None,
        substrate_id: Optional[torch.Tensor] = None,  # (B,) token
        substrate_coords: Optional[torch.Tensor] = None,  # (B, N_atoms, 3)
        substrate_types: Optional[torch.Tensor] = None,  # (B, N_atoms, 8)
        n_flow_steps: Optional[int] = None,
        n_mpnn_seqs: Optional[int] = None,
        use_rag: bool = False,
    ) -> Dict[str, torch.Tensor]:
        B, N_seq, L = msa_tokens.shape
        device = msa_tokens.device

        if constraints is not None:
            def expand_batch(value):
                if value is None or value.shape[0] == B:
                    return value
                if value.shape[0] == 1:
                    return value.expand(B, *value.shape[1:])
                raise ValueError(f"Constraint batch dimension must be 1 or {B}, got {value.shape[0]}")

            constraints = NRPSConstraints(
                fixed_mask=expand_batch(constraints.fixed_mask),
                stachelhaus_positions=constraints.stachelhaus_positions,
                domain_boundaries=expand_batch(constraints.domain_boundaries),
                module_boundaries=expand_batch(constraints.module_boundaries),
                icosahedral_face=expand_batch(constraints.icosahedral_face),
                ppt_serine_position=constraints.ppt_serine_position,
                hotspot_coords=constraints.hotspot_coords,
                hotspot_indices=constraints.hotspot_indices,
                target_substrate=constraints.target_substrate,
                fixed_sequence=expand_batch(constraints.fixed_sequence),
                domain_types=expand_batch(constraints.domain_types),
            )

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
        if not use_rag:
            rag_status = "DISABLED"
        elif self.structural_retriever.index_embs is None or self.structural_retriever.index_embs.shape[0] == 0:
            rag_status = "RAG_UNAVAILABLE: index not configured"
        else:
            rag_status = "RAG_UNAVAILABLE: query/index embedding spaces are not aligned"

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
                residue_coords=source_t,  # use source backbone Cα as residue coords
            )

        # ════════════════════════════════════════════════════════════════════
        # STAGE E: NRPS Constraint Encoding
        # ════════════════════════════════════════════════════════════════════
        if constraints is not None:
            constraint_cond = self.constraint_encoder(
                L=L,
                constraints=constraints,
                device=device,
                batch_size=B,
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

        steps = n_flow_steps or self.n_flow_steps

        R_final, t_final = self.flow_model.sample(
            R0=source_R,
            t0=source_t,
            pair_cond=pair_cond,
            evol_single=single_repr,
            n_steps=steps,
            fixed_mask=fixed_mask,
            substrate_coords=substrate_coords,
            evol_conditioning_fn=evol_conditioning_fn,
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
        base_node_repr = self.base_mpnn(backbone_coords, evol_node_feats)  # (B, L, 128)

        # Multi-scale hierarchical design (NEW v2)
        # Build geometric residue graph from the generated backbone.
        K_nn = 32  # k-NN
        edge_index, edge_feats, edge_mask = get_protein_graph(
            t_final, R_final, k_neighbors=K_nn
        )
        if edge_index.shape[-1] < K_nn:
            pad = K_nn - edge_index.shape[-1]
            edge_index = F.pad(edge_index, (0, pad), value=0)
            edge_feats = F.pad(edge_feats, (0, 0, 0, pad), value=0.0)
            edge_mask = F.pad(edge_mask, (0, pad), value=False)

        geometry = validate_backbone(backbone_coords, R_final)

        d_bounds = (
            constraints.domain_boundaries
            if constraints is not None
            else torch.tensor(
                [
                    [
                        [0, L // 5],
                        [L // 5, 2 * L // 5],
                        [2 * L // 5, 3 * L // 5],
                        [3 * L // 5, 4 * L // 5],
                        [4 * L // 5, L],
                    ]
                ],
                device=device,
            ).expand(B, -1, -1)
        )
        m_bounds = (
            constraints.module_boundaries
            if constraints is not None
            else torch.tensor(
                [[[0, L // 2], [L // 2, L], [0, 0], [0, 0], [0, 0]]], device=device
            ).expand(B, -1, -1)
        )
        face_id = (
            constraints.icosahedral_face
            if constraints is not None
            else torch.zeros(B, dtype=torch.long, device=device)
        )

        n_seqs = self.n_mpnn_seqs
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
                edge_mask=edge_mask,
            )  # (B, L, 20)
            all_logits.append(logits + torch.randn_like(logits))

        sequences = torch.stack(all_logits, dim=1)  # (B, n_seqs, L, 20)

        if constraints is not None and constraints.fixed_sequence is not None:
            fixed_sequence = constraints.fixed_sequence.to(device)
            if fixed_sequence.shape != (B, L):
                raise ValueError("fixed_sequence must have shape (B, L)")
            valid = fixed_sequence.ge(0)
            if valid.any() and fixed_sequence[valid].max() >= 20:
                raise ValueError("fixed_sequence values must be -1 or amino-acid token IDs 0..19")
            mask = valid.unsqueeze(1).unsqueeze(-1)
            one_hot = F.one_hot(fixed_sequence.clamp_min(0), num_classes=20).float()
            sequences = torch.where(
                mask,
                torch.where(one_hot.bool(), sequences, torch.full_like(sequences, -1e4)),
                sequences,
            )

        # ════════════════════════════════════════════════════════════════════
        # STAGE H: Pareto Multi-Objective Scoring (NEW v2)
        # ════════════════════════════════════════════════════════════════════
        # Compute objectives PER CANDIDATE, not just from mean sequence.
        # This allows ranking and comparing all generated candidates on Pareto frontier.
        # Reshape: (B, n_seqs, L, 20) → (B*n_seqs, L, 20)
        sequences_flat = sequences.reshape(B * n_seqs, L, 20)

        # Project logits to representation space
        seq_repr_flat = self._seq_to_repr(sequences_flat)  # (B*n_seqs, L, d_mpnn)

        # Score all candidates
        pareto_objectives_flat = self.pareto_head(seq_repr_flat)

        # Reshape objectives back: (B*n_seqs,) → (B, n_seqs)
        pareto_objectives = ParetoObjectives(
            evolutionary_plausibility=pareto_objectives_flat.evolutionary_plausibility.reshape(B, n_seqs),
            structural_stability=pareto_objectives_flat.structural_stability.reshape(B, n_seqs),
            expression_efficiency=pareto_objectives_flat.expression_efficiency.reshape(B, n_seqs),
            substrate_selectivity=pareto_objectives_flat.substrate_selectivity.reshape(B, n_seqs),
            assembly_compatibility=pareto_objectives_flat.assembly_compatibility.reshape(B, n_seqs),
        )

        # ════════════════════════════════════════════════════════════════════
        # OUTPUT PACKAGE
        # ════════════════════════════════════════════════════════════════════
        return {
            "sequences": sequences,  # (B, n_seqs, L, 20)
            "backbone_coords": backbone_coords,  # (B, L, 4, 3)
            "R_final": R_final,  # (B, L, 3, 3)
            "t_final": t_final,  # (B, L, 3)
            "evol_plausibility": pareto_objectives.evolutionary_plausibility,  # (B, n_seqs)
            "structural_stability": pareto_objectives.structural_stability,  # (B, n_seqs)
            "expression_efficiency": pareto_objectives.expression_efficiency,  # (B, n_seqs)
            "substrate_selectivity": pareto_objectives.substrate_selectivity,  # (B, n_seqs)
            "assembly_compat": pareto_objectives.assembly_compatibility,  # (B, n_seqs)
            "pareto_objectives": pareto_objectives,
            "pair_cond": pair_cond,  # for debugging
            "single_repr": single_repr,  # raw local MSA representation; no PoET scorer is connected
            "geometry_valid": torch.tensor(geometry.candidate_valid, dtype=torch.bool, device=device),
            "geometry_report": geometry.as_dict(),
            "rag_status": rag_status,
        }

    # ── High-Level Design API ────────────────────────────────────────────────

    @staticmethod
    def compute_expected_improvement(
        predicted_quality: torch.Tensor,  # (N_candidates,) mean predictions
        uncertainty: torch.Tensor,  # (N_candidates,) epistemic uncertainty
        best_observed: float,
    ) -> torch.Tensor:
        """
        Compute true Gaussian Expected Improvement (EI) acquisition function.

        EI(x) = (μ(x) - f_best) * Φ(Z) + σ(x) * φ(Z)

        where:
            μ(x) = predicted quality
            σ(x) = uncertainty
            f_best = explicit historical best or a caller-specified prior baseline
            Z = (μ(x) - f_best) / (σ(x) + ε)
            Φ(Z) = standard normal CDF
            φ(Z) = standard normal PDF
            ε = small epsilon to avoid division by zero

        Returns:
            ei: (N_candidates,) Expected Improvement scores
        """
        from scipy.stats import norm

        # Compute standardized improvement
        eps = 1e-8
        mu = predicted_quality.cpu().numpy()
        sigma = uncertainty.cpu().numpy()
        Z = (mu - best_observed) / (sigma + eps)

        # Gaussian EI
        ei = (mu - best_observed) * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei = torch.from_numpy(ei).to(predicted_quality.device).float()

        return ei

    def set_best_observed(self, value: Optional[float]) -> None:
        """Record an externally observed normalized utility, never a batch prediction."""
        if value is not None and not 0.0 <= float(value) <= 1.0:
            raise ValueError("best_observed must be normalized historical utility in [0,1]")
        self.best_observed = None if value is None else float(value)

    @torch.no_grad()
    def design(
        self,
        nrps_msa: torch.Tensor,
        source_backbone: Tuple[torch.Tensor, torch.Tensor],  # (R, t) bacterial
        initial_pair_features: torch.Tensor,
        constraints: Optional[NRPSConstraints] = None,
        target_substrate: str = "PHE",
        n_designs: int = 500,
        n_pareto_samples: int = 50,
        device: str = "cuda",
        flow_steps: Optional[int] = None,
        use_rag: bool = False,
        objective_weights: Optional[torch.Tensor] = None,
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
        L = nrps_msa.shape[-1]
        all_seqs, all_obj_vecs = [], []

        n_seqs = self.n_mpnn_seqs
        batch_size = min(16, max(1, (n_designs + n_seqs - 1) // n_seqs))
        n_batches = (n_designs + batch_size - 1) // batch_size

        # Substrate token
        SUBSTRATES = {
            s: i
            for i, s in enumerate(
                [
                    "ALA",
                    "ARG",
                    "ASN",
                    "ASP",
                    "CYS",
                    "GLN",
                    "GLU",
                    "GLY",
                    "HIS",
                    "ILE",
                    "LEU",
                    "LYS",
                    "MET",
                    "PHE",
                    "PRO",
                    "SER",
                    "THR",
                    "TRP",
                    "TYR",
                    "VAL",
                ]
            )
        }
        if target_substrate not in SUBSTRATES:
            raise ValueError(f"Unsupported substrate '{target_substrate}'")
        sub_token = torch.tensor(
            [SUBSTRATES[target_substrate]], device=device
        ).expand(batch_size)

        for batch_idx in range(n_batches):
            print(f"[CHIMERAv2.design] Batch {batch_idx+1}/{n_batches}")

            outputs = self(
                msa_tokens=nrps_msa.expand(batch_size, -1, -1).to(device),
                initial_pair_features=initial_pair_features.expand(batch_size, -1, -1, -1).to(device),
                source_R=source_R.expand(batch_size, -1, -1, -1).to(device),
                source_t=source_t.expand(batch_size, -1, -1).to(device),
                constraints=constraints,
                substrate_id=sub_token,
                n_flow_steps=flow_steps,
                n_mpnn_seqs=n_seqs,
                use_rag=use_rag,
            )

            # Preserve all candidates: (B, n_seqs, L, 20) → (B, n_seqs, L) token indices
            # Convert logits to token indices (argmax over amino acids, not over candidates)
            all_candidate_seqs = outputs["sequences"].argmax(dim=-1)  # (B, n_seqs, L)
            all_seqs.append(all_candidate_seqs.cpu())

            # Objective vectors for all candidates: (B, n_seqs, 5)
            # Stack objectives per candidate (not per batch, but per batch+candidate)
            candidate_tokens = outputs["sequences"].argmax(dim=-1).reshape(-1, L)
            candidate_coords = outputs["backbone_coords"].repeat_interleave(n_seqs, dim=0)
            candidate_rotations = outputs["R_final"].repeat_interleave(n_seqs, dim=0)
            evaluated = self.objective_evaluator.evaluate(
                candidate_tokens,
                candidate_coords,
                candidate_rotations,
            )
            obj_vec = torch.stack(
                [
                    evaluated["evolutionary_plausibility_proxy"],
                    evaluated["structural_validity"],
                    evaluated["expression_proxy"],
                    evaluated["selectivity_proxy"],
                    evaluated["assembly_proxy"],
                ],
                dim=-1,
            ).reshape(batch_size, n_seqs, 5)
            all_obj_vecs.append(obj_vec.cpu())

        # Flatten all batches and candidates together
        # From: [(B, n_seqs, L), ...] × n_batches → (B*n_seqs*n_batches, L) = (N_candidates, L)
        all_seqs = torch.cat(all_seqs, dim=0)  # (B*n_batches, n_seqs, L)
        all_seqs = all_seqs.reshape(-1, L)[:n_designs]  # Flatten to requested count

        # From: [(B, n_seqs, 5), ...] × n_batches → (B*n_seqs*n_batches, 5) = (N_candidates, 5)
        all_obj_vecs = torch.cat(all_obj_vecs, dim=0)  # (B*n_batches, n_seqs, 5)
        all_obj_vecs = all_obj_vecs.reshape(-1, 5)[:n_designs]  # Flatten to requested count

        # Pareto front
        pareto_front, pareto_idx = self.pareto_head.compute_pareto_frontier(
            all_obj_vecs, maximize=[True, True, True, True, True]
        )

        acquisition_scores = torch.zeros(all_obj_vecs.shape[0])
        acquisition_status = "not computed: explicit objective_weights and historical best_observed are required"
        if self.best_observed is not None and objective_weights is not None:
            candidate_tokens = all_seqs.to(device)
            candidate_repr = self._seq_to_repr(
                F.one_hot(candidate_tokens, num_classes=20).to(torch.float32)
            )

            def normalized_objectives(result):
                return torch.stack(
                    (
                        torch.sigmoid(result.evolutionary_plausibility),
                        result.structural_stability.clamp(0, 100) / 100.0,
                        result.expression_efficiency.clamp(0, 1),
                        result.substrate_selectivity.clamp(0, 1),
                        result.assembly_compatibility.clamp(0, 1),
                    ),
                    dim=-1,
                )

            posterior = self.uncertainty_estimator.predict_fixed_candidate(
                self.pareto_head,
                {"repr": candidate_repr},
                output_getter=normalized_objectives,
            )
            weights = torch.as_tensor(
                objective_weights,
                device=posterior["mean"].device,
                dtype=posterior["mean"].dtype,
            )
            utility_mean = BayesianUncertaintyEstimator.scalarize_objectives(
                posterior["mean"], weights=weights
            )
            normalized_weights = weights / weights.sum()
            utility_std = torch.sqrt(
                (posterior["epistemic_std"].square() * normalized_weights.square()).sum(-1)
            )
            acquisition_scores = BayesianUncertaintyEstimator.expected_improvement(
                utility_mean,
                utility_std,
                self.best_observed,
            ).detach().cpu()
            acquisition_status = "fixed-candidate MC-dropout EI using explicit utility weights and historical best"

        if acquisition_status.startswith("fixed-candidate"):
            frontier_order = torch.argsort(acquisition_scores[pareto_idx], descending=True)
            selected_indices = pareto_idx[frontier_order[:n_pareto_samples]]
        else:
            selected_indices = pareto_idx[:n_pareto_samples]

        return {
            "pareto_sequences": all_seqs[selected_indices],
            "pareto_scores": all_obj_vecs[selected_indices],
            "pareto_count": len(pareto_idx),
            "all_sequences": all_seqs,
            "all_objectives": all_obj_vecs,
            "acquisition_scores": acquisition_scores,
            "acquisition_status": acquisition_status,
            "total_generated": len(all_seqs),
        }

    # ── PROTEUS Integration ──────────────────────────────────────────────────

    def update_from_proteus(
        self,
        survivors: List[str],
        failures: List[str],
        msa: torch.Tensor,
        pair_features: torch.Tensor,
        n_dpo_steps: int = 50,
        learning_rate: float = 1e-5,
    ) -> Dict:
        """Disabled: legacy state cannot establish supervised policy validation."""
        raise RuntimeError(
            "LegacyCHIMERAv2 preference updates are disabled; use CanonicalTrainer after held-out sequence validation."
        )

    # ── Loss for Supervised Fine-Tuning ─────────────────────────────────────

    def compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        target_sequences: torch.Tensor,  # (B, L) ground truth
        target_R: Optional[torch.Tensor] = None,  # (B, L, 3, 3)
        target_t: Optional[torch.Tensor] = None,  # (B, L, 3)
        source_R: Optional[torch.Tensor] = None,
        source_t: Optional[torch.Tensor] = None,
        pair_cond: Optional[torch.Tensor] = None,
        evol_single: Optional[torch.Tensor] = None,
        objective_labels: Optional[Dict] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        """Disabled: the legacy mixed-loss API violates staged training contracts."""
        raise RuntimeError(
            "LegacyCHIMERAv2.compute_loss is disabled; use CanonicalTrainer with one explicit TrainingRegime."
        )

    def load_connectors(self, path: str):
        """Load previously saved connector weights."""
        state = torch.load(path, map_location="cpu")
        for name, sd in state.items():
            module_name = "_seq_to_repr" if name == "seq_to_repr" else name
            getattr(self, module_name).load_state_dict(sd)
        print(f"[CHIMERAv2] Connectors loaded from {path}")


# Legacy CHIMERAv2 compatibility implementation ends here.
