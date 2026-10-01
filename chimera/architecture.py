"""Canonical CHIMERA v2 composition root.

The local EvoFormer and ProteinMPNN backbones remain explicitly documented
approximations/stubs. Legacy ``CHIMERAv2`` remains isolated in
``chimera.chimera_v2`` for compatibility and is not instantiated here.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Optional, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings

from .bayesian import BayesianUncertaintyEstimator
from .conditioning import SubstratePocketConditioner
from .components import (
    EvoFormerBackbone,
    TriangularPairUpdateConnector,
    EvolCrossAttentionConnector,
    NodeProjectionConnector,
    NRPSConstraintEncoder,
    ProteinMPNNBackbone,
)
from .dpo import DPOBatch, DPOTrainer
from .domain_schema import AssemblySchema, DomainSpan, DomainType, NRPSConstraints
from .evaluators import BiologicalObjectiveEvaluator
from .flow_matching import FlowMatchingBackbone
from .geometry import validate_backbone
from .lie import so3_log
from .multi_objective import MultiScaleNRPSDesigner, ParetoObjectives, StructuralRetriever
from .pareto_pcgrad import MergeReadyParetoMultiObjectiveHead
from .pcgrad import PCGradOptimizer, pcgrad_step, project_conflicting_gradients
from .proteinmpnn import get_protein_graph
from .reproducibility import make_generator, seed_everything, seed_worker
from .schrodinger_bridge import SchrodingerBridge, SE3SchrodingerBridge
from .autoregressive_policy import AutoregressiveSequencePolicy


SUBSTRATE_TOKENS = {
    name: index
    for index, name in enumerate(
        (
            "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
            "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
        )
    )
}


class CanonicalCHIMERAv2(nn.Module):
    """Canonical model assembled directly from reusable components.

    Component modules preserve their historical names where feasible so
    connector checkpoints can be migrated by strict per-module loading.
    """

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
        **legacy_aliases,
    ) -> None:
        super().__init__()
        unsupported = set(legacy_aliases) - {"evoformer_layers", "flow_blocks", "mpnn_layers"}
        if unsupported:
            raise TypeError(f"Unexpected model configuration keys: {sorted(unsupported)}")
        if legacy_aliases.get("flow_blocks") is not None:
            n_flow_blocks = int(legacy_aliases["flow_blocks"])
        if min(d_evo_single, d_evo_pair, d_se3, d_pair_out, d_mpnn) <= 0:
            raise ValueError("model dimensions must be positive")
        if d_evo_single != d_se3:
            raise ValueError("d_evo_single must equal d_se3 for the configured velocity conditioning")
        if d_se3 % 8 or d_pair_out % 4 or d_mpnn % 4:
            raise ValueError("d_se3, d_pair_out and d_mpnn must be divisible by their attention head counts")
        if min(n_flow_blocks, n_flow_steps, n_mpnn_seqs, n_domains, n_modules) <= 0:
            raise ValueError("model depths, sample counts and hierarchy sizes must be positive")

        self.d_evo_single = d_evo_single
        self.d_evo_pair = d_evo_pair
        self.d_se3 = d_se3
        self.d_pair_out = d_pair_out
        self.d_mpnn = d_mpnn
        self.n_flow_steps = n_flow_steps
        self.n_mpnn_seqs = n_mpnn_seqs
        self.n_domains = n_domains
        self.n_modules = n_modules
        self.best_observed: Optional[float] = None
        self._is_canonical_composition = True
        self._reference_policy: Optional[AutoregressiveSequencePolicy] = None

        self.evoformer = EvoFormerBackbone(d_evo_single, d_evo_pair)
        self.flow_model = FlowMatchingBackbone(d_se3, d_pair_out, n_flow_blocks)
        self.base_mpnn = ProteinMPNNBackbone(d_mpnn)
        self.pair_connector = TriangularPairUpdateConnector(d_evo_pair, d_pair_out)
        self.evol_cross_attn = EvolCrossAttentionConnector(d_se3, d_evo_single)
        self.node_connector = NodeProjectionConnector(d_evo_single, d_mpnn)
        self.constraint_encoder = NRPSConstraintEncoder(d=d_pair_out)
        self.structural_retriever = StructuralRetriever(
            d_embed=d_evo_pair,
            d_context=d_pair_out,
            n_retrieve=n_retrieve,
        )
        self.substrate_conditioner = SubstratePocketConditioner(d_pair=d_pair_out)
        self.multi_scale_designer = MultiScaleNRPSDesigner(
            d_residue=d_mpnn,
            d_domain=256,
            d_module=512,
            d_assembly=256,
            n_domains=n_domains,
            n_modules=n_modules,
            edge_dim=28,
        )
        self.pareto_head = MergeReadyParetoMultiObjectiveHead(d_model=d_mpnn)
        self.objective_evaluator = BiologicalObjectiveEvaluator()
        self._seq_to_repr = nn.Linear(20, d_mpnn)
        self.sequence_policy = AutoregressiveSequencePolicy(d_mpnn)
        self.ret_proj = nn.Linear(30, d_pair_out)
        self.uncertainty_estimator = BayesianUncertaintyEstimator(n_samples=n_mc_dropout)
        self.dpo_trainer = DPOTrainer(beta=0.1)
        self.freeze_pretrained()

    @classmethod
    def from_pretrained(
        cls,
        evoformer_ckpt: Optional[str] = None,
        flow_ckpt: Optional[str] = None,
        mpnn_ckpt: Optional[str] = None,
        **kwargs,
    ) -> "CanonicalCHIMERAv2":
        model = cls(**kwargs)
        modules = (
            (model.evoformer, evoformer_ckpt, "EvoFormer approximation"),
            (model.flow_model, flow_ckpt, "canonical flow backbone"),
            (model.base_mpnn, mpnn_ckpt, "ProteinMPNN-inspired stub"),
        )
        for module, path, name in modules:
            if path is None:
                continue
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            state = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
            if name == "canonical flow backbone":
                state = cls._migrate_flow_state_dict(state, module.state_dict())
            try:
                module.load_state_dict(state, strict=True)
            except (RuntimeError, TypeError) as exc:
                raise RuntimeError(
                    f"{name} checkpoint is incompatible with this module: {path}; "
                    "native OpenFold/RFdiffusion/ProteinMPNN checkpoints need explicit adapters"
                ) from exc
        model.freeze_pretrained()
        return model

    @staticmethod
    def _migrate_flow_state_dict(state: Dict[str, torch.Tensor], target_state: Dict[str, torch.Tensor]):
        """Fill the explicit SB drift alias from historical flow-only weights."""
        migrated = dict(state)
        prefix = "flow_model.velocity_field."
        alias_prefix = "sb_model.drift_model."
        for key, value in tuple(migrated.items()):
            if key.startswith(prefix):
                alias = alias_prefix + key[len(prefix):]
                if alias in target_state:
                    migrated.setdefault(alias, value)
        return migrated

    def freeze_pretrained(self) -> None:
        frozen = (
            self.evoformer,
            self.flow_model,
            self.base_mpnn,
            self.pair_connector,
            self.evol_cross_attn,
            self.node_connector,
            self.constraint_encoder,
            self.structural_retriever,
            self.substrate_conditioner,
            self.multi_scale_designer,
            self.pareto_head,
            self.uncertainty_estimator,
            self.sequence_policy,
            self.ret_proj,
        )
        for module in frozen:
            for parameter in module.parameters():
                parameter.requires_grad_(False)
        for parameter in self._seq_to_repr.parameters():
            parameter.requires_grad_(True)

    def unfreeze_connectors(self) -> None:
        trainable = (
            self.pair_connector,
            self.evol_cross_attn,
            self.node_connector,
            self.constraint_encoder,
            self.structural_retriever,
            self.substrate_conditioner,
            self.multi_scale_designer,
            self.pareto_head,
            self._seq_to_repr,
            self.ret_proj,
            self.uncertainty_estimator,
            self.sequence_policy,
        )
        for module in trainable:
            for parameter in module.parameters():
                parameter.requires_grad_(True)

    def prepare_for_training(self) -> list[torch.nn.Parameter]:
        self.freeze_pretrained()
        self.unfreeze_connectors()
        return [parameter for parameter in self.parameters() if parameter.requires_grad]

    def count_frozen(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if not parameter.requires_grad)

    def count_trainable(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def init_dpo_reference(self) -> None:
        self._reference_policy = deepcopy(self.sequence_policy).eval()
        for parameter in self._reference_policy.parameters():
            parameter.requires_grad_(False)

    def build_retrieval_index(self, embeddings, metadata) -> None:
        self.structural_retriever.build_index(embeddings, metadata)

    @staticmethod
    def _frames_to_coords(R: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        offsets = R.new_tensor(
            [[-0.527, 1.359, 0.0], [0.0, 0.0, 0.0], [1.524, 0.0, 0.0], [2.2, -1.0, 0.0]]
        )
        offsets = offsets.view(1, 1, 4, 3).expand(R.shape[0], R.shape[1], -1, -1)
        return torch.einsum("blij,blkj->blki", R, offsets) + t.unsqueeze(2)

    @staticmethod
    def _expand_batch(value, batch_size: int):
        if value is None or value.shape[0] == batch_size:
            return value
        if value.shape[0] == 1:
            return value.expand(batch_size, *value.shape[1:])
        raise ValueError(f"constraint batch dimension must be 1 or {batch_size}")

    def _normalize_constraints(self, constraints, batch_size: int, length: int, device):
        if constraints is None:
            return None
        fields = {
            "fixed_mask": self._expand_batch(constraints.fixed_mask, batch_size),
            "stachelhaus_positions": constraints.stachelhaus_positions,
            "domain_boundaries": self._expand_batch(constraints.domain_boundaries, batch_size),
            "module_boundaries": self._expand_batch(constraints.module_boundaries, batch_size),
            "icosahedral_face": self._expand_batch(constraints.icosahedral_face, batch_size),
            "ppt_serine_position": constraints.ppt_serine_position,
            "hotspot_coords": constraints.hotspot_coords,
            "hotspot_indices": constraints.hotspot_indices,
            "target_substrate": constraints.target_substrate,
            "fixed_sequence": self._expand_batch(constraints.fixed_sequence, batch_size),
            "domain_types": self._expand_batch(constraints.domain_types, batch_size),
        }
        for key, value in tuple(fields.items()):
            if torch.is_tensor(value):
                fields[key] = value.to(device)
        normalized = NRPSConstraints(**fields)
        if normalized.stachelhaus_positions.ndim != 1:
            raise ValueError("stachelhaus_positions must be one-dimensional")
        if torch.any((normalized.stachelhaus_positions < 0) | (normalized.stachelhaus_positions >= length)):
            raise ValueError("stachelhaus positions must lie within the sequence")
        if normalized.domain_boundaries.shape != (batch_size, self.n_domains, 2):
            raise ValueError("domain_boundaries shape does not match configured domain count")
        if (
            normalized.module_boundaries.ndim != 3
            or normalized.module_boundaries.shape[0] != batch_size
            or normalized.module_boundaries.shape[-1] != 2
        ):
            raise ValueError("module_boundaries shape does not match configured module count")
        if normalized.module_boundaries.shape[1] > self.n_modules:
            extras = normalized.module_boundaries[:, self.n_modules :]
            if torch.any(extras != 0):
                raise ValueError("module_boundaries contains active modules beyond configured module count")
            fields["module_boundaries"] = normalized.module_boundaries[:, : self.n_modules]
            normalized = NRPSConstraints(**fields)
        elif normalized.module_boundaries.shape[1] < self.n_modules:
            raise ValueError("module_boundaries shape does not match configured module count")
        if torch.any(normalized.domain_boundaries[..., 0] >= normalized.domain_boundaries[..., 1]):
            raise ValueError("domain spans must be non-empty")
        for row in normalized.module_boundaries.tolist():
            for start, end in row:
                if (start, end) == (0, 0):
                    continue
                if start >= end:
                    raise ValueError("module spans must be non-empty or the (0,0) inactive sentinel")
        if torch.any(normalized.domain_boundaries < 0) or torch.any(normalized.domain_boundaries > length):
            raise ValueError("domain spans must lie inside sequence length")
        if torch.any(normalized.module_boundaries < 0) or torch.any(normalized.module_boundaries > length):
            raise ValueError("module spans must lie inside sequence length")
        domain_types = (DomainType.A, DomainType.T, DomainType.C, DomainType.TE, DomainType.LINKER)
        for batch_index in range(batch_size):
            spans = []
            for domain_index, (start, end) in enumerate(normalized.domain_boundaries[batch_index].tolist()):
                type_index = (
                    int(normalized.domain_types[batch_index, domain_index].item())
                    if normalized.domain_types is not None
                    else min(domain_index, len(domain_types) - 1)
                )
                if not 0 <= type_index < len(domain_types):
                    raise ValueError("domain type ID is outside the supported schema")
                spans.append(DomainSpan(int(start), int(end), domain_types[type_index]))
            modules = [
                (int(start), int(end))
                for start, end in normalized.module_boundaries[batch_index].tolist()
                if (int(start), int(end)) != (0, 0)
            ]
            if not modules:
                raise ValueError("each batch item must declare an active module span")
            schema = AssemblySchema(spans, modules)
            schema.validate(length)
        return normalized

    def forward(
        self,
        msa_tokens: torch.Tensor,
        initial_pair_features: torch.Tensor,
        source_R: torch.Tensor,
        source_t: torch.Tensor,
        constraints: Optional[NRPSConstraints] = None,
        substrate_id: Optional[torch.Tensor] = None,
        substrate_coords: Optional[torch.Tensor] = None,
        substrate_types: Optional[torch.Tensor] = None,
        n_flow_steps: Optional[int] = None,
        n_mpnn_seqs: Optional[int] = None,
        use_rag: bool = False,
        temperature: float = 1.0,
        generator: Optional[torch.Generator] = None,
    ) -> Dict[str, torch.Tensor]:
        if msa_tokens.ndim != 3:
            raise ValueError("msa_tokens must have shape (B,N,L)")
        B, _, L = msa_tokens.shape
        if L < 2:
            raise ValueError("canonical structure generation requires at least two residues")
        if constraints is None and L < max(self.n_domains, self.n_modules):
            raise ValueError(
                "sequence length must be at least the configured domain/module count"
            )
        device = msa_tokens.device
        if initial_pair_features.shape != (B, L, L, self.d_evo_pair):
            raise ValueError("initial_pair_features must have shape (B,L,L,d_evo_pair)")
        if source_R.shape != (B, L, 3, 3) or source_t.shape != (B, L, 3):
            raise ValueError("source backbone shapes must be (B,L,3,3) and (B,L,3)")
        if source_R.device != device or source_t.device != device or initial_pair_features.device != device:
            raise ValueError("all model inputs must be on the same device")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        steps = self.n_flow_steps if n_flow_steps is None else int(n_flow_steps)
        n_draws_requested = self.n_mpnn_seqs if n_mpnn_seqs is None else int(n_mpnn_seqs)
        if steps < 2:
            raise ValueError("n_flow_steps must be at least 2")
        if n_draws_requested < 1:
            raise ValueError("n_mpnn_seqs must be positive")

        constraints = self._normalize_constraints(constraints, B, L, device)
        with torch.no_grad():
            single_repr, pair_repr = self.evoformer(msa_tokens, initial_pair_features)

        retrieved_context = None
        if use_rag and substrate_id is not None and self.structural_retriever.index_embs is not None:
            raise RuntimeError(
                "Structural retrieval is provisional: the learned substrate query encoder "
                "is not aligned to the externally supplied structure embedding space."
            )
        pair_cond = self.pair_connector(pair_repr, retrieved_context)
        if substrate_id is not None:
            substrate_id = substrate_id.to(device=device, dtype=torch.long)
            pair_cond = self.substrate_conditioner(
                pair_cond,
                substrate_id,
                substrate_coords=substrate_coords.to(device) if substrate_coords is not None else None,
                substrate_types=substrate_types.to(device) if substrate_types is not None else None,
                residue_coords=source_t,
            )

        if constraints is not None:
            constraint_cond = self.constraint_encoder(L, constraints, device, B)
            diagonal = torch.arange(L, device=device)
            pair_cond = pair_cond.clone()
            pair_cond[:, diagonal, diagonal] += constraint_cond
            fixed_mask = constraints.fixed_mask
            face_id = constraints.icosahedral_face
            domain_bounds = constraints.domain_boundaries
            module_bounds = constraints.module_boundaries
        else:
            fixed_mask = None
            face_id = torch.zeros(B, dtype=torch.long, device=device)
            domain_bounds = torch.stack([
                torch.tensor([index * L // self.n_domains, (index + 1) * L // self.n_domains], device=device)
                for index in range(self.n_domains)
            ]).unsqueeze(0).expand(B, -1, -1)
            module_bounds = torch.stack([
                torch.tensor([index * L // self.n_modules, (index + 1) * L // self.n_modules], device=device)
                for index in range(self.n_modules)
            ]).unsqueeze(0).expand(B, -1, -1)
        if torch.any((face_id < 0) | (face_id >= 20)):
            raise ValueError("icosahedral_face values must be in [0,19]")

        def evo_conditioning_fn(nodes, flow_time):
            return self.evol_cross_attn(nodes, single_repr, flow_time)

        R_final, t_final = self.flow_model.sample(
            source_R, source_t, pair_cond, single_repr,
            n_steps=steps,
            fixed_mask=fixed_mask,
            substrate_coords=substrate_coords,
            evol_conditioning_fn=evo_conditioning_fn,
            generator=generator,
        )
        backbone_coords = self._frames_to_coords(R_final, t_final)
        pocket_mask = None
        if constraints is not None:
            pocket_mask = torch.zeros(B, L, dtype=torch.bool, device=device)
            for position in constraints.stachelhaus_positions:
                pocket_mask[:, max(0, int(position) - 2):min(L, int(position) + 3)] = True
        evol_nodes = self.node_connector(single_repr, pocket_mask)
        base_nodes = self.base_mpnn(backbone_coords, evol_nodes)
        edge_index, edge_features, edge_mask = get_protein_graph(t_final, R_final, k_neighbors=32)
        if edge_features.shape[-1] != 28:
            raise RuntimeError(f"protein graph edge width must be 28, got {edge_features.shape[-1]}")
        max_neighbors = 32
        if edge_index.shape[-1] < max_neighbors:
            pad = max_neighbors - edge_index.shape[-1]
            edge_index = F.pad(edge_index, (0, pad), value=0)
            edge_features = F.pad(edge_features, (0, 0, 0, pad), value=0.0)
            edge_mask = F.pad(edge_mask, (0, pad), value=False)
        edge_features = edge_features * edge_mask.unsqueeze(-1).to(edge_features.dtype)

        geometry = validate_backbone(backbone_coords, R_final)
        logits_per_draw = []
        for _ in range(n_draws_requested):
            logits_per_draw.append(self.multi_scale_designer(
                residue_feats=base_nodes,
                evol_node_feats=evol_nodes,
                edge_feats=edge_features,
                edge_index=edge_index,
                domain_boundaries=domain_bounds,
                module_boundaries=module_bounds,
                icosahedral_face=face_id,
            ))
        logits = torch.stack(logits_per_draw, dim=1)
        if constraints is not None and constraints.fixed_sequence is not None:
            fixed = constraints.fixed_sequence[:, None, :, None]
            fixed_mask_sequence = fixed >= 0
            one_hot_fixed = F.one_hot(constraints.fixed_sequence.clamp_min(0), num_classes=20)[:, None].to(logits.dtype)
            logits = torch.where(fixed_mask_sequence, torch.where(one_hot_fixed.bool(), logits, torch.full_like(logits, -1e4)), logits)

        n_draws = logits.shape[1]
        flat_logits = logits.reshape(B * n_draws, L, 20)
        policy_context = self._seq_to_repr(flat_logits)
        fixed_tokens = None
        if constraints is not None and constraints.fixed_sequence is not None:
            fixed_tokens = constraints.fixed_sequence[:, None, :].expand(B, n_draws, L).reshape(B * n_draws, L)
        sampled = self.sequence_policy.generate(
            policy_context,
            length=L,
            temperature=temperature,
            generator=generator,
            fixed_tokens=fixed_tokens,
        ).reshape(B, n_draws, L)
        sampled_repr = self._seq_to_repr(F.one_hot(sampled, num_classes=20).to(logits.dtype).reshape(B * n_draws, L, 20))
        objective_flat = self.pareto_head(sampled_repr)
        objectives = ParetoObjectives(
            evolutionary_plausibility=objective_flat.evolutionary_plausibility.reshape(B, n_draws),
            structural_stability=objective_flat.structural_stability.reshape(B, n_draws),
            expression_efficiency=objective_flat.expression_efficiency.reshape(B, n_draws),
            substrate_selectivity=objective_flat.substrate_selectivity.reshape(B, n_draws),
            assembly_compatibility=objective_flat.assembly_compatibility.reshape(B, n_draws),
        )
        return {
            "sequences": F.one_hot(sampled, num_classes=20).to(logits.dtype),
            "sequence_tokens": sampled,
            "sequence_logits": logits,
            "sequence_context": policy_context.reshape(B, n_draws, L, self.d_mpnn),
            "objective_context": sampled_repr.reshape(B, n_draws, L, self.d_mpnn),
            "backbone_coords": backbone_coords,
            "R_final": R_final,
            "t_final": t_final,
            "evol_plausibility": objectives.evolutionary_plausibility,
            "structural_stability": objectives.structural_stability,
            "expression_efficiency": objectives.expression_efficiency,
            "substrate_selectivity": objectives.substrate_selectivity,
            "assembly_compat": objectives.assembly_compatibility,
            "pareto_objectives": objectives,
            "pair_cond": pair_cond,
            "single_repr": single_repr,
            "geometry_valid": torch.tensor(geometry.valid, device=device),
            "geometry_report": geometry.as_dict(),
        }

    def sequence_policy_loss(
        self,
        conditioning_context: torch.Tensor,
        target_sequence: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Teacher-forced token loss, separate from no-grad sequence sampling."""
        logits = self.sequence_policy(conditioning_context, target_sequence)
        loss = F.cross_entropy(logits.transpose(1, 2), target_sequence, reduction="none")
        if padding_mask is None:
            return loss.mean()
        if padding_mask.shape != target_sequence.shape or padding_mask.dtype != torch.bool:
            raise ValueError("padding_mask must be bool and match target_sequence")
        valid = (~padding_mask).to(loss.dtype)
        return (loss * valid).sum() / valid.sum().clamp_min(1)

    def sequence_logprob(self, sequence_tokens, msa_tokens, pair_features, source_R, source_t):
        """Score candidate sequence under a deterministic MSA conditioning context."""
        with torch.no_grad():
            single, _ = self.evoformer(msa_tokens, pair_features)
            context = self.node_connector(single)
        return self.sequence_policy.logprob(context, sequence_tokens)

    def init_dpo_reference(self) -> None:
        self._reference_policy = deepcopy(self.sequence_policy).eval()
        for parameter in self._reference_policy.parameters():
            parameter.requires_grad_(False)

    def update_from_proteus(self, survivors, failures, msa, pair_features, n_dpo_steps=50, learning_rate=1e-5, best_context_batch_index=0):
        if not survivors or not failures:
            raise ValueError("Need at least one survivor and one failure for DPO")
        if self._reference_policy is None:
            self.init_dpo_reference()
        if msa.ndim != 3 or pair_features.ndim != 4 or msa.shape[0] != pair_features.shape[0]:
            raise ValueError("MSA and pair feature batch dimensions must match")
        if not 0 <= best_context_batch_index < msa.shape[0]:
            raise ValueError("best_context_batch_index is outside the MSA batch")
        with torch.no_grad():
            single, _ = self.evoformer(msa, pair_features)
            context = self.node_connector(single[best_context_batch_index:best_context_batch_index + 1])
        alphabet = "ACDEFGHIKLMNPQRSTVWY"
        if len({len(seq) for seq in survivors + failures}) != 1:
            raise ValueError("all PROTEUS sequences must have the same length")
        def encode(sequences):
            invalid = sorted({aa for seq in sequences for aa in seq if aa not in alphabet})
            if invalid:
                raise ValueError(f"invalid amino acids in PROTEUS sequences: {invalid}")
            return torch.tensor([[alphabet.index(aa) for aa in seq] for seq in sequences], device=context.device)
        n_pairs = min(len(survivors), len(failures))
        chosen, rejected = encode(survivors[:n_pairs]), encode(failures[:n_pairs])
        batch_context = context.expand(n_pairs, -1, -1).contiguous()
        mask = torch.ones_like(chosen, dtype=torch.bool)
        batch = DPOBatch(batch_context, chosen, rejected, mask, mask)
        trainable = [parameter for parameter in self.sequence_policy.parameters() if parameter.requires_grad]
        if not trainable:
            raise RuntimeError("Sequence policy is frozen; call prepare_for_training first")
        optimizer = torch.optim.AdamW(trainable, lr=learning_rate)
        metrics = {}
        for _ in range(n_dpo_steps):
            metrics = self.dpo_trainer.step(optimizer, self.sequence_policy, self._reference_policy, batch)
        return metrics

    def set_best_observed(self, value: Optional[float]) -> None:
        if value is not None and not 0 <= float(value) <= 1:
            raise ValueError("best_observed must be normalized historical utility in [0,1]")
        self.best_observed = None if value is None else float(value)

    def design(
        self,
        nrps_msa: torch.Tensor,
        source_backbone: Tuple[torch.Tensor, torch.Tensor],
        initial_pair_features: torch.Tensor,
        constraints: Optional[NRPSConstraints] = None,
        target_substrate: str = "PHE",
        n_designs: int = 500,
        n_pareto_samples: int = 50,
        device: Optional[str] = None,
        flow_steps: Optional[int] = None,
        use_rag: bool = False,
        objective_weights: Optional[torch.Tensor] = None,
    ) -> Dict:
        if n_designs < 1 or n_pareto_samples < 1 or n_pareto_samples > n_designs:
            raise ValueError("design counts must satisfy 1 <= n_pareto_samples <= n_designs")
        if target_substrate not in SUBSTRATE_TOKENS:
            raise ValueError(f"Unsupported substrate '{target_substrate}'")
        device_obj = torch.device(device) if device is not None else next(self.parameters()).device
        B = nrps_msa.shape[0]
        if B != 1:
            raise ValueError("design currently expects one conditioning example per call")
        if constraints is not None:
            for name in ("fixed_mask", "domain_boundaries", "module_boundaries", "icosahedral_face", "fixed_sequence", "domain_types"):
                value = getattr(constraints, name)
                if value is not None and value.shape[0] != 1:
                    raise ValueError(f"design constraints {name} must have batch size 1")
        batch_size = min(8, n_designs)
        sequences, objective_batches = [], []
        objective_contexts = []
        remaining = n_designs
        while remaining:
            current = min(batch_size, remaining)
            outputs = self(
                nrps_msa.expand(current, -1, -1).to(device_obj),
                initial_pair_features.expand(current, -1, -1, -1).to(device_obj),
                source_backbone[0].expand(current, -1, -1, -1).to(device_obj),
                source_backbone[1].expand(current, -1, -1).to(device_obj),
                constraints=constraints,
                substrate_id=torch.full((current,), SUBSTRATE_TOKENS[target_substrate], device=device_obj),
                n_flow_steps=flow_steps,
                n_mpnn_seqs=1,
                use_rag=use_rag,
            )
            sequences.append(outputs["sequence_tokens"][:, 0].detach().cpu())
            objective_contexts.append(outputs["objective_context"][:, 0].detach())
            objective_batches.append(torch.stack([
                outputs["evol_plausibility"][:, 0],
                outputs["structural_stability"][:, 0] / 100.0,
                outputs["expression_efficiency"][:, 0],
                outputs["substrate_selectivity"][:, 0],
                outputs["assembly_compat"][:, 0],
            ], dim=-1).detach().cpu())
            remaining -= current
        all_sequences = torch.cat(sequences)[:n_designs]
        all_objectives = torch.cat(objective_batches)[:n_designs]
        pareto_front, pareto_indices = self.pareto_head.compute_pareto_frontier(all_objectives)
        acquisition_scores = torch.zeros(n_designs)
        acquisition_status = "not computed: explicit objective_weights and historical best_observed are required"
        if self.best_observed is not None and objective_weights is not None:
            objective_context = torch.cat(objective_contexts, dim=0)[:n_designs]

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
                {"repr": objective_context},
                output_getter=normalized_objectives,
                n_samples=max(2, self.uncertainty_estimator.n_samples),
            )
            weights = torch.as_tensor(
                objective_weights,
                device=posterior["mean"].device,
                dtype=posterior["mean"].dtype,
            )
            utility_mean = BayesianUncertaintyEstimator.scalarize_objectives(
                posterior["mean"],
                weights=weights,
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
            frontier_order = torch.argsort(
                acquisition_scores.index_select(0, pareto_indices),
                descending=True,
            )
            selected = pareto_indices.index_select(0, frontier_order[:n_pareto_samples])
        else:
            selected = pareto_indices[:n_pareto_samples]
        return {
            "pareto_sequences": all_sequences.index_select(0, selected),
            "pareto_scores": all_objectives.index_select(0, selected),
            "pareto_count": int(pareto_indices.numel()),
            "all_sequences": all_sequences,
            "all_objectives": all_objectives,
            "acquisition_scores": acquisition_scores,
            "acquisition_status": acquisition_status,
            "total_generated": int(all_sequences.shape[0]),
        }

    @torch.no_grad()
    def compute_expected_improvement(self, mean, std, best_observed=None):
        historical = self.best_observed if best_observed is None else best_observed
        if historical is None:
            raise ValueError("Expected Improvement requires an explicit historical best")
        return BayesianUncertaintyEstimator.expected_improvement(mean, std, historical)

    def from_pretrained_not_supported(self):
        raise NotImplementedError

    def save(self, path: str) -> None:
        torch.save({name: getattr(self, name).state_dict() for name in (
            "evoformer", "flow_model", "base_mpnn", "pair_connector", "evol_cross_attn",
            "node_connector", "constraint_encoder", "structural_retriever", "substrate_conditioner",
            "multi_scale_designer", "pareto_head", "_seq_to_repr", "sequence_policy", "ret_proj",
        )}, path)

    def load_connectors(self, path: str) -> None:
        state = torch.load(path, map_location="cpu", weights_only=False)
        for name, values in state.items():
            module_name = "_seq_to_repr" if name == "seq_to_repr" else name
            if not hasattr(self, module_name):
                raise ValueError(f"checkpoint contains unknown module {name!r}")
            if module_name == "flow_model":
                values = self._migrate_flow_state_dict(
                    values,
                    self.flow_model.state_dict(),
                )
            if module_name == "multi_scale_designer":
                values = dict(values)
                legacy_edge_weight = values.get("edge_proj.weight")
                current_edge_weight = self.multi_scale_designer.edge_proj.weight
                if (
                    torch.is_tensor(legacy_edge_weight)
                    and legacy_edge_weight.ndim == 2
                    and legacy_edge_weight.shape[0] == current_edge_weight.shape[0]
                    and legacy_edge_weight.shape[1] == 16
                    and current_edge_weight.shape[1] == 28
                ):
                    migrated = current_edge_weight.detach().cpu().clone()
                    migrated[:, :16] = legacy_edge_weight
                    values["edge_proj.weight"] = migrated
                    warnings.warn(
                        "Migrated legacy 16-D edge projection to canonical 28-D geometry; "
                        "the 12 newly introduced geometry columns retain current initialization.",
                        UserWarning,
                        stacklevel=2,
                    )
            getattr(self, module_name).load_state_dict(values, strict=True)


CHIMERAv2 = CanonicalCHIMERAv2

__all__ = [
    "CHIMERAv2", "CanonicalCHIMERAv2", "NRPSConstraints", "BayesianUncertaintyEstimator",
    "DPOBatch", "DPOTrainer", "MergeReadyParetoMultiObjectiveHead", "PCGradOptimizer",
    "pcgrad_step", "project_conflicting_gradients", "seed_everything", "seed_worker",
    "make_generator", "SchrodingerBridge", "SE3SchrodingerBridge", "FlowMatchingBackbone",
    "MultiScaleNRPSDesigner", "SubstratePocketConditioner", "so3_log", "AutoregressiveSequencePolicy",
]
