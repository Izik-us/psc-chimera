"""Canonical CHIMERA v2 composition root.

The local MSA and ProteinMPNN-inspired backbones remain explicitly documented
approximations/stubs. Legacy ``CHIMERAv2`` remains isolated in
``chimera.chimera_v2`` for compatibility and is not instantiated here.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
from typing import Optional, Dict, Tuple
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings

from .bayesian import BayesianUncertaintyEstimator
from .conditioning import SubstratePocketConditioner
from .components import (
    MSARepresentationBackbone,
    TriangularPairUpdateConnector,
    EvolCrossAttentionConnector,
    NodeProjectionConnector,
    NRPSConstraintEncoder,
    ProteinMPNNBackbone,
)
from .dpo import DPOBatch, DPOTrainer
from .domain_schema import AssemblySchema, DomainSpan, DomainType, NRPSConstraints
from .evaluators import BiologicalObjectiveEvaluator
from .se3_flow import FlowMatchingBackbone
from .geometry import validate_backbone
from .lie import so3_log
from .objective_schema import validate_objective_target
from .pareto_pcgrad import (
    OBJECTIVE_FEATURE_PLAN,
    MergeReadyParetoMultiObjectiveHead,
    ObjectiveFeatureEncoder,
    ObjectiveFeatureSource,
    ParetoObjectives,
)
from .pcgrad import PCGradOptimizer, pcgrad_step, project_conflicting_gradients
from .proteinmpnn import get_protein_graph
from .reproducibility import make_generator, seed_everything, seed_worker
from .schrodinger_bridge import SchrodingerBridge, SE3SchrodingerBridge
from .autoregressive_policy import AutoregressiveSequencePolicy
from .retrieval import StructuralRetriever
from .sequence_design import MultiScaleNRPSDesigner


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
        self.n_flow_blocks = n_flow_blocks
        self.n_flow_steps = n_flow_steps
        self.n_mpnn_seqs = n_mpnn_seqs
        self.n_domains = n_domains
        self.n_modules = n_modules
        self.best_observed: Optional[float] = None
        self._is_canonical_composition = True
        self._reference_policy: Optional[AutoregressiveSequencePolicy] = None

        self.evoformer = MSARepresentationBackbone(d_evo_single, d_evo_pair)
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
        self.objective_feature_encoder = ObjectiveFeatureEncoder(d_mpnn, d_evo_single)
        self.pareto_head = MergeReadyParetoMultiObjectiveHead(d_model=d_mpnn)
        self.objective_evaluator = BiologicalObjectiveEvaluator()
        self._seq_to_repr = nn.Linear(20, d_mpnn)
        self.sequence_policy = AutoregressiveSequencePolicy(d_mpnn)
        self.ret_proj = nn.Linear(30, d_pair_out)
        self.uncertainty_estimator = BayesianUncertaintyEstimator(n_samples=n_mc_dropout)
        self.dpo_trainer = DPOTrainer(beta=0.1)
        self._component_names = (
            "evoformer",
            "flow_model",
            "base_mpnn",
            "pair_connector",
            "evol_cross_attn",
            "node_connector",
            "constraint_encoder",
            "structural_retriever",
            "substrate_conditioner",
            "multi_scale_designer",
            "objective_feature_encoder",
            "pareto_head",
            "_seq_to_repr",
            "sequence_policy",
            "ret_proj",
            "uncertainty_estimator",
        )
        self.component_status = {
            name: {
                "initialization": "random",
                "training_status": "not_started",
                "training_steps": 0,
                "checkpoint_source": None,
                "checkpoint_sha256": None,
                "migration_status": "none",
                "training_dataset_manifest": None,
            }
            for name in self._component_names
        }
        self.objective_calibration = {name: None for name in OBJECTIVE_FEATURE_PLAN}
        self.objective_status = {
            name: {
                "initialization": "random",
                "training_status": "not_started",
                "training_steps": 0,
                "label_sources": [],
                "label_kinds": [],
                "structure_data_sources": [],
                "training_dataset_manifest": None,
            }
            for name in self.objective_calibration
        }

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
            ("evoformer", model.evoformer, evoformer_ckpt, "CHIMERA MSA approximation"),
            ("flow_model", model.flow_model, flow_ckpt, "canonical SB flow backbone"),
            ("base_mpnn", model.base_mpnn, mpnn_ckpt, "ProteinMPNN-inspired local module"),
        )
        for component_name, module, path, name in modules:
            if path is None:
                continue
            checkpoint_path = Path(path).expanduser().resolve()
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            state = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
            if component_name == "flow_model":
                state = cls._migrate_flow_state_dict(state, module.state_dict())
            try:
                module.load_state_dict(state, strict=True)
            except (RuntimeError, TypeError) as exc:
                raise RuntimeError(
                    f"{name} checkpoint is incompatible with this module: {path}; "
                    "native OpenFold/RFdiffusion/ProteinMPNN checkpoints need explicit adapters"
                ) from exc
            model.component_status[component_name].update(
                initialization="loaded_unverified",
                checkpoint_source=str(checkpoint_path),
                checkpoint_sha256=hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
            )
        return model

    @staticmethod
    def _migrate_flow_state_dict(state: Dict[str, torch.Tensor], target_state: Dict[str, torch.Tensor]):
        """Migrate historical flow keys and the former floor-divided IPA layout.

        Old IPA checkpoints could use 12 heads even when ``d_single`` was not
        divisible by 12. Their per-head scalar features were truncated. The
        migration preserves overlapping heads/features, zero-pads newly valid
        scalar dimensions, and initializes new heads from the target module.
        The old frozen bridge was random, frozen, and dimension preserving; the
        canonical backbone replaces it with Identity, so those keys are dropped.
        """
        migrated = dict(state)
        removed_bridge_keys = [
            key for key in migrated
            if key == "frozen_bridge" or key.startswith("frozen_bridge.")
            or ".frozen_bridge." in key
        ]
        for key in removed_bridge_keys:
            migrated.pop(key)
        if removed_bridge_keys:
            warnings.warn(
                "Discarded legacy random frozen_bridge parameters; canonical flow uses Identity.",
                UserWarning,
                stacklevel=2,
            )

        prefix = "flow_model.velocity_field."
        alias_prefix = "sb_model.drift_model."
        for key, value in tuple(migrated.items()):
            if key.startswith(prefix):
                alias = alias_prefix + key[len(prefix):]
                if alias in target_state:
                    migrated.setdefault(alias, value)

        for target_gamma_key, target_gamma in target_state.items():
            if not target_gamma_key.endswith(".ipa.gamma"):
                continue
            module_prefix = target_gamma_key[:-len("gamma")]
            source_gamma = migrated.get(target_gamma_key)
            if source_gamma is None or source_gamma.shape == target_gamma.shape:
                continue
            old_heads = int(source_gamma.numel())
            new_heads = int(target_gamma.numel())
            copy_heads = min(old_heads, new_heads)
            source_q = migrated[module_prefix + "q_s.weight"]
            target_q = target_state[module_prefix + "q_s.weight"]
            old_head_dim = source_q.shape[0] // old_heads
            new_head_dim = target_q.shape[0] // new_heads
            pair_dim = target_state[module_prefix + "pair_bias.weight"].shape[1]

            for name in ("q_s.weight", "k_s.weight", "v_s.weight"):
                source = migrated[module_prefix + name]
                target = target_state[module_prefix + name].clone()
                source = source.reshape(old_heads, old_head_dim, source.shape[1])
                target_view = target.reshape(new_heads, new_head_dim, target.shape[1])
                for head in range(copy_heads):
                    width = min(old_head_dim, new_head_dim)
                    target_view[head, :width] = source[head, :width]
                migrated[module_prefix + name] = target

            for name in ("q_p.weight", "k_p.weight", "v_p.weight"):
                source = migrated[module_prefix + name]
                target = target_state[module_prefix + name].clone()
                points = source.shape[0] // (old_heads * 3)
                source_view = source.reshape(old_heads, points, 3, source.shape[1])
                target_view = target.reshape(new_heads, points, 3, target.shape[1])
                target_view[:copy_heads] = source_view[:copy_heads]
                migrated[module_prefix + name] = target

            for name in ("pair_bias.weight", "substrate_gate.weight"):
                source = migrated[module_prefix + name]
                target = target_state[module_prefix + name].clone()
                target[:copy_heads] = source[:copy_heads]
                migrated[module_prefix + name] = target

            substrate_bias_key = module_prefix + "substrate_gate.bias"
            source_substrate_bias = migrated[substrate_bias_key]
            target_substrate_bias = target_state[substrate_bias_key].clone()
            target_substrate_bias[:copy_heads] = source_substrate_bias[:copy_heads]
            migrated[substrate_bias_key] = target_substrate_bias

            source_gamma = migrated[module_prefix + "gamma"]
            target_gamma_value = target_state[module_prefix + "gamma"].clone()
            target_gamma_value[:copy_heads] = source_gamma[:copy_heads]
            migrated[module_prefix + "gamma"] = target_gamma_value

            source_out = migrated[module_prefix + "out.weight"]
            target_out = target_state[module_prefix + "out.weight"].clone()
            old_width = old_head_dim + 3 + pair_dim
            new_width = new_head_dim + 3 + pair_dim
            source_view = source_out.reshape(source_out.shape[0], old_heads, old_width)
            target_view = target_out.reshape(target_out.shape[0], new_heads, new_width)
            for head in range(copy_heads):
                target_view[:, head, :min(old_head_dim, new_head_dim)] = source_view[
                    :, head, :min(old_head_dim, new_head_dim)
                ]
                target_view[:, head, new_head_dim:new_head_dim + 3] = source_view[
                    :, head, old_head_dim:old_head_dim + 3
                ]
                target_view[:, head, new_head_dim + 3:] = source_view[
                    :, head, old_head_dim + 3:old_width
                ]
            migrated[module_prefix + "out.weight"] = target_out

            warnings.warn(
                f"Migrated truncated IPA heads at {module_prefix[:-1]} from "
                f"{old_heads} to {new_heads} divisible heads; overlapping head features were preserved.",
                UserWarning,
                stacklevel=2,
            )
        return migrated

    def freeze_pretrained(self) -> None:
        """Freeze only components explicitly marked trained, never random weights."""
        for name in self._component_names:
            if self.component_status[name]["training_status"] != "validated":
                continue
            module = getattr(self, name)
            for parameter in module.parameters():
                parameter.requires_grad_(False)

    @staticmethod
    def _migrate_pareto_state_dict(state, target_state):
        """Migrate legacy learned-assembly Pareto weights to typed four-head objectives."""
        migrated = dict(state)
        retired = [key for key in migrated if key.startswith("head_asm.")]
        for key in retired:
            migrated.pop(key)
        unexpected = set(migrated) - set(target_state)
        if unexpected:
            raise RuntimeError(f"unknown Pareto checkpoint keys: {sorted(unexpected)}")
        missing = set(target_state) - set(migrated)
        if any(not key.startswith("feature_fusions.") for key in missing):
            raise RuntimeError(f"Pareto checkpoint is missing required keys: {sorted(missing)}")
        for key in missing:
            migrated[key] = target_state[key].detach().cpu().clone()
        migration_status = None
        if retired or missing:
            migration_status = (
                "legacy learned assembly head retired; typed feature-fusion parameters retain current initialization"
            )
        return migrated, migration_status

    def unfreeze_connectors(self) -> None:
        self.prepare_for_training("sequence")

    def prepare_for_training(self, regime: str = "sequence") -> list[torch.nn.Parameter]:
        from .training import configure_trainable_components

        configure_trainable_components(self, regime)
        return [parameter for parameter in self.parameters() if parameter.requires_grad]

    def inference_readiness(self) -> dict:
        """Report component training state; module existence is not readiness."""
        required = (
            "evoformer", "flow_model", "base_mpnn", "pair_connector",
            "evol_cross_attn", "node_connector", "constraint_encoder",
            "substrate_conditioner", "multi_scale_designer", "pareto_head",
            "objective_feature_encoder", "_seq_to_repr", "sequence_policy",
        )
        statuses = {name: dict(self.component_status[name]) for name in required}
        incompatible = [name for name, state in statuses.items() if state["initialization"] == "incompatible_checkpoint"]
        trained = [
            name for name, state in statuses.items()
            if state["training_status"] == "validated"
            and state.get("training_dataset_manifest")
            and state.get("validation", {}).get("dataset_manifest")
        ]
        if incompatible:
            state = "INCOMPATIBLE_CHECKPOINT"
        elif (
            len(trained) == len(required)
            and all(
                state["training_status"] == "validated"
                and state.get("training_dataset_manifest")
                and state.get("validation", {}).get("dataset_manifest")
                for state in self.objective_status.values()
            )
            and all(self.objective_calibration.values())
        ):
            state = "TRAINED"
        elif trained:
            state = "PARTIALLY_TRAINED"
        else:
            state = "UNINITIALIZED"
        reasons = []
        if len(trained) != len(required):
            reasons.append("one or more required components are not marked trained")
        if not all(self.objective_calibration.values()):
            reasons.append("one or more objective predictors lack calibration evidence")
        if any(
            state["training_status"] != "validated"
            or not state.get("training_dataset_manifest")
            or not state.get("validation", {}).get("dataset_manifest")
            for state in self.objective_status.values()
        ):
            reasons.append("one or more objective heads have no supervised labels")
        if len(trained) != len(required):
            reasons.append("one or more model components lack held-out validation")
        return {"state": state, "components": statuses, "reasons": reasons}

    def model_configuration(self) -> dict[str, int]:
        """Return the canonical architecture identity used by checkpoint manifests."""
        return {
            "d_evo_single": self.d_evo_single,
            "d_evo_pair": self.d_evo_pair,
            "d_se3": self.d_se3,
            "d_pair_out": self.d_pair_out,
            "d_mpnn": self.d_mpnn,
            "n_flow_blocks": self.n_flow_blocks,
            "n_flow_steps": self.n_flow_steps,
            "n_retrieve": self.structural_retriever.n_retrieve,
            "n_mpnn_seqs": self.n_mpnn_seqs,
            "n_mc_dropout": self.uncertainty_estimator.n_samples,
            "n_domains": self.n_domains,
            "n_modules": self.n_modules,
        }

    def objective_provenance(self) -> dict[str, dict]:
        provenance = {
            name: {
                "source": (
                    "deterministic_evaluator"
                    if name == "assembly_compatibility"
                    else "validated_neural_surrogate"
                    if state["training_status"] == "validated"
                    else "loaded_unverified_neural_surrogate"
                    if state["initialization"] == "loaded_unverified"
                    else "random_neural_surrogate"
                    if state["training_steps"] == 0
                    else "unvalidated_neural_surrogate"
                ),
                "initialization": state["initialization"],
                "training_status": state["training_status"],
                "training_steps": state["training_steps"],
                "label_sources": list(state["label_sources"]),
                "label_kinds": list(state["label_kinds"]),
                "structure_data_sources": list(state["structure_data_sources"]),
                "generated_reference_distribution_shift_risk": (
                    "present_reference_training_vs_generated_inference"
                    if name == "structural_stability" and "reference" in state["structure_data_sources"]
                    else "not_established_or_not_applicable"
                ),
                "feature_sources": list(OBJECTIVE_FEATURE_PLAN.get(name, ("structural",))),
                "calibrated": (
                    False if name == "assembly_compatibility"
                    else self.objective_calibration[name] is not None
                ),
                "differentiable": name != "assembly_compatibility",
                "uncertainty_available": (
                    False if name == "assembly_compatibility"
                    else self.objective_calibration[name] is not None
                    and state["training_status"] == "validated"
                ),
                "biological_measurement": False,
            }
            for name, state in self.objective_status.items()
        }
        provenance["assembly_compatibility"] = {
            "source": "deterministic_evaluator",
            "initialization": "not_applicable",
            "training_status": "not_applicable",
            "training_steps": 0,
            "label_sources": ["icosahedral_interface_geometry_proxy"],
            "label_kinds": ["deterministic_evaluator"],
            "feature_sources": ["structural"],
            "calibrated": False,
            "differentiable": False,
            "uncertainty_available": False,
            "biological_measurement": False,
        }
        return provenance

    def record_objective_calibration(
        self,
        name: str,
        predicted_mean: torch.Tensor,
        epistemic_std: torch.Tensor,
        targets: torch.Tensor,
        *,
        calibration_manifest: str,
    ) -> dict[str, float | int | str]:
        """Record empirical held-out calibration evidence for one objective predictor."""
        if name not in self.objective_status:
            raise ValueError(f"unknown objective: {name}")
        status = self.objective_status[name]
        validation = status.get("validation")
        if status["training_status"] != "validated" or not validation:
            raise RuntimeError("objective must be trained and held-out validated before calibration")
        if not calibration_manifest:
            raise ValueError("calibration_manifest is required")
        used_manifests = {
            status.get("training_dataset_manifest"),
            validation.get("dataset_manifest"),
        }
        if calibration_manifest in used_manifests:
            raise ValueError("calibration data must be distinct from training and validation data")
        if predicted_mean.shape != epistemic_std.shape or predicted_mean.shape != targets.shape:
            raise ValueError("calibration predictions, uncertainty, and targets must have matching shapes")
        validate_objective_target(name, predicted_mean, targets)
        if predicted_mean.numel() < 30:
            raise ValueError("uncertainty calibration requires at least 30 held-out examples")
        if epistemic_std.dtype != torch.float32:
            raise TypeError("epistemic_std must have dtype torch.float32")
        if epistemic_std.ndim != 1 or not torch.isfinite(epistemic_std).all():
            raise ValueError("calibration data must be finite")
        if torch.any(epistemic_std < 0):
            raise ValueError("epistemic_std must be non-negative")
        residual = targets - predicted_mean
        one_sigma_coverage = float((residual.abs() <= epistemic_std).float().mean())
        if not 0.58 <= one_sigma_coverage <= 0.78:
            raise ValueError(
                "one-sigma coverage is outside the accepted calibration band [0.58, 0.78]"
            )
        record = {
            "method": "heldout_one_sigma_coverage",
            "calibration_manifest": calibration_manifest,
            "examples": int(predicted_mean.numel()),
            "rmse": float(residual.square().mean().sqrt()),
            "mae": float(residual.abs().mean()),
            "one_sigma_coverage": one_sigma_coverage,
        }
        self.objective_calibration[name] = record
        return record

    def require_inference_ready(self, *, allow_experimental: bool = False) -> dict:
        readiness = self.inference_readiness()
        if readiness["state"] != "TRAINED" and not allow_experimental:
            raise RuntimeError(
                f"Canonical CHIMERA is {readiness['state']}; production inference requires "
                "trained components and calibrated objectives. Use explicit experimental mode to proceed."
            )
        return readiness

    def count_frozen(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if not parameter.requires_grad)

    def count_trainable(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def init_dpo_reference(self) -> None:
        policy_status = self.component_status["sequence_policy"]
        if policy_status["training_status"] != "validated" or not policy_status.get("validation"):
            raise RuntimeError(
                "DPO reference initialization requires a supervised, validated sequence policy"
            )
        reference_policy = deepcopy(self.sequence_policy).eval()
        for parameter in reference_policy.parameters():
            parameter.requires_grad_(False)
        object.__setattr__(self, "_reference_policy", reference_policy)

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
        msa_padding_mask: Optional[torch.Tensor] = None,
        constraints: Optional[NRPSConstraints] = None,
        substrate_id: Optional[torch.Tensor] = None,
        substrate_coords: Optional[torch.Tensor] = None,
        substrate_types: Optional[torch.Tensor] = None,
        n_flow_steps: Optional[int] = None,
        n_mpnn_seqs: Optional[int] = None,
        use_rag: bool = False,
        temperature: float = 1.0,
        generator: Optional[torch.Generator] = None,
        validate_geometry: bool = True,
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
            single_repr, pair_repr = self.evoformer(
                msa_tokens,
                initial_pair_features,
                msa_padding_mask=msa_padding_mask,
            )

        retrieved_context = None
        if not use_rag:
            rag_status = "DISABLED"
        elif self.structural_retriever.index_embs is None or self.structural_retriever.index_embs.shape[0] == 0:
            rag_status = "RAG_UNAVAILABLE: index not configured"
        elif substrate_id is None:
            rag_status = "RAG_UNAVAILABLE: substrate query missing"
        else:
            rag_status = "RAG_UNAVAILABLE: query/index embedding spaces are not aligned"
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

        geometry = (
            validate_backbone(backbone_coords, R_final)
            if validate_geometry
            else None
        )
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
                edge_mask=edge_mask,
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
        objective_raw_features = {
            ObjectiveFeatureSource.SEQUENCE.value: F.one_hot(
                sampled.reshape(B * n_draws, L), num_classes=20
            ).to(logits.dtype),
            ObjectiveFeatureSource.EVOLUTIONARY.value: single_repr[:, None]
            .expand(-1, n_draws, -1, -1)
            .reshape(B * n_draws, L, self.d_evo_single),
        }
        face_features = F.one_hot(face_id.long(), num_classes=20).to(logits.dtype)
        structural_features = torch.cat(
            (
                self.objective_feature_encoder.structural_invariants(R_final, t_final),
                face_features[:, None].expand(-1, L, -1),
            ),
            dim=-1,
        )
        objective_raw_features[ObjectiveFeatureSource.STRUCTURAL.value] = structural_features[:, None]
        objective_raw_features[ObjectiveFeatureSource.STRUCTURAL.value] = objective_raw_features[
            ObjectiveFeatureSource.STRUCTURAL.value
        ].expand(-1, n_draws, -1, -1).reshape(B * n_draws, L, 52)
        if substrate_id is not None:
            objective_raw_features[ObjectiveFeatureSource.SUBSTRATE.value] = F.one_hot(
                substrate_id.long(), num_classes=20
            ).to(logits.dtype)[:, None].expand(-1, n_draws, -1).reshape(B * n_draws, 20)
        encoded_objective_features = self.objective_feature_encoder(objective_raw_features)
        available_objectives = tuple(
            name for name, sources in OBJECTIVE_FEATURE_PLAN.items()
            if set(sources).issubset(encoded_objective_features)
        )
        objective_flat = self.pareto_head(
            encoded_objective_features,
            objective_names=available_objectives,
        )
        candidate_coords = backbone_coords[:, None].expand(-1, n_draws, -1, -1, -1).reshape(
            B * n_draws, L, 4, 3
        )
        candidate_rotations = R_final[:, None].expand(-1, n_draws, -1, -1, -1).reshape(
            B * n_draws, L, 3, 3
        )
        candidate_faces = face_id[:, None].expand(-1, n_draws).reshape(B * n_draws)
        proxy_evaluation = self.objective_evaluator.evaluate(
            sampled.reshape(B * n_draws, L),
            candidate_coords,
            rotations=candidate_rotations,
            faces=candidate_faces,
        )
        proxy_outputs = proxy_evaluation["objective_outputs"]
        proxy_scores = torch.stack(
            [
                proxy_outputs["evolutionary_plausibility"].value,
                proxy_outputs["structural_validity"].value,
                proxy_outputs["expression_efficiency"].value,
                proxy_outputs["substrate_selectivity"].value,
                proxy_outputs["assembly_compatibility"].value,
            ],
            dim=-1,
        ).reshape(B, n_draws, 5)
        assembly_scores = proxy_outputs["assembly_compatibility"].value.reshape(B, n_draws)
        objectives = ParetoObjectives(
            evolutionary_plausibility=objective_flat.evolutionary_plausibility.reshape(B, n_draws),
            structural_stability=objective_flat.structural_stability.reshape(B, n_draws),
            expression_efficiency=objective_flat.expression_efficiency.reshape(B, n_draws),
            substrate_selectivity=objective_flat.substrate_selectivity.reshape(B, n_draws),
            assembly_compatibility=assembly_scores,
            available_objectives=tuple((*available_objectives, "assembly_compatibility")),
        )
        return {
            "sequences": F.one_hot(sampled, num_classes=20).to(logits.dtype),
            "sequence_tokens": sampled,
            "sequence_logits": logits,
            "sequence_context": policy_context.reshape(B, n_draws, L, self.d_mpnn),
            "objective_context": sampled_repr.reshape(B, n_draws, L, self.d_mpnn),
            "objective_feature_embeddings": {
                name: value.reshape(B, n_draws, self.d_mpnn)
                for name, value in encoded_objective_features.items()
            },
            "objective_predictions_available": objectives.available_objectives,
            "backbone_coords": backbone_coords,
            "R_final": R_final,
            "t_final": t_final,
            "evol_plausibility": objectives.evolutionary_plausibility,
            "structural_stability": objectives.structural_stability,
            "expression_efficiency": objectives.expression_efficiency,
            "substrate_selectivity": objectives.substrate_selectivity,
            "assembly_compat": assembly_scores,
            "pareto_objectives": objectives,
            "pair_cond": pair_cond,
            "single_repr": single_repr,
            "geometry_valid": (
                torch.tensor(geometry.candidate_valid, dtype=torch.bool, device=device)
                if geometry is not None else None
            ),
            "geometry_report": geometry.as_dict() if geometry is not None else None,
            "objective_provenance": self.objective_provenance(),
            "objective_proxy_outputs": proxy_outputs,
            "objective_proxy_scores": proxy_scores,
            "rag_status": rag_status,
        }

    def sequence_policy_loss(
        self,
        conditioning_context: torch.Tensor,
        target_sequence: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Teacher-forced token loss, separate from no-grad sequence sampling."""
        if target_sequence.ndim != 2:
            raise ValueError("target_sequence must have shape (B,L)")
        if torch.any((target_sequence < 0) | (target_sequence > self.sequence_policy.vocab_size)):
            raise ValueError("target_sequence contains tokens outside amino-acid vocabulary plus padding")
        if padding_mask is None:
            padding_mask = target_sequence.eq(self.sequence_policy.vocab_size)
        if padding_mask.shape != target_sequence.shape or padding_mask.dtype != torch.bool:
            raise ValueError("padding_mask must be bool and match target_sequence")
        padding_mask = padding_mask | target_sequence.eq(self.sequence_policy.vocab_size)
        valid = ~padding_mask
        if not valid.any():
            raise ValueError("target_sequence must contain at least one supervised residue")
        safe_targets = target_sequence.masked_fill(padding_mask, 0)
        logits = self.sequence_policy(conditioning_context, safe_targets)
        loss = F.cross_entropy(logits.transpose(1, 2), safe_targets, reduction="none")
        return (loss * valid.to(loss.dtype)).sum() / valid.sum()

    def sequence_logprob(self, sequence_tokens, msa_tokens, pair_features, source_R, source_t):
        """Score candidate sequence under a deterministic MSA conditioning context."""
        with torch.no_grad():
            single, _ = self.evoformer(msa_tokens, pair_features)
            context = self.node_connector(single)
        return self.sequence_policy.logprob(context, sequence_tokens)

    def update_from_proteus(self, survivors, failures, msa, pair_features, n_dpo_steps=50, learning_rate=1e-5, best_context_batch_index=0):
        if not survivors or not failures:
            raise ValueError("Need at least one survivor and one failure for DPO")
        policy_status = self.component_status["sequence_policy"]
        if policy_status["training_status"] != "validated" or not policy_status.get("validation"):
            raise RuntimeError(
                "PROTEUS preference updates require a supervised, validated sequence policy"
            )
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
        from .training import CanonicalTrainer, TrainingRegime

        trainer = CanonicalTrainer(
            self,
            TrainingRegime.PREFERENCE,
            learning_rate=learning_rate,
        )
        metrics = {}
        for _ in range(n_dpo_steps):
            metrics = trainer.train_preference_step(batch)
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
        experimental: bool = False,
    ) -> Dict:
        readiness = self.require_inference_ready(allow_experimental=experimental)
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
        sequences, objective_batches, proxy_batches = [], [], []
        objective_feature_batches: dict[str, list[torch.Tensor]] = {}
        deterministic_assembly_batches = []
        rag_statuses = []
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
            rag_statuses.append(outputs["rag_status"])
            for source, embedding in outputs["objective_feature_embeddings"].items():
                objective_feature_batches.setdefault(source, []).append(embedding[:, 0].detach())
            objective_batches.append(torch.stack([
                outputs["evol_plausibility"][:, 0],
                outputs["structural_stability"][:, 0] / 100.0,
                outputs["expression_efficiency"][:, 0],
                outputs["substrate_selectivity"][:, 0],
                outputs["assembly_compat"][:, 0],
            ], dim=-1).detach().cpu())
            proxy_batches.append(outputs["objective_proxy_scores"][:, 0].detach().cpu())
            deterministic_assembly_batches.append(outputs["assembly_compat"][:, 0].detach())
            remaining -= current
        all_sequences = torch.cat(sequences)[:n_designs]
        if readiness["state"] == "TRAINED":
            all_objectives = torch.cat(objective_batches)[:n_designs]
            ranking_source = "validated_calibrated_neural_surrogates"
            objective_names = [
                "evolutionary_plausibility", "structural_stability",
                "expression_efficiency", "substrate_selectivity", "assembly_compatibility",
            ]
        else:
            all_objectives = torch.cat(proxy_batches)[:n_designs]
            ranking_source = "deterministic_proxy_evaluator"
            objective_names = [
                "normalized_sequence_entropy_proxy", "backbone_sanity_validity",
                "rule_based_codon_optimization_proxy", "target_profile_match_proxy",
                "icosahedral_interface_geometry_proxy",
            ]
        pareto_front, pareto_indices = self.pareto_head.compute_pareto_frontier(all_objectives)
        acquisition_scores = torch.zeros(n_designs)
        acquisition_status = "not computed: explicit objective_weights and historical best_observed are required"
        if (
            self.best_observed is not None
            and objective_weights is not None
            and readiness["state"] == "TRAINED"
            and self.component_status["pareto_head"]["training_status"] == "validated"
        ):
            objective_feature_inputs = {
                source: torch.cat(values, dim=0)[:n_designs]
                for source, values in objective_feature_batches.items()
            }
            deterministic_assembly = torch.cat(deterministic_assembly_batches, dim=0)[:n_designs]

            def normalized_objectives(result):
                return torch.stack(
                    (
                        torch.sigmoid(result.evolutionary_plausibility),
                        result.structural_stability.clamp(0, 100) / 100.0,
                        result.expression_efficiency.clamp(0, 1),
                        result.substrate_selectivity.clamp(0, 1),
                        deterministic_assembly.to(result.substrate_selectivity).clamp(0, 1),
                    ),
                    dim=-1,
                )

            posterior = self.uncertainty_estimator.predict_fixed_candidate(
                self.pareto_head,
                {"features": objective_feature_inputs},
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
            "readiness": readiness,
            "objective_provenance": self.objective_provenance(),
            "ranking_source": ranking_source,
            "objective_names": objective_names,
            "rag_status": rag_statuses[0] if rag_statuses and len(set(rag_statuses)) == 1 else "MIXED",
        }

    @torch.no_grad()
    def compute_expected_improvement(self, mean, std, best_observed=None):
        historical = self.best_observed if best_observed is None else best_observed
        if historical is None:
            raise ValueError("Expected Improvement requires an explicit historical best")
        return BayesianUncertaintyEstimator.expected_improvement(mean, std, historical)

    def save(self, path: str) -> None:
        """Save component-transfer weights; this is not a resumable training checkpoint."""
        torch.save({
            "artifact_type": "component_transfer_checkpoint",
            "format_version": 1,
            "components": {
                name: getattr(self, name).state_dict() for name in (
                    "evoformer", "flow_model", "base_mpnn", "pair_connector", "evol_cross_attn",
                    "node_connector", "constraint_encoder", "structural_retriever", "substrate_conditioner",
                    "multi_scale_designer", "objective_feature_encoder", "pareto_head", "_seq_to_repr",
                    "sequence_policy", "ret_proj",
                )
            },
        }, path)

    def load_connectors(self, path: str) -> None:
        state = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(state, dict):
            raise TypeError("component-transfer checkpoint must contain a mapping")
        artifact_type = state.get("artifact_type")
        if artifact_type == "canonical_training_checkpoint":
            raise ValueError("canonical training checkpoints must be loaded with CanonicalTrainer.resume")
        if artifact_type == "component_transfer_checkpoint":
            if state.get("format_version") != 1 or not isinstance(state.get("components"), dict):
                raise ValueError("unsupported or malformed component-transfer checkpoint")
            state = state["components"]
        elif artifact_type is not None:
            raise ValueError(f"unsupported checkpoint artifact type: {artifact_type!r}")
        checkpoint_path = Path(path).expanduser().resolve()
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
                edge_type_key = "edge_type_embedding.weight"
                if edge_type_key not in values:
                    values[edge_type_key] = self.multi_scale_designer.edge_type_embedding.weight.detach().cpu().clone()
                    self.component_status[module_name]["migration_status"] = "new edge-type embeddings retain random initialization"
                    warnings.warn(
                        "Legacy multi-scale checkpoint has no edge-type embeddings; initialized these new parameters randomly and left them trainable.",
                        UserWarning,
                        stacklevel=2,
                    )
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
            if module_name == "pareto_head":
                values, migration_status = self._migrate_pareto_state_dict(
                    values,
                    self.pareto_head.state_dict(),
                )
                if migration_status is not None:
                    self.component_status[module_name]["migration_status"] = migration_status
                    warnings.warn(migration_status, UserWarning, stacklevel=2)
            getattr(self, module_name).load_state_dict(values, strict=True)
            if module_name in self.component_status:
                self.component_status[module_name].update(
                    initialization="loaded_unverified",
                    checkpoint_source=str(checkpoint_path),
                    checkpoint_sha256=hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
                )


CHIMERAv2 = CanonicalCHIMERAv2

__all__ = [
    "CHIMERAv2", "CanonicalCHIMERAv2", "NRPSConstraints", "BayesianUncertaintyEstimator",
    "DPOBatch", "DPOTrainer", "MergeReadyParetoMultiObjectiveHead", "PCGradOptimizer",
    "pcgrad_step", "project_conflicting_gradients", "seed_everything", "seed_worker",
    "make_generator", "SchrodingerBridge", "SE3SchrodingerBridge", "FlowMatchingBackbone",
    "MultiScaleNRPSDesigner", "SubstratePocketConditioner", "so3_log", "AutoregressiveSequencePolicy",
]
