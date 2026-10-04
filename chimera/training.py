"""Canonical staged training, gradient auditing, and checkpoint ownership."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from enum import Enum
import hashlib
import json
import random
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional

import numpy as np
import torch
import torch.nn.functional as F

from .checkpoint import (
    CheckpointManifest,
    config_hash,
    git_provenance,
    runtime_provenance,
    state_schema_hash,
    validate_checkpoint_compatibility,
)
from .dpo import DPOBatch
from .proteinmpnn import get_protein_graph
from .pareto_pcgrad import OBJECTIVE_FEATURE_PLAN, ObjectiveFeatureSource
from .objective_schema import OBJECTIVE_SCHEMA, objective_schema_hash


def _qualified_name(value: Any) -> str:
    cls = value if isinstance(value, type) else type(value)
    return f"{cls.__module__}.{cls.__qualname__}"


def _stable_config_value(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, Mapping):
        return {str(key): _stable_config_value(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (tuple, list)):
        return [_stable_config_value(item) for item in value]
    if callable(value):
        code = getattr(value, "__code__", None)
        if code is not None:
            return {
                "callable": f"{getattr(value, '__module__', '')}.{getattr(value, '__qualname__', type(value).__qualname__)}",
                "bytecode": code.co_code.hex(),
                "constants": repr(code.co_consts),
                "defaults": _stable_config_value(getattr(value, "__defaults__", None)),
            }
        return {"callable_class": _qualified_name(value)}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def _optimizer_fingerprint(optimizer: torch.optim.Optimizer, model) -> tuple[str, str]:
    parameter_names = {id(parameter): name for name, parameter in model.named_parameters()}
    groups = []
    for group in optimizer.param_groups:
        groups.append({
            "parameter_names": [parameter_names.get(id(parameter), "unregistered") for parameter in group["params"]],
            "configuration": _stable_config_value({key: value for key, value in group.items() if key != "params"}),
        })
    payload = {
        "class": _qualified_name(optimizer),
        "defaults": _stable_config_value(optimizer.defaults),
        "parameter_groups": groups,
    }
    return payload["class"], config_hash(payload)


def _scheduler_fingerprint(scheduler: Optional[Any]) -> tuple[str, str]:
    if scheduler is None:
        return "none", config_hash({"scheduler": "none"})
    runtime_fields = {"optimizer", "last_epoch", "_step_count", "_last_lr", "_get_lr_called_within_step"}
    configuration = {
        key: _stable_config_value(value)
        for key, value in vars(scheduler).items()
        if key not in runtime_fields
    }
    return _qualified_name(scheduler), config_hash({"class": _qualified_name(scheduler), "configuration": configuration})


class TrainingRegime(str, Enum):
    REPRESENTATION = "representation"
    FLOW = "flow"
    SEQUENCE = "sequence"
    CONSTRAINT = "constraint"
    OBJECTIVE = "objective"
    PREFERENCE = "preference"


class ObjectiveLabelKind(str, Enum):
    PROXY = "proxy"
    SURROGATE = "surrogate"
    VALIDATED_SURROGATE = "validated_surrogate"
    DETERMINISTIC_EVALUATOR = "deterministic_evaluator"
    EXPERIMENTAL_MEASUREMENT = "experimental_measurement"


_REGIME_MODULES = {
    TrainingRegime.REPRESENTATION: ("evoformer",),
    TrainingRegime.FLOW: (
        "evoformer", "pair_connector", "flow_model", "evol_cross_attn",
        "constraint_encoder", "substrate_conditioner",
    ),
    TrainingRegime.SEQUENCE: (
        "evoformer", "node_connector", "base_mpnn", "multi_scale_designer",
        "_seq_to_repr", "sequence_policy",
    ),
    TrainingRegime.CONSTRAINT: (
        "evoformer", "pair_connector", "flow_model", "evol_cross_attn",
        "constraint_encoder", "substrate_conditioner",
    ),
    TrainingRegime.OBJECTIVE: ("objective_feature_encoder", "pareto_head"),
    TrainingRegime.PREFERENCE: ("sequence_policy",),
}


@dataclass
class CanonicalTrainingBatch:
    msa_tokens: torch.Tensor
    pair_features: torch.Tensor
    source_R: Optional[torch.Tensor] = None
    source_t: Optional[torch.Tensor] = None
    target_R: Optional[torch.Tensor] = None
    target_t: Optional[torch.Tensor] = None
    target_sequence: Optional[torch.Tensor] = None
    msa_padding_mask: Optional[torch.Tensor] = None
    sequence_padding_mask: Optional[torch.Tensor] = None
    constraints: Any = None
    structure_source: Optional[str] = None
    substrate_id: Optional[torch.Tensor] = None
    substrate_coords: Optional[torch.Tensor] = None
    substrate_types: Optional[torch.Tensor] = None
    objective_labels: Optional[Mapping[str, torch.Tensor]] = None
    objective_label_sources: Optional[Mapping[str, str]] = None
    objective_label_kinds: Optional[Mapping[str, ObjectiveLabelKind | str]] = None
    objective_features: Optional[Mapping[ObjectiveFeatureSource | str, torch.Tensor]] = None
    preference_batch: Optional[DPOBatch] = None
    residue_index: Optional[torch.Tensor] = None
    residue_mask: Optional[torch.Tensor] = None

    def to(self, device: torch.device | str) -> "CanonicalTrainingBatch":
        values = {}
        for item in fields(self):
            value = getattr(self, item.name)
            if torch.is_tensor(value):
                value = value.to(device)
            elif item.name == "objective_labels" and value is not None:
                value = {
                    name: label.to(device) if torch.is_tensor(label) else label
                    for name, label in value.items()
                }
            elif item.name == "objective_features" and value is not None:
                value = {name: feature.to(device) for name, feature in value.items()}
            elif item.name == "constraints" and value is not None:
                updates = {
                    field.name: getattr(value, field.name).to(device)
                    if torch.is_tensor(getattr(value, field.name))
                    else getattr(value, field.name)
                    for field in fields(value)
                }
                value = type(value)(**updates)
            elif item.name == "preference_batch" and value is not None:
                updates = {
                    field.name: getattr(value, field.name).to(device)
                    if torch.is_tensor(getattr(value, field.name))
                    else getattr(value, field.name)
                    for field in fields(value)
                }
                value = type(value)(**updates)
            values[item.name] = value
        return type(self)(**values)


def configure_trainable_components(model, regime: TrainingRegime | str) -> tuple[str, ...]:
    """Freeze everything outside one explicit stage; random modules remain trainable in-stage."""
    regime = TrainingRegime(regime)
    selected = _REGIME_MODULES[regime]
    for parameter in model.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    for name in selected:
        module = getattr(model, name)
        for parameter in module.parameters():
            parameter.requires_grad_(True)
    for name in model._component_names:
        module = getattr(model, name)
        model.component_status[name]["frozen"] = not any(
            parameter.requires_grad for parameter in module.parameters()
        )
    return selected


def gradient_flow_report(model) -> dict[str, Any]:
    """Report every parameter's trainability and observed gradient state."""
    parameters = []
    components: dict[str, dict[str, Any]] = {}
    for name, parameter in model.named_parameters():
        gradient = parameter.grad
        finite = bool(torch.isfinite(gradient).all()) if gradient is not None else None
        norm = float(gradient.detach().norm().cpu()) if gradient is not None else None
        parameters.append({
            "name": name,
            "requires_grad": bool(parameter.requires_grad),
            "gradient_present": gradient is not None,
            "gradient_norm": norm,
            "gradient_finite": finite,
        })
        component = name.split(".", 1)[0]
        row = components.setdefault(component, {
            "trainable_parameters": 0,
            "gradient_parameters": 0,
            "gradient_norm_squared": 0.0,
            "all_gradients_finite": True,
        })
        if parameter.requires_grad:
            row["trainable_parameters"] += parameter.numel()
        if gradient is not None:
            row["gradient_parameters"] += parameter.numel()
            row["gradient_norm_squared"] += norm * norm
            row["all_gradients_finite"] &= bool(finite)
    for row in components.values():
        row["gradient_norm"] = row.pop("gradient_norm_squared") ** 0.5
    return {"parameters": parameters, "components": components}


class CanonicalTrainer:
    """Own one regime's trainable modules, loss, optimizer, scheduler, and resume state."""

    def __init__(
        self,
        model,
        regime: TrainingRegime | str,
        *,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        dataset_manifest: str | Mapping[str, Any] | None = None,
        dataset_path: str | Path | None = None,
        dataset_hash: str | None = None,
        dataset_version: str | None = None,
        preprocessing_hash: str | None = None,
        preprocessing_config: Mapping[str, Any] | None = None,
        random_seed: int | None = None,
        min_validation_examples: int = 30,
        generator: Optional[torch.Generator] = None,
    ):
        self.model = model
        self.regime = TrainingRegime(regime)
        if isinstance(dataset_manifest, Mapping):
            manifest_data = dict(dataset_manifest)
            self.dataset_manifest = json.dumps(
                manifest_data, sort_keys=True, separators=(",", ":"), default=str
            )
            dataset_version = dataset_version or manifest_data.get("dataset_version")
        else:
            self.dataset_manifest = str(dataset_manifest) if dataset_manifest else None
        if dataset_path is not None:
            dataset_bytes = Path(dataset_path).expanduser().resolve().read_bytes()
            file_hash = hashlib.sha256(dataset_bytes).hexdigest()
            if dataset_hash is not None and dataset_hash != file_hash:
                raise ValueError("dataset_hash does not match the supplied dataset_path")
            dataset_hash = file_hash
        self.dataset_hash = dataset_hash
        self.dataset_version = dataset_version
        if preprocessing_hash is not None and preprocessing_config is not None:
            if preprocessing_hash != config_hash(preprocessing_config):
                raise ValueError("preprocessing_hash does not match preprocessing_config")
        self.preprocessing_hash = preprocessing_hash or (
            config_hash(preprocessing_config) if preprocessing_config is not None else None
        )
        self.random_seed = random_seed
        if self.random_seed is None and generator is not None:
            self.random_seed = int(generator.initial_seed())
        if self.random_seed is not None and (not isinstance(self.random_seed, int) or self.random_seed < 0):
            raise ValueError("random_seed must be a non-negative integer or None")
        self.generator = generator
        if not isinstance(min_validation_examples, int) or min_validation_examples < 1:
            raise ValueError("min_validation_examples must be a positive integer")
        self.min_validation_examples = min_validation_examples
        self.epoch = 0
        self.global_step = 0
        self._last_objective_sources: dict[str, str] = {}
        self._last_objective_kinds: dict[str, str] = {}
        self._last_objective_names: tuple[str, ...] = ()
        if self.regime == TrainingRegime.PREFERENCE:
            policy_state = model.component_status["sequence_policy"]
            validation = policy_state.get("validation")
            continuing_preference = (
                model._reference_policy is not None
                and policy_state.get("preference_base_validation") is not None
            )
            if not continuing_preference and (
                policy_state["training_status"] != "validated" or not validation
            ):
                raise RuntimeError(
                    "Preference training requires a supervised, validated sequence policy"
                )
            if model._reference_policy is None:
                model.init_dpo_reference()
            selected = configure_trainable_components(model, self.regime)
        else:
            selected = configure_trainable_components(model, self.regime)
        parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        if not parameters:
            raise ValueError(f"training regime {self.regime.value} has no trainable parameters")
        if optimizer is None:
            optimizer = torch.optim.AdamW(
                parameters,
                lr=learning_rate,
                weight_decay=weight_decay,
            )
        else:
            optimizer_parameters = [
                parameter
                for group in optimizer.param_groups
                for parameter in group["params"]
            ]
            expected_ids = {id(parameter) for parameter in parameters}
            supplied_ids = {id(parameter) for parameter in optimizer_parameters}
            if len(optimizer_parameters) != len(supplied_ids) or supplied_ids != expected_ids:
                raise ValueError(
                    "optimizer parameters must exactly match the selected regime's trainable parameters"
                )
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.optimizer_class, self.optimizer_hash = _optimizer_fingerprint(optimizer, model)
        self.scheduler_class, self.scheduler_hash = _scheduler_fingerprint(scheduler)
        self.trainable_components = selected
        self._resume_source_manifest: Optional[dict[str, Any]] = None

    @staticmethod
    def _default_bounds(count: int, length: int, batch_size: int, device: torch.device) -> torch.Tensor:
        bounds = torch.tensor(
            [[index * length // count, (index + 1) * length // count] for index in range(count)],
            dtype=torch.long,
            device=device,
        )
        return bounds.unsqueeze(0).expand(batch_size, -1, -1)

    def _conditioning(self, batch: CanonicalTrainingBatch):
        model = self.model
        batch_size, _, length = batch.msa_tokens.shape
        single, pair = model.evoformer(
            batch.msa_tokens,
            batch.pair_features,
            msa_padding_mask=batch.msa_padding_mask,
            residue_mask=batch.residue_mask,
            residue_index=batch.residue_index,
        )
        pair_cond = model.pair_connector(pair)
        constraints = None
        if batch.constraints is not None:
            constraints = model._normalize_constraints(
                batch.constraints,
                batch_size,
                length,
                batch.msa_tokens.device,
            )
            encoded = model.constraint_encoder(length, constraints, pair_cond.device, batch_size)
            indices = torch.arange(length, device=pair_cond.device)
            pair_cond = pair_cond.clone()
            pair_cond[:, indices, indices] = pair_cond[:, indices, indices] + encoded
        if batch.substrate_id is not None:
            pair_cond = model.substrate_conditioner(
                pair_cond,
                batch.substrate_id.long(),
                substrate_coords=batch.substrate_coords,
                substrate_types=batch.substrate_types,
                residue_coords=batch.source_t,
            )
        return single, pair_cond, constraints

    def _flow_loss(self, batch: CanonicalTrainingBatch) -> torch.Tensor:
        if any(value is None for value in (batch.source_R, batch.source_t, batch.target_R, batch.target_t)):
            raise ValueError("flow training requires source and target rotations/translations")
        single, pair_cond, constraints = self._conditioning(batch)
        fixed_mask = constraints.fixed_mask if constraints is not None else None
        conditioning = lambda nodes, time: self.model.evol_cross_attn(nodes, single, time)
        return self.model.flow_model.loss(
            batch.source_R,
            batch.source_t,
            batch.target_R,
            batch.target_t,
            pair_cond,
            single,
            fixed_mask=fixed_mask,
            substrate_coords=batch.substrate_coords,
            evol_conditioning_fn=conditioning,
        )

    def _sequence_loss(self, batch: CanonicalTrainingBatch) -> torch.Tensor:
        if batch.target_sequence is None:
            raise ValueError("sequence recovery training requires target_sequence labels")
        structure_R = batch.target_R if batch.target_R is not None else batch.source_R
        structure_t = batch.target_t if batch.target_t is not None else batch.source_t
        if structure_R is None or structure_t is None:
            raise ValueError("sequence recovery training requires ground-truth backbone frames")
        single, _, constraints = self._conditioning(batch)
        batch_size, _, length = batch.msa_tokens.shape
        node_features = self.model.node_connector(single)
        coords = self.model._frames_to_coords(structure_R, structure_t)
        residue_features = self.model.base_mpnn(coords, node_features)
        edge_index, edge_features, edge_mask = get_protein_graph(structure_t, structure_R, k_neighbors=32)
        edge_features = edge_features * edge_mask.unsqueeze(-1).to(edge_features.dtype)
        if constraints is None:
            domain_bounds = self._default_bounds(self.model.n_domains, length, batch_size, single.device)
            module_bounds = self._default_bounds(self.model.n_modules, length, batch_size, single.device)
            face = torch.zeros(batch_size, dtype=torch.long, device=single.device)
        else:
            domain_bounds = constraints.domain_boundaries
            module_bounds = constraints.module_boundaries
            face = constraints.icosahedral_face
        logits = self.model.multi_scale_designer(
            residue_features,
            node_features,
            edge_features,
            edge_index,
            domain_bounds,
            module_bounds,
            face,
            edge_mask=edge_mask,
        )
        context = self.model._seq_to_repr(logits)
        return self.model.sequence_policy_loss(
            context,
            batch.target_sequence,
            padding_mask=batch.sequence_padding_mask,
        )

    def _objective_inputs(self, batch: CanonicalTrainingBatch):
        if batch.target_sequence is None:
            raise ValueError("objective training requires target sequences")
        labels = {
            name: value
            for name, value in (batch.objective_labels or {}).items()
            if value is not None
        }
        if "assembly_compatibility" in labels:
            raise ValueError(
                "assembly_compatibility is a deterministic inference-time evaluator, not a supervised head"
            )
        if "structural_stability" in labels and batch.structure_source not in {
            "reference", "generated", "synthetic"
        }:
            raise ValueError(
                "structural_stability labels require structure_source: reference, generated, or synthetic"
            )
        unknown = set(labels) - set(self.model.objective_status)
        if unknown:
            raise ValueError(f"unknown objective labels: {sorted(unknown)}")
        sources = dict(batch.objective_label_sources or {})
        kinds = {
            name: ObjectiveLabelKind(kind).value
            for name, kind in (batch.objective_label_kinds or {}).items()
        }
        feature_inputs: dict[str, torch.Tensor] = {
            ObjectiveFeatureSource.SEQUENCE.value: F.one_hot(
                batch.target_sequence.long(), num_classes=20
            ).to(batch.pair_features.dtype)
        }
        if batch.msa_tokens is not None:
            with torch.no_grad():
                evolutionary, _ = self.model.evoformer(
                    batch.msa_tokens,
                    batch.pair_features,
                    msa_padding_mask=batch.msa_padding_mask,
                    residue_mask=batch.residue_mask,
                    residue_index=batch.residue_index,
                )
            feature_inputs[ObjectiveFeatureSource.EVOLUTIONARY.value] = evolutionary.detach()

        structure_R = batch.target_R if batch.target_R is not None else batch.source_R
        structure_t = batch.target_t if batch.target_t is not None else batch.source_t
        if structure_R is not None and structure_t is not None:
            face = getattr(batch.constraints, "icosahedral_face", None)
            if face is None:
                face_features = batch.pair_features.new_zeros((batch.target_sequence.shape[0], 20))
            else:
                if torch.any((face < 0) | (face >= 20)):
                    raise ValueError("icosahedral_face values must be in [0,19]")
                face_features = F.one_hot(face.long(), num_classes=20).to(batch.pair_features.dtype)
            face_features = face_features[:, None, :].expand(-1, structure_R.shape[1], -1)
            feature_inputs[ObjectiveFeatureSource.STRUCTURAL.value] = torch.cat(
                (
                    self.model.objective_feature_encoder.structural_invariants(structure_R, structure_t),
                    face_features,
                ),
                dim=-1,
            ).detach()
        if batch.substrate_id is not None:
            if torch.any((batch.substrate_id < 0) | (batch.substrate_id >= 20)):
                raise ValueError("substrate_id values must be in [0,19]")
            feature_inputs[ObjectiveFeatureSource.SUBSTRATE.value] = F.one_hot(
                batch.substrate_id.long(), num_classes=20
            ).to(batch.pair_features.dtype)
        for source, value in (batch.objective_features or {}).items():
            name = ObjectiveFeatureSource(source).value
            if name == ObjectiveFeatureSource.SEQUENCE.value:
                raise ValueError("sequence objective features are derived from target_sequence")
            feature_inputs[name] = value

        if not labels:
            raise ValueError("objective training requires labels or a usable deterministic evaluator target")
        missing_sources = {
            name for name in labels
            if not isinstance(sources.get(name), str) or not sources[name].strip()
        }
        if missing_sources:
            raise ValueError(f"objective label provenance is required for: {sorted(missing_sources)}")
        missing_kinds = set(labels) - set(kinds)
        if missing_kinds:
            raise ValueError(f"objective label kinds are required for: {sorted(missing_kinds)}")
        for name in labels:
            if kinds[name] not in OBJECTIVE_SCHEMA[name]["label_kinds"]:
                raise ValueError(f"{name} does not accept label kind {kinds[name]!r}")
        supervised_names = tuple(labels)
        encoded_features = self.model.objective_feature_encoder(feature_inputs)
        label_keys = {
            "evolutionary_plausibility": "evol",
            "structural_stability": "stab",
            "expression_efficiency": "expr",
            "substrate_selectivity": "sel",
            "assembly_compatibility": "asm",
        }
        labels = {
            short_name: labels.get(full_name)
            for full_name, short_name in label_keys.items()
        }
        objectives = self.model.pareto_head(encoded_features, objective_names=supervised_names)
        self._last_objective_sources = {name: sources[name] for name in supervised_names}
        self._last_objective_kinds = {name: kinds[name] for name in supervised_names}
        self._last_objective_names = supervised_names
        self._last_structure_source = batch.structure_source
        return objectives, labels, encoded_features

    def _objective_loss(self, batch: CanonicalTrainingBatch) -> torch.Tensor:
        objectives, labels, _ = self._objective_inputs(batch)
        loss, _ = self.model.pareto_head.pcgrad_loss(objectives, labels)
        return loss

    def _objective_validation_loss(self, batch: CanonicalTrainingBatch) -> torch.Tensor:
        objectives, labels, _ = self._objective_inputs(batch)
        losses = self.model.pareto_head._task_losses(objectives, labels)
        if not losses:
            raise ValueError("objective validation has no supported labels")
        return torch.stack(list(losses.values())).mean()

    def _loss(self, batch: CanonicalTrainingBatch) -> torch.Tensor:
        if self.regime == TrainingRegime.REPRESENTATION:
            loss, _ = self.model.evoformer.representation_loss(
                batch.msa_tokens,
                batch.pair_features,
                msa_padding_mask=batch.msa_padding_mask,
                residue_mask=batch.residue_mask,
                generator=self.generator,
                residue_index=batch.residue_index,
            )
            return loss
        if self.regime in (TrainingRegime.FLOW, TrainingRegime.CONSTRAINT):
            if self.regime == TrainingRegime.CONSTRAINT and batch.constraints is None:
                raise ValueError("constraint regime requires NRPS constraints")
            return self._flow_loss(batch)
        if self.regime == TrainingRegime.SEQUENCE:
            return self._sequence_loss(batch)
        if self.regime == TrainingRegime.OBJECTIVE:
            return self._objective_loss(batch)
        raise ValueError("preference regime requires train_preference_step")

    def _required_gradient_components(self, batch: CanonicalTrainingBatch) -> tuple[str, ...]:
        if self.regime == TrainingRegime.REPRESENTATION:
            return ("evoformer",)
        if self.regime in (TrainingRegime.FLOW, TrainingRegime.CONSTRAINT):
            required = ["evoformer", "pair_connector", "flow_model", "evol_cross_attn"]
            if batch.constraints is not None:
                required.append("constraint_encoder")
            if batch.substrate_id is not None:
                required.append("substrate_conditioner")
            return tuple(required)
        if self.regime == TrainingRegime.SEQUENCE:
            return ("evoformer", "node_connector", "base_mpnn", "multi_scale_designer", "_seq_to_repr", "sequence_policy")
        if self.regime == TrainingRegime.OBJECTIVE:
            return ("objective_feature_encoder", "pareto_head")
        return ("sequence_policy",)

    def train_step(self, batch: CanonicalTrainingBatch) -> dict[str, Any]:
        if self.regime == TrainingRegime.PREFERENCE:
            return self.train_preference_step(batch.preference_batch)
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        loss = self._loss(batch)
        if loss.ndim != 0 or not torch.isfinite(loss):
            raise FloatingPointError("canonical regime loss must be a finite scalar")
        loss.backward()
        audit = gradient_flow_report(self.model)
        component_rows = audit["components"]
        missing = [
            name for name in self._required_gradient_components(batch)
            if component_rows.get(name, {}).get("gradient_parameters", 0) == 0
        ]
        nonfinite = [
            name for name in self._required_gradient_components(batch)
            if not component_rows.get(name, {}).get("all_gradients_finite", True)
        ]
        if missing or nonfinite:
            raise RuntimeError(f"gradient contract failed; missing={missing}, nonfinite={nonfinite}")
        self.optimizer.step()
        for name in self._required_gradient_components(batch):
            state = self.model.component_status[name]
            state["training_status"] = "in_progress"
            state["training_steps"] += 1
            if self.dataset_manifest:
                state["training_dataset_manifest"] = self.dataset_manifest
        if self.regime == TrainingRegime.OBJECTIVE:
            for name in self._last_objective_names:
                state = self.model.objective_status[name]
                state["training_status"] = "in_progress"
                state["training_steps"] += 1
                if self.dataset_manifest:
                    state["training_dataset_manifest"] = self.dataset_manifest
                source = self._last_objective_sources[name]
                if source not in state["label_sources"]:
                    state["label_sources"].append(source)
                kind = self._last_objective_kinds[name]
                if kind not in state["label_kinds"]:
                    state["label_kinds"].append(kind)
                if name == "structural_stability" and self._last_structure_source is not None:
                    if self._last_structure_source not in state["structure_data_sources"]:
                        state["structure_data_sources"].append(self._last_structure_source)
        for parameter in self.model.parameters():
            if parameter.requires_grad and not torch.isfinite(parameter).all():
                raise FloatingPointError("optimizer produced non-finite trainable parameters")
        if self.scheduler is not None:
            self.scheduler.step()
        self.global_step += 1
        return {"loss": float(loss.detach()), "gradient_flow": audit, "global_step": self.global_step}

    def validate(
        self,
        batches: Iterable[CanonicalTrainingBatch] | CanonicalTrainingBatch,
        *,
        validation_manifest: str | Mapping[str, Any],
        min_validation_examples: int | None = None,
    ) -> dict[str, Any]:
        if self.regime == TrainingRegime.PREFERENCE:
            raise ValueError("preference-only validation cannot validate the base policy")
        if not self.dataset_manifest:
            raise ValueError("training requires a dataset manifest before held-out validation")
        if isinstance(validation_manifest, Mapping):
            validation_identity = json.dumps(
                dict(validation_manifest), sort_keys=True, separators=(",", ":"), default=str
            )
        else:
            validation_identity = str(validation_manifest)
        if not validation_identity or validation_identity == self.dataset_manifest:
            raise ValueError("validation requires a distinct held-out dataset manifest")
        minimum = self.min_validation_examples if min_validation_examples is None else min_validation_examples
        if not isinstance(minimum, int) or minimum < 1:
            raise ValueError("min_validation_examples must be a positive integer")
        if isinstance(batches, CanonicalTrainingBatch):
            batches = (batches,)
        was_training = self.model.training
        self.model.eval()
        total_loss = 0.0
        example_count = 0
        batch_count = 0
        objective_loss_sums: dict[str, float] = {}
        objective_example_counts: dict[str, int] = {}
        validation_components: set[str] = set()
        validated_objectives: set[str] = set()
        structure_source_counts: dict[str, int] = {}
        try:
            with torch.no_grad():
                for batch in batches:
                    if not isinstance(batch, CanonicalTrainingBatch):
                        raise TypeError("validation dataset must yield CanonicalTrainingBatch values")
                    batch_examples = int(batch.msa_tokens.shape[0])
                    if batch_examples < 1:
                        raise ValueError("validation batches must contain at least one example")
                    if self.regime == TrainingRegime.OBJECTIVE:
                        objectives, labels, _ = self._objective_inputs(batch)
                        task_losses = self.model.pareto_head._task_losses(objectives, labels)
                        if not task_losses:
                            raise ValueError("objective validation has no supported labels")
                        batch_loss = torch.stack(list(task_losses.values())).mean()
                        for name, task_loss in task_losses.items():
                            objective_loss_sums[name] = objective_loss_sums.get(name, 0.0) + (
                                float(task_loss) * batch_examples
                            )
                            objective_example_counts[name] = (
                                objective_example_counts.get(name, 0) + batch_examples
                            )
                        validated_objectives.update(self._last_objective_names)
                        if "structural_stability" in self._last_objective_names:
                            source = self._last_structure_source
                            structure_source_counts[source] = (
                                structure_source_counts.get(source, 0) + batch_examples
                            )
                    else:
                        batch_loss = self._loss(batch)
                    if batch_loss.ndim != 0 or not torch.isfinite(batch_loss):
                        raise FloatingPointError("validation loss must be a finite scalar")
                    total_loss += float(batch_loss) * batch_examples
                    example_count += batch_examples
                    batch_count += 1
                    validation_components.update(self._required_gradient_components(batch))
        finally:
            self.model.train(was_training)
        if batch_count == 0:
            raise ValueError("validation dataset must contain at least one batch")
        enough_examples = example_count >= minimum
        validation_record = {
            "loss": total_loss / example_count,
            "examples": example_count,
            "batch_count": batch_count,
            "objective_losses": {
                name: value / objective_example_counts[name]
                for name, value in objective_loss_sums.items()
            },
            "calibration_statistics": (
                "not_computed_requires_heldout_predictions_and_uncertainty"
            ),
            "finite": True,
            "minimum_examples": minimum,
            "dataset_manifest": validation_identity,
            "validation_manifest": {
                "dataset_identity": validation_identity,
                "minimum_validation_examples": minimum,
                "observed_examples": example_count,
                "batch_count": batch_count,
            },
            "structure_source_counts": structure_source_counts,
            "generated_reference_distribution_shift_risk": (
                "present"
                if {"reference", "generated"}.issubset(structure_source_counts)
                else "not_established"
            ),
            "status": "validated" if enough_examples else "insufficient_validation_examples",
        }
        for name in validation_components:
            status = self.model.component_status[name]
            if status["training_steps"] > 0:
                if status.get("training_dataset_manifest") != self.dataset_manifest:
                    raise ValueError(f"component {name} was trained on a different dataset manifest")
                status["validation"] = validation_record
                if enough_examples:
                    status["training_status"] = "validated"
        if self.regime == TrainingRegime.OBJECTIVE:
            for name in validated_objectives:
                status = self.model.objective_status[name]
                if status["training_steps"] > 0:
                    if status.get("training_dataset_manifest") != self.dataset_manifest:
                        raise ValueError(f"objective {name} was trained on a different dataset manifest")
                    status["validation"] = validation_record
                    if enough_examples:
                        status["training_status"] = "validated"
        return validation_record

    def validate_sequence_policy(
        self,
        batches: Iterable[CanonicalTrainingBatch] | CanonicalTrainingBatch,
        *,
        validation_manifest: str | Mapping[str, Any],
    ) -> float:
        if self.regime != TrainingRegime.SEQUENCE:
            raise ValueError("sequence-policy validation requires the sequence regime")
        result = self.validate(batches, validation_manifest=validation_manifest)
        policy_status = self.model.component_status["sequence_policy"]
        if policy_status.get("validation") is not None:
            policy_status["validation"]["sequence_nll"] = result["loss"]
        return float(result["loss"])

    def train_preference_step(self, batch: Optional[DPOBatch]) -> dict[str, Any]:
        if self.regime != TrainingRegime.PREFERENCE:
            raise ValueError("train_preference_step requires the preference regime")
        if batch is None:
            raise ValueError("preference training requires a DPOBatch")
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        loss, metrics = self.model.dpo_trainer.loss(
            self.model.sequence_policy,
            self.model._reference_policy,
            batch,
        )
        loss.backward()
        audit = gradient_flow_report(self.model)
        policy_gradients = audit["components"].get("sequence_policy", {})
        if policy_gradients.get("gradient_parameters", 0) == 0 or not policy_gradients.get("all_gradients_finite", False):
            raise RuntimeError("preference gradient contract failed for sequence_policy")
        torch.nn.utils.clip_grad_norm_(self.model.sequence_policy.parameters(), 1.0)
        self.optimizer.step()
        status = self.model.component_status["sequence_policy"]
        if status.get("validation") is not None:
            status["preference_base_validation"] = status["validation"]
        status["training_status"] = "in_progress"
        status["training_steps"] += 1
        status["validation"] = None
        status["training_dataset_manifest"] = None
        self.global_step += 1
        if self.scheduler is not None:
            self.scheduler.step()
        return {**metrics, "gradient_flow": audit, "global_step": self.global_step}

    def fit(
        self,
        batches: Iterable[CanonicalTrainingBatch],
        epochs: int,
        *,
        evaluation_hook: Optional[Callable[["CanonicalTrainer", int], Any]] = None,
    ) -> list[dict[str, Any]]:
        if epochs < 1:
            raise ValueError("epochs must be positive")
        if iter(batches) is batches:
            batches = tuple(batches)
        history = []
        for _ in range(epochs):
            epoch_rows = [self.train_step(batch) for batch in batches]
            if not epoch_rows:
                raise ValueError("training batches must not be empty")
            self.epoch += 1
            event = {"epoch": self.epoch, "steps": epoch_rows}
            if evaluation_hook is not None:
                event["evaluation"] = evaluation_hook(self, self.epoch)
            history.append(event)
        return history

    def _manifest(self) -> CheckpointManifest:
        config = self.model.model_configuration()
        git = git_provenance(Path(__file__).resolve().parents[1])
        runtime = runtime_provenance()
        return CheckpointManifest(
            config_hash=config_hash(config),
            state_schema_hash=state_schema_hash(self.model.state_dict()),
            objective_schema_hash=objective_schema_hash(),
            dataset_manifest=self.dataset_manifest,
            dataset_version=self.dataset_version,
            dataset_hash=self.dataset_hash,
            preprocessing_hash=self.preprocessing_hash,
            git_commit=git["git_commit"],
            git_worktree_clean=git["git_worktree_clean"],
            source_tree_hash=git["source_tree_hash"],
            environment_hash=runtime["environment_hash"],
            python_version=runtime["python_version"],
            torch_version=runtime["torch_version"],
            platform=runtime["platform"],
            numpy_version=runtime["numpy_version"],
            cuda_available=runtime["cuda_available"],
            cuda_version=runtime["cuda_version"],
            cudnn_version=runtime["cudnn_version"],
            cuda_device_names=runtime["cuda_device_names"],
            deterministic_algorithms_enabled=runtime["deterministic_algorithms_enabled"],
            cudnn_deterministic=runtime["cudnn_deterministic"],
            cudnn_benchmark=runtime["cudnn_benchmark"],
            matmul_allow_tf32=runtime["matmul_allow_tf32"],
            cudnn_allow_tf32=runtime["cudnn_allow_tf32"],
            training_regime=self.regime.value,
            random_seed=self.random_seed,
            min_validation_examples=self.min_validation_examples,
            optimizer_class=self.optimizer_class,
            optimizer_hash=self.optimizer_hash,
            scheduler_class=self.scheduler_class,
            scheduler_hash=self.scheduler_hash,
        )

    def save_checkpoint(self, path: str | Path) -> None:
        manifest = self._manifest()
        if not manifest.has_complete_provenance():
            raise ValueError("canonical checkpoints require complete, verified provenance")
        payload = {
            "artifact_type": "canonical_training_checkpoint",
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict() if self.scheduler is not None else None,
            "manifest": asdict(manifest),
            "training": {
                "regime": self.regime.value,
                "epoch": self.epoch,
                "global_step": self.global_step,
                "component_status": self.model.component_status,
                "objective_calibration": self.model.objective_calibration,
                "objective_status": self.model.objective_status,
                "preference_reference_state": (
                    self.model._reference_policy.state_dict()
                    if self.model._reference_policy is not None
                    else None
                ),
                "resumed_from_manifest": self._resume_source_manifest,
                "rng_state": {
                    "python": random.getstate(),
                    "numpy": np.random.get_state(),
                    "torch": torch.get_rng_state(),
                    "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                    "generator": self.generator.get_state() if self.generator is not None else None,
                    "generator_device": str(self.generator.device) if self.generator is not None else None,
                },
            },
        }
        torch.save(payload, path)

    def resume(self, path: str | Path, *, migration_id: str | None = None) -> None:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(payload, dict) and payload.get("artifact_type") == "component_transfer_checkpoint":
            raise ValueError("component-transfer checkpoints cannot be resumed as canonical training state")
        if not isinstance(payload, dict) or not all(
            key in payload for key in ("model_state", "optimizer_state", "manifest", "training")
        ):
            raise ValueError("checkpoint is not a canonical training checkpoint")
        if payload.get("artifact_type", "canonical_training_checkpoint") != "canonical_training_checkpoint":
            raise ValueError("checkpoint artifact type is not canonical training state")
        actual = CheckpointManifest(**payload["manifest"])
        actual.validate()
        expected = self._manifest()
        training = payload["training"]
        compatibility = validate_checkpoint_compatibility(
            actual, expected, migration_id=migration_id
        )
        if compatibility in ("incompatible", "unverified-provenance"):
            raise ValueError(f"checkpoint provenance is not compatible with this trainer: {compatibility}")
        if actual.dataset_manifest != expected.dataset_manifest or actual.dataset_hash != expected.dataset_hash:
            raise ValueError("checkpoint dataset identity does not match this trainer")
        if actual.dataset_version != expected.dataset_version:
            raise ValueError("checkpoint dataset version does not match this trainer")
        if actual.preprocessing_hash != expected.preprocessing_hash:
            raise ValueError("checkpoint preprocessing is incompatible")
        if actual.config_hash != expected.config_hash:
            raise ValueError("checkpoint model configuration is incompatible")
        if actual.state_schema_hash != expected.state_schema_hash:
            raise ValueError("checkpoint model state schema is incompatible")
        if actual.training_regime != self.regime.value or training["regime"] != self.regime.value:
            raise ValueError("checkpoint training regime does not match this trainer")
        if actual.random_seed != expected.random_seed:
            raise ValueError("checkpoint random seed does not match this trainer")
        self.model.load_state_dict(payload["model_state"], strict=True)
        self.optimizer.load_state_dict(payload["optimizer_state"])
        saved_scheduler = payload.get("scheduler_state")
        if (saved_scheduler is None) != (self.scheduler is None):
            raise ValueError("checkpoint scheduler configuration does not match this trainer")
        if self.scheduler is not None:
            self.scheduler.load_state_dict(saved_scheduler)
        self.model.component_status = training["component_status"]
        self.model.objective_calibration = training["objective_calibration"]
        self.model.objective_status = training["objective_status"]
        reference_state = training.get("preference_reference_state")
        if reference_state is not None:
            reference_policy = self.model.sequence_policy.__class__(
                context_dim=self.model.d_mpnn,
                vocab_size=self.model.sequence_policy.vocab_size,
            )
            reference_policy.load_state_dict(reference_state, strict=True)
            reference_policy.eval()
            for parameter in reference_policy.parameters():
                parameter.requires_grad_(False)
            object.__setattr__(self.model, "_reference_policy", reference_policy)
        self.epoch = int(training["epoch"])
        self.global_step = int(training["global_step"])
        self._resume_source_manifest = asdict(actual)
        rng_state = training.get("rng_state")
        if rng_state is not None:
            random.setstate(rng_state["python"])
            np.random.set_state(rng_state["numpy"])
            torch.set_rng_state(rng_state["torch"])
            saved_cuda_state = rng_state.get("cuda")
            if saved_cuda_state is not None:
                if not torch.cuda.is_available() or len(saved_cuda_state) != torch.cuda.device_count():
                    raise ValueError("checkpoint CUDA RNG state does not match the current environment")
                torch.cuda.set_rng_state_all(saved_cuda_state)
            generator_state = rng_state.get("generator")
            if generator_state is not None:
                if self.generator is None:
                    self.generator = torch.Generator(device=rng_state.get("generator_device", "cpu"))
                self.generator.set_state(generator_state)

    @classmethod
    def from_checkpoint(
        cls,
        model,
        path: str | Path,
        *,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        migration_id: str | None = None,
    ) -> "CanonicalTrainer":
        """Construct a regime trainer and strictly restore a saved training state."""
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if (
            not isinstance(payload, dict)
            or payload.get("artifact_type", "canonical_training_checkpoint") != "canonical_training_checkpoint"
            or "training" not in payload
        ):
            raise ValueError("checkpoint is not a canonical training checkpoint")
        training = payload["training"]
        manifest = CheckpointManifest(**payload["manifest"])
        manifest.validate()
        regime = TrainingRegime(training["regime"])
        if regime == TrainingRegime.PREFERENCE:
            if training.get("preference_reference_state") is None:
                raise ValueError("preference checkpoint is missing its frozen reference policy")
            model.load_state_dict(payload["model_state"], strict=True)
            model.component_status = training["component_status"]
            model.objective_calibration = training["objective_calibration"]
            model.objective_status = training["objective_status"]
            reference_policy = model.sequence_policy.__class__(
                context_dim=model.d_mpnn,
                vocab_size=model.sequence_policy.vocab_size,
            )
            reference_policy.load_state_dict(training["preference_reference_state"], strict=True)
            reference_policy.eval()
            for parameter in reference_policy.parameters():
                parameter.requires_grad_(False)
            object.__setattr__(model, "_reference_policy", reference_policy)
        trainer = cls(
            model,
            regime,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            optimizer=optimizer,
            scheduler=scheduler,
            dataset_manifest=manifest.dataset_manifest,
            dataset_hash=manifest.dataset_hash,
            dataset_version=manifest.dataset_version,
            preprocessing_hash=manifest.preprocessing_hash,
            random_seed=manifest.random_seed,
            min_validation_examples=manifest.min_validation_examples or 30,
        )
        trainer.resume(path, migration_id=migration_id)
        return trainer


__all__ = [
    "CanonicalTrainingBatch",
    "CanonicalTrainer",
    "ObjectiveLabelKind",
    "TrainingRegime",
    "configure_trainable_components",
    "gradient_flow_report",
]
