"""Canonical staged training, gradient auditing, and checkpoint ownership."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional

import torch
import torch.nn.functional as F

from .checkpoint import CheckpointManifest, config_hash, state_schema_hash
from .dpo import DPOBatch
from .proteinmpnn import get_protein_graph


class TrainingRegime(str, Enum):
    REPRESENTATION = "representation"
    FLOW = "flow"
    SEQUENCE = "sequence"
    CONSTRAINT = "constraint"
    OBJECTIVE = "objective"
    PREFERENCE = "preference"


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
    TrainingRegime.OBJECTIVE: ("_seq_to_repr", "pareto_head"),
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
    substrate_id: Optional[torch.Tensor] = None
    substrate_coords: Optional[torch.Tensor] = None
    substrate_types: Optional[torch.Tensor] = None
    objective_labels: Optional[Mapping[str, torch.Tensor]] = None
    objective_label_sources: Optional[Mapping[str, str]] = None
    preference_batch: Optional[DPOBatch] = None

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
    if regime == TrainingRegime.REPRESENTATION:
        for module in (model.evoformer.pair_init, model.evoformer.frozen_feature_expander):
            for parameter in module.parameters():
                parameter.requires_grad_(False)
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
        dataset_manifest: str = "",
        generator: Optional[torch.Generator] = None,
    ):
        self.model = model
        self.regime = TrainingRegime(regime)
        self.dataset_manifest = dataset_manifest
        self.generator = generator
        self.epoch = 0
        self.global_step = 0
        self._last_objective_sources: dict[str, str] = {}
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
        self.trainable_components = selected

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

    def _objective_loss(self, batch: CanonicalTrainingBatch) -> torch.Tensor:
        if batch.target_sequence is None or not batch.objective_labels:
            raise ValueError("objective training requires target sequences and explicit objective labels")
        unknown = set(batch.objective_labels) - set(self.model.objective_status)
        if unknown:
            raise ValueError(f"unknown objective labels: {sorted(unknown)}")
        missing_sources = set(batch.objective_labels) - set(batch.objective_label_sources or {})
        if missing_sources:
            raise ValueError(f"objective label provenance is required for: {sorted(missing_sources)}")
        one_hot = F.one_hot(batch.target_sequence.long(), num_classes=20).to(batch.pair_features.dtype)
        context = self.model._seq_to_repr(one_hot)
        objectives = self.model.pareto_head(context)
        label_keys = {
            "evolutionary_plausibility": "evol",
            "structural_stability": "stab",
            "expression_efficiency": "expr",
            "substrate_selectivity": "sel",
            "assembly_compatibility": "asm",
        }
        labels = {
            short_name: batch.objective_labels.get(full_name)
            for full_name, short_name in label_keys.items()
        }
        self._last_objective_sources = dict(batch.objective_label_sources or {})
        self._last_objective_names = tuple(batch.objective_labels)
        loss, _ = self.model.pareto_head.pcgrad_loss(objectives, labels)
        return loss

    def _objective_validation_loss(self, batch: CanonicalTrainingBatch) -> torch.Tensor:
        if batch.target_sequence is None or not batch.objective_labels:
            raise ValueError("objective validation requires target sequences and explicit labels")
        one_hot = F.one_hot(batch.target_sequence.long(), num_classes=20).to(batch.pair_features.dtype)
        objectives = self.model.pareto_head(self.model._seq_to_repr(one_hot))
        label_keys = {
            "evolutionary_plausibility": "evol",
            "structural_stability": "stab",
            "expression_efficiency": "expr",
            "substrate_selectivity": "sel",
            "assembly_compatibility": "asm",
        }
        labels = {
            short_name: batch.objective_labels.get(full_name)
            for full_name, short_name in label_keys.items()
        }
        losses = self.model.pareto_head._task_losses(objectives, labels)
        if not losses:
            raise ValueError("objective validation has no supported labels")
        return torch.stack(list(losses.values())).mean()

    def _loss(self, batch: CanonicalTrainingBatch) -> torch.Tensor:
        if self.regime == TrainingRegime.REPRESENTATION:
            return self.model.evoformer.masked_reconstruction_loss(
                batch.msa_tokens,
                msa_padding_mask=batch.msa_padding_mask,
                generator=self.generator,
            )
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
            return ("_seq_to_repr", "pareto_head")
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
        for parameter in self.model.parameters():
            if parameter.requires_grad and not torch.isfinite(parameter).all():
                raise FloatingPointError("optimizer produced non-finite trainable parameters")
        if self.scheduler is not None:
            self.scheduler.step()
        self.global_step += 1
        return {"loss": float(loss.detach()), "gradient_flow": audit, "global_step": self.global_step}

    def validate(
        self,
        batch: CanonicalTrainingBatch,
        *,
        validation_manifest: str,
    ) -> float:
        if self.regime == TrainingRegime.PREFERENCE:
            raise ValueError("preference-only validation cannot validate the base policy")
        if not self.dataset_manifest:
            raise ValueError("training requires a dataset manifest before held-out validation")
        if not validation_manifest or validation_manifest == self.dataset_manifest:
            raise ValueError("validation requires a distinct held-out dataset manifest")
        was_training = self.model.training
        self.model.eval()
        try:
            with torch.no_grad():
                    loss = (
                        self._objective_validation_loss(batch)
                        if self.regime == TrainingRegime.OBJECTIVE
                        else self._loss(batch)
                    )
        finally:
            self.model.train(was_training)
        if not torch.isfinite(loss):
            raise FloatingPointError("validation loss is non-finite")
        validation_record = {
            "loss": float(loss),
            "examples": int(batch.msa_tokens.shape[0]),
            "dataset_manifest": validation_manifest,
        }
        for name in self._required_gradient_components(batch):
            status = self.model.component_status[name]
            if status["training_steps"] > 0:
                if status.get("training_dataset_manifest") != self.dataset_manifest:
                    raise ValueError(f"component {name} was trained on a different dataset manifest")
                status["training_status"] = "validated"
                status["validation"] = validation_record
        if self.regime == TrainingRegime.OBJECTIVE:
            for name in self._last_objective_names:
                status = self.model.objective_status[name]
                if status["training_steps"] > 0:
                    if status.get("training_dataset_manifest") != self.dataset_manifest:
                        raise ValueError(f"objective {name} was trained on a different dataset manifest")
                    status["training_status"] = "validated"
                    status["validation"] = validation_record
        return float(loss)

    def validate_sequence_policy(
        self,
        batch: CanonicalTrainingBatch,
        *,
        validation_manifest: str,
    ) -> float:
        if self.regime != TrainingRegime.SEQUENCE:
            raise ValueError("sequence-policy validation requires the sequence regime")
        result = self.validate(batch, validation_manifest=validation_manifest)
        policy_status = self.model.component_status["sequence_policy"]
        policy_status["validation"]["sequence_nll"] = result
        return result

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
        config = {
            "d_evo_single": self.model.d_evo_single,
            "d_evo_pair": self.model.d_evo_pair,
            "d_se3": self.model.d_se3,
            "d_pair_out": self.model.d_pair_out,
            "d_mpnn": self.model.d_mpnn,
            "n_flow_blocks": self.model.n_flow_blocks,
            "n_flow_steps": self.model.n_flow_steps,
            "n_retrieve": self.model.structural_retriever.n_retrieve,
            "n_mpnn_seqs": self.model.n_mpnn_seqs,
            "n_mc_dropout": self.model.uncertainty_estimator.n_samples,
            "n_domains": self.model.n_domains,
            "n_modules": self.model.n_modules,
        }
        return CheckpointManifest(
            config_hash=config_hash(config),
            state_schema_hash=state_schema_hash(self.model.state_dict()),
            dataset_manifest=self.dataset_manifest,
        )

    def save_checkpoint(self, path: str | Path) -> None:
        manifest = self._manifest()
        payload = {
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
            },
        }
        torch.save(payload, path)

    def resume(self, path: str | Path) -> None:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict) or not all(
            key in payload for key in ("model_state", "optimizer_state", "manifest", "training")
        ):
            raise ValueError("checkpoint is not a canonical training checkpoint")
        actual = CheckpointManifest(**payload["manifest"])
        actual.validate()
        expected = self._manifest()
        if actual.dataset_manifest != expected.dataset_manifest:
            raise ValueError("checkpoint dataset manifest does not match this trainer")
        if actual.config_hash != expected.config_hash:
            raise ValueError("checkpoint model configuration is incompatible")
        if actual.state_schema_hash != expected.state_schema_hash:
            raise ValueError("checkpoint model state schema is incompatible")
        training = payload["training"]
        if training["regime"] != self.regime.value:
            raise ValueError("checkpoint training regime does not match this trainer")
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

    @classmethod
    def from_checkpoint(
        cls,
        model,
        path: str | Path,
        *,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
        scheduler: Optional[Any] = None,
    ) -> "CanonicalTrainer":
        """Construct a regime trainer and strictly restore a saved training state."""
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict) or "training" not in payload:
            raise ValueError("checkpoint is not a canonical training checkpoint")
        training = payload["training"]
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
            scheduler=scheduler,
            dataset_manifest=payload["manifest"].get("dataset_manifest", ""),
        )
        trainer.resume(path)
        return trainer


__all__ = [
    "CanonicalTrainingBatch",
    "CanonicalTrainer",
    "TrainingRegime",
    "configure_trainable_components",
    "gradient_flow_report",
]
