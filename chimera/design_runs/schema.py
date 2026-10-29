"""Versioned task contract for bounded CHIMERA design runs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from ..configuration import InferenceConfig
from ..errors import ConfigurationError

DESIGN_TASK_SCHEMA_VERSION = 1


def _nonempty_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError(f"{field} must be a non-empty string")
    return value.strip()


@dataclass(frozen=True)
class DesignTask:
    """A declarative, JSON-serializable request for one bounded design job."""

    task_id: str
    objective: str
    target_context: Mapping[str, Any]
    input_bundle: str
    checkpoint_path: str
    inference_config: InferenceConfig
    candidate_count: int
    input_shape: Mapping[str, int]
    schema_version: int = DESIGN_TASK_SCHEMA_VERSION
    hard_constraints: tuple[Mapping[str, Any], ...] = ()
    soft_objectives: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if isinstance(self.schema_version, bool) or self.schema_version != DESIGN_TASK_SCHEMA_VERSION:
            raise ConfigurationError(
                f"unsupported design-task schema_version={self.schema_version}"
            )
        _nonempty_string(self.task_id, "task_id")
        _nonempty_string(self.objective, "objective")
        _nonempty_string(self.input_bundle, "input_bundle")
        _nonempty_string(self.checkpoint_path, "checkpoint_path")
        if not isinstance(self.target_context, Mapping):
            raise ConfigurationError("target_context must be a JSON object")
        if not isinstance(self.inference_config, InferenceConfig):
            raise ConfigurationError("inference_config must be a validated InferenceConfig")
        if (
            isinstance(self.candidate_count, bool)
            or not isinstance(self.candidate_count, int)
            or not 1 <= self.candidate_count <= 100_000
        ):
            raise ConfigurationError("candidate_count must be an integer in [1, 100000]")
        if not isinstance(self.input_shape, Mapping):
            raise ConfigurationError("input_shape must be a JSON object")
        for name in ("sequence_length", "msa_depth", "pair_channels"):
            value = self.input_shape.get(name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ConfigurationError(f"input_shape.{name} must be a positive integer")
        if self.input_shape["sequence_length"] < 2:
            raise ConfigurationError("input_shape.sequence_length must be at least 2")
        if self.input_shape["sequence_length"] > self.inference_config.max_sequence_length:
            raise ConfigurationError(
                "input_shape.sequence_length exceeds inference_config.max_sequence_length"
            )
        if not isinstance(self.hard_constraints, (tuple, list)) or any(
            not isinstance(item, Mapping) for item in self.hard_constraints
        ):
            raise ConfigurationError("hard_constraints must be an array of JSON objects")
        if not isinstance(self.soft_objectives, (tuple, list)) or any(
            not isinstance(item, str) or not item.strip() for item in self.soft_objectives
        ):
            raise ConfigurationError("soft_objectives must be an array of non-empty strings")
        object.__setattr__(self, "hard_constraints", tuple(dict(x) for x in self.hard_constraints))
        object.__setattr__(self, "soft_objectives", tuple(self.soft_objectives))
        object.__setattr__(self, "target_context", dict(self.target_context))
        object.__setattr__(self, "input_shape", dict(self.input_shape))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "DesignTask":
        if not isinstance(value, Mapping):
            raise ConfigurationError("design task must be a JSON object")
        data = dict(value)
        try:
            config_data = data.pop("inference_config")
        except KeyError as exc:
            raise ConfigurationError("design task is missing inference_config") from exc
        data["inference_config"] = InferenceConfig.from_dict(config_data)
        for name in ("hard_constraints", "soft_objectives"):
            if name in data:
                if not isinstance(data[name], list):
                    raise ConfigurationError(f"{name} must be a JSON array")
                data[name] = tuple(data[name])
        try:
            return cls(**data)
        except TypeError as exc:
            raise ConfigurationError(f"invalid design-task fields: {exc}") from exc

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "task_id": self.task_id,
            "objective": self.objective,
            "target_context": dict(self.target_context),
            "input_bundle": self.input_bundle,
            "checkpoint_path": self.checkpoint_path,
            "inference_config": self.inference_config.to_dict(),
            "candidate_count": self.candidate_count,
            "input_shape": dict(self.input_shape),
            "hard_constraints": [dict(x) for x in self.hard_constraints],
            "soft_objectives": list(self.soft_objectives),
        }
