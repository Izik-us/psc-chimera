"""Versioned JSON-only inference configuration and canonical identity."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass
from typing import Any, Mapping

from .errors import ConfigurationError


CONFIGURATION_SCHEMA_VERSION = 1
CANONICAL_OBJECTIVES = (
    "evolutionary_plausibility",
    "structural_stability",
    "expression_efficiency",
    "substrate_selectivity",
    "assembly_compatibility",
)


def _validate_json_value(value: Any, path: str = "configuration") -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ConfigurationError(f"{path} must not contain NaN or infinity")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json_value(item, f"{path}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ConfigurationError(f"{path} keys must be strings, got {type(key).__name__}")
            _validate_json_value(item, f"{path}.{key}")
        return
    raise ConfigurationError(
        f"{path} contains non-JSON value {type(value).__name__}; use explicit scalar/list/object data"
    )


def canonical_config_json(value: Mapping[str, Any]) -> str:
    """Serialize a JSON-compatible mapping in a stable, strict representation."""
    if not isinstance(value, Mapping):
        raise ConfigurationError("configuration must be a JSON object")
    data = dict(value)
    _validate_json_value(data)
    return json.dumps(
        data,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


@dataclass(frozen=True)
class InferenceConfig:
    """Explicit, serializable configuration for canonical inference."""

    model_id: str
    artifact_id: str
    checkpoint_id: str
    preprocessing_id: str
    schema_version: int = CONFIGURATION_SCHEMA_VERSION
    device: str = "cpu"
    dtype: str = "float32"
    sampler: str = "stochastic_euler_maruyama"
    inference_steps: int = 20
    sequence_temperature: float = 1.0
    batch_size: int = 1
    microbatch_size: int | None = None
    max_sequence_length: int = 1024
    padding_policy: str = "longest_in_batch"
    objective_set: tuple[str, ...] = CANONICAL_OBJECTIVES
    retrieval_id: str | None = None
    random_seed: int | None = 0
    deterministic: bool = True
    cpu_threads: int = 1
    worker_count: int = 0
    memory_limit_bytes: int | None = None
    timeout_seconds: int = 3600

    def __post_init__(self) -> None:
        if (
            isinstance(self.schema_version, bool)
            or not isinstance(self.schema_version, int)
            or self.schema_version != CONFIGURATION_SCHEMA_VERSION
        ):
            raise ConfigurationError(
                f"unsupported inference configuration schema_version={self.schema_version}"
            )
        for name in ("model_id", "artifact_id", "checkpoint_id", "preprocessing_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ConfigurationError(f"{name} must be a non-empty identity string")
        if not isinstance(self.device, str) or not (
            self.device == "cpu" or re.fullmatch(r"cuda(?::\d+)?", self.device)
        ):
            raise ConfigurationError("device must be 'cpu' or an explicit 'cuda[:index]' value")
        if self.dtype != "float32":
            raise ConfigurationError(
                "dtype must be 'float32'; reduced-precision canonical inference is not verified"
            )
        if self.sampler != "stochastic_euler_maruyama":
            raise ConfigurationError("unsupported inference sampler")
        if (
            isinstance(self.sequence_temperature, bool)
            or not isinstance(self.sequence_temperature, (int, float))
            or not math.isfinite(self.sequence_temperature)
            or self.sequence_temperature <= 0
        ):
            raise ConfigurationError("sequence_temperature must be a finite positive number")
        if not isinstance(self.deterministic, bool):
            raise ConfigurationError("deterministic must be a boolean")
        if self.deterministic and self.random_seed is None:
            raise ConfigurationError("deterministic inference requires an explicit random_seed")
        if (
            isinstance(self.inference_steps, bool)
            or not isinstance(self.inference_steps, int)
            or self.inference_steps < 2
        ):
            raise ConfigurationError("inference_steps must be at least 2")
        for name in (
            "batch_size",
            "max_sequence_length",
            "cpu_threads",
            "timeout_seconds",
        ):
            if (
                isinstance(getattr(self, name), bool)
                or not isinstance(getattr(self, name), int)
                or getattr(self, name) < 1
            ):
                raise ConfigurationError(f"{name} must be a positive integer")
        if self.microbatch_size is not None and (
            isinstance(self.microbatch_size, bool)
            or not isinstance(self.microbatch_size, int)
            or self.microbatch_size < 1
            or self.microbatch_size > self.batch_size
        ):
            raise ConfigurationError("microbatch_size must be between 1 and batch_size")
        if (
            isinstance(self.worker_count, bool)
            or not isinstance(self.worker_count, int)
            or self.worker_count < 0
        ):
            raise ConfigurationError("worker_count must be a non-negative integer")
        if self.random_seed is not None and (
            isinstance(self.random_seed, bool)
            or not isinstance(self.random_seed, int)
            or self.random_seed < 0
        ):
            raise ConfigurationError("random_seed must be a non-negative integer or None")
        if self.memory_limit_bytes is not None and (
            isinstance(self.memory_limit_bytes, bool)
            or not isinstance(self.memory_limit_bytes, int)
            or self.memory_limit_bytes < 1
        ):
            raise ConfigurationError("memory_limit_bytes must be a positive integer or None")
        if self.padding_policy != "longest_in_batch":
            raise ConfigurationError(
                "padding_policy must be 'longest_in_batch'; other policies are unsupported"
            )
        if not isinstance(self.objective_set, (tuple, list)):
            raise ConfigurationError("objective_set must contain objective names")
        if any(not isinstance(name, str) or not name.strip() for name in self.objective_set):
            raise ConfigurationError("objective_set entries must be non-empty strings")
        if not self.objective_set or len(set(self.objective_set)) != len(self.objective_set):
            raise ConfigurationError("objective_set must contain unique objective names")
        unknown = sorted(set(self.objective_set) - set(CANONICAL_OBJECTIVES))
        if unknown:
            raise ConfigurationError(
                f"objective_set contains unsupported objective names: {', '.join(unknown)}"
            )
        if self.retrieval_id is not None and (
            not isinstance(self.retrieval_id, str) or not self.retrieval_id.strip()
        ):
            raise ConfigurationError("retrieval_id must be a non-empty identity string or None")
        object.__setattr__(self, "objective_set", tuple(self.objective_set))

    def to_dict(self) -> dict[str, Any]:
        """Return the canonical JSON data shape, using arrays for tuple fields."""
        result = asdict(self)
        result["objective_set"] = list(self.objective_set)
        return result

    def to_json(self) -> str:
        return canonical_config_json(self.to_dict())

    @property
    def config_id(self) -> str:
        return hashlib.sha256(self.to_json().encode("utf-8")).hexdigest()

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "InferenceConfig":
        if not isinstance(data, Mapping):
            raise ConfigurationError("inference configuration must be an object")
        values = dict(data)
        if "objective_set" in values:
            objectives = values["objective_set"]
            if not isinstance(objectives, list):
                raise ConfigurationError("objective_set must be a JSON array")
            values["objective_set"] = tuple(objectives)
        try:
            return cls(**values)
        except TypeError as exc:
            raise ConfigurationError(f"invalid inference configuration fields: {exc}") from exc
