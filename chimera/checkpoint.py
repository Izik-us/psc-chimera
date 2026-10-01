"""Versioned checkpoint manifests for merge-safe CHIMERA experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping

import hashlib
import json
import platform
import sys
from pathlib import Path


def _runtime_snapshot() -> dict[str, str]:
    try:
        import torch

        torch_version = str(torch.__version__)
    except Exception:
        torch_version = "unknown"
    return {
        "python_version": platform.python_version(),
        "torch_version": torch_version,
        "platform": platform.platform(),
    }


@dataclass(frozen=True)
class CheckpointManifest:
    """Machine-readable description of the architecture and provenance a checkpoint belongs to."""

    format_version: int = 2
    chimera_version: str = "2.0.0"
    transport: str = "se3_schrodinger_bridge"
    sequence_policy: str = "autoregressive"
    geometry_edge_dim: int = 28
    objective_count: int = 5
    uncertainty: str = "mc_dropout_epistemic"
    acquisition: str = "gaussian_expected_improvement"
    config_hash: str = ""
    state_schema_hash: str = ""
    git_commit: str = ""
    dataset_manifest: str = ""
    dataset_hash: str = ""
    preprocessing_hash: str = ""
    environment_hash: str = ""
    python_version: str = ""
    torch_version: str = ""
    platform: str = ""

    def __post_init__(self) -> None:
        runtime = _runtime_snapshot()
        if not self.python_version:
            object.__setattr__(self, "python_version", runtime["python_version"])
        if not self.torch_version:
            object.__setattr__(self, "torch_version", runtime["torch_version"])
        if not self.platform:
            object.__setattr__(self, "platform", runtime["platform"])
        if not self.environment_hash:
            payload = json.dumps(
                {
                    "python_version": self.python_version,
                    "torch_version": self.torch_version,
                    "platform": self.platform,
                    "transport": self.transport,
                    "sequence_policy": self.sequence_policy,
                    "chimera_version": self.chimera_version,
                    "dataset_manifest": self.dataset_manifest,
                    "dataset_hash": self.dataset_hash,
                    "preprocessing_hash": self.preprocessing_hash,
                    "git_commit": self.git_commit,
                },
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            )
            object.__setattr__(self, "environment_hash", hashlib.sha256(payload.encode("utf-8")).hexdigest())

    def validate(self, expected: "CheckpointManifest" | None = None) -> None:
        if self.format_version not in (1, 2):
            raise ValueError(f"unsupported checkpoint format_version={self.format_version}")
        if self.transport != "se3_schrodinger_bridge":
            raise ValueError("checkpoint does not declare the canonical SE(3) SB transport")
        if self.sequence_policy != "autoregressive":
            raise ValueError("CHIMERA sequence policy must be autoregressive")
        if self.geometry_edge_dim != 28:
            raise ValueError("checkpoint geometry contract must use 28-D ProteinMPNN-inspired edges")
        if self.objective_count != 5:
            raise ValueError("CHIMERA expects five objective channels")
        if self.uncertainty != "mc_dropout_epistemic":
            raise ValueError("checkpoint uncertainty contract must be MC-dropout epistemic")
        if self.acquisition != "gaussian_expected_improvement":
            raise ValueError("checkpoint acquisition contract must be Gaussian expected improvement")
        for name in (
            "config_hash",
            "state_schema_hash",
            "git_commit",
            "dataset_manifest",
            "dataset_hash",
            "preprocessing_hash",
            "environment_hash",
            "python_version",
            "torch_version",
            "platform",
        ):
            value = getattr(self, name)
            if not isinstance(value, str):
                raise ValueError(f"{name} must be a string")
        if expected is not None:
            expected_dict = asdict(expected)
            actual_dict = asdict(self)
            # A v1 manifest can be compared to a v2 expected contract only when
            # the newly introduced fingerprints are intentionally unspecified.
            if self.format_version == 1:
                for key in (
                    "config_hash",
                    "state_schema_hash",
                    "git_commit",
                    "dataset_manifest",
                    "dataset_hash",
                    "preprocessing_hash",
                    "environment_hash",
                    "python_version",
                    "torch_version",
                    "platform",
                ):
                    actual_dict.pop(key, None)
                    expected_dict.pop(key, None)
                expected_dict["format_version"] = 1
            if actual_dict != expected_dict:
                raise ValueError("checkpoint manifest does not match the expected architecture contract")


def state_schema_hash(state_dict: Mapping[str, Any]) -> str:
    """Hash parameter names, tensor shapes and dtypes without hashing tensor data."""
    entries = []
    for name in sorted(state_dict):
        value = state_dict[name]
        if hasattr(value, "shape") and hasattr(value, "dtype"):
            entries.append((name, tuple(value.shape), str(value.dtype)))
        else:
            entries.append((name, type(value).__name__))
    payload = json.dumps(entries, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def config_hash(config: Mapping[str, Any]) -> str:
    """Stable SHA-256 fingerprint for a JSON-serializable model configuration."""
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def save_manifest(path: str | Path, manifest: CheckpointManifest | None = None) -> None:
    manifest = manifest or CheckpointManifest()
    manifest.validate()
    Path(path).write_text(json.dumps(asdict(manifest), indent=2) + "\n", encoding="utf-8")


def load_manifest(path: str | Path) -> CheckpointManifest:
    data: Dict[str, Any] = json.loads(Path(path).read_text(encoding="utf-8"))
    manifest = CheckpointManifest(**data)
    manifest.validate()
    return manifest


def validate_checkpoint_compatibility(
    actual: CheckpointManifest,
    expected: CheckpointManifest,
) -> str:
    """Return a compatibility verdict for checkpoint provenance validation."""
    if actual == expected:
        return "exact-compatible"
    try:
        actual.validate(expected)
        return "expected-compatible"
    except ValueError:
        return "incompatible"
