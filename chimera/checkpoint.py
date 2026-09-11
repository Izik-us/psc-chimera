"""Versioned checkpoint manifests for merge-safe CHIMERA experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping

import hashlib
import json
from pathlib import Path


@dataclass(frozen=True)
class CheckpointManifest:
    """Machine-readable description of the architecture a checkpoint belongs to."""

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
        for name in ("config_hash", "state_schema_hash", "git_commit"):
            value = getattr(self, name)
            if not isinstance(value, str):
                raise ValueError(f"{name} must be a string")
        if expected is not None:
            expected_dict = asdict(expected)
            actual_dict = asdict(self)
            # A v1 manifest can be compared to a v2 expected contract only when
            # the newly introduced fingerprints are intentionally unspecified.
            if self.format_version == 1:
                actual_dict.pop("config_hash", None)
                actual_dict.pop("state_schema_hash", None)
                actual_dict.pop("git_commit", None)
                expected_dict.pop("config_hash", None)
                expected_dict.pop("state_schema_hash", None)
                expected_dict.pop("git_commit", None)
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
