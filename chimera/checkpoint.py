"""Versioned checkpoint manifests for merge-safe CHIMERA experiments."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Any

import json
from pathlib import Path


@dataclass(frozen=True)
class CheckpointManifest:
    """Machine-readable description of the architecture a checkpoint belongs to."""

    format_version: int = 1
    chimera_version: str = "2.0.0"
    transport: str = "se3_schrodinger_bridge"
    sequence_policy: str = "autoregressive"
    geometry_edge_dim: int = 28
    objective_count: int = 5
    uncertainty: str = "mc_dropout_epistemic"
    acquisition: str = "gaussian_expected_improvement"

    def validate(self, expected: "CheckpointManifest" | None = None) -> None:
        if self.format_version != 1:
            raise ValueError(f"unsupported checkpoint format_version={self.format_version}")
        if self.transport != "se3_schrodinger_bridge":
            raise ValueError("checkpoint does not declare the canonical SE(3) SB transport")
        if self.sequence_policy != "autoregressive":
            raise ValueError("CHIMERA sequence policy must be autoregressive")
        if self.geometry_edge_dim != 28:
            raise ValueError("checkpoint geometry contract must use 28-D ProteinMPNN-inspired edges")
        if self.objective_count != 5:
            raise ValueError("CHIMERA expects five objective channels")
        if expected is not None and asdict(self) != asdict(expected):
            raise ValueError("checkpoint manifest does not match the expected architecture contract")


def save_manifest(path: str | Path, manifest: CheckpointManifest | None = None) -> None:
    manifest = manifest or CheckpointManifest()
    manifest.validate()
    Path(path).write_text(json.dumps(asdict(manifest), indent=2) + "\n", encoding="utf-8")


def load_manifest(path: str | Path) -> CheckpointManifest:
    data: Dict[str, Any] = json.loads(Path(path).read_text(encoding="utf-8"))
    manifest = CheckpointManifest(**data)
    manifest.validate()
    return manifest
