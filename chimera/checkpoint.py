"""Versioned checkpoint manifests for merge-safe CHIMERA experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping

import hashlib
import json
import platform
from pathlib import Path
import subprocess


@dataclass(frozen=True)
class CheckpointManifest:
    """Machine-readable description of the architecture and provenance a checkpoint belongs to."""

    format_version: int = 3
    chimera_version: str = "2.0.0"
    transport: str = "se3_schrodinger_bridge"
    sequence_policy: str = "autoregressive"
    geometry_edge_dim: int = 28
    objective_count: int = 5
    objective_schema_hash: str | None = None
    uncertainty: str = "mc_dropout_epistemic"
    acquisition: str = "gaussian_expected_improvement"
    config_hash: str | None = None
    state_schema_hash: str | None = None
    git_commit: str | None = None
    git_worktree_clean: bool | None = None
    source_tree_hash: str | None = None
    dataset_manifest: str | None = None
    dataset_version: str | None = None
    dataset_hash: str | None = None
    preprocessing_hash: str | None = None
    environment_hash: str | None = None
    python_version: str | None = None
    torch_version: str | None = None
    platform: str | None = None
    training_regime: str | None = None
    random_seed: int | None = None

    def has_complete_provenance(self) -> bool:
        required = (
            "config_hash",
            "state_schema_hash",
            "objective_schema_hash",
            "git_commit",
            "source_tree_hash",
            "dataset_manifest",
            "dataset_version",
            "dataset_hash",
            "preprocessing_hash",
            "environment_hash",
            "python_version",
            "torch_version",
            "platform",
            "training_regime",
        )
        return all(getattr(self, name) for name in required) and self.git_worktree_clean is not None

    def validate(self, expected: "CheckpointManifest" | None = None) -> None:
        if self.format_version not in (1, 2, 3):
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
            "config_hash", "state_schema_hash", "objective_schema_hash", "git_commit", "source_tree_hash", "dataset_manifest",
            "dataset_version", "dataset_hash", "preprocessing_hash", "environment_hash",
            "python_version", "torch_version", "platform", "training_regime",
        ):
            value = getattr(self, name)
            if value is not None and not isinstance(value, str):
                raise ValueError(f"{name} must be a string or None")
        if self.random_seed is not None and (not isinstance(self.random_seed, int) or self.random_seed < 0):
            raise ValueError("random_seed must be a non-negative integer or None")
        if self.git_worktree_clean is not None and not isinstance(self.git_worktree_clean, bool):
            raise ValueError("git_worktree_clean must be a boolean or None")
        if expected is not None:
            expected_dict = asdict(expected)
            actual_dict = asdict(self)
            if self.format_version in (1, 2):
                unavailable_fields = (
                    (
                        "config_hash", "state_schema_hash", "objective_schema_hash", "git_commit",
                        "git_worktree_clean", "source_tree_hash", "dataset_manifest", "dataset_version",
                        "dataset_hash", "preprocessing_hash", "environment_hash", "python_version",
                        "torch_version", "platform", "training_regime", "random_seed",
                    )
                    if self.format_version == 1
                    else (
                        "objective_schema_hash", "dataset_version", "training_regime", "random_seed",
                        "git_worktree_clean", "source_tree_hash",
                    )
                )
                for key in unavailable_fields:
                    actual_dict.pop(key, None)
                    expected_dict.pop(key, None)
                expected_dict["format_version"] = self.format_version
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


def runtime_provenance() -> dict[str, str | None]:
    try:
        import torch

        torch_version = str(torch.__version__)
    except Exception:
        torch_version = None
    result = {
        "python_version": platform.python_version(),
        "torch_version": torch_version,
        "platform": platform.platform(),
    }
    result["environment_hash"] = (
        config_hash(result) if all(result.values()) else None
    )
    return result


def git_provenance(repository_root: str | Path) -> dict[str, Any]:
    """Return commit and source-tree identity without inventing missing Git metadata."""
    root = Path(repository_root).resolve()
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=root,
            text=True,
            stderr=subprocess.DEVNULL,
        )
        paths = subprocess.check_output(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
            cwd=root,
            stderr=subprocess.DEVNULL,
        ).decode("utf-8").split("\0")
    except (OSError, subprocess.CalledProcessError):
        return {"git_commit": None, "git_worktree_clean": None, "source_tree_hash": None}

    source_extensions = {".ini", ".json", ".md", ".py", ".pyi", ".sh", ".toml", ".txt", ".yaml", ".yml"}
    source_files = {}
    try:
        for relative in sorted(set(paths) - {""}):
            path = root / relative
            if path.suffix.lower() in source_extensions and path.is_file():
                source_files[relative.replace("\\", "/")] = hashlib.sha256(path.read_bytes()).hexdigest()
        tree_hash = config_hash(source_files)
    except OSError:
        tree_hash = None
    return {
        "git_commit": commit or None,
        "git_worktree_clean": not bool(status.strip()),
        "source_tree_hash": tree_hash,
    }


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
    try:
        actual.validate()
        expected.validate()
    except (TypeError, ValueError):
        return "incompatible"
    if actual == expected:
        return "exact-compatible" if actual.has_complete_provenance() else "unverified-provenance"

    if actual.format_version == expected.format_version == 3:
        actual_dict = asdict(actual)
        expected_dict = asdict(expected)
        for field in ("git_commit", "git_worktree_clean", "source_tree_hash"):
            actual_dict.pop(field)
            expected_dict.pop(field)
        if actual_dict == expected_dict:
            if actual.has_complete_provenance() and expected.has_complete_provenance():
                return "expected-compatible"
            return "unverified-provenance"
    try:
        actual.validate(expected)
    except ValueError:
        return "incompatible"
    return "expected-compatible" if actual.has_complete_provenance() else "unverified-provenance"
